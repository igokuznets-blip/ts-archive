"""HTTP API архива: справочники, таймлайн, позиционирование, HLS-плейлисты поверх исходных TS,
раздача фрагментов по Range, выгрузки, статистика."""
from __future__ import annotations

import logging
import shutil
import tempfile
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, PlainTextResponse, Response
from psycopg.types.json import Jsonb
from pydantic import BaseModel, Field

from .. import hls, media
from ..indexer import concat_list
from ..config import settings
from ..db import tx

log = logging.getLogger(__name__)
app = FastAPI(title="TS Archive API", version="1.0.0", docs_url="/api/docs", openapi_url="/api/openapi.json", redoc_url=None)


# ---------------------------------------------------------------- helpers
def actor(request: Request) -> str:
    return request.headers.get("x-remote-user") or "anonymous"


def audit(conn, request: Request, action: str, details: dict) -> None:
    conn.execute("INSERT INTO audit_log(actor, action, details) VALUES (%s,%s,%s)",
                 (actor(request), action, Jsonb(details)))


def serve_file(rel: str, media_type: str, download_name: str | None = None,
               cache: str | None = None) -> Response:
    """Отдаёт файл архива. За Nginx — через X-Accel-Redirect: Range-запросы, sendfile и кэш
    обрабатывает Nginx. Без Nginx — FileResponse (тоже поддерживает Range)."""
    path = (settings.archive_root / rel).resolve()
    if settings.archive_root.resolve() not in path.parents:
        raise HTTPException(400, "недопустимый путь")
    headers = {}
    if cache:
        headers["Cache-Control"] = cache
    if download_name:
        headers["Content-Disposition"] = f'attachment; filename="{download_name}"'
    if settings.use_x_accel:
        headers["X-Accel-Redirect"] = settings.x_accel_prefix + rel
        return Response(status_code=200, media_type=media_type, headers=headers)
    if not path.exists():
        raise HTTPException(404, "файл не найден")
    return FileResponse(path, media_type=media_type, headers=headers)


def iso(dt: datetime | None) -> str | None:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z") if dt else None


def wall_to_offset(conn, seg_id: int, t: datetime) -> float:
    """Абсолютное время → позиция на шкале сегмента (плейлиста) с учётом мелких разрывов."""
    f = conn.execute(
        "SELECT offset_in_segment AS off, ts_start, duration FROM fragment "
        "WHERE segment_id=%s AND ts_start <= %s ORDER BY ts_start DESC LIMIT 1", (seg_id, t)).fetchone()
    if not f:
        return 0.0
    return f["off"] + min(f["duration"], max(0.0, (t - f["ts_start"]).total_seconds()))


def segment_media_urls(seg: dict) -> dict:
    """hls — плейлист поверх исходных TS (основной режим); video — прокси MP4 (только для
    кодеков, которые браузер не воспроизводит)."""
    proxy = bool(seg.get("proxy_path"))
    return {
        "hls": None if proxy else f"/api/segments/{seg['id']}/playlist.m3u8",
        "video": f"/api/segments/{seg['id']}/video" if proxy else None,
        "sprite": f"/api/segments/{seg['id']}/sprite" if seg.get("sprite_path") else None,
    }


def detection_json(d: dict) -> dict:
    out = {"id": d["id"], "segment_id": d["segment_id"], "label": d["label"], "start": iso(d["ts_start"]),
           "end": iso(d["ts_end"]), "count": d["max_count"], "score": round(float(d["peak_score"]), 2),
           "zones": d["zones"] or []}
    if "thumb" in d:
        out["thumb"] = f"/api/detections/{d['id']}/thumb" if d["thumb"] else None
    if "off_start" in d:
        out["off_start"] = round(d["off_start"], 3)
        out["off_end"] = round(d["off_end"], 3)
    return out


# ---------------------------------------------------------------- справочники
@app.get("/api/health")
def health():
    with tx() as conn:
        conn.execute("SELECT 1")
    return {"status": "ok"}


@app.get("/api/vehicles")
def vehicles():
    with tx() as conn:
        rows = conn.execute(
            """SELECT v.id AS vehicle_id, v.code AS vehicle_code, v.name AS vehicle_name, v.type,
                      v.pipeline_section, c.id AS camera_id, c.code AS camera_code, c.name AS camera_name,
                      r.data_from, r.data_to, r.segments
               FROM vehicle v JOIN camera c ON c.vehicle_id=v.id
               LEFT JOIN LATERAL (SELECT min(ts_start) data_from, max(ts_end) data_to, count(*) segments
                                  FROM segment s WHERE s.camera_id=c.id AND s.status='done') r ON true
               ORDER BY v.pipeline_section NULLS LAST, v.code, c.code""").fetchall()
    out: dict[int, dict] = {}
    for r in rows:
        v = out.setdefault(r["vehicle_id"], {
            "id": r["vehicle_id"], "code": r["vehicle_code"], "name": r["vehicle_name"], "type": r["type"],
            "pipeline_section": r["pipeline_section"], "cameras": []})
        v["cameras"].append({"id": r["camera_id"], "code": r["camera_code"], "name": r["camera_name"],
                             "data_from": iso(r["data_from"]), "data_to": iso(r["data_to"]),
                             "segments": r["segments"]})
    return list(out.values())


class VehiclePatch(BaseModel):
    name: str | None = None
    type: str | None = None
    pipeline_section: str | None = None


@app.patch("/api/vehicles/{vehicle_id}")
def patch_vehicle(vehicle_id: int, body: VehiclePatch, request: Request):
    data = body.model_dump(exclude_unset=True)
    if not data:
        return {"updated": 0}
    sets = ", ".join(f"{k}=%s" for k in data)
    with tx() as conn:
        n = conn.execute(f"UPDATE vehicle SET {sets} WHERE id=%s", (*data.values(), vehicle_id)).rowcount
        audit(conn, request, "vehicle.update", {"id": vehicle_id, **data})
    return {"updated": n}


class CameraPatch(BaseModel):
    name: str | None = None


@app.patch("/api/cameras/{camera_id}")
def patch_camera(camera_id: int, body: CameraPatch, request: Request):
    with tx() as conn:
        n = conn.execute("UPDATE camera SET name=%s WHERE id=%s", (body.name, camera_id)).rowcount
        audit(conn, request, "camera.update", {"id": camera_id, "name": body.name})
    return {"updated": n}


# ---------------------------------------------------------------- таймлайн и позиционирование
@app.get("/api/timeline")
def timeline(vehicle_id: int | None = None, camera_id: list[int] | None = Query(None),
             ts_from: datetime = Query(..., alias="from"), ts_to: datetime = Query(..., alias="to"),
             limit: int = Query(20000, le=100000)):
    """Покрытие записью по камерам за интервал: сегменты, статусы, разрывы, события."""
    if ts_to <= ts_from:
        raise HTTPException(400, "to должен быть больше from")
    with tx() as conn:
        if camera_id:
            cams = conn.execute("SELECT id, code, name, vehicle_id FROM camera WHERE id = ANY(%s) ORDER BY code",
                                (camera_id,)).fetchall()
        elif vehicle_id:
            cams = conn.execute("SELECT id, code, name, vehicle_id FROM camera WHERE vehicle_id=%s ORDER BY code",
                                (vehicle_id,)).fetchall()
        else:
            raise HTTPException(400, "нужен vehicle_id или camera_id")
        result = []
        for c in cams:
            segs = conn.execute(
                """SELECT id, ts_start, ts_end, duration, status, gap_total_sec, fragments_count, detect_status
                   FROM segment WHERE camera_id=%s
                     AND tstzrange(ts_start, ts_end, '[]') && tstzrange(%s, %s, '[]')
                   ORDER BY ts_start LIMIT %s""", (c["id"], ts_from, ts_to, limit)).fetchall()
            events = conn.execute(
                "SELECT id, ts, kind, note, author FROM event WHERE camera_id=%s AND ts BETWEEN %s AND %s "
                "ORDER BY ts LIMIT 5000", (c["id"], ts_from, ts_to)).fetchall()
            dets = conn.execute(
                "SELECT id, segment_id, label, ts_start, ts_end, max_count, peak_score, zones FROM detection "
                "WHERE camera_id=%s AND tstzrange(ts_start, ts_end, '[]') && tstzrange(%s, %s, '[]') "
                "ORDER BY ts_start LIMIT 20000", (c["id"], ts_from, ts_to)).fetchall()
            pending = conn.execute(
                "SELECT count(*) AS n FROM fragment WHERE camera_id=%s AND status IN ('new','probing','probed') "
                "AND ts_start BETWEEN %s AND %s", (c["id"], ts_from, ts_to)).fetchone()["n"]
            result.append({
                "camera_id": c["id"], "code": c["code"], "name": c["name"], "pending_fragments": pending,
                "segments": [{"id": s["id"], "start": iso(s["ts_start"]), "end": iso(s["ts_end"]),
                              "duration": s["duration"], "status": s["status"],
                              "gap_total_sec": round(s["gap_total_sec"], 3),
                              "fragments": s["fragments_count"],
                              "detect": s["detect_status"]} for s in segs],
                "detections": [detection_json(d) for d in dets],
                "events": [{"id": e["id"], "ts": iso(e["ts"]), "kind": e["kind"], "note": e["note"],
                            "author": e["author"]} for e in events],
            })
    return {"from": iso(ts_from), "to": iso(ts_to), "cameras": result}


@app.get("/api/segments/{seg_id}")
def segment(seg_id: int, request: Request):
    """Сегмент + карта фрагментов [offset, start_ms, duration] для точного пересчёта
    позиции плеера в абсолютное время и обратно."""
    with tx() as conn:
        s = conn.execute(
            """SELECT s.*, c.code AS camera_code, v.code AS vehicle_code FROM segment s
               JOIN camera c ON c.id=s.camera_id JOIN vehicle v ON v.id=c.vehicle_id WHERE s.id=%s""",
            (seg_id,)).fetchone()
        if not s:
            raise HTTPException(404, "сегмент не найден")
        frags = conn.execute(
            "SELECT offset_in_segment AS off, ts_start, duration FROM fragment WHERE segment_id=%s "
            "ORDER BY offset_in_segment", (seg_id,)).fetchall()
        if s["status"] == "done" and s["detect_status"] == "pending" and s["detect_priority"] < 10:
            # сегмент открыли в интерфейсе — разметить его раньше архива
            conn.execute("UPDATE segment SET detect_priority=10 WHERE id=%s", (seg_id,))
        audit(conn, request, "segment.view", {"segment_id": seg_id, "camera_id": s["camera_id"]})
    return {
        "id": s["id"], "camera_id": s["camera_id"], "camera_code": s["camera_code"],
        "vehicle_code": s["vehicle_code"], "start": iso(s["ts_start"]), "end": iso(s["ts_end"]),
        "duration": s["duration"], "status": s["status"], "error": s["error"],
        "vcodec": s["vcodec"], "width": s["width"], "height": s["height"],
        "acodec": s["codec_sig"].split("|a:")[1].split(":")[0] if s["codec_sig"] and "|a:" in s["codec_sig"] else None,
        "proxy_transcoded": s["proxy_transcoded"], "thumbs": s["thumbs"],
        "fragments": [[round(f["off"], 4), int(f["ts_start"].timestamp() * 1000), round(f["duration"], 4)]
                      for f in frags],
        "urls": segment_media_urls(s),
        "detect": {"status": s["detect_status"], "model": s["detect_model"], "error": s["detect_error"],
                   "boxes": f"/api/segments/{s['id']}/boxes" if s["detect_status"] == "done" and s["detect_path"] else None},
    }


@app.get("/api/segments/{seg_id}/playlist.m3u8")
def segment_playlist(seg_id: int):
    """HLS VOD-плейлист сегмента: куски исходных TS-файлов по EXT-X-BYTERANGE."""
    with tx() as conn:
        s = conn.execute("SELECT duration, status FROM segment WHERE id=%s", (seg_id,)).fetchone()
        if not s or s["status"] != "done":
            raise HTTPException(404, "сегмент не готов")
        frags = conn.execute(
            "SELECT id, offset_in_segment AS off, size_bytes, start_pts, kf_t, kf_pos FROM fragment "
            "WHERE segment_id=%s ORDER BY offset_in_segment", (seg_id,)).fetchall()
    if not frags:
        raise HTTPException(404, "у сегмента нет фрагментов")
    text = hls.build_playlist(frags, s["duration"], lambda f: f"/api/fragments/{f['id']}/media",
                              settings.hls_chunk_sec)
    return Response(text, media_type="application/vnd.apple.mpegurl",
                    headers={"Cache-Control": f"private, max-age={settings.playlist_cache_sec}"})


@app.api_route("/api/fragments/{frag_id}/media", methods=["GET", "HEAD"])
def fragment_media(frag_id: int):
    """Исходный TS-фрагмент. Плеер запрашивает его кусками (Range); за Nginx байты отдаёт Nginx."""
    with tx() as conn:
        f = conn.execute("SELECT rel_path FROM fragment WHERE id=%s AND status='grouped'", (frag_id,)).fetchone()
    if not f:
        raise HTTPException(404, "фрагмент не найден")
    # оригиналы неизменны — кэш браузера безопасен
    return serve_file(f["rel_path"], "video/mp2t", cache="private, max-age=86400, immutable")


@app.get("/api/fragments/{frag_id}/download")
def fragment_download(frag_id: int, request: Request):
    """Скачать исходный фрагмент целиком (байт в байт, sha256 как при приёме)."""
    with tx() as conn:
        f = conn.execute("SELECT rel_path, orig_name FROM fragment WHERE id=%s AND status='grouped'",
                         (frag_id,)).fetchone()
        if not f:
            raise HTTPException(404, "фрагмент не найден")
        audit(conn, request, "fragment.download", {"fragment_id": frag_id})
    return serve_file(f["rel_path"], "video/mp2t", Path(f["orig_name"]).name)


@app.api_route("/api/segments/{seg_id}/video", methods=["GET", "HEAD"])
def segment_video(seg_id: int):
    """Прокси H.264 (только для сегментов с кодеком, который браузер не воспроизводит)."""
    with tx() as conn:
        s = conn.execute("SELECT proxy_path FROM segment WHERE id=%s AND status='done'", (seg_id,)).fetchone()
    if not s or not s["proxy_path"]:
        raise HTTPException(404, "прокси для сегмента не строился — используйте playlist.m3u8")
    return serve_file(s["proxy_path"], "video/mp4")


@app.api_route("/api/segments/{seg_id}/sprite", methods=["GET", "HEAD"])
def segment_sprite(seg_id: int):
    with tx() as conn:
        s = conn.execute("SELECT sprite_path FROM segment WHERE id=%s", (seg_id,)).fetchone()
    if not s or not s["sprite_path"]:
        raise HTTPException(404, "нет превью")
    return serve_file(s["sprite_path"], "image/jpeg")


@app.get("/api/seek")
def seek(camera_id: int, ts: datetime, direction: str = Query("forward", pattern="^(forward|backward)$")):
    """Момент времени → сегмент и смещение. Если в момент ts записи нет — ближайший сегмент."""
    with tx() as conn:
        s = conn.execute(
            "SELECT id, ts_start, ts_end FROM segment WHERE camera_id=%s AND status='done' "
            "AND ts_start <= %s AND ts_end >= %s ORDER BY ts_start DESC LIMIT 1", (camera_id, ts, ts)).fetchone()
        exact = s is not None
        if not s:
            if direction == "forward":
                s = conn.execute("SELECT id, ts_start, ts_end FROM segment WHERE camera_id=%s AND status='done' "
                                 "AND ts_start > %s ORDER BY ts_start LIMIT 1", (camera_id, ts)).fetchone()
            else:
                s = conn.execute("SELECT id, ts_start, ts_end FROM segment WHERE camera_id=%s AND status='done' "
                                 "AND ts_end < %s ORDER BY ts_end DESC LIMIT 1", (camera_id, ts)).fetchone()
        if not s:
            raise HTTPException(404, "записи нет")
        t = ts if exact else (s["ts_start"] if direction == "forward" else s["ts_end"])
        offset = wall_to_offset(conn, s["id"], t)
    return {"segment_id": s["id"], "offset": round(offset, 3), "exact": exact, "ts": iso(t),
            "segment_start": iso(s["ts_start"]), "segment_end": iso(s["ts_end"])}


# ---------------------------------------------------------------- события
class EventIn(BaseModel):
    camera_id: int
    ts: datetime
    kind: str = Field("mark", max_length=64)
    note: str | None = Field(None, max_length=2000)


@app.post("/api/events")
def add_event(body: EventIn, request: Request):
    with tx() as conn:
        eid = conn.execute("INSERT INTO event(camera_id, ts, kind, note, author) VALUES (%s,%s,%s,%s,%s) "
                           "RETURNING id", (body.camera_id, body.ts, body.kind, body.note, actor(request))
                           ).fetchone()["id"]
        audit(conn, request, "event.create", {"id": eid, "camera_id": body.camera_id})
    return {"id": eid}


@app.delete("/api/events/{event_id}")
def delete_event(event_id: int, request: Request):
    with tx() as conn:
        n = conn.execute("DELETE FROM event WHERE id=%s", (event_id,)).rowcount
        audit(conn, request, "event.delete", {"id": event_id})
    return {"deleted": n}


# ---------------------------------------------------------------- разметка (люди в кадре)
@app.get("/api/detections")
def detections(camera_id: list[int] = Query(...), ts_from: datetime = Query(..., alias="from"),
               ts_to: datetime = Query(..., alias="to"), label: str = "person", zone: str | None = None,
               min_count: int = 1, limit: int = Query(500, le=5000)):
    """Поиск интервалов с людьми: по камерам, времени, зоне и числу людей (с миниатюрами)."""
    with tx() as conn:
        rows = conn.execute(
            """SELECT d.*, c.code AS camera_code, v.code AS vehicle_code FROM detection d
               JOIN camera c ON c.id=d.camera_id JOIN vehicle v ON v.id=c.vehicle_id
               WHERE d.camera_id = ANY(%s) AND d.label=%s AND d.max_count >= %s
                 AND tstzrange(d.ts_start, d.ts_end, '[]') && tstzrange(%s, %s, '[]')
                 AND (%s::text IS NULL OR %s = ANY(d.zones))
               ORDER BY d.ts_start DESC LIMIT %s""",
            (camera_id, label, min_count, ts_from, ts_to, zone, zone, limit)).fetchall()
    return [{**detection_json(r), "camera_id": r["camera_id"], "camera_code": r["camera_code"],
             "vehicle_code": r["vehicle_code"]} for r in rows]


@app.api_route("/api/detections/{det_id}/thumb", methods=["GET", "HEAD"])
def detection_thumb(det_id: int):
    with tx() as conn:
        d = conn.execute("SELECT thumb FROM detection WHERE id=%s", (det_id,)).fetchone()
    if not d or not d["thumb"]:
        raise HTTPException(404, "нет миниатюры")
    return serve_file(d["thumb"], "image/jpeg", cache="private, max-age=3600")


@app.api_route("/api/segments/{seg_id}/boxes", methods=["GET", "HEAD"])
def segment_boxes(seg_id: int):
    """Рамки по кадрам для наложения в плеере: {fps, labels, frames: [[t, [[x1,y1,x2,y2,score,label,zone],...]],...]}
    t — позиция на шкале сегмента (video.currentTime), координаты — доли кадра."""
    with tx() as conn:
        s = conn.execute("SELECT detect_path FROM segment WHERE id=%s AND detect_status='done'", (seg_id,)).fetchone()
    if not s or not s["detect_path"]:
        raise HTTPException(404, "сегмент ещё не размечен")
    return serve_file(f"{s['detect_path']}/boxes.json", "application/json", cache="private, max-age=300")


class DetectRequest(BaseModel):
    priority: int = Field(20, ge=0, le=100)


@app.post("/api/segments/{seg_id}/detect")
def segment_detect(seg_id: int, body: DetectRequest, request: Request):
    """Поставить сегмент на (пере)разметку с приоритетом (например, после смены модели или зон)."""
    from ..detect_queue import requeue
    n = requeue(seg_id=seg_id, priority=body.priority)
    with tx() as conn:
        audit(conn, request, "segment.detect", {"segment_id": seg_id, "priority": body.priority})
    return {"queued": n}


class DetectConfig(BaseModel):
    enabled: bool = True
    ignore: list[list[list[float]]] = []
    zones: dict[str, list[list[float]]] = {}
    min_conf: float | None = Field(None, ge=0.05, le=0.99)
    min_height: float | None = Field(None, ge=0, le=1)
    helmet: bool | None = None  # проверять каски на этой камере (по умолчанию да, если есть модель)
    requeue: bool = False  # пересчитать архив камеры с новыми зонами


@app.get("/api/cameras/{camera_id}/detect-config")
def get_detect_config(camera_id: int):
    with tx() as conn:
        c = conn.execute("SELECT detect_enabled, detect_config FROM camera WHERE id=%s", (camera_id,)).fetchone()
    if not c:
        raise HTTPException(404, "камера не найдена")
    return {"enabled": c["detect_enabled"], **(c["detect_config"] or {})}


@app.put("/api/cameras/{camera_id}/detect-config")
def put_detect_config(camera_id: int, body: DetectConfig, request: Request):
    """Зоны и исключения камеры (координаты — доли кадра 0..1, многоугольники)."""
    cfg = body.model_dump(exclude={"enabled", "requeue"}, exclude_none=True)
    for poly in [*cfg.get("ignore", []), *cfg.get("zones", {}).values()]:
        if len(poly) < 3 or any(len(pt) != 2 or not (0 <= pt[0] <= 1 and 0 <= pt[1] <= 1) for pt in poly):
            raise HTTPException(400, "многоугольник: не меньше 3 точек [x, y] в долях кадра 0..1")
    with tx() as conn:
        n = conn.execute("UPDATE camera SET detect_enabled=%s, detect_config=%s WHERE id=%s",
                         (body.enabled, Jsonb(cfg), camera_id)).rowcount
        audit(conn, request, "camera.detect_config", {"id": camera_id, **cfg, "enabled": body.enabled})
    queued = 0
    if n and body.requeue:
        from ..detect_queue import requeue
        queued = requeue(camera_id=camera_id, priority=1)
    return {"updated": n, "queued": queued}


# ---------------------------------------------------------------- выгрузка
class ExportIn(BaseModel):
    camera_id: int
    ts_from: datetime = Field(alias="from")
    ts_to: datetime = Field(alias="to")


@app.post("/api/exports")
def create_export(body: ExportIn, request: Request):
    """Вырезка интервала прямо из исходных TS в один MP4 без перекодирования видео
    (начало — по ключевому кадру). Через разрывы записи части склеиваются подряд."""
    span = (body.ts_to - body.ts_from).total_seconds()
    if span <= 0:
        raise HTTPException(400, "пустой интервал")
    if span > settings.export_max_sec:
        raise HTTPException(400, f"интервал больше {settings.export_max_sec} с")
    with tx() as conn:
        cam = conn.execute("SELECT c.code, v.code AS vcode FROM camera c JOIN vehicle v ON v.id=c.vehicle_id "
                           "WHERE c.id=%s", (body.camera_id,)).fetchone()
        if not cam:
            raise HTTPException(404, "камера не найдена")
        segs = conn.execute(
            """SELECT id, ts_start, ts_end, duration, codec_sig FROM segment WHERE camera_id=%s AND status='done'
               AND tstzrange(ts_start, ts_end, '[]') && tstzrange(%s, %s, '[]') ORDER BY ts_start""",
            (body.camera_id, body.ts_from, body.ts_to)).fetchall()
        if not segs:
            raise HTTPException(404, "в интервале нет записи")
        cuts = []
        for s in segs:
            a = wall_to_offset(conn, s["id"], max(body.ts_from, s["ts_start"]))
            b = wall_to_offset(conn, s["id"], min(body.ts_to, s["ts_end"]))
            if b - a <= 0.05:
                continue
            frags = conn.execute(
                "SELECT rel_path, offset_in_segment AS off, acodec FROM fragment WHERE segment_id=%s "
                "ORDER BY offset_in_segment", (s["id"],)).fetchall()
            ends = [f["off"] for f in frags[1:]] + [s["duration"]]
            # только фрагменты, пересекающие [a, b]: ffmpeg не читает лишние файлы по 80 МБ
            sel = [(f, e - f["off"]) for f, e in zip(frags, ends) if e > a and f["off"] < b]
            if sel:
                cuts.append((sel, a - sel[0][0]["off"], b - sel[0][0]["off"], sel[0][0]["acodec"]))
    if not cuts:
        raise HTTPException(404, "в интервале нет записи")

    export_id = uuid.uuid4()
    name = f"{cam['vcode']}_{cam['code']}_{body.ts_from.strftime('%Y%m%d_%H%M%S')}_{int(span)}s.mp4"
    rel = Path("exports") / f"{export_id}.mp4"
    out = settings.archive_root / rel
    work = Path(tempfile.mkdtemp(dir=settings.tmp, prefix="export_"))
    try:
        parts = []
        for i, (sel, a, b, acodec) in enumerate(cuts):
            lst = work / f"src{i:04d}.ffconcat"
            concat_list([f for f, _ in sel], [d for _, d in sel], lst)
            p = work / f"part{i:04d}.mp4"
            media.cut_to_mp4(lst, a, b, p, acodec)
            parts.append(p)
        if len(parts) == 1:
            shutil.move(parts[0], out)
        else:
            lst = work / "parts.ffconcat"
            media.write_concat_list(parts, None, lst)
            media.concat_mp4_parts(lst, out)
    except media.MediaError as e:
        raise HTTPException(500, f"ошибка выгрузки: {e}")
    finally:
        shutil.rmtree(work, ignore_errors=True)
    size = out.stat().st_size
    with tx() as conn:
        conn.execute(
            "INSERT INTO export(id, camera_id, ts_from, ts_to, rel_path, size_bytes, author, expires_at) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
            (export_id, body.camera_id, body.ts_from, body.ts_to, rel.as_posix(), size, actor(request),
             datetime.now(timezone.utc) + timedelta(hours=settings.export_ttl_hours)))
        audit(conn, request, "export.create", {"id": str(export_id), "camera_id": body.camera_id,
                                               "from": iso(body.ts_from), "to": iso(body.ts_to)})
    return {"id": str(export_id), "name": name, "size_bytes": size, "segments": len(cuts),
            "url": f"/api/exports/{export_id}/download?name={name}"}


@app.get("/api/exports/{export_id}/download")
def download_export(export_id: uuid.UUID, request: Request, name: str = "export.mp4"):
    with tx() as conn:
        e = conn.execute("SELECT rel_path FROM export WHERE id=%s", (export_id,)).fetchone()
        if not e:
            raise HTTPException(404, "выгрузка не найдена или истекла")
        audit(conn, request, "export.download", {"id": str(export_id)})
    safe = "".join(ch for ch in name if ch.isalnum() or ch in "._-") or "export.mp4"
    return serve_file(e["rel_path"], "video/mp4", safe)


# ---------------------------------------------------------------- эксплуатация
def _stats() -> dict:
    with tx() as conn:
        fr = {r["status"]: r["n"] for r in conn.execute(
            "SELECT status, count(*) n FROM fragment GROUP BY status").fetchall()}
        sg = {r["status"]: r["n"] for r in conn.execute(
            "SELECT status, count(*) n FROM segment GROUP BY status").fetchall()}
        dt = {r["detect_status"]: r["n"] for r in conn.execute(
            "SELECT detect_status, count(*) n FROM segment WHERE status='done' GROUP BY detect_status").fetchall()}
        dperf = conn.execute(
            "SELECT count(*) n, coalesce(sum(duration),0) dur, coalesce(sum(detect_sec),0) sec FROM segment "
            "WHERE detect_status='done' AND detect_done_at > now() - interval '1 day'").fetchone()
        by_label = {r["label"]: r["n"] for r in conn.execute(
            "SELECT label, count(*) n FROM detection GROUP BY label").fetchall()}
        ndet = sum(by_label.values())
        lag = conn.execute(
            "SELECT extract(epoch FROM now() - min(ingested_at)) AS lag FROM fragment "
            "WHERE status IN ('new','probing','probed')").fetchone()["lag"]
        errors = conn.execute(
            "SELECT id, camera_id, ts_start, error FROM segment WHERE status='error' ORDER BY id DESC LIMIT 20"
        ).fetchall()
        sizes = conn.execute("SELECT coalesce(sum(size_bytes),0) AS raw FROM fragment").fetchone()["raw"]
        proxy_size = conn.execute("SELECT coalesce(sum(size_bytes),0) - coalesce(sum(r.raw),0) AS s FROM segment s "
                                  "JOIN LATERAL (SELECT sum(size_bytes) raw FROM fragment f WHERE f.segment_id=s.id) r ON true "
                                  "WHERE s.proxy_path IS NOT NULL").fetchone()["s"]
    du = shutil.disk_usage(settings.archive_root)
    incoming = sum(1 for _ in settings.incoming.rglob("*.ts")) if settings.incoming.exists() else 0
    return {"fragments": fr, "segments": sg, "incoming_files": incoming,
            "detect": {"segments": dt, "detections": ndet, "by_label": by_label,
                       "realtime_factor_24h": round(dperf["dur"] / dperf["sec"], 1) if dperf["sec"] else None},
            "pipeline_lag_sec": float(lag) if lag else 0.0,
            "raw_bytes": int(sizes), "proxy_bytes": int(proxy_size or 0),
            "disk": {"total": du.total, "used": du.used, "free": du.free},
            "recent_errors": [{"segment_id": e["id"], "camera_id": e["camera_id"], "start": iso(e["ts_start"]),
                               "error": e["error"]} for e in errors]}


@app.get("/api/stats")
def stats():
    return _stats()


@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    s = _stats()
    lines = []
    for st, n in s["fragments"].items():
        lines.append(f'tsarchive_fragments{{status="{st}"}} {n}')
    for st, n in s["segments"].items():
        lines.append(f'tsarchive_segments{{status="{st}"}} {n}')
    lines += [f"tsarchive_incoming_files {s['incoming_files']}",
              f"tsarchive_pipeline_lag_seconds {s['pipeline_lag_sec']:.1f}",
              f"tsarchive_raw_bytes {s['raw_bytes']}",
              f"tsarchive_proxy_bytes {s['proxy_bytes']}",
              f"tsarchive_disk_free_bytes {s['disk']['free']}",
              f"tsarchive_detections_total {s['detect']['detections']}",
              *[f'tsarchive_detections{{label="{k}"}} {v}' for k, v in s["detect"]["by_label"].items()]]
    for st, n in s["detect"]["segments"].items():
        lines.append(f'tsarchive_detect_segments{{status="{st}"}} {n}')
    return "\n".join(lines) + "\n"
