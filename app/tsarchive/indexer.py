"""Этап 4. Индексация сегмента: точная временная шкала фрагментов и превью для таймлайна.

Склеенные файлы не создаются: сегмент — это запись в БД, а видео воспроизводится напрямую
из исходных TS через HLS-плейлист с EXT-X-BYTERANGE (см. hls.py). Исключение — кодеки,
которые браузер не воспроизводит (MPEG-2 Video и т.п.): для них при PROXY_TRANSCODE=auto
строится прокси H.264.
"""
from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path

from psycopg.types.json import Jsonb

from . import media
from .config import settings
from .db import tx

log = logging.getLogger(__name__)


def _claim() -> dict | None:
    with tx() as conn:
        return conn.execute(
            """UPDATE segment s SET status='processing', claimed_at=now(), attempts=attempts+1
               WHERE s.id = (SELECT id FROM segment WHERE status='ready'
                             ORDER BY ts_start LIMIT 1 FOR UPDATE SKIP LOCKED)
               RETURNING s.*,
                 (SELECT v.code FROM camera c JOIN vehicle v ON v.id=c.vehicle_id WHERE c.id=s.camera_id) AS vehicle_code,
                 (SELECT c.code FROM camera c WHERE c.id=s.camera_id) AS camera_code""").fetchone()


def segment_fragments(conn, seg_id: int) -> list[dict]:
    return conn.execute(
        "SELECT id, rel_path, size_bytes, duration, start_pts, acodec, ts_start, offset_in_segment "
        "FROM fragment WHERE segment_id=%s ORDER BY ts_start", (seg_id,)).fetchall()


def offsets(durs: list[float]) -> list[float]:
    out, acc = [], 0.0
    for d in durs:
        out.append(acc)
        acc += d
    return out


def concat_list(frags: list[dict], durs: list[float], list_path: Path) -> None:
    media.write_concat_list([settings.archive_root / f["rel_path"] for f in frags], durs, list_path)


def build(seg: dict) -> dict:
    with tx() as conn:
        frags = segment_fragments(conn, seg["id"])
    if not frags:
        raise media.MediaError("у сегмента нет фрагментов")
    durs = media.concat_durations([f["start_pts"] for f in frags], [f["duration"] for f in frags])
    total = sum(durs)

    day = seg["ts_start"].strftime("%Y/%m/%d")
    stem = f"{seg['id']}_{seg['ts_start'].strftime('%H%M%S')}"
    thumb_dir_rel = Path("thumbs") / seg["vehicle_code"] / seg["camera_code"] / day
    (settings.archive_root / thumb_dir_rel).mkdir(parents=True, exist_ok=True)
    sprite_rel = thumb_dir_rel / f"{stem}.jpg"

    transcode = (settings.proxy_transcode == "always" or
                 (settings.proxy_transcode == "auto" and seg["vcodec"] not in settings.browser_codecs))
    proxy_rel = None
    work = Path(tempfile.mkdtemp(dir=settings.tmp, prefix=f"seg{seg['id']}_"))
    try:
        lst = work / "list.ffconcat"
        concat_list(frags, durs, lst)
        if transcode:
            proxy_dir_rel = Path("proxies") / seg["vehicle_code"] / seg["camera_code"] / day
            (settings.archive_root / proxy_dir_rel).mkdir(parents=True, exist_ok=True)
            proxy_rel = proxy_dir_rel / f"{stem}.mp4"
            out = settings.archive_root / proxy_rel
            tmp = out.with_suffix(".mp4.part")
            try:
                media.transcode_proxy(lst, tmp)
                tmp.replace(out)
            finally:
                tmp.unlink(missing_ok=True)
        thumbs = None
        try:
            thumbs = media.make_sprite(lst, settings.archive_root / sprite_rel, total, seg["width"], seg["height"])
        except media.MediaError as e:
            log.warning("сегмент %s: не удалось сделать превью: %s", seg["id"], e)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    size = sum(f["size_bytes"] for f in frags)
    if proxy_rel:
        size += (settings.archive_root / proxy_rel).stat().st_size
    return {"proxy_path": proxy_rel.as_posix() if proxy_rel else None,
            "sprite_path": sprite_rel.as_posix() if thumbs else None, "thumbs": thumbs,
            "size_bytes": size, "proxy_transcoded": transcode, "duration": total,
            "ids": [f["id"] for f in frags], "offsets": offsets(durs)}


def run_once() -> int:
    seg = _claim()
    if not seg:
        return 0
    try:
        out = build(seg)
    except Exception as e:  # noqa: BLE001
        status = "error" if seg["attempts"] >= settings.max_attempts else "ready"
        with tx() as conn:
            conn.execute("UPDATE segment SET status=%s, error=%s, claimed_at=NULL WHERE id=%s",
                         (status, str(e)[:2000], seg["id"]))
        log.error("сегмент %s: ошибка индексации (%s): %s", seg["id"], status, e)
        return 1
    with tx() as conn:
        # точные позиции фрагментов на шкале сегмента — общие для HLS-плейлиста, превью и UI
        conn.execute("""UPDATE fragment AS fr SET offset_in_segment=v.off
                        FROM unnest(%s::bigint[], %s::float8[]) AS v(id, off) WHERE fr.id=v.id""",
                     (out["ids"], out["offsets"]))
        conn.execute(
            """UPDATE segment SET status='done', proxy_path=%s, sprite_path=%s, thumbs=%s, size_bytes=%s,
               proxy_transcoded=%s, duration=%s, error=NULL, claimed_at=NULL, done_at=now(),
               detect_status=CASE WHEN detect_status='processing' THEN detect_status ELSE 'pending' END,
               detect_attempts=0 WHERE id=%s""",
            (out["proxy_path"], out["sprite_path"], Jsonb(out["thumbs"]) if out["thumbs"] else None,
             out["size_bytes"], out["proxy_transcoded"], out["duration"], seg["id"]))
    log.info("сегмент %s готов: %s фрагм., %.1f с%s", seg["id"], seg["fragments_count"], out["duration"],
             f" → прокси {out['proxy_path']}" if out["proxy_path"] else " (воспроизведение из исходных TS)")
    return 1


def reindex(seg_id: int) -> None:
    """Повторная индексация сегмента (CLI)."""
    with tx() as conn:
        conn.execute("UPDATE segment SET status='ready', attempts=0, error=NULL WHERE id=%s", (seg_id,))
