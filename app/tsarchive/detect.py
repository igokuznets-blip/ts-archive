"""Этап 6. Предварительная разметка: интервалы «человек в кадре» по каждому сегменту.

Работает после индексации (segment.status='done') и не мешает просмотру: видео не копируется
и не перекодируется на диск. FFmpeg декодирует исходные TS по тому же списку ffconcat, что и
превью, прореживает кадры (DETECT_FPS) и отдаёт их в OpenVINO уже вписанными во вход модели.

Проверка касок (HELMET_MODE):
  * heads (основной) — детектор голов «в каске / без каски» (heads.py) на том же кадре, что и
    детектор людей. Головы сопоставляются с рамками людей; люди, слитые в одну рамку или видные
    только по голове, добавляются к числу людей в кадре;
  * crop (запасной) — FFmpeg отдаёт второй поток в исходном разрешении (не шире HELMET_MAX_WIDTH),
    по каждой рамке человека вырезается квадрат головы и классифицируется (helmet.py).
Интервалы, где кто-то без каски, сохраняются с label='no_helmet'.

Результат по сегменту:
  * строки таблицы detection — интервалы (начало, конец, макс. число людей, зоны, миниатюра);
  * detections/<техника>/<камера>/<день>/<id>/boxes.json — рамки по кадрам для наложения в плеере;
  * миниатюры лучшего кадра каждого интервала (с рамками) там же.

Очередь — в PostgreSQL (SKIP LOCKED): сначала сегменты с повышенным приоритетом (их открыли в
интерфейсе или поставили на пересчёт), затем новые, затем архив (дозаполнение после включения).
"""
from __future__ import annotations

import json
import logging
import os
import queue as _queue
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path

import numpy as np

from . import media
from .config import settings
from .db import tx
from .detect_queue import requeue  # noqa: F401  (реэкспорт для CLI)
from .detector import COCO, Letterbox, Reorder
from .helmet import H_NO, H_UNKNOWN, H_YES  # noqa: F401

log = logging.getLogger(__name__)
_detector = None
_detector_failed_at = 0.0
_helmet = None
_helmet_checked_at = 0.0
_heads = None
_heads_checked_at = 0.0
NO_HELMET = "no_helmet"


# ---------------------------------------------------------------- геометрия и зоны
def point_in_poly(x: float, y: float, poly: list) -> bool:
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1:
            inside = not inside
    return inside


@dataclass
class CameraRules:
    """Настройки камеры (camera.detect_config, координаты в долях кадра 0..1):
    {"ignore": [[[x,y],...], ...],          — области, где люди не учитываются (оператор в кабине)
     "zones": {"Опасная зона": [[x,y],...]}, — именованные зоны: попадание ног (низ рамки)
     "min_conf": 0.5, "min_height": 0.04,    — переопределение порогов для камеры
     "helmet": true}                          — проверять каски (false — камера в кабине и т. п.)"""
    ignore: list = field(default_factory=list)
    zones: dict = field(default_factory=dict)
    min_conf: float = 0.0
    min_height: float = 0.0
    helmet: bool = True

    @classmethod
    def from_json(cls, cfg: dict | None) -> "CameraRules":
        cfg = cfg or {}
        return cls(ignore=cfg.get("ignore") or [], zones=cfg.get("zones") or {},
                   min_conf=float(cfg.get("min_conf") or settings.detect_conf),
                   min_height=float(cfg.get("min_height") if cfg.get("min_height") is not None
                                    else settings.detect_min_height),
                   helmet=bool(cfg.get("helmet", True)))

    def apply(self, boxes: np.ndarray, scores: np.ndarray) -> list[tuple[list[float], float, list[str]]]:
        """boxes нормированы 0..1. → [(box, score, [зоны])] после фильтров."""
        out = []
        for b, s in zip(boxes, scores):
            if s < self.min_conf or (b[3] - b[1]) < self.min_height:
                continue
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            if any(point_in_poly(cx, cy, p) for p in self.ignore):
                continue
            foot = (cx, b[3])
            zones = [name for name, p in self.zones.items() if point_in_poly(*foot, p)]
            out.append(([round(float(v), 4) for v in b], round(float(s), 3), zones))
        return out


# ---------------------------------------------------------------- интервалы
@dataclass
class Interval:
    label: str
    start: float
    last: float
    hits: int = 0
    max_count: int = 0
    peak: float = 0.0
    best_t: float = 0.0
    best_key: tuple = (0, 0.0)
    best_frame: object = None
    best_boxes: list = field(default_factory=list)
    zones: set = field(default_factory=set)

    def end(self, step: float) -> float:
        return self.last + step


class IntervalBuilder:
    """Срабатывания по кадрам → интервалы. Соседние кадры с объектом объединяются, если пауза
    не больше gap (кратковременные пропуски детектора); короткие интервалы отбрасываются."""

    def __init__(self, fps: float, gap: float, min_dur: float, min_hits: int, on_close=None):
        self.step = 1.0 / fps
        self.gap, self.min_dur, self.min_hits = gap, min_dur, min_hits
        self.open: dict[str, Interval] = {}
        self.done: list[Interval] = []
        self.on_close = on_close

    def feed(self, t: float, label: str, dets: list, frame=None) -> None:
        cur = self.open.get(label)
        if cur and t - cur.last > self.gap + 1e-6:
            self._close(label)
            cur = None
        if not dets:
            return
        if cur is None:
            cur = self.open[label] = Interval(label, start=t, last=t)
        cur.last = t
        cur.hits += 1
        n = len(dets)
        top = max(d[1] for d in dets)
        cur.max_count = max(cur.max_count, n)
        cur.peak = max(cur.peak, top)
        for d in dets:
            cur.zones.update(d[2])
        key = (n, top)
        if key > cur.best_key:  # лучший кадр: больше людей, затем выше уверенность
            cur.best_key, cur.best_t, cur.best_boxes = key, t, dets
            if frame is not None:
                cur.best_frame = frame

    def tick(self, t: float) -> None:
        """Закрыть интервалы, у которых пауза уже больше gap (для потоковой записи миниатюр)."""
        for label in list(self.open):
            if t - self.open[label].last > self.gap + 1e-6:
                self._close(label)

    def _close(self, label: str) -> None:
        iv = self.open.pop(label)
        if iv.hits >= self.min_hits and iv.end(self.step) - iv.start >= self.min_dur - 1e-6:
            self.done.append(iv)
            if self.on_close:
                self.on_close(iv)
        iv.best_frame = None

    def finish(self) -> list[Interval]:
        for label in list(self.open):
            self._close(label)
        return self.done


def offset_to_wall(frags: list[dict], off: float):
    """Позиция на шкале сегмента → абсолютное время (по карте фрагментов)."""
    f = frags[0]
    for x in frags:
        if x["offset_in_segment"] <= off:
            f = x
        else:
            break
    return f["ts_start"] + timedelta(seconds=min(f["duration"] or 0, max(0.0, off - f["offset_in_segment"])))


# ---------------------------------------------------------------- миниатюры
def draw_boxes(frame: np.ndarray, boxes_px: list, colors: list) -> np.ndarray:
    img = frame.copy()
    h, w = img.shape[:2]
    for (x1, y1, x2, y2), col in zip(boxes_px, colors):
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w - 1, int(x2)), min(h - 1, int(y2))
        if x2 <= x1 or y2 <= y1:
            continue
        for d in range(2):  # рамка 2 px
            img[max(0, y1 - d), x1:x2] = col
            img[min(h - 1, y2 + d), x1:x2] = col
            img[y1:y2, max(0, x1 - d)] = col
            img[y1:y2, min(w - 1, x2 + d)] = col
    return img


class ThumbWriter:
    """Один процесс FFmpeg на сегмент: кадры подаются в stdin, на выходе 0001.jpg, 0002.jpg ...
    Запускается при первой миниатюре — в сегментах без людей FFmpeg не стартует."""

    def __init__(self, out_dir: Path, w: int, h: int):
        self.n = 0
        self.out_dir, self.w, self.h = out_dir, w, h
        self.proc = None

    def _start(self):
        tw = settings.detect_thumb_width
        self.proc = subprocess.Popen(
            [settings.ffmpeg_bin, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
             "-pix_fmt", "bgr24", "-s", f"{self.w}x{self.h}", "-r", "1", "-i", "pipe:0",
             "-vf", f"scale={tw}:-2", "-q:v", "4", "-start_number", "1", str(self.out_dir / "%04d.jpg")],
            stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> str:
        if self.proc is None:
            self._start()
        self.proc.stdin.write(np.ascontiguousarray(frame[: self.h, : self.w]).tobytes())
        self.n += 1
        return f"{self.n:04d}.jpg"

    def close(self, check: bool = True) -> None:
        if self.proc is None:
            return
        self.proc.stdin.close()
        err = self.proc.stderr.read()
        if self.proc.wait(timeout=120) != 0 and check:
            raise media.MediaError("ffmpeg (миниатюры): " + err.decode(errors="replace")[-800:])


# ---------------------------------------------------------------- очередь
def _claim() -> dict | None:
    with tx() as conn:
        return conn.execute(
            """UPDATE segment s SET detect_status='processing', detect_claimed_at=now(),
                      detect_attempts=detect_attempts+1
               WHERE s.id = (SELECT g.id FROM segment g JOIN camera c ON c.id=g.camera_id
                             WHERE g.status='done' AND g.detect_status='pending' AND c.detect_enabled
                             ORDER BY g.detect_priority DESC, g.ts_start DESC LIMIT 1 FOR UPDATE OF g SKIP LOCKED)
               RETURNING s.*,
                 (SELECT v.code FROM camera c JOIN vehicle v ON v.id=c.vehicle_id WHERE c.id=s.camera_id) AS vehicle_code,
                 (SELECT c.code FROM camera c WHERE c.id=s.camera_id) AS camera_code,
                 (SELECT c.detect_config FROM camera c WHERE c.id=s.camera_id) AS detect_config""").fetchone()


def get_detector():
    """Модель загружается один раз на процесс. Если файла модели нет — этап простаивает."""
    global _detector, _detector_failed_at
    if _detector is None:
        if time.monotonic() - _detector_failed_at < 300 and _detector_failed_at:
            return None
        path = Path(settings.detect_model)
        if not path.exists():
            log.error("модель детектора не найдена: %s (см. tools/get_models.sh)", path)
            _detector_failed_at = time.monotonic()
            return None
        from .detector import Detector
        _detector = Detector(path, device=settings.detect_device, hint=settings.detect_hint,
                             threads=settings.detect_threads, streams=settings.detect_streams,
                             cache_dir=str(settings.tmp / "ov_cache"), decoded=settings.detect_decoded)
    return _detector


def get_helmet():
    """Классификатор касок: загружается, если файл модели есть (проверка раз в 5 минут)."""
    global _helmet, _helmet_checked_at
    if _helmet is None and time.monotonic() - _helmet_checked_at > 300:
        _helmet_checked_at = time.monotonic()
        path = Path(settings.helmet_model)
        if path.exists():
            from .helmet import HelmetCheck, HelmetClassifier
            clf = HelmetClassifier(path, device=settings.detect_device, threads=settings.helmet_threads,
                                   cache_dir=str(settings.tmp / "ov_cache"))
            _helmet = HelmetCheck(clf, settings.helmet_yes, settings.helmet_no, settings.helmet_min_head_px,
                      tuple(float(x) for x in settings.helmet_scales.split(",") if x.strip()))
        else:
            log.info("модель касок не найдена (%s) — проверка касок выключена", path)
    return _helmet


def get_heads():
    """Детектор голов: загружается, если файл модели есть (проверка раз в 5 минут)."""
    global _heads, _heads_checked_at
    if _heads is None and time.monotonic() - _heads_checked_at > 300:
        _heads_checked_at = time.monotonic()
        path = Path(settings.heads_model)
        if path.exists():
            from .heads import HeadDetector
            _heads = HeadDetector(path, device=settings.detect_device, threads=settings.helmet_threads,
                                  cache_dir=str(settings.tmp / "ov_cache"), conf=settings.heads_conf,
                                  no_conf=settings.heads_no_conf, min_px=settings.heads_min_px,
                                  nms=settings.heads_nms)
        else:
            log.info("детектор голов не найден (%s)", path)
    return _heads


def helmet_backend(rules: "CameraRules", names: list[str], det):
    """→ ("heads", HeadDetector) | ("crop", HelmetCheck) | (None, None) по HELMET_MODE и наличию моделей."""
    mode = settings.helmet_mode.lower()
    if mode == "off" or not rules.helmet or "person" not in names:
        return None, None
    if mode in ("auto", "heads"):
        h = get_heads()
        if h is not None and (h.W, h.H) == (det.W, det.H):
            return "heads", h
        if h is not None:
            log.error("вход детектора голов %dx%d не совпадает с детектором людей %dx%d", h.W, h.H, det.W, det.H)
        if mode == "heads":
            return None, None
    c = get_helmet()
    return ("crop", c) if c is not None else (None, None)


def full_size(w: int, h: int) -> tuple[int, int]:
    """Размер кадра для вырезки голов: исходный, но не шире HELMET_MAX_WIDTH, чётные стороны."""
    if w > settings.helmet_max_width:
        h = h * settings.helmet_max_width / w
        w = settings.helmet_max_width
    return int(w) // 2 * 2, int(round(h / 2)) * 2


class FullFrames:
    """Второй выход FFmpeg (кадры в исходном разрешении) читается отдельным потоком."""

    def __init__(self, fd: int, w: int, h: int):
        self.w, self.h = w, h
        self.q: _queue.Queue = _queue.Queue(maxsize=32)
        self.f = os.fdopen(fd, "rb", buffering=w * h * 3 * 2)
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()

    def _run(self):
        n = self.w * self.h * 3
        try:
            while True:
                raw = self.f.read(n)
                if not raw or len(raw) < n:
                    break
                self.q.put(np.frombuffer(raw, np.uint8).reshape(self.h, self.w, 3))
        finally:
            self.q.put(None)
            self.f.close()

    def get(self, timeout: float = 120) -> np.ndarray | None:
        return self.q.get(timeout=timeout)


def labels_cfg() -> tuple[list[int], list[str]]:
    names = [c.strip() for c in settings.detect_classes.split(",") if c.strip()]
    ids = [COCO.index(n) for n in names]
    return ids, names


def analyze(seg: dict, det) -> dict:
    """Декодирование + инференс + интервалы. Пишет во временный каталог, возвращает итог."""
    with tx() as conn:
        frags = conn.execute(
            "SELECT id, rel_path, duration, start_pts, ts_start, offset_in_segment FROM fragment "
            "WHERE segment_id=%s ORDER BY offset_in_segment", (seg["id"],)).fetchall()
    if not frags:
        raise media.MediaError("у сегмента нет фрагментов")
    if not seg["width"] or not seg["height"]:
        raise media.MediaError("неизвестен размер кадра")
    fps = settings.detect_fps
    class_ids, names = labels_cfg()
    rules = CameraRules.from_json(seg.get("detect_config"))
    hmode, hb = helmet_backend(rules, names, det)
    helmet = hb if hmode == "crop" else None   # квадрат головы + классификатор (второй поток FFmpeg)
    heads_det = hb if hmode == "heads" else None
    person_li = names.index("person") if "person" in names else -1
    lb = Letterbox(seg["width"], seg["height"], det.W, det.H)
    sw, sh = lb.scaled
    fx, fy = 1.0 / (seg["width"] * lb.r), 1.0 / (seg["height"] * lb.r)  # пиксели входа → доли кадра

    work = Path(tempfile.mkdtemp(dir=settings.tmp, prefix=f"det{seg['id']}_"))
    out_dir = work / "out"
    out_dir.mkdir()
    lst = work / "list.ffconcat"
    durs = media.concat_durations([f["start_pts"] for f in frags], [f["duration"] for f in frags])
    media.write_concat_list([settings.archive_root / f["rel_path"] for f in frags], durs, lst)

    thumbs = ThumbWriter(out_dir, sw, sh)
    builders = {}
    thumb_of: dict[int, str] = {}

    def on_close(iv: Interval):
        if iv.best_frame is None:
            return
        px = [[b[0] / fx, b[1] / fy, b[2] / fx, b[3] / fy] for b, _, _ in iv.best_boxes]
        if iv.label == NO_HELMET:
            cols = [(220, 40, 220)] * len(iv.best_boxes)  # BGR: пурпурная — без каски
        else:
            cols = [(40, 40, 230) if z else (60, 200, 60) for _, _, z in iv.best_boxes]  # в зоне — красная
        thumb_of[id(iv)] = thumbs.write(draw_boxes(iv.best_frame, px, cols))

    for name in names:
        builders[name] = IntervalBuilder(fps, settings.detect_gap_sec, settings.detect_min_dur_sec,
                                         settings.detect_min_hits, on_close)
    if helmet or heads_det:
        builders[NO_HELMET] = IntervalBuilder(fps, settings.detect_gap_sec, settings.detect_min_dur_sec,
                                              settings.helmet_min_hits, on_close)

    frame_bytes = det.W * det.H * 3
    cmd = [settings.ffmpeg_bin, "-nostdin", "-hide_banner", "-loglevel", "error",
           "-threads", str(settings.detect_decode_threads), "-fflags", "+discardcorrupt",
           "-f", "concat", "-safe", "0", "-i", str(lst)]
    full = None
    rfd = wfd = None
    if helmet:
        fw, fh = full_size(seg["width"], seg["height"])
        rfd, wfd = os.pipe()
        cmd += ["-filter_complex", f"[0:v:0]fps={fps},split=2[a][b];[a]{lb.ffmpeg_filter()}[lb];"
                                   f"[b]scale={fw}:{fh}:flags=bilinear,setsar=1[full]",
                "-map", "[lb]", "-pix_fmt", "bgr24", "-f", "rawvideo", "pipe:1",
                "-map", "[full]", "-pix_fmt", "bgr24", "-f", "rawvideo", f"pipe:{wfd}"]
    else:
        cmd += ["-map", "0:v:0", "-an", "-sn", "-dn", "-vf", f"fps={fps}," + lb.ffmpeg_filter(),
                "-pix_fmt", "bgr24", "-f", "rawvideo", "pipe:1"]
    t0 = time.monotonic()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=frame_bytes * 4,
                            pass_fds=(wfd,) if wfd is not None else ())
    if helmet:
        os.close(wfd)
        full = FullFrames(rfd, fw, fh)
    helmet_stat = [0, 0, 0]  # не определено / в каске / без каски (по людям-кадрам)
    err_buf: list[bytes] = []
    threading.Thread(target=lambda: err_buf.append(proc.stderr.read()), daemon=True).start()

    reorder = Reorder()
    frames_keep: dict[int, np.ndarray] = {}
    boxes_json: list = []
    n_frames = 0
    n_with = 0

    full_keep: dict[int, np.ndarray] = {}
    full_next = [0]  # индекс следующего кадра из второго выхода FFmpeg (кадры идут по порядку)

    def full_frame(idx: int):
        """Кадр idx в исходном разрешении. Обработка идёт по возрастанию idx, старые кадры удаляются."""
        while full_next[0] <= idx:
            f = full.get()
            if f is None:
                break
            full_keep[full_next[0]] = f
            full_next[0] += 1
        for k in [k for k in full_keep if k < idx]:
            del full_keep[k]
        return full_keep.pop(idx, None)

    meta = [0, 0]  # людей в кадре / без каски (для текущего кадра, режим heads)

    def heads_step(t: float, frame, dets: list):
        """Головы на кадре → (dets для интервала «люди» с учётом лишних голов, коды рамок, строки голов)."""
        from .heads import associate, people_count, person_codes
        if frame is None or not (dets or settings.heads_all_frames):
            return dets, [(H_UNKNOWN, 0.0)] * len(dets), []
        hs = heads_det(frame, fx, fy)
        hs = [h for h in hs if not any(point_in_poly((h.box[0] + h.box[2]) / 2, (h.box[1] + h.box[3]) / 2, p)
                                       for p in rules.ignore)]
        owned = associate([d[0] for d in dets], hs)
        hc = person_codes(owned, hs)
        _, extra = people_count(owned, hs, settings.heads_alone_conf)
        feed = dets + [(hs[i].box, hs[i].score, []) for i in extra]
        bad = [(h.box, h.score, dets[h.person][2] if h.person >= 0 else []) for h in hs if h.code == H_NO
               and (h.person >= 0 or h.score >= settings.heads_alone_conf)]
        builders[NO_HELMET].feed(t, NO_HELMET, bad, frame if bad else None)
        for h in hs:
            helmet_stat[h.code] += 1
        helmet_stat[H_UNKNOWN] += sum(1 for o in owned if not o)
        rows = [h.box + [h.score, h.code, h.person] for h in hs]
        meta[0], meta[1] = len(feed), len(bad)
        return feed, hc, rows

    def handle(idx: int, out: np.ndarray):
        nonlocal n_with
        frame = frames_keep.pop(idx, None)
        t = round(idx / fps, 3)
        boxes, scores, cls = det.decoder(out, class_ids, min(rules.min_conf, settings.detect_conf),
                                         settings.detect_nms)
        if boxes.size:
            boxes = boxes * np.array([fx, fy, fx, fy], dtype=np.float32)
            boxes = boxes.clip(0, 1)
        row = []
        heads_row = []
        meta[0] = meta[1] = 0
        for li, name in enumerate(names):
            m = cls == class_ids[li]
            dets = rules.apply(boxes[m], scores[m]) if boxes.size else []
            hc = None
            if heads_det and li == person_li:
                feed_dets, hc, heads_row = heads_step(t, frame, dets)
                builders[name].feed(t, name, feed_dets, frame if feed_dets else None)
            else:
                builders[name].feed(t, name, dets, frame if dets else None)
            if helmet and li == person_li:
                ff = full_frame(idx)
                hc = helmet(ff, [d[0] for d in dets]) if (ff is not None and dets) else [(H_UNKNOWN, 0.0)] * len(dets)
                bad = [d for d, (c, _) in zip(dets, hc) if c == H_NO]
                builders[NO_HELMET].feed(t, NO_HELMET, bad, frame if bad else None)
                for c, _ in hc:
                    helmet_stat[c] += 1
            for k, (b, s, z) in enumerate(dets):
                row.append(b + [s, li, 1 if z else 0] + ([hc[k][0], hc[k][1]] if hc else []))
        if row or heads_row:
            n_with += 1
            # v3: [t, люди, головы, людей в кадре, без каски]
            boxes_json.append([t, row, heads_row, meta[0], meta[1]] if heads_det else [t, row])
        for b in builders.values():
            b.tick(t)

    q = det.queue(lambda idx, out: reorder.put(idx, out))
    try:
        buf_view = proc.stdout
        while True:
            raw = buf_view.read(frame_bytes)
            if not raw or len(raw) < frame_bytes:
                break
            frame = np.frombuffer(raw, np.uint8).reshape(det.H, det.W, 3)
            frames_keep[n_frames] = frame
            q.start_async(frame[None], userdata=n_frames)
            n_frames += 1
            for idx_out in _drain(reorder):
                handle(*idx_out)
        q.wait_all()
        for idx_out in _drain(reorder):
            handle(*idx_out)
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait()
        if full:
            while full.t.is_alive():  # дочитать, чтобы поток завершился
                if full.q.get(timeout=30) is None:
                    break
    elapsed = time.monotonic() - t0
    if proc.returncode not in (0, None) and n_frames == 0:
        thumbs.close(check=False)
        shutil.rmtree(work, ignore_errors=True)
        raise media.MediaError("ffmpeg: " + (b"".join(err_buf).decode(errors="replace")[-1500:] or "ошибка декодирования"))

    intervals = []
    for b in builders.values():
        b.finish()
        intervals.extend(b.done)
    thumbs.close()
    intervals.sort(key=lambda iv: iv.start)
    (out_dir / "boxes.json").write_text(json.dumps(
        {"v": 3 if heads_det else 2, "fps": fps, "model": model_name(det, hb), "labels": names,
         "helmet": bool(hb), "helmet_mode": hmode,
         "frames": boxes_json},
        separators=(",", ":")), encoding="utf-8")

    step = 1.0 / fps
    rows = []
    for iv in intervals:
        a, b = iv.start, min(seg["duration"], iv.end(step))
        rows.append({"label": iv.label, "off_start": a, "off_end": b, "best_off": iv.best_t,
                     "ts_start": offset_to_wall(frags, a), "ts_end": offset_to_wall(frags, b),
                     "max_count": iv.max_count, "peak_score": iv.peak, "zones": sorted(iv.zones),
                     "thumb": thumb_of.get(id(iv))})
    return {"work": work, "out_dir": out_dir, "rows": rows, "frames": n_frames, "frames_with": n_with,
            "elapsed": elapsed, "model": model_name(det, hb), "expected": int(seg["duration"] * fps),
            "helmet": helmet_stat if hb else None, "helmet_mode": hmode}


def model_name(det, hb) -> str:
    if hb is None:
        return det.model_name
    return det.model_name + "+" + (hb.clf.model_name if hasattr(hb, "clf") else hb.model_name)


def _drain(reorder: Reorder):
    return reorder.pop_ready_pairs()


def run_once() -> int:
    det = get_detector()
    if det is None:
        return 0
    seg = _claim()
    if not seg:
        return 0
    res = None
    try:
        res = analyze(seg, det)
        day = seg["ts_start"].strftime("%Y/%m/%d")
        rel_dir = Path("detections") / seg["vehicle_code"] / seg["camera_code"] / day / str(seg["id"])
        final = settings.archive_root / rel_dir
        final.parent.mkdir(parents=True, exist_ok=True)
        old = None
        if final.exists():
            old = final.with_name(final.name + f".old{int(time.time())}")
            final.rename(old)
        res["out_dir"].rename(final)
        if old:
            shutil.rmtree(old, ignore_errors=True)
        with tx() as conn:
            conn.execute("DELETE FROM detection WHERE segment_id=%s", (seg["id"],))
            for r in res["rows"]:
                conn.execute(
                    """INSERT INTO detection(camera_id, segment_id, label, ts_start, ts_end, off_start, off_end,
                           best_off, max_count, peak_score, zones, thumb, model)
                       VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                    (seg["camera_id"], seg["id"], r["label"], r["ts_start"], r["ts_end"], r["off_start"],
                     r["off_end"], r["best_off"], r["max_count"], r["peak_score"], r["zones"],
                     (rel_dir / r["thumb"]).as_posix() if r["thumb"] else None, res["model"]))
            conn.execute(
                """UPDATE segment SET detect_status='done', detect_error=NULL, detect_claimed_at=NULL,
                       detect_done_at=now(), detect_model=%s, detect_path=%s, detect_frames=%s,
                       detect_sec=%s, detect_priority=0 WHERE id=%s""",
                (res["model"], rel_dir.as_posix(), res["frames"], round(res["elapsed"], 2), seg["id"]))
        speed = res["frames"] / max(res["elapsed"], 1e-6)
        log.info("сегмент %s: детекция %d кадров за %.1f с (%.1f кадр/с, ×%.0f реального времени), "
                 "интервалов %d, кадров с объектами %d", seg["id"], res["frames"], res["elapsed"], speed,
                 seg["duration"] / max(res["elapsed"], 1e-6), len(res["rows"]), res["frames_with"])
        if res["helmet"]:
            u, y, n = res["helmet"]
            if res.get("helmet_mode") == "heads":
                log.info("сегмент %s: головы — в каске %d, без каски %d; рамок людей без найденной головы %d "
                         "(по кадрам)", seg["id"], y, n, u)
            else:
                log.info("сегмент %s: каски — в каске %d, без каски %d, не определить %d (люди×кадры)",
                         seg["id"], y, n, u)
        if res["frames"] < 0.8 * res["expected"]:
            log.warning("сегмент %s: декодировано %d кадров из ~%d ожидаемых", seg["id"], res["frames"],
                        res["expected"])
    except Exception as e:  # noqa: BLE001
        status = "error" if seg["detect_attempts"] >= settings.max_attempts else "pending"
        with tx() as conn:
            conn.execute("UPDATE segment SET detect_status=%s, detect_error=%s, detect_claimed_at=NULL WHERE id=%s",
                         (status, str(e)[:2000], seg["id"]))
        log.error("сегмент %s: ошибка детекции (%s): %s", seg["id"], status, e)
    finally:
        if res:
            shutil.rmtree(res["work"], ignore_errors=True)
    return 1
