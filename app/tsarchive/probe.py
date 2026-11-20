"""Этап 2. Анализ фрагментов через ffprobe: длительность, PTS, параметры потоков,
индекс ключевых кадров (время → байтовое смещение) для воспроизведения без склейки."""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path

from . import media
from .config import settings
from .db import tx
from .ingest import _move

log = logging.getLogger(__name__)


def _claim(limit: int) -> list[dict]:
    with tx() as conn:
        return conn.execute(
            """UPDATE fragment SET status='probing', claimed_at=now()
               WHERE id IN (SELECT id FROM fragment WHERE status='new'
                            ORDER BY camera_id, ts_start LIMIT %s FOR UPDATE SKIP LOCKED)
               RETURNING id, rel_path, ts_start""", (limit,)).fetchall()


def _probe_one(row: dict) -> tuple[dict, tuple | None, str | None]:
    path = settings.archive_root / row["rel_path"]
    try:
        err = media.check_ts_header(path)
        if err:
            return row, None, err
        res = media.probe(path)
        if res.duration <= 0:
            return row, None, "нулевая длительность"
        kf_t, kf_pos = media.keyframe_index(path, res.start_pts)
        return row, (res, kf_t, kf_pos), None
    except FileNotFoundError:
        return row, None, "файл отсутствует"
    except media.MediaError as e:
        return row, None, str(e)


def run_once() -> int:
    rows = _claim(settings.probe_batch)
    if not rows:
        return 0
    with ThreadPoolExecutor(max_workers=settings.probe_threads) as ex:
        results = list(ex.map(_probe_one, rows))
    with tx() as conn:
        for row, res, err in results:
            if res is None:
                qrel = None
                src = settings.archive_root / row["rel_path"]
                if src.exists():
                    qrel = Path("quarantine/probe_failed") / Path(row["rel_path"]).relative_to("raw")
                    _move(src, settings.archive_root / qrel)
                conn.execute("UPDATE fragment SET status='quarantine', error=%s, rel_path=coalesce(%s, rel_path), "
                             "probed_at=now() WHERE id=%s",
                             (err, qrel.as_posix() if qrel else None, row["id"]))
                log.warning("фрагмент %s в карантине: %s", row["rel_path"], err)
                continue
            res, kf_t, kf_pos = res
            conn.execute(
                """UPDATE fragment SET status='probed', duration=%s, ts_end=%s, start_pts=%s, vcodec=%s,
                   width=%s, height=%s, fps=%s, acodec=%s, codec_sig=%s, starts_with_key=%s,
                   kf_t=%s, kf_pos=%s, probed_at=now(), claimed_at=NULL, error=NULL WHERE id=%s""",
                (res.duration, row["ts_start"] + timedelta(seconds=res.duration), res.start_pts, res.vcodec,
                 res.width, res.height, res.fps, res.acodec, res.codec_sig, res.starts_with_key,
                 kf_t, kf_pos, row["id"]))
    return len(rows)
