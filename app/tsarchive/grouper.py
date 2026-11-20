"""Этап 3. Группировка проанализированных фрагментов в сегменты.

Сегмент закрывается, если:
  * разрыв между концом предыдущего и началом следующего фрагмента > GAP_THRESHOLD_SEC;
  * фрагменты перекрываются сильнее OVERLAP_TOLERANCE_SEC (повтор/сбой часов);
  * изменилась сигнатура потоков (кодек, разрешение, аудио) — иначе -c copy невозможен;
  * длительность достигла SEGMENT_MAX_SEC;
  * последний фрагмент камеры принят более IDLE_CLOSE_SEC назад (техника выключена).
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from .config import settings
from .db import tx

log = logging.getLogger(__name__)
LOCK_KEY = 727002


def _split(frags: list[dict]) -> list[tuple[list[dict], bool]]:
    """Делит упорядоченные фрагменты на группы. Возвращает (группа, закрыта_по_разрыву)."""
    groups: list[tuple[list[dict], bool]] = []
    cur: list[dict] = []
    cur_dur = 0.0
    for f in frags:
        if cur:
            prev = cur[-1]
            gap = (f["ts_start"] - prev["ts_end"]).total_seconds()
            breaks = (
                gap > settings.gap_threshold_sec
                or gap < -settings.overlap_tolerance_sec
                or f["codec_sig"] != prev["codec_sig"]
                or cur_dur + f["duration"] > settings.segment_max_sec
            )
            if breaks:
                groups.append((cur, True))
                cur, cur_dur = [], 0.0
        cur.append(f)
        cur_dur += f["duration"]
    if cur:
        groups.append((cur, False))
    return groups


def _create_segment(conn, camera_id: int, group: list[dict]) -> int:
    first, last = group[0], group[-1]
    duration = sum(f["duration"] for f in group)
    span = (last["ts_end"] - first["ts_start"]).total_seconds()
    seg_id = conn.execute(
        """INSERT INTO segment(camera_id, ts_start, ts_end, duration, fragments_count, codec_sig,
                               vcodec, width, height, gap_total_sec, status)
           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,'ready') RETURNING id""",
        (camera_id, first["ts_start"], last["ts_end"], duration, len(group), first["codec_sig"],
         first["vcodec"], first["width"], first["height"], max(0.0, span - duration))).fetchone()["id"]
    offset = 0.0
    ids, offsets = [], []
    for f in group:
        ids.append(f["id"])
        offsets.append(offset)
        offset += f["duration"]
    conn.execute(
        """UPDATE fragment AS fr SET status='grouped', segment_id=%s, offset_in_segment=v.off
           FROM unnest(%s::bigint[], %s::float8[]) AS v(id, off) WHERE fr.id = v.id""",
        (seg_id, ids, offsets))
    return seg_id


def run_once() -> int:
    created = 0
    with tx() as conn:
        if not conn.execute("SELECT pg_try_advisory_xact_lock(%s) AS ok", (LOCK_KEY,)).fetchone()["ok"]:
            return 0  # группировку уже выполняет другой воркер
        cams = conn.execute("SELECT DISTINCT camera_id FROM fragment WHERE status='probed'").fetchall()
        now = datetime.now(timezone.utc)
        for c in cams:
            cid = c["camera_id"]
            # не группируем дальше первого ещё не проанализированного фрагмента камеры
            barrier = conn.execute(
                "SELECT min(ts_start) AS t FROM fragment WHERE camera_id=%s AND status IN ('new','probing')",
                (cid,)).fetchone()["t"]
            last = conn.execute(
                "SELECT max(ingested_at) AS t, max(duration) AS d FROM fragment WHERE camera_id=%s",
                (cid,)).fetchone()
            last_ingest = last["t"]
            # файл в несколько минут приходит раз в несколько минут: не закрываем сегмент раньше,
            # чем пройдут две длительности фрагмента
            idle_limit = max(settings.idle_close_sec, 2 * (last["d"] or 0))
            q = ("SELECT id, ts_start, ts_end, duration, codec_sig, vcodec, width, height FROM fragment "
                 "WHERE camera_id=%s AND status='probed'")
            params: list = [cid]
            if barrier is not None:
                q += " AND ts_start < %s"
                params.append(barrier)
            frags = conn.execute(q + " ORDER BY ts_start, id", params).fetchall()
            if not frags:
                continue
            idle = (now - last_ingest).total_seconds() > idle_limit
            for group, closed in _split(frags):
                dur = sum(f["duration"] for f in group)
                full = dur >= settings.segment_max_sec - 1.0
                if closed or full or (idle and barrier is None):
                    _create_segment(conn, cid, group)
                    created += 1
    if created:
        log.info("создано сегментов: %d", created)
    return created
