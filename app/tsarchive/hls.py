"""HLS-плейлист сегмента поверх исходных TS-файлов (EXT-X-BYTERANGE).

Склеенных копий нет: плеер (hls.js / Safari) читает куски исходных фрагментов
по Range-запросам. Куски начинаются с PAT/PMT перед ключевым кадром (см. media.keyframe_index),
поэтому перемотка возможна в любое место архива.

Временная шкала плейлиста совпадает с картой фрагментов сегмента (offset_in_segment):
fragment.offset = сумма EXTINF всех предыдущих кусков. Это позволяет UI пересчитывать
позицию плеера в абсолютное время записи так же, как раньше для склеенного MP4.
"""
from __future__ import annotations

import math
from typing import Callable, Iterable

from .media import PTS_WRAP


def fragment_chunks(dur: float, size: int, kf_t: Iterable[float] | None, kf_pos: Iterable[int] | None,
                    target: float) -> list[tuple[float, int, int]]:
    """Нарезка фрагмента на куски по ключевым кадрам: [(длительность, длина в байтах, смещение)].

    Первый кусок всегда начинается с байта 0 (там PAT/PMT и начало файла), последний
    заканчивается концом файла. Сумма длительностей равна dur.
    """
    bounds: list[tuple[float, int]] = [(0.0, 0)]
    for t, p in zip(kf_t or [], kf_pos or []):
        if p <= 0 or p >= size or t <= 0.05 or t >= dur - 0.05:
            continue
        if t - bounds[-1][0] >= target - 1e-6:
            bounds.append((t, p))
    chunks = []
    for i, (t, p) in enumerate(bounds):
        t_next, p_next = bounds[i + 1] if i + 1 < len(bounds) else (dur, size)
        if p_next <= p:
            continue
        chunks.append((max(0.001, t_next - t), p_next - p, p))
    return chunks


def is_discontinuity(prev_start_pts: float | None, prev_dur: float, start_pts: float | None) -> bool:
    """Разрыв PTS между соседними фрагментами (перезапуск регистратора, другой энкодер)."""
    if prev_start_pts is None or start_pts is None:
        return True
    diff = start_pts - prev_start_pts
    if diff < -PTS_WRAP / 2:
        diff += PTS_WRAP
    return abs(diff - prev_dur) > 0.5


def build_playlist(frags: list[dict], seg_duration: float, url: Callable[[dict], str], target: float) -> str:
    """frags: id, off, size_bytes, start_pts, kf_t, kf_pos — в порядке offset_in_segment."""
    lines: list[str] = []
    max_d = 1.0
    body: list[str] = []
    for i, f in enumerate(frags):
        end = frags[i + 1]["off"] if i + 1 < len(frags) else seg_duration
        dur = max(0.001, end - f["off"])
        if i > 0:
            p = frags[i - 1]
            p_dur = f["off"] - p["off"]
            if is_discontinuity(p["start_pts"], p_dur, f["start_pts"]):
                body.append("#EXT-X-DISCONTINUITY")
        u = url(f)
        for d, length, offset in fragment_chunks(dur, f["size_bytes"], f["kf_t"], f["kf_pos"], target):
            max_d = max(max_d, d)
            body.append(f"#EXTINF:{d:.6f},")
            body.append(f"#EXT-X-BYTERANGE:{length}@{offset}")
            body.append(u)
    lines += ["#EXTM3U", "#EXT-X-VERSION:4", "#EXT-X-PLAYLIST-TYPE:VOD",
              f"#EXT-X-TARGETDURATION:{math.ceil(max_d)}", "#EXT-X-MEDIA-SEQUENCE:0"]
    lines += body
    lines.append("#EXT-X-ENDLIST")
    return "\n".join(lines) + "\n"
