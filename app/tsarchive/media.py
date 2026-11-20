"""Обёртки над ffprobe/ffmpeg и индекс ключевых кадров для воспроизведения исходных TS по HLS byte-range."""
from __future__ import annotations

import json
import logging
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .config import settings

log = logging.getLogger(__name__)


class MediaError(RuntimeError):
    pass


def _run(args: list[str], timeout: int | None = None) -> str:
    log.debug("exec: %s", " ".join(args))
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=timeout or settings.ffmpeg_timeout)
    except subprocess.TimeoutExpired as e:
        raise MediaError(f"таймаут: {args[0]}") from e
    if p.returncode != 0:
        tail = (p.stderr or "").strip().splitlines()[-8:]
        raise MediaError(f"{Path(args[0]).name} код {p.returncode}: " + " | ".join(tail))
    return p.stdout


@dataclass
class ProbeResult:
    duration: float
    start_pts: float | None
    vcodec: str | None
    width: int | None
    height: int | None
    fps: str | None
    acodec: str | None
    codec_sig: str
    starts_with_key: bool | None


def check_ts_header(path: Path) -> str | None:
    """Быстрая проверка транспортного потока: sync byte 0x47 каждые 188 байт."""
    with open(path, "rb") as f:
        head = f.read(188 * 4)
    if len(head) < 188:
        return "файл меньше одного TS-пакета"
    if any(head[i] != 0x47 for i in range(0, len(head) - 187, 188)):
        return "нет sync byte 0x47 — не MPEG-TS"
    return None


def probe(path: Path) -> ProbeResult:
    # один вызов: формат, потоки и флаги первых пакетов (проверка ключевого кадра в начале)
    out = _run([settings.ffprobe_bin, "-v", "error", "-show_format", "-show_streams",
                "-show_entries", "packet=codec_type,flags", "-read_intervals", "%+#16",
                "-of", "json", str(path)], timeout=60)
    data = json.loads(out)
    streams = data.get("streams", [])
    fmt = data.get("format", {})
    v = next((s for s in streams if s.get("codec_type") == "video"), None)
    a = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if v is None:
        raise MediaError("в фрагменте нет видеопотока")

    # длительность берём по видеопотоку: format.duration в TS включает «хвосты» аудио
    # и даёт накопительную ошибку в несколько процентов на сотнях фрагментов
    duration = _f(v.get("duration"))
    if not duration or duration <= 0:
        duration = _duration_by_packets(path) or _f(fmt.get("duration"))
    # start_time формата — именно от него concat demuxer отсчитывает таймстемпы файла
    start = _f(fmt.get("start_time"))
    first_v = next((p for p in data.get("packets", []) if p.get("codec_type") == "video"), None)
    starts_with_key = first_v["flags"].startswith("K") if first_v and first_v.get("flags") else None

    sig_parts = [f"v:{v.get('codec_name')}:{v.get('width')}x{v.get('height')}:{v.get('pix_fmt')}"]
    if a is not None:
        sig_parts.append(f"a:{a.get('codec_name')}:{a.get('sample_rate')}:{a.get('channels')}")
    return ProbeResult(
        duration=float(duration or 0.0), start_pts=start, vcodec=v.get("codec_name"),
        width=v.get("width"), height=v.get("height"), fps=v.get("r_frame_rate"),
        acodec=a.get("codec_name") if a else None, codec_sig="|".join(sig_parts),
        starts_with_key=starts_with_key,
    )


def _duration_by_packets(path: Path) -> float:
    out = _run([settings.ffprobe_bin, "-v", "error", "-select_streams", "v:0", "-show_entries",
                "packet=pts_time,duration_time", "-of", "csv=p=0", str(path)], timeout=60)
    pts, last_dur = [], 0.0
    for line in out.splitlines():
        parts = line.split(",")
        if parts and parts[0] not in ("", "N/A"):
            pts.append(float(parts[0]))
            if len(parts) > 1 and parts[1] not in ("", "N/A"):
                last_dur = float(parts[1])
    if not pts:
        return 0.0
    return max(pts) - min(pts) + last_dur


def _is_pat(buf: bytes, i: int) -> bool:
    """TS-пакет с PID 0 (PAT) и началом секции."""
    return buf[i] == 0x47 and (buf[i + 1] & 0x1F) == 0 and buf[i + 2] == 0 and (buf[i + 1] & 0x40)


def keyframe_index(path: Path, start_pts: float | None) -> tuple[list[float], list[int]]:
    """Индекс ключевых кадров фрагмента: время от начала файла (с) и байтовое смещение,
    с которого можно начать чтение для воспроизведения.

    Смещение сдвигается назад к ближайшему PAT перед ключевым кадром: тогда каждый кусок
    HLS byte-range начинается с PAT/PMT, и демультиплексор плеера может стартовать
    с любого места файла. Если PAT рядом нет, берётся позиция самого кадра.
    Декодирования нет — ffprobe только читает заголовки пакетов (~0,2 с на файл 80 МБ).
    """
    out = _run([settings.ffprobe_bin, "-v", "error", "-select_streams", "v:0", "-show_entries",
                "packet=pts_time,pos,flags", "-of", "csv=p=0", str(path)], timeout=300)
    base = start_pts or 0.0
    raw: list[tuple[float, int]] = []
    for line in out.splitlines():
        parts = line.split(",")
        if len(parts) < 3 or not parts[2].startswith("K") or parts[0] in ("", "N/A") or parts[1] in ("", "N/A"):
            continue
        raw.append((float(parts[0]) - base, int(parts[1])))
    raw.sort(key=lambda x: x[1])
    times, positions = [], []
    window = settings.pat_search_bytes
    with open(path, "rb") as f:
        for t, pos in raw:
            start = pos
            if pos % 188 == 0 and pos > 0:
                lo = max(0, pos - window)
                lo -= lo % 188
                f.seek(lo)
                buf = f.read(pos - lo)
                i = len(buf) - 188
                while i >= 0:
                    if _is_pat(buf, i):
                        start = lo + i
                        break
                    i -= 188
            if positions and start <= positions[-1]:
                continue
            times.append(round(t, 6))
            positions.append(start)
    return times, positions


def _f(x) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _q(p: Path) -> str:
    return "'" + str(p).replace("'", "'\\''") + "'"


PTS_WRAP = 2 ** 33 / 90000.0  # переполнение 33-битного PTS MPEG-TS (~26,5 ч)


def concat_durations(start_pts: list[float | None], durations: list[float]) -> list[float]:
    """Длительности для директив `duration` в ffconcat.

    Бортовые регистраторы режут один непрерывный поток, поэтому PTS соседних фрагментов
    непрерывны. Разница start_time соседних файлов — точное смещение, при котором concat
    demuxer кладёт пакеты на исходную временную шкалу: без дублей и пропусков кадров на стыках.
    (Длительность видеопотока даёт дрожание до длительности аудиокадра.)
    Если PTS не непрерывны (рестарт регистратора), берётся длительность видео.
    """
    out = []
    for i, d in enumerate(durations):
        nd = d
        if i + 1 < len(durations) and start_pts[i] is not None and start_pts[i + 1] is not None:
            diff = start_pts[i + 1] - start_pts[i]
            if diff < -PTS_WRAP / 2:
                diff += PTS_WRAP
            if 0.5 * d <= diff <= 1.5 * d + 0.2:
                nd = diff
        out.append(nd)
    return out


def write_concat_list(paths: list[Path], durations: list[float] | None, list_path: Path) -> None:
    lines = ["ffconcat version 1.0"]
    for i, p in enumerate(paths):
        lines.append(f"file {_q(p)}")
        if durations and durations[i]:
            # явная длительность делает таймлайн склейки детерминированным
            lines.append(f"duration {durations[i]:.6f}")
    list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


MP4_AUDIO_COPY = ("aac", "mp3", "opus")


def audio_args(acodec: str | None) -> list[str]:
    """Аудио для MP4: AAC/MP3 копируются, прочее (G.711 и т.п.) перекодируется в AAC —
    это дёшево и не затрагивает видео. AUDIO_MODE=drop отключает звук."""
    if acodec is None or settings.audio_mode == "drop":
        return ["-an"] if settings.audio_mode == "drop" else []
    if acodec in MP4_AUDIO_COPY:
        return ["-c:a", "copy"] + (["-bsf:a", "aac_adtstoasc"] if acodec == "aac" else [])
    return ["-c:a", "aac", "-b:a", "64k"]


def transcode_proxy(list_path: Path, out_mp4: Path) -> None:
    """Прокси H.264 для кодеков, которые браузер не воспроизводит (например, MPEG-2 Video).
    Это единственный случай, когда на диске появляется второй экземпляр видео."""
    _run([settings.ffmpeg_bin, "-hide_banner", "-loglevel", "error", "-y",
          "-fflags", "+genpts+discardcorrupt", "-f", "concat", "-safe", "0", "-i", str(list_path),
          "-map", "0:v", "-map", "0:a?", "-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
          "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart", "-f", "mp4", str(out_mp4)])


def make_sprite(list_path: Path, out_jpg: Path, duration: float, width: int | None, height: int | None) -> dict:
    """Полоса превью для таймлайна прямо из исходных фрагментов (список ffconcat)."""
    interval = max(1, settings.thumb_interval_sec)
    count = max(1, int(math.floor(max(0.0, duration - 0.1) / interval)) + 1)
    cols = max(1, min(settings.thumb_cols, count))
    rows = max(1, math.ceil(count / cols))
    tw = settings.thumb_width
    th = int(round(tw * (height or 9) / (width or 16) / 2) * 2) or 90
    # декодируются только ключевые кадры (-skip_frame nokey): для часового сегмента из файлов
    # по 80 МБ это в десятки раз быстрее полного декодирования; точность — один GOP (1–2 с)
    _run([settings.ffmpeg_bin, "-hide_banner", "-loglevel", "error", "-y",
          "-skip_frame", "nokey", "-f", "concat", "-safe", "0", "-i", str(list_path),
          # первый ключевой кадр в каждой корзине [k*N, (k+1)*N) — без накопления сдвига
          # (fps=1/N взял бы последний кадр корзины — сдвиг до N/2 с)
          "-vf", f"select='isnan(prev_selected_t)+gt(floor(t/{interval})\\,floor(prev_selected_t/{interval}))',"
                 f"scale={tw}:{th},tile={cols}x{rows}",
          "-fps_mode", "passthrough", "-frames:v", "1", "-q:v", "5", str(out_jpg)], timeout=1800)
    return {"interval": interval, "cols": cols, "rows": rows, "w": tw, "h": th, "count": count}


def cut_to_mp4(list_path: Path, start: float, end: float, out: Path, acodec: str | None) -> None:
    """Вырезка интервала из исходных фрагментов (список ffconcat) в MP4 без перекодирования видео.
    Начало округляется вниз до ключевого кадра."""
    _run([settings.ffmpeg_bin, "-hide_banner", "-loglevel", "error", "-y",
          "-fflags", "+genpts+discardcorrupt", "-ss", f"{max(0.0, start):.3f}",
          "-f", "concat", "-safe", "0", "-i", str(list_path), "-t", f"{max(0.1, end - start):.3f}",
          "-map", "0:v", "-map", "0:a?", "-c:v", "copy", *audio_args(acodec),
          "-avoid_negative_ts", "make_zero", "-movflags", "+faststart", "-f", "mp4", str(out)])


def concat_mp4_parts(list_path: Path, out: Path) -> None:
    """Склейка готовых MP4-частей выгрузки (через разрывы записи) в один файл, -c copy."""
    _run([settings.ffmpeg_bin, "-hide_banner", "-loglevel", "error", "-y",
          "-f", "concat", "-safe", "0", "-i", str(list_path), "-map", "0:v", "-map", "0:a?", "-c", "copy",
          "-movflags", "+faststart", "-f", "mp4", str(out)])
