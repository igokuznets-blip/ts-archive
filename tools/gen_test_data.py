#!/usr/bin/env python3
"""Генератор тестовых файлов MPEG-TS ~80 МБ, имитирующих бортовые регистраторы.

Пример (файлы ~80 МБ по 160 с, 720p, 4 Мбит/с):
    python3 tools/gen_test_data.py --out data/archive/incoming --vehicles PT-101 EX-202 \
        --cameras cam1 cam2 --minutes 30 --start "2026-01-15 10:00:00"

Создаёт: разрыв записи в середине, один повреждённый файл и один дубликат.
"""
import argparse
import shutil
import subprocess
import tempfile
from datetime import datetime, timedelta
from pathlib import Path


def gen_camera(out: Path, vehicle: str, cam: str, start: datetime, minutes: float, frag_sec: float,
               gap_at: float, gap_len: float, bitrate: str, audio: bool, hue: int, vcodec: str = "libx264",
               size: str = "1280x720") -> int:
    tmp = Path(tempfile.mkdtemp())
    total = int(minutes * 60)
    fps = 25
    vf = f"hue=h={hue}"
    args = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", f"testsrc=size={size}:rate={fps}:duration={total}"]
    if audio:
        args += ["-f", "lavfi", "-i", f"sine=frequency={300 + hue}:sample_rate=16000:duration={total}"]
    # CBR (nal-hrd=cbr) — размер файла как у реального регистратора даже на простой тестовой картинке
    cbr = ["-preset", "veryfast", "-x264-params", "nal-hrd=cbr", "-minrate", bitrate] if vcodec == "libx264" else []
    args += ["-vf", vf, "-c:v", vcodec] + cbr + ["-b:v", bitrate, "-maxrate", bitrate,
             "-bufsize", bitrate, "-g", str(fps * 2), "-keyint_min", str(fps * 2),
             "-force_key_frames", f"expr:gte(t,n_forced*{frag_sec})",
             "-sc_threshold", "0", "-pix_fmt", "yuv420p"]
    if audio:
        args += ["-c:a", "aac", "-b:a", "32k"]
    args += ["-f", "segment", "-segment_time", str(frag_sec), "-segment_format", "mpegts",
             "-reset_timestamps", "0", str(tmp / "f%06d.ts")]
    subprocess.run(args, check=True)
    dst = out / vehicle / cam
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for i, f in enumerate(sorted(tmp.glob("f*.ts"))):
        t = i * frag_sec
        if gap_at <= t < gap_at + gap_len:
            continue  # техника была выключена — разрыв в записи
        ts = start + timedelta(seconds=t)
        shutil.move(str(f), dst / f"{ts.strftime('%Y%m%d_%H%M%S')}_{ts.microsecond // 1000:03d}.ts")
        n += 1
    shutil.rmtree(tmp)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/archive/incoming")
    ap.add_argument("--vehicles", nargs="+", default=["PT-101", "EX-202"])
    ap.add_argument("--cameras", nargs="+", default=["cam1", "cam2"])
    ap.add_argument("--start", default="2026-01-15 10:00:00")
    ap.add_argument("--minutes", type=float, default=12)
    ap.add_argument("--frag-sec", type=float, default=160, help="длительность файла, с")
    ap.add_argument("--gap-at", type=float, default=300, help="начало разрыва, с от старта")
    ap.add_argument("--gap-len", type=float, default=60)
    ap.add_argument("--bitrate", default="4000k", help="4 Мбит/с × 160 с ≈ 80 МБ на файл")
    ap.add_argument("--size", default="1280x720")
    ap.add_argument("--no-audio", action="store_true")
    ap.add_argument("--no-defects", action="store_true")
    ap.add_argument("--vcodec", default="libx264", help="libx264 | mpeg2video | libx265")
    a = ap.parse_args()

    out = Path(a.out)
    start = datetime.strptime(a.start, "%Y-%m-%d %H:%M:%S")
    hue = 0
    for vi, v in enumerate(a.vehicles):
        for c in a.cameras:
            n = gen_camera(out, v, c, start + timedelta(seconds=vi * 30), a.minutes, a.frag_sec,
                           a.gap_at, a.gap_len, a.bitrate, not a.no_audio, hue, a.vcodec, a.size)
            hue += 70
            print(f"{v}/{c}: {n} фрагментов")
    if not a.no_defects:
        first = sorted((out / a.vehicles[0] / a.cameras[0]).glob("*.ts"))
        (out / a.vehicles[0] / a.cameras[0] / "20260115_235959_000.ts").write_bytes(b"\x00" * 4096)
        shutil.copy(first[0], out / a.vehicles[0] / a.cameras[0] / "20260115_235958_000.ts")
        (out / "garbage_name.ts").write_bytes(first[1].read_bytes())
        print("добавлены дефекты: битый файл, дубликат, файл с неверным именем")


if __name__ == "__main__":
    main()
