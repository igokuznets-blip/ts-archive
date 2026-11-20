"""Квантование модели детектора в INT8 (NNCF, post-training) на кадрах из вашего же архива.

На Xeon с AVX-512 VNNI (Ice Lake: Gold 5320 и т.п.) INT8 обычно в 2–3 раза быстрее FP32
при потере точности в пределах долей mAP. Калибровка идёт на реальных кадрах камер — это
важнее, чем на COCO: ночь, пыль, блики, ракурсы спецтехники.

    python -m tsarchive quantize /models/yolox_s.onnx /models/yolox_s_int8.xml [кадров=300]
"""
from __future__ import annotations

import logging
import random
import subprocess
from pathlib import Path

import numpy as np

from . import media
from .config import settings
from .db import tx
from .detector import Letterbox

log = logging.getLogger(__name__)


def calibration_frames(W: int, H: int, n: int) -> list[np.ndarray]:
    """n кадров из случайных готовых сегментов всех камер (равномерно по камерам)."""
    with tx() as conn:
        segs = conn.execute(
            "SELECT DISTINCT ON (camera_id, date_trunc('hour', ts_start)) id, width, height, duration "
            "FROM segment WHERE status='done' AND width IS NOT NULL ORDER BY camera_id, date_trunc('hour', ts_start), random()"
        ).fetchall()
        random.shuffle(segs)
        per_seg = max(1, -(-n // max(1, len(segs))))  # с округлением вверх
        frames: list[np.ndarray] = []
        for s in segs:
            if len(frames) >= n:
                break
            frags = conn.execute("SELECT rel_path FROM fragment WHERE segment_id=%s ORDER BY offset_in_segment",
                                 (s["id"],)).fetchall()
            lb = Letterbox(s["width"], s["height"], W, H)
            for _ in range(per_seg):
                f = random.choice(frags)
                t = random.uniform(0, 60)
                cmd = [settings.ffmpeg_bin, "-hide_banner", "-loglevel", "error", "-ss", f"{t:.2f}",
                       "-i", str(settings.archive_root / f["rel_path"]), "-frames:v", "1",
                       "-vf", lb.ffmpeg_filter(), "-pix_fmt", "bgr24", "-f", "rawvideo", "pipe:1"]
                raw = subprocess.run(cmd, capture_output=True, timeout=60).stdout
                if len(raw) == W * H * 3:
                    frames.append(np.frombuffer(raw, np.uint8).reshape(H, W, 3))
    return frames[:n]


def quantize(src: str, dst: str, n: int = 300) -> None:
    import nncf
    import openvino as ov

    core = ov.Core()
    model = core.read_model(src)
    _, _, H, W = [d.get_length() for d in model.input(0).get_partial_shape()]
    frames = calibration_frames(W, H, n)
    if len(frames) < 20:
        raise SystemExit(f"мало кадров для калибровки ({len(frames)}): нужны готовые сегменты в архиве")
    log.info("калибровка на %d кадрах", len(frames))
    ds = nncf.Dataset(frames, lambda fr: fr.transpose(2, 0, 1)[None].astype(np.float32))
    # MIXED: симметричные веса, асимметричные активации — стандарт для детекторов семейства YOLO
    q = nncf.quantize(model, ds, preset=nncf.QuantizationPreset.MIXED, subset_size=len(frames))
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    ov.save_model(q, dst)
    log.info("сохранено: %s", dst)


__all__ = ["quantize", "calibration_frames", "media"]
