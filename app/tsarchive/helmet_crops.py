"""Выгрузка квадратов голов из размеченного архива — для проверки и дообучения модели касок.

    python -m tsarchive helmet-crops <camera_id> <каталог> [кадров=300]

Берёт кадры с людьми из boxes.json размеченных сегментов камеры (равномерно по архиву),
вырезает головы так же, как этап detect, и раскладывает по подкаталогам helmet/, no_helmet/,
unclear/ по решению текущей модели (если модели нет — в unsorted/). Оператор переносит ошибочные
файлы в правильный каталог, после чего каталог передаётся в tools/train_helmet.py --extra.
"""
from __future__ import annotations

import json
import random
import subprocess
from pathlib import Path

import numpy as np

from .config import settings
from .db import tx
from .detect import full_size, get_helmet
from .helmet import CLASSES, H_NO, H_UNKNOWN, H_YES, crop_square, head_square


def _frame(path: Path, ss: float, w: int, h: int) -> np.ndarray | None:
    r = subprocess.run([settings.ffmpeg_bin, "-nostdin", "-hide_banner", "-loglevel", "error", "-ss", f"{ss:.3f}",
                        "-i", str(path), "-map", "0:v:0", "-frames:v", "1", "-vf", f"scale={w}:{h},setsar=1",
                        "-pix_fmt", "bgr24", "-f", "rawvideo", "pipe:1"], capture_output=True, timeout=60)
    if len(r.stdout) < w * h * 3:
        return None
    return np.frombuffer(r.stdout[: w * h * 3], np.uint8).reshape(h, w, 3)


def export(camera_id: int, out: Path, n_frames: int = 300, seed: int = 1) -> dict:
    from PIL import Image  # есть в образе detect (зависимость nncf); в app-образе не нужен

    rng = random.Random(seed)
    with tx() as conn:
        segs = conn.execute("SELECT id, width, height, detect_path FROM segment WHERE camera_id=%s "
                            "AND detect_status='done' AND detect_path IS NOT NULL", (camera_id,)).fetchall()
    cand = []
    for s in segs:
        bj = settings.archive_root / s["detect_path"] / "boxes.json"
        if not bj.exists():
            continue
        data = json.loads(bj.read_text())
        li = data["labels"].index("person") if "person" in data["labels"] else -1
        for t, rows in data["frames"]:
            boxes = [r[:4] for r in rows if r[5] == li]
            if boxes:
                cand.append((s, t, boxes))
    rng.shuffle(cand)
    cand = sorted(cand[:n_frames], key=lambda c: (c[0]["id"], c[1]))
    helmet = get_helmet()
    names = {H_YES: "helmet", H_NO: "no_helmet", H_UNKNOWN: "unclear"}
    for d in (*CLASSES, "unsorted"):
        (out / d).mkdir(parents=True, exist_ok=True)
    stat = {k: 0 for k in (*CLASSES, "unsorted", "small")}
    frags_cache: dict[int, list] = {}
    for s, t, boxes in cand:
        if s["id"] not in frags_cache:
            with tx() as conn:
                frags_cache[s["id"]] = conn.execute(
                    "SELECT rel_path, offset_in_segment FROM fragment WHERE segment_id=%s ORDER BY offset_in_segment",
                    (s["id"],)).fetchall()
        fr = [f for f in frags_cache[s["id"]] if f["offset_in_segment"] <= t] or frags_cache[s["id"]][:1]
        f = fr[-1]
        w, h = full_size(s["width"], s["height"])
        img = _frame(settings.archive_root / f["rel_path"], t - f["offset_in_segment"], w, h)
        if img is None:
            continue
        res = helmet(img, boxes) if helmet else [(None, 0.0)] * len(boxes)
        for k, (b, (code, pno)) in enumerate(zip(boxes, res)):
            cx, top, side = head_square(b[0] * w, b[1] * h, b[2] * w, b[3] * h)
            if side / 1.6 < settings.helmet_min_head_px:
                stat["small"] += 1
                continue
            sub = names[code] if code is not None else "unsorted"
            name = f"s{s['id']}_t{t:08.2f}_{k}_p{int(pno * 100):02d}.jpg"
            crop = crop_square(img, cx, top, side, 96)
            Image.fromarray(crop[:, :, ::-1]).save(out / sub / name, quality=92)
            stat[sub] += 1
    return {"frames": len(cand), **stat}
