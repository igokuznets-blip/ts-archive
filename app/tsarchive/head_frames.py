"""Выгрузка кадров с людьми и предразметкой голов (COCO) — для проверки и дообучения детектора голов.

    python -m tsarchive head-frames <camera_id> <каталог> [кадров=300]

Результат: <каталог>/images/*.jpg (исходное разрешение) и <каталог>/coco.json с рамками голов
текущей модели (категории helmet и head). Каталог импортируется в CVAT / Label Studio как
«COCO 1.0», оператор исправляет рамки, экспортирует обратно в COCO и передаёт файл в
tools/train_heads.py --extra <каталог>/coco.json.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

from .config import settings
from .db import tx
from .detect import get_heads
from .detector import Letterbox
from .helmet import H_NO
from .helmet_crops import _frame


def export(camera_id: int, out: Path, n_frames: int = 300, seed: int = 1) -> dict:
    from PIL import Image

    rng = random.Random(seed)
    with tx() as conn:
        segs = conn.execute("SELECT id, width, height, detect_path FROM segment WHERE camera_id=%s "
                            "AND detect_status='done' AND detect_path IS NOT NULL", (camera_id,)).fetchall()
    cand = []
    for s in segs:
        bj = settings.archive_root / s["detect_path"] / "boxes.json"
        if bj.exists():
            for fr in json.loads(bj.read_text())["frames"]:
                if fr[1] or (len(fr) > 2 and fr[2]):
                    cand.append((s, fr[0]))
    rng.shuffle(cand)
    cand = sorted(cand[:n_frames], key=lambda c: (c[0]["id"], c[1]))
    heads = get_heads()
    (out / "images").mkdir(parents=True, exist_ok=True)
    coco = {"images": [], "annotations": [],
            "categories": [{"id": 1, "name": "helmet"}, {"id": 2, "name": "head"}]}
    frags_cache: dict[int, list] = {}
    for s, t in cand:
        if s["id"] not in frags_cache:
            with tx() as conn:
                frags_cache[s["id"]] = conn.execute(
                    "SELECT rel_path, offset_in_segment FROM fragment WHERE segment_id=%s ORDER BY offset_in_segment",
                    (s["id"],)).fetchall()
        fr = [f for f in frags_cache[s["id"]] if f["offset_in_segment"] <= t] or frags_cache[s["id"]][:1]
        f = fr[-1]
        w, h = s["width"], s["height"]
        img = _frame(settings.archive_root / f["rel_path"], t - f["offset_in_segment"], w, h)
        if img is None:
            continue
        name = f"images/s{s['id']}_t{t:08.2f}.jpg"
        Image.fromarray(img[:, :, ::-1]).save(out / name, quality=93)
        iid = len(coco["images"]) + 1
        coco["images"].append({"id": iid, "file_name": name, "width": w, "height": h})
        if heads is None:
            continue
        lb = Letterbox(w, h, heads.W, heads.H)
        sw, sh = lb.scaled
        x = np.full((heads.H, heads.W, 3), 114, np.uint8)
        x[:sh, :sw] = np.asarray(Image.fromarray(img[:, :, ::-1]).resize((sw, sh), Image.BILINEAR))[:, :, ::-1]
        for hd in heads(x, 1.0 / (w * lb.r), 1.0 / (h * lb.r)):
            x1, y1, x2, y2 = hd.box[0] * w, hd.box[1] * h, hd.box[2] * w, hd.box[3] * h
            coco["annotations"].append({"id": len(coco["annotations"]) + 1, "image_id": iid,
                                        "category_id": 2 if hd.code == H_NO else 1, "iscrowd": 0,
                                        "bbox": [round(x1, 1), round(y1, 1), round(x2 - x1, 1), round(y2 - y1, 1)],
                                        "area": round((x2 - x1) * (y2 - y1), 1), "score": hd.score})
    (out / "coco.json").write_text(json.dumps(coco, ensure_ascii=False), encoding="utf-8")
    return {"frames": len(coco["images"]), "heads": len(coco["annotations"]), "model": bool(heads)}
