#!/usr/bin/env python3
"""Сквозная проверка двухступенчатой схемы на размеченных изображениях (как в этапе detect):
YOLOX находит людей → квадрат головы от рамки человека → классификатор каски.
Каждая голова из разметки сопоставляется с человеком, у которого она в квадрате головы.

    python tools/eval_helmet.py --data /data/hardhat --list val.txt \
        --det models/yolox_s_int8.xml --cls models/helmet_cls.onnx

Печатает: долю голов, найденных через людей, и матрицу «разметка × решение» (в каске / без
каски / не определено) при текущих порогах HELMET_*.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from tsarchive.config import settings  # noqa: E402
from tsarchive.detector import COCO, Detector  # noqa: E402
from tsarchive.helmet import H_NO, H_UNKNOWN, H_YES, HelmetCheck, HelmetClassifier, head_square  # noqa: E402


def letterbox(img, W, H):
    h, w = img.shape[:2]
    r = min(W / w, H / h)
    nw, nh = int(round(w * r)) // 2 * 2, int(round(h * r)) // 2 * 2
    out = np.full((H, W, 3), 114, np.uint8)
    out[:nh, :nw] = np.asarray(Image.fromarray(img[:, :, ::-1]).resize((nw, nh), Image.BILINEAR))[:, :, ::-1]
    return out, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--list", type=Path, help="файл со списком изображений (по имени), иначе все")
    ap.add_argument("--det", required=True)
    ap.add_argument("--cls", required=True)
    ap.add_argument("--upscale", type=float, default=1.0, help="увеличить изображение перед обработкой")
    a = ap.parse_args()
    det = Detector(Path(a.det), device="CPU", hint="LATENCY", threads=0, streams="", cache_dir=None,
                   decoded=settings.detect_decoded)
    chk = HelmetCheck(HelmetClassifier(Path(a.cls)), settings.helmet_yes, settings.helmet_no,
                      settings.helmet_min_head_px,
                      tuple(float(x) for x in settings.helmet_scales.split(",") if x.strip()))
    names = set(Path(x).name for x in a.list.read_text().split()) if a.list else None
    samples = json.load(open(a.data / "samples.json"))["samples"]
    pid = COCO.index("person")
    M = np.zeros((2, 3), int)        # строки: helmet, head; столбцы: в каске, без каски, не определено
    missed = [0, 0]
    n_img = 0
    for s in samples:
        p = a.data / s["filepath"]
        if (names is not None and p.name not in names) or not p.exists():
            continue
        img = np.asarray(Image.open(p).convert("RGB"))[:, :, ::-1]
        if a.upscale != 1:
            img = np.asarray(Image.fromarray(img[:, :, ::-1]).resize(
                (int(img.shape[1] * a.upscale), int(img.shape[0] * a.upscale)), Image.BILINEAR))[:, :, ::-1]
        H, W = img.shape[:2]
        lb, r = letterbox(img, det.W, det.H)
        boxes, scores, cls = det.decoder(det.infer(lb), [pid], settings.detect_conf, settings.detect_nms)
        pb = [[b[0] / r / W, b[1] / r / H, b[2] / r / W, b[3] / r / H] for b in boxes]
        res = chk(img, pb) if pb else []
        sq = []
        for b in pb:
            cx, top, side = head_square(b[0] * W, b[1] * H, b[2] * W, b[3] * H)
            sq.append((cx - side / 2, top, cx + side / 2, top + side))
        n_img += 1
        for d in s["ground_truth"]["detections"]:
            if d["label"] not in ("helmet", "head"):
                continue
            x, y, bw, bh = d["bounding_box"]
            if bh * s["metadata"]["height"] * a.upscale < settings.helmet_min_head_px:
                continue
            gi = 0 if d["label"] == "helmet" else 1
            hx, hy = (x + bw / 2) * W, (y + bh / 2) * H
            cand = [k for k, q in enumerate(sq) if q[0] <= hx <= q[2] and q[1] <= hy <= q[3]]
            if not cand:
                missed[gi] += 1
                continue
            k = min(cand, key=lambda k: abs((sq[k][0] + sq[k][2]) / 2 - hx))
            code = res[k][0]
            M[gi, {H_YES: 0, H_NO: 1, H_UNKNOWN: 2}[code]] += 1
    tot = M.sum(1) + np.array(missed)
    print(f"изображений {n_img}; голов: в каске {tot[0]}, без каски {tot[1]}")
    print(f"человек не найден детектором: в каске {missed[0]}, без каски {missed[1]}")
    print("разметка \\ решение     в каске  без каски  не определено")
    for i, n in enumerate(("в каске", "без каски")):
        print(f"  {n:<20}{M[i, 0]:>8}{M[i, 1]:>11}{M[i, 2]:>15}")
    found = M[1].sum()
    if found:
        print(f"полнота «без каски» среди найденных людей: {M[1, 1] / found:.3f}")
    if M[:, 1].sum():
        print(f"точность «без каски»: {M[1, 1] / M[:, 1].sum():.3f}  (ложных тревог: {M[0, 1]})")


if __name__ == "__main__":
    main()
