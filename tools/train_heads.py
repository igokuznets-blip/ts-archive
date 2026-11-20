#!/usr/bin/env python3
"""Обучение детектора голов «в каске / без каски» (YOLOX-nano/tiny, Apache-2.0) на CPU.

Классы: 0 — helmet (голова в каске), 1 — head (голова без каски).
Данные: Hard Hat Workers (tools/get_hardhat.py, CC0) + свои кадры в формате COCO
(экспорт из CVAT/Label Studio, категории helmet и head) через --extra.

    python tools/train_heads.py --data /data/hardhat --weights yolox_nano.pth \
        --out models/heads_nano.onnx --epochs 30
    # дообучение на своих кадрах (повтор ×4, старт от готовой модели):
    python tools/train_heads.py --data /data/hardhat --extra /data/my_heads/coco.json \
        --resume models/heads_nano.pt --epochs 10 --out models/heads_nano.onnx

Вход ONNX: 1×3×640×640, BGR 0..255 (как у основной модели YOLOX), выход без декодирования —
его разбирает tsarchive.detector.YoloxDecoder.
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from yolox_min import YOLOX, YOLOPAFPN, YOLOXHead  # noqa: E402

CLASSES = ("helmet", "head")
LABEL_MAP = {"helmet": 0, "hardhat": 0, "head": 1, "no_helmet": 1, "nohelmet": 1}
GRAY = 114
SIZES = {"nano": (0.33, 0.25, True), "tiny": (0.33, 0.375, False)}


# ---------------------------------------------------------------- данные
def load_hardhat(root: Path) -> list[tuple[Path, np.ndarray]]:
    """samples.json FiftyOne → [(путь, [[cls,x1,y1,x2,y2] в долях])]."""
    out = []
    for s in json.load(open(root / "samples.json"))["samples"]:
        p = root / s["filepath"]
        if not p.exists():
            continue
        rows = []
        for d in (s.get("ground_truth") or {}).get("detections") or []:
            c = LABEL_MAP.get(d["label"])
            if c is None:
                continue
            x, y, w, h = d["bounding_box"]
            rows.append([c, x, y, x + w, y + h])
        out.append((p, np.array(rows, np.float32).reshape(-1, 5)))
    return out


def load_coco(path: Path) -> list[tuple[Path, np.ndarray]]:
    """COCO json (CVAT: «Export → COCO 1.0»), изображения рядом в images/ или по file_name."""
    j = json.load(open(path))
    cats = {c["id"]: LABEL_MAP.get(c["name"].lower()) for c in j["categories"]}
    imgs = {i["id"]: i for i in j["images"]}
    rows: dict[int, list] = {i: [] for i in imgs}
    for a in j["annotations"]:
        c = cats.get(a["category_id"])
        if c is None:
            continue
        im = imgs[a["image_id"]]
        x, y, w, h = a["bbox"]
        rows[a["image_id"]].append([c, x / im["width"], y / im["height"], (x + w) / im["width"],
                                    (y + h) / im["height"]])
    base = path.parent
    out = []
    for i, im in imgs.items():
        p = base / im["file_name"]
        if not p.exists():
            p = base / "images" / im["file_name"]
        if p.exists():
            out.append((p, np.array(rows[i], np.float32).reshape(-1, 5)))
    return out


def read_bgr(p: Path) -> np.ndarray:
    return np.asarray(Image.open(p).convert("RGB"))[:, :, ::-1]


def resize(img: np.ndarray, w: int, h: int) -> np.ndarray:
    return np.asarray(Image.fromarray(img[:, :, ::-1]).resize((max(1, w), max(1, h)), Image.BILINEAR))[:, :, ::-1]


def place(canvas, img, boxes, x0, y0, w, h, out_boxes):
    """Вписать img (масштаб w×h) в canvas в точку (x0,y0); рамки — доли исходного."""
    r = resize(img, w, h)
    H, W = canvas.shape[:2]
    cx0, cy0 = max(0, x0), max(0, y0)
    cx1, cy1 = min(W, x0 + w), min(H, y0 + h)
    if cx1 <= cx0 or cy1 <= cy0:
        return
    canvas[cy0:cy1, cx0:cx1] = r[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0]
    for c, a, b, cc, d in boxes:
        bx = np.clip([x0 + a * w, y0 + b * h, x0 + cc * w, y0 + d * h], [cx0, cy0, cx0, cy0], [cx1, cy1, cx1, cy1])
        ow, oh = (cc - a) * w, (d - b) * h
        nw, nh = bx[2] - bx[0], bx[3] - bx[1]
        if nw >= 3 and nh >= 3 and nw * nh >= 0.4 * ow * oh:  # обрезанные краем меньше чем на 60% — оставляем
            out_boxes.append([c, *bx])


def hsv_jitter(img: np.ndarray, rnd: random.Random) -> np.ndarray:
    f = img.astype(np.float32)
    f = f * rnd.uniform(0.7, 1.3) + rnd.uniform(-25, 25)  # яркость/контраст
    if rnd.random() < 0.3:
        g = f.mean(axis=2, keepdims=True)
        f = g + (f - g) * rnd.uniform(0.0, 0.6)  # «ночь/ИК» — почти без цвета
    return np.clip(f, 0, 255).astype(np.uint8)


def sample_train(items, S: int, rnd: random.Random):
    canvas = np.full((S, S, 3), GRAY, np.uint8)
    boxes: list = []
    if rnd.random() < 0.6:  # мозаика 2×2: головы мельче и разнообразнее фон
        xc, yc = int(S * rnd.uniform(0.3, 0.7)), int(S * rnd.uniform(0.3, 0.7))
        cells = [(0, 0, xc, yc), (xc, 0, S - xc, yc), (0, yc, xc, S - yc), (xc, yc, S - xc, S - yc)]
        for (x0, y0, cw, ch) in cells:
            p, b = rnd.choice(items)
            img = read_bgr(p)
            k = rnd.uniform(0.5, 1.2) * max(cw, ch) / max(img.shape[:2])
            w, h = int(img.shape[1] * k), int(img.shape[0] * k)
            ox = x0 + rnd.randint(min(0, cw - w), max(0, cw - w))
            oy = y0 + rnd.randint(min(0, ch - h), max(0, ch - h))
            sub = np.full((ch, cw, 3), GRAY, np.uint8)
            tmp: list = []
            place(sub, img, b, ox - x0, oy - y0, w, h, tmp)
            canvas[y0:y0 + ch, x0:x0 + cw] = sub
            boxes += [[c, a + x0, bb + y0, cc + x0, d + y0] for c, a, bb, cc, d in tmp]
    else:  # одно изображение, случайный масштаб (в т.ч. сильно уменьшенное — дальний план)
        p, b = rnd.choice(items)
        img = read_bgr(p)
        k = S / max(img.shape[:2]) * rnd.uniform(0.25, 1.3)
        w, h = int(img.shape[1] * k), int(img.shape[0] * k)
        place(canvas, img, b, rnd.randint(min(0, S - w), max(0, S - w)),
              rnd.randint(min(0, S - h), max(0, S - h)), w, h, boxes)
    if rnd.random() < 0.5:
        canvas = canvas[:, ::-1]
        boxes = [[c, S - x2, y1, S - x1, y2] for c, x1, y1, x2, y2 in boxes]
    canvas = hsv_jitter(np.ascontiguousarray(canvas), rnd)
    return canvas, np.array(boxes, np.float32).reshape(-1, 5)


def letterbox(img: np.ndarray, S: int):
    r = min(S / img.shape[0], S / img.shape[1])
    w, h = int(img.shape[1] * r), int(img.shape[0] * r)
    c = np.full((S, S, 3), GRAY, np.uint8)
    c[:h, :w] = resize(img, w, h)
    return c, r


def to_targets(batch_boxes, max_labels=120):
    t = torch.zeros(len(batch_boxes), max_labels, 5)
    for i, b in enumerate(batch_boxes):
        b = b[:max_labels]
        if len(b):
            cx, cy = (b[:, 1] + b[:, 3]) / 2, (b[:, 2] + b[:, 4]) / 2
            w, h = b[:, 3] - b[:, 1], b[:, 4] - b[:, 2]
            t[i, :len(b)] = torch.from_numpy(np.stack([b[:, 0], cx, cy, w, h], 1))
    return t


# ---------------------------------------------------------------- модель
def build(size: str) -> YOLOX:
    depth, width, dw = SIZES[size]
    in_ch = [256, 512, 1024]
    m = YOLOX(YOLOPAFPN(depth, width, in_channels=in_ch, depthwise=dw),
              YOLOXHead(len(CLASSES), width, in_channels=in_ch, depthwise=dw))
    for mod in m.modules():
        if isinstance(mod, torch.nn.BatchNorm2d):
            mod.eps, mod.momentum = 1e-3, 0.03
    return m


def load_weights(m: torch.nn.Module, path: Path) -> None:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck.get("model", ck)
    own = m.state_dict()
    ok = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}
    m.load_state_dict(ok, strict=False)
    print(f"веса {path.name}: загружено {len(ok)}/{len(own)} тензоров (cls_preds — заново)")


def decode(out: torch.Tensor, conf: float, nms_thr: float):
    """out: [N, A, 5+C] декодированный (eval-режим YOLOX) → список [K,6] (x1,y1,x2,y2,score,cls)."""
    import torchvision
    res = []
    for p in out:
        sc, cl = (p[:, 5:] * p[:, 4:5]).max(1)
        m = sc >= conf
        p, sc, cl = p[m], sc[m], cl[m]
        b = torch.stack([p[:, 0] - p[:, 2] / 2, p[:, 1] - p[:, 3] / 2, p[:, 0] + p[:, 2] / 2, p[:, 1] + p[:, 3] / 2], 1)
        k = torchvision.ops.batched_nms(b, sc, cl, nms_thr)
        res.append(torch.cat([b[k], sc[k, None], cl[k, None].float()], 1).numpy())
    return res


def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(br - tl, 0, None), 2)
    aa = np.prod(a[:, 2:] - a[:, :2], 1)
    bb = np.prod(b[:, 2:] - b[:, :2], 1)
    return inter / (aa[:, None] + bb[None] - inter + 1e-9)


def evaluate(m, items, S: int, conf: float = 0.35, batch: int = 8, upscale: float = 1.0):
    """AP50 по классам + точность/полнота при пороге conf (IoU 0,5).
    upscale<1 — изображение уменьшается перед вписыванием (имитация дальнего плана)."""
    m.eval()
    dets = {c: [] for c in range(len(CLASSES))}
    npos = {c: 0 for c in range(len(CLASSES))}
    with torch.no_grad():
        for i in range(0, len(items), batch):
            chunk = items[i:i + batch]
            ims, rs, gts = [], [], []
            for p, b in chunk:
                img = read_bgr(p)
                if upscale != 1.0:
                    img = resize(img, int(img.shape[1] * upscale), int(img.shape[0] * upscale))
                lb, r = letterbox(img, S)
                ims.append(lb)
                rs.append(r)
                g = b.copy()
                g[:, [1, 3]] *= img.shape[1]
                g[:, [2, 4]] *= img.shape[0]
                gts.append(g)
            x = torch.from_numpy(np.stack(ims)).permute(0, 3, 1, 2).float()
            for d, r, g in zip(decode(m(x), 0.01, 0.5), rs, gts):
                d[:, :4] /= r
                for c in dets:
                    gc = g[g[:, 0] == c][:, 1:]
                    npos[c] += len(gc)
                    dc = d[d[:, 5] == c]
                    dc = dc[np.argsort(-dc[:, 4])]
                    used = np.zeros(len(gc), bool)
                    ious = iou(dc[:, :4], gc) if len(gc) and len(dc) else np.zeros((len(dc), 0))
                    for j in range(len(dc)):
                        tp = 0
                        if ious.shape[1]:
                            k = int(ious[j].argmax())
                            if ious[j, k] >= 0.5 and not used[k]:
                                used[k] = True
                                tp = 1
                        dets[c].append((float(dc[j, 4]), tp))
    res = {}
    for c, name in enumerate(CLASSES):
        d = sorted(dets[c], key=lambda t: -t[0])
        tp = np.cumsum([t for _, t in d]) if d else np.zeros(0)
        fp = np.cumsum([1 - t for _, t in d]) if d else np.zeros(0)
        rec = tp / max(1, npos[c])
        prec = tp / np.maximum(1, tp + fp)
        ap = 0.0
        for t in np.linspace(0, 1, 101):
            pr = prec[rec >= t]
            ap += (pr.max() if pr.size else 0) / 101
        k = sum(1 for s, _ in d if s >= conf)
        res[name] = {"ap50": round(ap, 3), "gt": npos[c],
                     "recall": round(float(rec[k - 1]) if k else 0.0, 3),
                     "precision": round(float(prec[k - 1]) if k else 0.0, 3)}
    m.train()
    return res


def export(m, path: Path, S: int) -> None:
    m.eval()
    m.head.decode_in_inference = False
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(m, torch.zeros(1, 3, S, S), str(path), input_names=["images"], output_names=["output"],
                      opset_version=13, dynamo=False)
    m.head.decode_in_inference = True
    print("ONNX:", path)


# ---------------------------------------------------------------- обучение
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True, help="каталог get_hardhat.py (samples.json, data/)")
    ap.add_argument("--extra", type=Path, action="append", default=[], help="COCO json своих кадров")
    ap.add_argument("--extra-repeat", type=int, default=4)
    ap.add_argument("--size", choices=sorted(SIZES), default="nano")
    ap.add_argument("--weights", type=Path, help="предобученные веса COCO (yolox_nano.pth)")
    ap.add_argument("--resume", type=Path, help="продолжить с .pt этой же модели")
    ap.add_argument("--out", type=Path, default=Path("models/heads_nano.onnx"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--iters", type=int, default=0, help="итераций в эпохе (0 — по числу изображений)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--img", type=int, default=416, help="размер при обучении")
    ap.add_argument("--export-img", type=int, default=640, help="вход ONNX")
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--val", type=float, default=0.1)
    ap.add_argument("--val-list", type=Path, help="готовый список проверочных изображений (по имени файла)")
    ap.add_argument("--exclude", type=Path, help="не использовать эти изображения вовсе (тестовое видео)")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--eval-every", type=int, default=1, help="проверка раз в N эпох")
    ap.add_argument("--time-limit", type=float, default=0, help="остановиться через N минут")
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    rnd = random.Random(a.seed)
    torch.manual_seed(a.seed)

    items = load_hardhat(a.data)
    if a.exclude:
        ex = {Path(x.strip()).name for x in open(a.exclude) if x.strip()}
        items = [it for it in items if it[0].name not in ex]
    if a.val_list:
        vn = {Path(x.strip()).name for x in open(a.val_list) if x.strip()}
        val = [it for it in items if it[0].name in vn]
        train = [it for it in items if it[0].name not in vn]
    else:
        rnd.shuffle(items)
        nv = int(len(items) * a.val)
        val, train = items[:nv], items[nv:]
    for e in a.extra:
        ex_items = load_coco(e)
        print(f"{e}: {len(ex_items)} изображений")
        train += ex_items * a.extra_repeat
    print(f"обучение {len(train)}, проверка {len(val)}")

    m = build(a.size)
    if a.resume:
        m.load_state_dict(torch.load(a.resume, map_location="cpu"))
    elif a.weights:
        load_weights(m, a.weights)
    m.train()
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=a.lr, weight_decay=5e-4)
    iters = a.iters or max(1, len(train) // a.batch)
    total = iters * a.epochs
    warm = min(iters, 300)
    best, t0, step = -1.0, time.time(), 0
    pt = a.out.with_suffix(".pt")
    for ep in range(a.epochs):
        tl = 0.0
        for it in range(iters):
            lr = a.lr * (step / warm if step < warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm))))
            lr = max(lr, a.lr * 0.02)
            for g in opt.param_groups:
                g["lr"] = lr
            ims, bxs = zip(*(sample_train(train, a.img, rnd) for _ in range(a.batch)))
            x = torch.from_numpy(np.stack(ims)).permute(0, 3, 1, 2).float()
            loss = m(x, to_targets(bxs))["total_loss"]
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 10.0)
            opt.step()
            tl += float(loss)
            step += 1
            if it % 50 == 0:
                print(f"  эпоха {ep + 1} итерация {it}/{iters} loss {float(loss):.3f} lr {lr:.5f} "
                      f"{time.time() - t0:.0f} с", flush=True)
            if a.time_limit and time.time() - t0 > a.time_limit * 60:
                break
        last = ep + 1 == a.epochs or (a.time_limit and time.time() - t0 > a.time_limit * 60)
        if (ep + 1) % a.eval_every and not last:
            continue
        r = evaluate(m, val, a.export_img)
        score = r["head"]["ap50"] + r["helmet"]["ap50"]
        print(f"эпоха {ep + 1}: loss {tl / max(1, it + 1):.3f} {json.dumps(r, ensure_ascii=False)} "
              f"{time.time() - t0:.0f} с", flush=True)
        if score > best:
            best = score
            torch.save(m.state_dict(), pt)
            print("  лучшая, сохранено", pt)
        if a.time_limit and time.time() - t0 > a.time_limit * 60:
            print("лимит времени")
            break
    m.load_state_dict(torch.load(pt, map_location="cpu"))
    export(m, a.out, a.export_img)
    meta = {"classes": list(CLASSES), "size": a.size, "input": a.export_img, "train_images": len(train),
            "val_images": len(val), "val": evaluate(m, val, a.export_img),
            "val_far_x0.5": evaluate(m, val, a.export_img, upscale=0.5)}
    a.out.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    print(json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()
