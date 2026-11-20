#!/usr/bin/env python3
"""Обучение классификатора каски (helmet / no_helmet / unclear) для этапа detect.

Запускается отдельно от архива, на любой машине с Python 3.10+ (GPU не нужен):

    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    pip install numpy pillow onnx openvino
    python tools/get_hardhat.py --out /data/hardhat           # датасет Hard Hat Workers (CC0)
    python tools/train_helmet.py --data /data/hardhat --out models/helmet_cls.onnx
    # дообучение на кадрах своего архива:
    python tools/train_helmet.py --data /data/hardhat --extra /data/my_heads --out models/helmet_cls.onnx

--extra — каталог с подкаталогами helmet/, no_helmet/, unclear/ (jpg/png) — например, квадраты
голов, выгруженные командой `python -m tsarchive helmet-crops` и разложенные вручную. Такие
примеры повторяются ×--extra-repeat за эпоху, чтобы перевесить общий датасет.

Квадраты голов для обучения строятся так же, как в работе (tsarchive.helmet.head_square), но
с разбросом масштаба и сдвига: на работе квадрат берётся от рамки человека, а не от рамки
головы из разметки. Класс unclear — квадраты туловища, фона и частично срезанных голов.
На выходе ONNX: вход N×96×96×3 (BGR, 0..255, float), выход — логиты 3 классов.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from tsarchive.helmet import CLASSES, crop_square  # noqa: E402

SIZE = 96


def load_hardhat(root: Path):
    """Формат FiftyOne (samples.json, Voxel51/hard-hat-detection) или Pascal VOC (annotations/*.xml)."""
    items = []
    sj = root / "samples.json"
    if sj.exists():
        for s in json.load(open(sj))["samples"]:
            p = root / s["filepath"]
            if not p.exists():
                p = root / "data" / Path(s["filepath"]).name
            if not p.exists():
                continue
            w, h = s["metadata"]["width"], s["metadata"]["height"]
            boxes = []
            for d in s["ground_truth"]["detections"]:
                if d["label"] not in ("helmet", "head"):
                    continue
                x, y, bw, bh = d["bounding_box"]
                boxes.append((d["label"], x * w, y * h, (x + bw) * w, (y + bh) * h))
            items.append((p, boxes))
        return items
    import xml.etree.ElementTree as ET
    for xf in sorted((root / "annotations").glob("*.xml")):
        t = ET.parse(xf).getroot()
        p = root / "images" / t.findtext("filename")
        boxes = []
        objs = list(t.iter("object"))
        # SHWD: hat — каска, person — голова без каски; Hard Hat (VOC): helmet, head, person = силуэт
        shwd = any(o.findtext("name") == "hat" for o in objs)
        for o in objs:
            name = o.findtext("name")
            if name == "person" and not shwd:
                continue
            if name not in ("helmet", "head", "hat", "person"):
                continue
            lab = {"hat": "helmet", "person": "head"}.get(name, name)
            bb = o.find("bndbox")
            boxes.append((lab, *(float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))))
        if p.exists():
            items.append((p, boxes))
    return items


def overlap(a, b) -> float:
    """Доля площади b внутри a."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix * iy / max(1e-6, (b[2] - b[0]) * (b[3] - b[1]))


def degrade(patch: np.ndarray, rng: random.Random, min_head: float) -> np.ndarray:
    """Имитация реальной камеры: низкое разрешение головы, размытие, ИК-ночь (ч/б), шум, JPEG."""
    im = Image.fromarray(patch[:, :, ::-1])
    r = rng.uniform(min_head * 1.6, SIZE) if rng.random() < 0.6 else SIZE
    if r < SIZE - 2:
        im = im.resize((int(r), int(r)), Image.BILINEAR).resize((SIZE, SIZE), Image.BILINEAR)
    if rng.random() < 0.15:
        im = im.filter(ImageFilter.GaussianBlur(rng.uniform(0.4, 1.2)))
    a = np.asarray(im, dtype=np.float32)
    if rng.random() < 0.2:
        a = a.mean(axis=2, keepdims=True).repeat(3, axis=2)
    a = a * rng.uniform(0.6, 1.35) + rng.uniform(-30, 30)
    if rng.random() < 0.3:
        a = a + np.random.default_rng(rng.randrange(1 << 30)).normal(0, rng.uniform(2, 8), a.shape)
    return np.clip(a, 0, 255).astype(np.uint8)[:, :, ::-1]


def make_crops(items, rng: random.Random, n_var: dict, min_head: float, train: bool):
    X, Y = [], []
    for p, boxes in items:
        img = np.asarray(Image.open(p).convert("RGB"))[:, :, ::-1]
        H, W = img.shape[:2]
        heads = [b[1:] for b in boxes]
        for lab, x1, y1, x2, y2 in boxes:
            hs = max(x2 - x1, y2 - y1)
            if hs < min_head:
                continue
            cls = 0 if lab == "helmet" else 1
            for _ in range(n_var[cls] if train else 1):
                # квадрат как от рамки человека: голова в верхней части, со сдвигом и масштабом
                s = hs * (rng.uniform(1.25, 2.2) if train else 1.6)
                cx = (x1 + x2) / 2 + (rng.uniform(-0.17, 0.17) * s if train else 0)
                top = y1 - (rng.uniform(0.0, 0.2) * s if train else 0.08 * s)
                patch = crop_square(img, cx, top, s, SIZE)
                if train and rng.random() < 0.5:
                    patch = patch[:, ::-1]
                X.append(degrade(patch, rng, min_head) if train else patch)
                Y.append(cls)
            # unclear: туловище под головой (человек наклонился / голова за кадром или закрыта)
            if rng.random() < (0.5 if train else 0.3):
                s = hs * rng.uniform(1.4, 2.4)
                top = y2 + rng.uniform(0.3, 1.5) * hs
                sq = ((x1 + x2) / 2 - s / 2, top, (x1 + x2) / 2 + s / 2, top + s)
                if top + s * 0.5 < H and all(overlap(sq, h) < 0.25 for h in heads):
                    patch = crop_square(img, (x1 + x2) / 2 + rng.uniform(-0.2, 0.2) * s, top, s, SIZE)
                    X.append(degrade(patch, rng, min_head) if train else patch)
                    Y.append(2)
        # unclear: фон без голов
        for _ in range(2 if train else 1):
            s = rng.uniform(24, min(H, W) / 2.5)
            cx, top = rng.uniform(s / 2, W - s / 2), rng.uniform(0, H - s)
            sq = (cx - s / 2, top, cx + s / 2, top + s)
            if all(overlap(sq, h) < 0.15 for h in heads):
                patch = crop_square(img, cx, top, s, SIZE)
                X.append(degrade(patch, rng, min_head) if train else patch)
                Y.append(2)
    return X, Y


def load_extra(root: Path, repeat: int, rng: random.Random, min_head: float):
    X, Y = [], []
    for ci, name in enumerate(CLASSES):
        for f in sorted((root / name).glob("*")):
            if f.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            a = np.asarray(Image.open(f).convert("RGB").resize((SIZE, SIZE), Image.BILINEAR))[:, :, ::-1]
            for k in range(repeat):
                X.append(degrade(a[:, ::-1] if k % 2 else a, rng, min_head) if k else a)
                Y.append(ci)
    return X, Y


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--extra", type=Path)
    ap.add_argument("--extra-repeat", type=int, default=6)
    ap.add_argument("--out", type=Path, default=Path("models/helmet_cls.onnx"))
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--val", type=float, default=0.15, help="доля изображений на проверку")
    ap.add_argument("--min-head", type=float, default=12, help="меньше — не обучаем (пиксели)")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--split-out", type=Path, help="сохранить список проверочных изображений")
    a = ap.parse_args()

    import torch
    import torch.nn as nn
    import torchvision

    if a.threads:
        torch.set_num_threads(a.threads)
    rng = random.Random(a.seed)
    torch.manual_seed(a.seed)
    items = load_hardhat(a.data)
    if not items:
        sys.exit(f"в {a.data} не найдено изображений с разметкой")
    rng.shuffle(items)
    nv = int(len(items) * a.val)
    val_items, tr_items = items[:nv], items[nv:]
    if a.split_out:
        a.split_out.write_text("\n".join(str(p) for p, _ in val_items))
    t0 = time.time()
    # на каску ~3,3 примера без каски: голов без каски дублируем с разными искажениями
    Xv, Yv = make_crops(val_items, random.Random(7), {}, a.min_head, train=False)
    print(f"изображений: обучение {len(tr_items)}, проверка {len(val_items)}; "
          f"проверочных квадратов {len(Xv)} ({np.bincount(Yv, minlength=3).tolist()})", flush=True)
    Xv = torch.from_numpy(np.stack(Xv))
    Yv = torch.tensor(Yv)

    class Net(nn.Module):
        """BGR uint8-диапазон NHWC → нормализация ImageNet → MobileNetV3-Small."""

        def __init__(self):
            super().__init__()
            m = torchvision.models.mobilenet_v3_small(
                weights=torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
            m.classifier[3] = nn.Linear(m.classifier[3].in_features, len(CLASSES))
            self.m = m
            self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1) * 255)
            self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) * 255)

        def forward(self, x):
            x = x.permute(0, 3, 1, 2).flip(1)  # NHWC BGR → NCHW RGB
            return self.m((x - self.mean) / self.std)

    net = Net()
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    steps_per_epoch = None
    sched = None
    best = (-1.0, None)
    for ep in range(a.epochs):
        Xt, Yt = make_crops(tr_items, rng, {0: 1, 1: 3}, a.min_head, train=True)
        if a.extra:
            xe, ye = load_extra(a.extra, a.extra_repeat, rng, a.min_head)
            Xt += xe
            Yt += ye
        Xt = torch.from_numpy(np.stack(Xt))
        Yt = torch.tensor(Yt)
        cnt = torch.bincount(Yt, minlength=3).float()
        wcls = (cnt.sum() / (3 * cnt.clamp(min=1)))
        lossf = nn.CrossEntropyLoss(weight=wcls, label_smoothing=0.05)
        n = len(Yt)
        if sched is None:
            steps_per_epoch = (n + a.batch - 1) // a.batch
            sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps_per_epoch * a.epochs + 50,
                                                        pct_start=0.15)
        net.train()
        perm = torch.randperm(n)
        tl, te = 0.0, time.time()
        for i in range(0, n, a.batch):
            j = perm[i:i + a.batch]
            out = net(Xt[j].float())
            loss = lossf(out, Yt[j])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tl += float(loss.detach()) * len(j)
        net.eval()
        with torch.no_grad():
            pv = torch.cat([net(Xv[i:i + 512].float()).argmax(1) for i in range(0, len(Yv), 512)])
        rec = [float(((pv == c) & (Yv == c)).sum() / max(1, int((Yv == c).sum()))) for c in range(3)]
        prec_no = float(((pv == 1) & (Yv == 1)).sum() / max(1, int((pv == 1).sum())))
        bal = sum(rec) / 3
        print(f"эпоха {ep + 1}/{a.epochs}: примеров {n} {cnt.int().tolist()}, loss {tl / n:.3f}, "
              f"полнота helmet {rec[0]:.3f} no_helmet {rec[1]:.3f} unclear {rec[2]:.3f}, "
              f"точность no_helmet {prec_no:.3f}, {time.time() - te:.0f} с", flush=True)
        if bal > best[0]:
            best = (bal, {k: v.clone() for k, v in net.state_dict().items()})
    net.load_state_dict(best[1])
    net.eval()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best[1], a.out.with_suffix(".pt"))  # веса — на случай ошибки экспорта
    dummy = torch.zeros(1, SIZE, SIZE, 3)
    torch.onnx.export(net, dummy, str(a.out), input_names=["images"], output_names=["logits"],
                      dynamic_axes={"images": {0: "n"}, "logits": {0: "n"}}, opset_version=17, dynamo=False)
    print(f"сохранено: {a.out} (лучшая сбалансированная полнота {best[0]:.3f}), всего {time.time() - t0:.0f} с")


if __name__ == "__main__":
    main()
