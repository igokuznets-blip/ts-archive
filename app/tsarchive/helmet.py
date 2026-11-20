"""Проверка каски у найденных людей (второй шаг этапа detect).

Схема двухступенчатая: YOLOX находит людей, затем по каждой рамке человека вырезается
квадрат головы из кадра в исходном разрешении, и маленький классификатор (MobileNetV3-Small,
вход 96×96) относит его к одному из классов:

    0 helmet     — голова в каске
    1 no_helmet  — голова без каски
    2 unclear    — головы в квадрате нет / не видна (человек наклонился, закрыт техникой, спиной
                   в капюшоне и т. п.)

Решение по человеку: no_helmet, если p(no_helmet) ≥ HELMET_NO_THRESH; helmet, если
p(helmet) ≥ HELMET_YES_THRESH; иначе «не определено». Голова меньше HELMET_MIN_HEAD_PX пикселей
исходного кадра не классифицируется — на такой картинке каску не различить.

Модель обучается скриптом tools/train_helmet.py на открытом датасете Hard Hat Workers (CC0)
и при необходимости — на кадрах собственного архива (см. README).
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

CLASSES = ("helmet", "no_helmet", "unclear")
# коды в boxes.json
H_UNKNOWN, H_YES, H_NO = 0, 1, 2

# Геометрия квадрата головы относительно рамки человека. Та же функция используется при
# обучении (tools/train_helmet.py имитирует её разбросом), поэтому менять согласованно.
HEAD_K_H = 0.22   # сторона ≥ 22% высоты человека
HEAD_K_W = 0.45   # и ≥ 45% ширины
HEAD_TOP = 0.08   # квадрат начинается чуть выше рамки: верх каски часто выходит за неё


def head_square(x1: float, y1: float, x2: float, y2: float) -> tuple[float, float, float]:
    """Рамка человека (пиксели) → (cx, top, side) квадрата головы."""
    w, h = max(1.0, x2 - x1), max(1.0, y2 - y1)
    s = max(HEAD_K_H * h, HEAD_K_W * w)
    s = min(s, w * 1.1, h * 0.6)
    return (x1 + x2) / 2, y1 - HEAD_TOP * s, s


def crop_square(img: np.ndarray, cx: float, top: float, s: float, out: int) -> np.ndarray:
    """Квадрат из кадра (с серыми полями за краем) → out×out: выборка по сетке k×k на пиксель
    результата с усреднением (антиалиасинг при уменьшении), без OpenCV."""
    H, W = img.shape[:2]
    x0, y0 = cx - s / 2, top
    k = int(min(4, max(1, np.ceil(s / out))))
    g = (np.arange(out * k) + 0.5) / (out * k) * s
    xs = np.floor(x0 + g).astype(np.int64)
    ys = np.floor(y0 + g).astype(np.int64)
    vx = (xs >= 0) & (xs < W)
    vy = (ys >= 0) & (ys < H)
    patch = img[np.clip(ys, 0, H - 1)][:, np.clip(xs, 0, W - 1)].astype(np.float32)
    patch[~vy, :] = 114
    patch[:, ~vx] = 114
    if k > 1:
        patch = patch.reshape(out, k, out, k, 3).mean(axis=(1, 3))
    return patch.astype(np.uint8)


class HelmetClassifier:
    def __init__(self, path: Path, device: str = "CPU", size: int = 96, threads: int = 0,
                 cache_dir: str | None = None):
        import openvino as ov
        from openvino.preprocess import PrePostProcessor

        core = ov.Core()
        if cache_dir:
            core.set_property({"CACHE_DIR": cache_dir})
        model = core.read_model(str(path))
        inp = model.input(0)
        shape = inp.get_partial_shape()
        self.size = int(shape[1].get_length()) if shape[1].is_static and shape[3].is_static and \
            shape[3].get_length() == 3 else size
        ppp = PrePostProcessor(model)
        ppp.input().tensor().set_element_type(ov.Type.u8)
        ppp.input().preprocess().convert_element_type(ov.Type.f32)
        model = ppp.build()
        cfg = {"PERFORMANCE_HINT": "LATENCY"}
        if threads:
            cfg["INFERENCE_NUM_THREADS"] = int(threads)
        self.compiled = core.compile_model(model, device, cfg)
        self.req = self.compiled.create_infer_request()
        self.model_name = Path(path).stem
        self.calls, self.crops, self.sec = 0, 0, 0.0
        log.info("модель касок %s: вход %d×%d, классы %s", self.model_name, self.size, self.size,
                 ",".join(CLASSES))

    def probs(self, crops: np.ndarray) -> np.ndarray:
        """crops: N×S×S×3 uint8 BGR → N×3 вероятностей."""
        if not len(crops):
            return np.zeros((0, 3), np.float32)
        t0 = time.monotonic()
        out = self.req.infer({0: np.ascontiguousarray(crops)})
        logits = np.asarray(next(iter(out.values())), dtype=np.float32)
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        p = e / e.sum(axis=1, keepdims=True)
        self.calls += 1
        self.crops += len(crops)
        self.sec += time.monotonic() - t0
        return p


def decide(p: np.ndarray, yes_thr: float, no_thr: float) -> int:
    if p[1] >= no_thr:
        return H_NO
    if p[0] >= yes_thr:
        return H_YES
    return H_UNKNOWN


class HelmetCheck:
    """Классификация всех людей кадра. Рамки — доли кадра 0..1, кадр — исходное разрешение."""

    def __init__(self, clf: HelmetClassifier, yes_thr: float, no_thr: float, min_head_px: float,
                 scales: tuple[float, ...] = (1.0,)):
        self.clf, self.yes_thr, self.no_thr, self.min_head_px = clf, yes_thr, no_thr, min_head_px
        self.scales = scales or (1.0,)

    def __call__(self, frame: np.ndarray, boxes: list[list[float]]) -> list[tuple[int, float]]:
        """→ [(код H_*, p(no_helmet))] по каждой рамке."""
        H, W = frame.shape[:2]
        res: list[tuple[int, float]] = [(H_UNKNOWN, 0.0)] * len(boxes)
        idx, crops = [], []
        ns = len(self.scales)
        for i, b in enumerate(boxes):
            cx, top, s = head_square(b[0] * W, b[1] * H, b[2] * W, b[3] * H)
            if s / 1.6 < self.min_head_px:  # голова ≈ сторона квадрата / 1,6
                continue
            idx.append(i)
            for k in self.scales:  # несколько масштабов квадрата: поза и посадка рамки человека разные
                crops.append(crop_square(frame, cx, top - (k - 1) * s * 0.15, s * k, self.clf.size))
        if crops:
            p = self.clf.probs(np.stack(crops)).reshape(len(idx), ns, 3)
            for i, pp in zip(idx, p):
                # квадраты, где голова видна (unclear < 0,5), усредняются; если таких нет — все
                ok = pp[pp[:, 2] < 0.5]
                pi = (ok if len(ok) else pp).mean(axis=0)
                res[i] = (decide(pi, self.yes_thr, self.no_thr), round(float(pi[1]), 3))
        return res
