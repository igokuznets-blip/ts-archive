"""Детектор голов «в каске / без каски» (YOLOX-nano, tools/train_heads.py) и сопоставление голов
с рамками людей.

Почему отдельный детектор, а не квадрат головы из рамки человека: если люди стоят вплотную,
YOLOX выдаёт одну общую рамку или вовсе не находит закрытого человека, а квадрат, вычисленный
из рамки, попадает между головами. Детектор голов находит каждую голову сам, поэтому:
  * число людей в кадре = рамки людей + «лишние» головы внутри общей рамки + уверенные головы
    без рамки человека (видна только голова и плечи);
  * решение «без каски» принимается по самой голове, а не по смещённому квадрату.

Модель работает на том же кадре, что и детектор людей (вписан во вход 640×640), второго
декодирования видео нет.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .helmet import H_NO, H_UNKNOWN, H_YES

log = logging.getLogger(__name__)
CLASSES = ("helmet", "head")  # 0 — в каске, 1 — без каски


@dataclass
class Head:
    box: list[float]       # доли кадра 0..1
    score: float
    code: int              # H_YES / H_NO
    person: int = -1       # индекс рамки человека или -1


class HeadDetector:
    def __init__(self, path: Path, device: str = "CPU", threads: int = 0, cache_dir: str | None = None,
                 conf: float = 0.35, no_conf: float = 0.45, min_px: float = 6, nms: float = 0.45):
        from .detector import Detector
        self.det = Detector(path, device=device, hint="LATENCY", threads=threads, cache_dir=cache_dir)
        self.model_name = self.det.model_name
        self.W, self.H = self.det.W, self.det.H
        self.conf, self.no_conf, self.min_px, self.nms = conf, no_conf, min_px, nms
        log.info("детектор голов %s: вход %dx%d, классы %s", self.model_name, self.W, self.H, ",".join(CLASSES))

    def __call__(self, frame_lb: np.ndarray, fx: float, fy: float) -> list[Head]:
        """frame_lb — кадр, вписанный во вход детектора людей (того же размера). fx, fy — пиксели → доли."""
        if frame_lb.shape[0] != self.H or frame_lb.shape[1] != self.W:
            raise ValueError(f"вход детектора голов {self.W}x{self.H}, кадр {frame_lb.shape[1]}x{frame_lb.shape[0]}")
        out = self.det.infer(frame_lb)
        boxes, scores, cls = self.det.decoder(out, [0, 1], self.conf, self.nms)
        if len(boxes) > 1:  # одна голова не может быть и «в каске», и «без каски»: NMS без учёта класса
            from .detector import nms
            keep = sorted(nms(boxes, scores, self.nms), key=lambda k: -scores[k])
            boxes, scores, cls = boxes[keep], scores[keep], cls[keep]
        heads = []
        for b, s, c in zip(boxes, scores, cls):
            if min(b[2] - b[0], b[3] - b[1]) < self.min_px:
                continue
            if c == 1 and s < self.no_conf:
                continue  # неуверенное «без каски» не считаем (ни туда, ни сюда)
            nb = [round(float(np.clip(v * k, 0, 1)), 4) for v, k in zip(b, (fx, fy, fx, fy))]
            heads.append(Head(nb, round(float(s), 3), H_NO if c == 1 else H_YES))
        return heads


def associate(persons: list[list[float]], heads: list[Head]) -> list[list[int]]:
    """Каждой голове — рамка человека, в верхней части которой лежит её центр (ближайшая по
    горизонтали, если таких несколько). → для каждой рамки человека список индексов голов."""
    owned: list[list[int]] = [[] for _ in persons]
    for hi, h in enumerate(heads):
        cx, cy = (h.box[0] + h.box[2]) / 2, (h.box[1] + h.box[3]) / 2
        hw = h.box[2] - h.box[0]
        best, bd = -1, 1e9
        for pi, p in enumerate(persons):
            pw, ph = p[2] - p[0], p[3] - p[1]
            if not (p[0] - 0.3 * hw <= cx <= p[2] + 0.3 * hw):
                continue
            if not (p[1] - 0.6 * hw <= cy <= p[1] + max(0.45 * ph, 1.5 * hw)):
                continue  # голова — в верхней части рамки (или чуть выше, если рамка срезала макушку)
            d = abs(cx - (p[0] + p[2]) / 2) / max(pw, 1e-6)
            if d < bd:
                best, bd = pi, d
        h.person = best
        if best >= 0:
            owned[best].append(hi)
    return owned


def person_codes(owned: list[list[int]], heads: list[Head]) -> list[tuple[int, float]]:
    """Код рамки человека по её головам: есть голова без каски → H_NO, есть в каске → H_YES."""
    res = []
    for idx in owned:
        no = [heads[i].score for i in idx if heads[i].code == H_NO]
        yes = [heads[i].score for i in idx if heads[i].code == H_YES]
        if no:
            res.append((H_NO, max(no)))
        elif yes:
            res.append((H_YES, 0.0))
        else:
            res.append((H_UNKNOWN, 0.0))
    return res


def people_count(owned: list[list[int]], heads: list[Head], alone_conf: float) -> tuple[int, list[int]]:
    """Число людей: каждая рамка — минимум один человек, в общей рамке — по числу голов, плюс
    уверенные головы вне рамок. → (число, индексы голов, добавленных как отдельные люди)."""
    n = sum(max(1, len(o)) for o in owned)
    extra = []
    for o in owned:
        extra += o[1:]  # вторая и следующие головы в одной рамке
    alone = [i for i, h in enumerate(heads) if h.person < 0 and h.score >= alone_conf]
    return n + len(alone), extra + alone
