"""Детектор объектов (люди) на OpenVINO: модель YOLOX в формате ONNX или OpenVINO IR (FP32/INT8).

Предобработка (BGR uint8 NHWC → float32 NCHW) встроена в граф модели через PrePostProcessor,
поэтому на Python остаётся только постобработка (декодирование сетки YOLOX + NMS) на numpy.
Буквенный бокс (letterbox) делает FFmpeg при декодировании: scale с сохранением пропорций и
заполнение справа и снизу цветом 114, как в обучении YOLOX.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

COCO = ("person bicycle car motorcycle airplane bus train truck boat traffic_light fire_hydrant stop_sign "
        "parking_meter bench bird cat dog horse sheep cow elephant bear zebra giraffe backpack umbrella handbag "
        "tie suitcase frisbee skis snowboard sports_ball kite baseball_bat baseball_glove skateboard surfboard "
        "tennis_racket bottle wine_glass cup fork knife spoon bowl banana apple sandwich orange broccoli carrot "
        "hot_dog pizza donut cake chair couch potted_plant bed dining_table toilet tv laptop mouse remote keyboard "
        "cell_phone microwave oven toaster sink refrigerator book clock vase scissors teddy_bear hair_drier "
        "toothbrush").split()


@dataclass
class Letterbox:
    """Геометрия: исходный кадр w×h вписан в вход модели W×H с коэффициентом r (слева сверху)."""
    src_w: int
    src_h: int
    W: int
    H: int

    @property
    def r(self) -> float:
        return min(self.W / self.src_w, self.H / self.src_h)

    @property
    def scaled(self) -> tuple[int, int]:
        # чётные размеры — требование yuv-фильтров FFmpeg
        sw = max(2, int(self.src_w * self.r) // 2 * 2)
        sh = max(2, int(self.src_h * self.r) // 2 * 2)
        return min(sw, self.W), min(sh, self.H)

    def ffmpeg_filter(self) -> str:
        sw, sh = self.scaled
        return f"scale={sw}:{sh}:flags=bilinear,pad={self.W}:{self.H}:0:0:color=0x727272"


def nms(boxes: np.ndarray, scores: np.ndarray, thr: float) -> list[int]:
    """Классический NMS по IoU. boxes: [N,4] x1y1x2y2."""
    order = scores.argsort()[::-1]
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    keep = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]
        w = (np.minimum(x2[i], x2[rest]) - np.maximum(x1[i], x1[rest])).clip(0)
        h = (np.minimum(y2[i], y2[rest]) - np.maximum(y1[i], y1[rest])).clip(0)
        inter = w * h
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[iou <= thr]
    return keep


class YoloxDecoder:
    """Декодирование «сырого» выхода YOLOX [1, N, 5+C]: (cx, cy, w, h) в долях сетки + obj + классы."""

    def __init__(self, W: int, H: int, decoded: bool = False, strides=(8, 16, 32)):
        self.decoded = decoded
        grids, st = [], []
        for s in strides:
            gx, gy = np.meshgrid(np.arange(W // s), np.arange(H // s))
            g = np.stack((gx, gy), 2).reshape(-1, 2)
            grids.append(g)
            st.append(np.full((g.shape[0], 1), s))
        self.grid = np.concatenate(grids).astype(np.float32)
        self.stride = np.concatenate(st).astype(np.float32)

    def __call__(self, out: np.ndarray, class_ids: list[int], conf: float, nms_thr: float):
        """→ boxes [K,4] в пикселях входа модели, scores [K], cls [K]."""
        p = out.reshape(-1, out.shape[-1])
        obj = p[:, 4]
        cls_scores = p[:, 5:][:, class_ids] * obj[:, None]
        best = cls_scores.argmax(1)
        score = cls_scores[np.arange(len(best)), best]
        idx = np.nonzero(score >= conf)[0]
        if not idx.size:
            return np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.int64)
        q = p[idx, :4]
        if not self.decoded:
            if self.grid.shape[0] != p.shape[0]:
                raise ValueError(f"выход модели {p.shape[0]} якорей не совпадает с сеткой {self.grid.shape[0]}")
            xy = (q[:, :2] + self.grid[idx]) * self.stride[idx]
            wh = np.exp(np.minimum(q[:, 2:4], 10.0)) * self.stride[idx]
        else:
            xy, wh = q[:, :2], q[:, 2:4]
        boxes = np.concatenate((xy - wh / 2, xy + wh / 2), 1)
        score, cls = score[idx], best[idx]
        keep = []
        for c in np.unique(cls):  # NMS по каждому классу отдельно
            m = np.nonzero(cls == c)[0]
            keep.extend(m[nms(boxes[m], score[m], nms_thr)])
        keep = np.array(sorted(keep, key=lambda k: -score[k]), dtype=np.int64)
        return boxes[keep], score[keep], np.array(class_ids)[cls[keep]]


class Detector:
    """Скомпилированная модель OpenVINO + очередь асинхронных запросов (режим THROUGHPUT)."""

    def __init__(self, model_path: str | Path, device: str = "CPU", hint: str = "THROUGHPUT",
                 threads: int = 0, streams: str = "", cache_dir: str | None = None, decoded: bool = False):
        import openvino as ov
        from openvino.preprocess import PrePostProcessor

        core = ov.Core()
        if cache_dir:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            core.set_property({"CACHE_DIR": str(cache_dir)})
        model = core.read_model(str(model_path))
        shape = model.input(0).get_partial_shape()
        if shape.is_dynamic:
            raise ValueError("нужна модель с фиксированным размером входа (например, 1x3x640x640)")
        _, _, self.H, self.W = [d.get_length() for d in shape]
        ppp = PrePostProcessor(model)
        ppp.input().tensor().set_element_type(ov.Type.u8).set_layout(ov.Layout("NHWC"))
        ppp.input().model().set_layout(ov.Layout("NCHW"))
        ppp.input().preprocess().convert_element_type(ov.Type.f32)
        model = ppp.build()
        cfg = {"PERFORMANCE_HINT": hint}
        if threads:
            cfg["INFERENCE_NUM_THREADS"] = int(threads)
        if streams:
            cfg["NUM_STREAMS"] = streams
        self.compiled = core.compile_model(model, device, cfg)
        self.nreq = max(1, self.compiled.get_property("OPTIMAL_NUMBER_OF_INFER_REQUESTS"))
        self.decoder = YoloxDecoder(self.W, self.H, decoded=decoded)
        self.model_name = Path(model_path).stem
        self._ov = ov
        log.info("модель %s: вход %dx%d, устройство %s, запросов %d", self.model_name, self.W, self.H,
                 device, self.nreq)

    def queue(self, on_result):
        """AsyncInferQueue: on_result(userdata, output ndarray) вызывается из потока OpenVINO."""
        q = self._ov.AsyncInferQueue(self.compiled, self.nreq)

        def cb(req, userdata):
            on_result(userdata, req.get_output_tensor(0).data.copy())
        q.set_callback(cb)
        return q

    def infer(self, frame_hwc: np.ndarray) -> np.ndarray:
        """Синхронный вызов (тесты, калибровка)."""
        return self.compiled(frame_hwc[None])[0]


class Reorder:
    """Результаты асинхронных запросов приходят не по порядку — выдаём строго по индексу кадра."""

    def __init__(self):
        self.lock = threading.Condition()
        self.ready: dict[int, object] = {}
        self.next = 0

    def put(self, idx: int, item) -> None:
        with self.lock:
            self.ready[idx] = item
            self.lock.notify_all()

    def pop_ready_pairs(self) -> list[tuple[int, object]]:
        """[(индекс, результат)] — непрерывная по индексам часть, готовая к обработке."""
        out = []
        with self.lock:
            while self.next in self.ready:
                out.append((self.next, self.ready.pop(self.next)))
                self.next += 1
        return out
