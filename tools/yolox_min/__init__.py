"""Минимальная копия моделей YOLOX (Megvii-BaseDetection/YOLOX, Apache-2.0, см. LICENSE) для
обучения детектора голов на CPU (tools/train_heads.py). Изменены только импорты: loguru → logging,
yolox.utils → .utils."""
from .yolo_head import YOLOXHead
from .yolo_pafpn import YOLOPAFPN
from .yolox import YOLOX
