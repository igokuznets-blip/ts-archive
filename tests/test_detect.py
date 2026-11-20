"""Тесты логики разметки без OpenVINO и БД: декодирование YOLOX, NMS, зоны, интервалы."""
from datetime import datetime, timedelta, timezone

import numpy as np

from tsarchive import detect
from tsarchive.detector import Letterbox, YoloxDecoder, nms


def test_letterbox_geometry():
    lb = Letterbox(1280, 720, 640, 640)
    assert lb.scaled == (640, 360)
    assert lb.ffmpeg_filter() == "scale=640:360:flags=bilinear,pad=640:640:0:0:color=0x727272"
    lb = Letterbox(720, 576, 416, 416)          # PAL D1, MPEG-2
    assert lb.scaled[1] == 332 and lb.scaled[0] <= 416


def test_yolox_decode_and_nms():
    W = H = 64
    dec = YoloxDecoder(W, H)
    n = dec.grid.shape[0]
    assert n == (8 * 8) + (4 * 4) + (2 * 2)
    out = np.full((1, n, 85), -10.0, np.float32)
    out[..., 4:] = 0.0
    # якорь сетки stride 8 в клетке (3, 2): центр (3.5*8, 2.5*8), размер exp(1)*8
    k = 2 * 8 + 3
    out[0, k, :4] = [0.5, 0.5, 1.0, 1.0]
    out[0, k, 4] = 0.9
    out[0, k, 5] = 0.95          # person
    out[0, k + 1, :4] = [0.4, 0.5, 1.0, 1.0]   # почти та же рамка — уйдёт в NMS
    out[0, k + 1, 4] = 0.8
    out[0, k + 1, 5] = 0.9
    boxes, scores, cls = dec(out, [0], 0.4, 0.45)
    assert len(scores) == 1 and cls[0] == 0
    cx, cy = (boxes[0, 0] + boxes[0, 2]) / 2, (boxes[0, 1] + boxes[0, 3]) / 2
    assert abs(cx - 28) < 1e-3 and abs(cy - 20) < 1e-3
    assert abs((boxes[0, 2] - boxes[0, 0]) - np.e * 8) < 1e-3
    assert abs(scores[0] - 0.9 * 0.95) < 1e-5


def test_nms_keeps_separate_people():
    b = np.array([[0, 0, 10, 20], [1, 0, 11, 20], [30, 0, 40, 20]], np.float32)
    assert sorted(nms(b, np.array([0.9, 0.8, 0.7]), 0.45)) == [0, 2]


def test_rules_ignore_and_zones():
    rules = detect.CameraRules(
        ignore=[[[0, 0], [0.3, 0], [0.3, 1], [0, 1]]],                    # левая треть — оператор в кабине
        zones={"Под стрелой": [[0.5, 0.5], [1, 0.5], [1, 1], [0.5, 1]]},   # правый нижний угол
        min_conf=0.5, min_height=0.05)
    boxes = np.array([[0.05, 0.2, 0.2, 0.8],     # в исключённой области
                      [0.6, 0.3, 0.7, 0.9],      # ноги в зоне
                      [0.4, 0.1, 0.45, 0.4],     # вне зоны
                      [0.8, 0.1, 0.81, 0.12],    # слишком мала
                      [0.4, 0.5, 0.45, 0.9]],    # низкая уверенность
                     np.float32)
    scores = np.array([0.9, 0.9, 0.8, 0.9, 0.3], np.float32)
    got = rules.apply(boxes, scores)
    assert [z for _, _, z in got] == [["Под стрелой"], []]


def test_interval_builder_merges_and_filters():
    closed = []
    b = detect.IntervalBuilder(fps=2, gap=3.0, min_dur=1.0, min_hits=2, on_close=closed.append)
    person = [([0.1, 0.1, 0.2, 0.5], 0.8, [])]
    two = person + [([0.5, 0.1, 0.6, 0.5], 0.9, ["Зона"])]
    seq = {0.0: person, 0.5: person, 1.0: [], 2.5: two, 3.0: person,   # пропуск 1.5 с — тот же интервал
           10.0: person,                                                # одиночное срабатывание — шум
           20.0: person, 20.5: person, 21.0: person}
    t = 0.0
    while t <= 25:
        b.feed(t, "person", seq.get(t, []), frame=np.zeros((2, 2, 3), np.uint8) if seq.get(t) else None)
        b.tick(t)
        t = round(t + 0.5, 3)
    ivs = b.finish()
    assert [(iv.start, iv.end(0.5)) for iv in ivs] == [(0.0, 3.5), (20.0, 21.5)]
    assert ivs[0].max_count == 2 and ivs[0].zones == {"Зона"} and ivs[0].best_t == 2.5
    assert len(closed) == 2


def test_offset_to_wall_and_polygon():
    t0 = datetime(2026, 1, 15, 7, 0, tzinfo=timezone.utc)
    frags = [{"offset_in_segment": 0.0, "ts_start": t0, "duration": 160.0},
             {"offset_in_segment": 160.0, "ts_start": t0 + timedelta(seconds=161.5), "duration": 160.0}]
    assert detect.offset_to_wall(frags, 10) == t0 + timedelta(seconds=10)
    assert detect.offset_to_wall(frags, 170) == t0 + timedelta(seconds=171.5)  # мелкий разрыв 1,5 с учтён
    sq = [[0, 0], [1, 0], [1, 1], [0, 1]]
    assert detect.point_in_poly(0.5, 0.5, sq) and not detect.point_in_poly(1.5, 0.5, sq)
