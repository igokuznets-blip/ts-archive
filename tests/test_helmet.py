import numpy as np

from tsarchive.helmet import H_NO, H_UNKNOWN, H_YES, crop_square, decide, head_square


def test_head_square_standing_person():
    # стоящий человек 110×300 px: голова ~40 px, квадрат ~66 px сверху по центру
    cx, top, s = head_square(100, 50, 210, 350)
    assert cx == 155
    assert 60 <= s <= 70
    assert top < 50 and top > 40


def test_head_square_crouching_limited_by_width():
    cx, top, s = head_square(0, 0, 130, 150)
    assert s <= 130 * 1.1 and s <= 150 * 0.6
    assert s >= 0.45 * 130 - 1e-6


def test_crop_square_size_and_padding():
    img = np.zeros((100, 200, 3), np.uint8)
    img[:, :100] = 255
    c = crop_square(img, 100, -20, 80, 32)      # квадрат выходит за верх кадра
    assert c.shape == (32, 32, 3)
    assert (c[0] == 114).all()                  # поля за краем — серые
    assert c[-1, 0, 0] == 255 and c[-1, -1, 0] == 0


def test_crop_square_downscale_averages():
    img = np.zeros((64, 64, 3), np.uint8)
    img[::2, ::2] = 255                          # «шахматка» усредняется, а не пропускается
    c = crop_square(img, 32, 0, 64, 16)
    assert 40 < c.mean() < 90


def test_decide_thresholds():
    assert decide(np.array([0.1, 0.8, 0.1]), 0.6, 0.7) == H_NO
    assert decide(np.array([0.7, 0.2, 0.1]), 0.6, 0.7) == H_YES
    assert decide(np.array([0.4, 0.5, 0.1]), 0.6, 0.7) == H_UNKNOWN
    assert decide(np.array([0.1, 0.1, 0.8]), 0.6, 0.7) == H_UNKNOWN
