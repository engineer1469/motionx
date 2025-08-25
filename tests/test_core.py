import numpy as np
import cv2
from motionx import MotionExtractor, DiffParams

def test_diff_basic():
    h, w = 64, 64
    f1 = np.zeros((h, w, 3), dtype=np.uint8)
    f2 = f1.copy()
    cv2.rectangle(f2, (10, 10), (20, 20), (255, 255, 255), -1)

    me = MotionExtractor(method="diff")
    m1 = me.process(f1)
    m2 = me.process(f2)
    assert m1.sum() == 0
    assert m2.sum() > 0

def test_diff_thresholding():
    h, w = 32, 32
    f1 = np.zeros((h, w, 3), dtype=np.uint8)
    f2 = f1.copy()
    cv2.rectangle(f2, (5, 5), (10, 10), (5, 5, 5), -1)  # small delta
    f3 = f1.copy()
    cv2.rectangle(f3, (5, 5), (10, 10), (255, 255, 255), -1)  # large delta

    params = DiffParams(thresh=10, blur_kernel=(0, 0), morph_kernel=(0, 0), morph_iters=0)
    me = MotionExtractor(method="diff", diff_params=params)
    m1 = me.process(f1)
    m2 = me.process(f2)
    m3 = me.process(f3)
    assert m1.sum() == 0
    # delta below threshold -> no motion
    assert m2.sum() == 0
    # large delta -> motion detected
    assert m3.sum() > 0

def test_weighted_first_frame_neutral_gray():
    h, w = 48, 48
    f1 = np.zeros((h, w, 3), dtype=np.uint8)
    me = MotionExtractor(method="weighted")
    out = me.process(f1)
    assert out.dtype == np.uint8
    assert out.shape == f1.shape
    # neutral gray for default alpha=beta=0.5
    assert np.all(out == 127)

def test_weighted_highlights_motion_roi():
    h, w = 64, 64
    f1 = np.zeros((h, w, 3), dtype=np.uint8)
    f2 = f1.copy()
    cv2.rectangle(f2, (16, 16), (48, 48), (255, 255, 255), -1)

    me = MotionExtractor(method="weighted")
    _ = me.process(f1)  # warm up buffer
    out = me.process(f2)
    # background should be near 127, ROI should be much brighter
    bg = out[0:10, 0:10]
    roi = out[20:44, 20:44]
    bg_mean = float(bg.mean())
    roi_mean = float(roi.mean())
    assert roi_mean > bg_mean + 50

def test_weighted_offset_behavior():
    h, w = 32, 32
    f = np.zeros((h, w, 3), dtype=np.uint8)
    me = MotionExtractor(method="weighted")
    # default offset=1 -> first frame neutral gray, second frame influenced by previous
    out1 = me.process(f)
    out2 = me.process(f)
    assert np.all(out1 == 127)
    # second should still be valid image and not all 127 due to blending with inverted prev
    assert out2.shape == f.shape
    assert out2.dtype == np.uint8
    assert not np.all(out2 == 127)
