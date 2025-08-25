import numpy as np
import cv2
from motionx import MotionExtractor

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
