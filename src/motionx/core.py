from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import cv2
import numpy as np

@dataclass
class DiffParams:
    thresh: int = 25
    blur_kernel: Tuple[int, int] = (5, 5)
    morph_kernel: Tuple[int, int] = (3, 3)
    morph_iters: int = 1

@dataclass
class Mog2Params:
    history: int = 500
    var_threshold: float = 16.0
    detect_shadows: bool = True

class MotionExtractor:
    """
    Simple motion extraction:
    - 'diff': abs difference of consecutive grayscale frames + threshold + morphology
    - 'mog2': OpenCV BackgroundSubtractorMOG2
    """
    def __init__(
        self,
        method: str = "diff",
        diff_params: Optional[DiffParams] = None,
        mog2_params: Optional[Mog2Params] = None,
    ) -> None:
        method = method.lower()
        if method not in {"diff", "mog2"}:
            raise ValueError("method must be 'diff' or 'mog2'")
        self.method = method
        self.diff_params = diff_params or DiffParams()
        self.mog2_params = mog2_params or Mog2Params()
        self._prev_gray: Optional[np.ndarray] = None
        self._mog2 = None
        if self.method == "mog2":
            self._mog2 = cv2.createBackgroundSubtractorMOG2(
                history=self.mog2_params.history,
                varThreshold=self.mog2_params.var_threshold,
                detectShadows=self.mog2_params.detect_shadows,
            )

    def reset(self) -> None:
        self._prev_gray = None

    def _preprocess(self, frame_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if self.diff_params.blur_kernel != (0, 0):
            gray = cv2.GaussianBlur(gray, self.diff_params.blur_kernel, 0)
        return gray

    def process(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Returns a uint8 mask (0/255) of motion regions for the given frame.
        """
        if self.method == "diff":
            gray = self._preprocess(frame_bgr)
            if self._prev_gray is None:
                self._prev_gray = gray
                return np.zeros_like(gray, dtype=np.uint8)

            diff = cv2.absdiff(self._prev_gray, gray)
            _, mask = cv2.threshold(diff, self.diff_params.thresh, 255, cv2.THRESH_BINARY)

            if self.diff_params.morph_kernel != (0, 0):
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_RECT, self.diff_params.morph_kernel
                )
                if self.diff_params.morph_iters > 0:
                    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
                    mask = cv2.dilate(mask, kernel, iterations=self.diff_params.morph_iters)

            self._prev_gray = gray
            return mask

        else:  # mog2
            assert self._mog2 is not None
            fgmask = self._mog2.apply(frame_bgr)  # shadows ~ 127 if detectShadows=True
            # binarize to 0/255 (ignore shadows)
            _, mask = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
            return mask
