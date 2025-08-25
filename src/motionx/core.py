from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import cv2
import numpy as np
import collections

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

@dataclass
class WeightedParams:
    """
    Parameters for the weighted motion extraction algorithm.

    ``alpha`` and ``beta`` are the blending weights for the current frame and the
    inverted previous frame, respectively.  ``offset`` specifies how many
    frames to delay before computing the difference; a value of 1 means the
    immediate previous frame.
    """
    alpha: float = 0.5
    beta: float = 0.5
    offset: int = 1

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
        weighted_params: Optional[WeightedParams] = None,
    ) -> None:
        method = method.lower()
        if method not in {"diff", "mog2", "weighted"}:
            raise ValueError("method must be 'diff', 'mog2' or 'weighted'")
        self.method = method
        self.diff_params = diff_params or DiffParams()
        self.mog2_params = mog2_params or Mog2Params()
        self.weighted_params = weighted_params or WeightedParams()
        self._prev_gray: Optional[np.ndarray] = None
        self._frame_buffer: Optional[collections.deque] = None
        self._mog2 = None
        if self.method == "mog2":
            self._mog2 = cv2.createBackgroundSubtractorMOG2(
                history=self.mog2_params.history,
                varThreshold=self.mog2_params.var_threshold,
                detectShadows=self.mog2_params.detect_shadows,
            )
        elif self.method == "weighted":
            # initialise a ring buffer for the weighted method
            self._frame_buffer = collections.deque(maxlen=self.weighted_params.offset)

    def reset(self) -> None:
        self._prev_gray = None
        # reset weighted buffer if necessary
        if self.method == "weighted":
            self._frame_buffer = collections.deque(maxlen=self.weighted_params.offset)

    def _preprocess(self, frame_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if self.diff_params.blur_kernel != (0, 0):
            gray = cv2.GaussianBlur(gray, self.diff_params.blur_kernel, 0)
        return gray

    def process(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Returns a motion representation for the given frame.  For the 'diff' and
        'mog2' methods this will be a binary mask (0/255).  For the 'weighted'
        method this will be a full colour image where static pixels are neutral
        grey and motion is highlighted.
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

        elif self.method == "weighted":
            # ensure buffer exists
            assert self._frame_buffer is not None
            # if buffer not filled yet, append and return neutral grey image
            if len(self._frame_buffer) < self.weighted_params.offset:
                self._frame_buffer.append(frame_bgr.copy())
                return np.full_like(frame_bgr, 127, dtype=np.uint8)
            # pick the oldest frame in the buffer
            prev_frame = self._frame_buffer[0]
            # compute weighted blend of current frame and inverted previous frame
            motion_frame = cv2.addWeighted(
                frame_bgr,
                self.weighted_params.alpha,
                cv2.bitwise_not(prev_frame),
                self.weighted_params.beta,
                0,
            )
            # update buffer
            self._frame_buffer.append(frame_bgr.copy())
            return motion_frame

        else:  # mog2
            assert self._mog2 is not None
            fgmask = self._mog2.apply(frame_bgr)  # shadows ~ 127 if detectShadows=True
            # binarize to 0/255 (ignore shadows)
            _, mask = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
            return mask
