from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Deque
import collections
import cv2
import numpy as np

@dataclass
class DiffParams:
    """
    Parameters for the 'diff' method (binary frame differencing).
    """
    thresh: int = 25
    blur_kernel: Tuple[int, int] = (5, 5)
    morph_kernel: Tuple[int, int] = (3, 3)
    morph_iters: int = 1

@dataclass
class Mog2Params:
    """
    Parameters for the MOG2 background subtraction method.
    """
    history: int = 500
    var_threshold: float = 16.0
    detect_shadows: bool = True

@dataclass
class WeightedParams:
    """
    Parameters for the 'weighted' motion extraction algorithm.

    alpha and beta are the blending weights for the current frame and the
    inverted previous frame, respectively. offset specifies how many
    frames to delay before computing the difference (>= 1). gain adjusts
    the contrast around mid-gray (gain > 1 amplifies subtle motion).
    """
    alpha: float = 0.5
    beta: float = 0.5
    offset: int = 1
    gain: float = 1.0

class MotionExtractor:
    """
    High-level motion extraction.

    Supported methods:
    - 'diff'      → binary frame differencing
    - 'diffgray'  → grayscale absolute difference (no threshold)
    - 'mog2'      → background subtraction (MOG2)
    - 'weighted'  → blend current frame with inverted previous frame (Posy-style)
    """
    def __init__(
        self,
        method: str = "diff",
        diff_params: Optional[DiffParams] = None,
        mog2_params: Optional[Mog2Params] = None,
        weighted_params: Optional[WeightedParams] = None,
    ) -> None:
        method = method.lower()
        if method not in {"diff", "mog2", "weighted", "diffgray"}:
            raise ValueError("method must be 'diff', 'mog2', 'weighted' or 'diffgray'")
        self.method = method
        self.diff_params = diff_params or DiffParams()
        self.mog2_params = mog2_params or Mog2Params()
        self.weighted_params = weighted_params or WeightedParams()
        self._prev_gray: Optional[np.ndarray] = None
        self._mog2: Optional[cv2.BackgroundSubtractor] = None
        self._frame_buffer: Optional[Deque[np.ndarray]] = None

        if self.method == "mog2":
            self._mog2 = cv2.createBackgroundSubtractorMOG2(
                history=self.mog2_params.history,
                varThreshold=self.mog2_params.var_threshold,
                detectShadows=self.mog2_params.detect_shadows,
            )
        elif self.method == "weighted":
            # initialize a ring buffer for the weighted method
            self._frame_buffer = collections.deque(maxlen=self.weighted_params.offset)

    def reset(self) -> None:
        self._prev_gray = None
        # reset the weighted buffer if necessary
        if self.method == "weighted":
            self._frame_buffer = collections.deque(maxlen=self.weighted_params.offset)

    def _preprocess(self, frame_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        kx, ky = self.diff_params.blur_kernel
        if kx > 0 and ky > 0:
            # enforce odd kernel sizes for GaussianBlur
            if kx % 2 == 0:
                kx += 1
            if ky % 2 == 0:
                ky += 1
            gray = cv2.GaussianBlur(gray, (kx, ky), 0)
        return gray

    def process(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Process a single frame and return the motion representation.

        - For 'diff' and 'mog2', returns a binary mask (uint8).
        - For 'diffgray', returns a grayscale difference image (uint8).
        - For 'weighted', returns a BGR image where static pixels are neutral
          gray and motion edges are highlighted. If WeightedParams.gain ≠ 1,
          applies a gain around mid-gray to enhance subtle motion.
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

        elif self.method == "diffgray":
            gray = self._preprocess(frame_bgr)
            if self._prev_gray is None:
                self._prev_gray = gray
                return np.zeros_like(gray, dtype=np.uint8)
            diff = cv2.absdiff(self._prev_gray, gray)
            self._prev_gray = gray
            return diff

        elif self.method == "weighted":
            assert self._frame_buffer is not None
            if len(self._frame_buffer) < self.weighted_params.offset:
                # warm up: return neutral gray until buffer fills
                self._frame_buffer.append(frame_bgr.copy())
                return np.full_like(frame_bgr, 127, dtype=np.uint8)
            prev_frame = self._frame_buffer[0]
            motion_frame = cv2.addWeighted(
                frame_bgr,
                self.weighted_params.alpha,
                cv2.bitwise_not(prev_frame),
                self.weighted_params.beta,
                0,
            )
            # apply gain if != 1.0
            gain = self.weighted_params.gain
            if gain != 1.0:
                temp = motion_frame.astype(np.int16)
                temp = 127 + gain * (temp - 127)
                temp = np.clip(temp, 0, 255)
                motion_frame = temp.astype(np.uint8)
            self._frame_buffer.append(frame_bgr.copy())
            return motion_frame

        else:  # mog2
            assert self._mog2 is not None
            fgmask = self._mog2.apply(frame_bgr)
            # binarize foreground mask to 0/255 (ignore shadows)
            _, mask = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
            return mask
