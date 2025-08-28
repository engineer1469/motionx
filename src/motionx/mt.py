from __future__ import annotations
from typing import Deque, Dict, Tuple
import collections
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np

from .core import DiffParams, WeightedParams

# ------------------------------
# Stateless per-frame operations
# ------------------------------

def _ensure_odd(k: int) -> int:
    if k <= 0:
        return 0
    return k if (k % 2 == 1) else (k + 1)

def _preprocess_gray(frame_bgr: np.ndarray, blur_kernel: Tuple[int, int]) -> np.ndarray:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    kx, ky = blur_kernel
    if kx > 0 and ky > 0:
        kx = _ensure_odd(kx)
        ky = _ensure_odd(ky)
        gray = cv2.GaussianBlur(gray, (kx, ky), 0)
    return gray

def _diff_mask(prev_gray: np.ndarray, gray: np.ndarray, params: DiffParams) -> np.ndarray:
    diff = cv2.absdiff(prev_gray, gray)
    _, mask = cv2.threshold(diff, params.thresh, 255, cv2.THRESH_BINARY)
    if params.morph_kernel != (0, 0):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, params.morph_kernel)
        if params.morph_iters > 0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            mask = cv2.dilate(mask, kernel, iterations=params.morph_iters)
    return mask

def _diff_gray(prev_gray: np.ndarray, gray: np.ndarray) -> np.ndarray:
    return cv2.absdiff(prev_gray, gray)

def _weighted_bgr(curr_bgr: np.ndarray, prev_bgr: np.ndarray, params: WeightedParams) -> np.ndarray:
    motion = cv2.addWeighted(
        curr_bgr, params.alpha,
        cv2.bitwise_not(prev_bgr), params.beta,
        0,
    )
    if params.gain != 1.0:
        tmp = motion.astype(np.int16)
        tmp = 127 + params.gain * (tmp - 127)
        motion = np.clip(tmp, 0, 255).astype(np.uint8)
    return motion


# ---------------------------------------------
# Parallel runner for diff/diffgray/weighted
# ---------------------------------------------

def run_parallel(
    method: str,
    fs,                      # FrameSource
    sink,                    # FrameSink
    diff_params: DiffParams,
    weighted_params: WeightedParams,
    workers: int = 2,
    max_futures: int = 32,
):
    """
    Data-parallel execution for methods with fixed frame offsets:
      - diff:     uses offset k=1 on GRAY frames
      - diffgray: uses offset k=1 on GRAY frames
      - weighted: uses offset k=weighted_params.offset on BGR frames

    Results are written in-order. Memory is bounded by max_futures + small buffers.
    """
    assert workers >= 1 and max_futures >= 1
    method = method.lower()
    if method == "weighted":
        k = max(1, int(weighted_params.offset))
    elif method in ("diff", "diffgray"):
        k = 1
    else:
        raise ValueError("run_parallel only supports 'diff', 'diffgray', or 'weighted'")

    # State
    next_write = 0
    futures: Dict[int, "object"] = {}   # idx -> Future
    results: Dict[int, np.ndarray] = {} # idx -> processed frame

    # For diff/diffgray we carry the previous gray frame; to make tasks independent
    # we pass copies into each job.
    last_gray: np.ndarray | None = None

    # For weighted we need a ring buffer of the last k BGR frames.
    bgr_buffer: Deque[np.ndarray] | None = collections.deque(maxlen=k) if method == "weighted" else None

    def drain_ready():
        # Move finished futures to results
        done = [idx for idx, fut in list(futures.items()) if fut.done()]
        for idx in done:
            results[idx] = futures.pop(idx).result()
        # Write in order
        nonlocal next_write
        while next_write in results:
            sink.write(results.pop(next_write))
            next_write += 1

    with ThreadPoolExecutor(max_workers=workers) as ex:
        idx = 0
        while True:
            ok, frame = fs.read()
            if not ok:
                break

            if idx < k:
                # Warmup frames before we have a reference
                if method == "weighted":
                    results[idx] = np.full_like(frame, 127, dtype=np.uint8)
                    bgr_buffer.append(frame.copy())
                else:
                    gray = _preprocess_gray(frame, diff_params.blur_kernel)
                    results[idx] = np.zeros_like(gray, dtype=np.uint8)
                    last_gray = gray
                idx += 1
                drain_ready()
                continue

            # Backpressure: don't exceed max_futures in flight
            while len(futures) >= max_futures:
                drain_ready()

            if method == "weighted":
                assert bgr_buffer is not None and len(bgr_buffer) == k
                prev_bgr = bgr_buffer[0]
                curr_bgr = frame.copy()
                futures[idx] = ex.submit(_weighted_bgr, curr_bgr, prev_bgr.copy(), weighted_params)
                bgr_buffer.append(curr_bgr)
            elif method == "diff":
                gray = _preprocess_gray(frame, diff_params.blur_kernel)
                assert last_gray is not None
                prev_gray = last_gray.copy()
                futures[idx] = ex.submit(_diff_mask, prev_gray, gray.copy(), diff_params)
                last_gray = gray
            else: # diffgray
                gray = _preprocess_gray(frame, diff_params.blur_kernel)
                assert last_gray is not None
                prev_gray = last_gray.copy()
                futures[idx] = ex.submit(_diff_gray, prev_gray, gray.copy())
                last_gray = gray

            idx += 1
            drain_ready()

        # End of stream: wait for all to finish and write remaining
        while futures:
            drain_ready()

        # Any remaining (shouldn't be) — write in order
        drain_ready()
