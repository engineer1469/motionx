from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Tuple, Union
import glob
import os
import cv2
import numpy as np

_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".mpg", ".mpeg", ".m4v"}
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}

def _is_video_file(p: Union[str, Path]) -> bool:
    ext = Path(p).suffix.lower()
    return ext in _VIDEO_EXTS

def _looks_like_pattern(s: str) -> bool:
    # glob (*, ?) or printf (%d, %0Nd)
    return any(x in s for x in ["*", "?", "%d", "%0"])

@dataclass
class StreamInfo:
    width: int
    height: int
    fps: float

class FrameSource:
    """
    Unified reader:
      - video file (via cv2.VideoCapture)
      - webcam index (int: "0", "1", ...)
      - printf image sequence (cv2.VideoCapture supports "img_%06d.png")
      - directory or glob of images (manual reader)
    """
    def __init__(self, src: Union[str, int], in_fps: Optional[float] = None):
        self.src = src
        self.in_fps = in_fps
        self._cap: Optional[cv2.VideoCapture] = None
        self._files: Optional[List[str]] = None
        self._idx = 0
        self._info: Optional[StreamInfo] = None

        if isinstance(src, int) or (isinstance(src, str) and src.isdigit()):
            # webcam
            idx = int(src)
            self._cap = cv2.VideoCapture(idx)
        elif isinstance(src, str) and (_is_video_file(src) or _looks_like_pattern(src)):
            # video file or printf-pattern handled by VideoCapture
            self._cap = cv2.VideoCapture(src)
        else:
            # directory or glob -> manual image list
            p = Path(str(src))
            if p.is_dir():
                # collect images in dir
                files = []
                for ext in _IMAGE_EXTS:
                    files.extend(glob.glob(str(p / f"*{ext}")))
            else:
                # glob string
                files = glob.glob(str(src))
            self._files = sorted(files)
            if not self._files:
                raise FileNotFoundError(f"No frames found for: {src}")

        # probe basic info
        self._probe()

    def _probe(self) -> None:
        if self._cap is not None and self._cap.isOpened():
            w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self._cap.get(cv2.CAP_PROP_FPS) or 0.0
            if fps <= 1e-3:  # some sources report 0
                fps = self.in_fps or 30.0
            self._info = StreamInfo(width=w, height=h, fps=fps)
        else:
            # image list
            assert self._files is not None
            first = cv2.imread(self._files[0], cv2.IMREAD_COLOR)
            if first is None:
                raise RuntimeError(f"Could not read first frame: {self._files[0]}")
            h, w = first.shape[:2]
            fps = self.in_fps or 30.0
            self._info = StreamInfo(width=w, height=h, fps=fps)

    @property
    def info(self) -> StreamInfo:
        assert self._info is not None
        return self._info

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        if self._cap is not None and self._cap.isOpened():
            ok, frame = self._cap.read()
            return ok, frame if ok else None
        else:
            assert self._files is not None
            if self._idx >= len(self._files):
                return False, None
            f = self._files[self._idx]
            self._idx += 1
            img = cv2.imread(f, cv2.IMREAD_COLOR)
            if img is None:
                return False, None
            return True, img

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()

class FrameSink:
    """
    Unified writer:
      - video file (mp4/avi) via VideoWriter
      - image sequence via template: "out/frame_%06d.png"
      - image directory via out_dir + auto "frame_%06d.png"
    """
    def __init__(
        self,
        output: Optional[str],
        frame_size: Tuple[int, int],
        fps: float,
        out_seq: Optional[str] = None,
        out_dir: Optional[str] = None,
    ):
        self._writer: Optional[cv2.VideoWriter] = None
        self._template: Optional[str] = None
        self._counter = 0

        w, h = frame_size
        # Decide mode:
        if out_seq:
            # image sequence template
            self._template = out_seq
            # ensure directory exists
            Path(out_seq).parent.mkdir(parents=True, exist_ok=True)
        elif out_dir:
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            self._template = str(Path(out_dir) / "frame_%06d.png")
        elif output and _is_video_file(output):
            fourcc = cv2.VideoWriter_fourcc(*("mp4v" if output.lower().endswith(".mp4") else "XVID"))
            self._writer = cv2.VideoWriter(output, fourcc, fps, (w, h), isColor=False)
            if not self._writer.isOpened():
                raise RuntimeError(f"Could not create video writer: {output}")
        elif output and _looks_like_pattern(output):
            # treat as image sequence template too
            self._template = output
            Path(output).parent.mkdir(parents=True, exist_ok=True)
        else:
            raise ValueError(
                "Specify a video file path (.mp4/.avi), or use --out-seq TEMPLATE, or --out-dir DIR."
            )

    def write(self, frame_gray_or_mask: np.ndarray) -> None:
        if self._writer is not None:
            self._writer.write(frame_gray_or_mask)
        elif self._template:
            fname = self._template % self._counter
            self._counter += 1
            cv2.imwrite(fname, frame_gray_or_mask)
        else:
            raise RuntimeError("FrameSink not initialized")

    def release(self) -> None:
        if self._writer is not None:
            self._writer.release()
