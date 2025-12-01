from pathlib import Path
from typing import Iterator, Tuple, Optional

import cv2
import numpy as np


class VideoReader:
    """Lightweight wrapper around OpenCV VideoCapture.

    This abstracts away frame iteration so the rest of the pipeline
    can treat a video as an iterator over (frame_idx, frame_array).
    """

    def __init__(self, path: str | Path):
        self.path = str(path)
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video: {self.path}")
        self._frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._fps = float(self.cap.get(cv2.CAP_PROP_FPS))

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def fps(self) -> float:
        return self._fps

    def iter_frames(
        self,
        start: int = 0,
        end: Optional[int] = None,
        stride: int = 1,
    ) -> Iterator[Tuple[int, np.ndarray]]:
        """Yield (frame_idx, frame) from start to end (exclusive)."""
        if end is None:
            end = self._frame_count
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        idx = start
        while idx < end:
            ret, frame = self.cap.read()
            if not ret:
                break
            yield idx, frame
            # skip frames if stride > 1
            if stride > 1:
                next_idx = idx + stride
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, next_idx)
                idx = next_idx
            else:
                idx += 1

    def read_frame(self, idx: int):
        """Random access: read a single frame by index."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError(f"Could not read frame {idx}")
        return frame

    def close(self):
        self.cap.release()
