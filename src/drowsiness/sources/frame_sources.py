"""Frame sources.

All sources give monotonic timestamps in seconds:
  * WebcamSource     - wall-clock (time.monotonic) since start
  * VideoFileSource  - frame index / FPS, so processing a file is deterministic
                       and independent of how fast the laptop is
  * read_features    - recorded FrameFeatures (no pixels at all) for replay

``read()`` returns a Frame, or None when the camera failed to deliver one
(the pipeline turns that into a camera-health observation). ``finished``
becomes True only at the real end of a file.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from pathlib import Path

import cv2

from drowsiness.schemas.detection import Frame, FrameFeatures


class WebcamSource:
    def __init__(self, index: int = 0, width: int = 640, height: int = 480, mirror: bool = True) -> None:
        self._cap = cv2.VideoCapture(index)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._mirror = mirror
        self._t0 = time.monotonic()
        self._index = 0
        self.finished = False

    @property
    def opened(self) -> bool:
        return bool(self._cap.isOpened())

    def now(self) -> float:
        return time.monotonic() - self._t0

    def read(self) -> Frame | None:
        ok, img = self._cap.read()
        t = self.now()
        if not ok or img is None:
            return None
        if self._mirror:
            img = cv2.flip(img, 1)  # a mirrored view is natural for the person in front of it
        self._index += 1
        return Frame(img, t, self._index)

    def close(self) -> None:
        self._cap.release()


class VideoFileSource:
    def __init__(self, path: str | Path) -> None:
        self._cap = cv2.VideoCapture(str(path))
        if not self._cap.isOpened():
            raise FileNotFoundError(f"cannot open video: {path}")
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps if fps and fps > 1 else 30.0
        self._index = 0
        self.finished = False

    @property
    def opened(self) -> bool:
        return True

    def now(self) -> float:
        return self._index / self.fps

    def read(self) -> Frame | None:
        ok, img = self._cap.read()
        if not ok or img is None:
            self.finished = True
            return None
        frame = Frame(img, self._index / self.fps, self._index)
        self._index += 1
        return frame

    def close(self) -> None:
        self._cap.release()


def read_features(path: str | Path) -> Iterator[FrameFeatures]:
    """Iterate a features.jsonl recording."""
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield FrameFeatures.from_dict(json.loads(line))
