"""Frame sources: a local webcam, or frames pushed from the phone over a websocket.

The phone is a dumb eye. It streams JPEG frames to the laptop over the local
network (no internet). The StreamFrameSource is a thread-safe buffer that the
ingest websocket writes into and the pipeline reads from.
"""
from __future__ import annotations

import threading
from abc import ABC, abstractmethod

import cv2
import numpy as np


class FrameSource(ABC):
    @abstractmethod
    def read(self) -> np.ndarray | None:
        """Return the most recent frame (BGR ndarray) or None if unavailable."""

    def release(self) -> None:  # noqa: B027 - optional override
        pass


class WebcamFrameSource(FrameSource):
    """Pulls from a local camera index — useful for laptop-only testing."""

    def __init__(self, index: int = 0) -> None:
        self._cap = cv2.VideoCapture(index)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open webcam index {index}")

    def read(self) -> np.ndarray | None:
        ok, frame = self._cap.read()
        return frame if ok else None

    def release(self) -> None:
        self._cap.release()


class StreamFrameSource(FrameSource):
    """Holds the latest frame pushed by the phone over the ingest websocket."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest: np.ndarray | None = None

    def push_jpeg(self, jpeg_bytes: bytes) -> bool:
        arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return False
        with self._lock:
            self._latest = frame
        return True

    def read(self) -> np.ndarray | None:
        with self._lock:
            return None if self._latest is None else self._latest.copy()


def make_source(cfg) -> FrameSource:
    """Build a FrameSource from the capture config."""
    if cfg.source == "webcam":
        return WebcamFrameSource(cfg.webcam_index)
    return StreamFrameSource()


def encode_jpeg(frame: np.ndarray, quality: int = 80) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else b""
