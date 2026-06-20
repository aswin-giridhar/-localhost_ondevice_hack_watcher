"""Frame sources: a local webcam, or frames pushed from the phone over a websocket.

The phone is a dumb eye. It streams JPEG frames to the laptop over the local
network (no internet). The StreamFrameSource is a thread-safe buffer that the
ingest websocket writes into and the pipeline reads from.
"""
from __future__ import annotations

import threading
import time
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


class DemoFrameSource(FrameSource):
    """Synthesizes a changing scene so the dashboard shows life without a phone.

    Every few seconds it adds/removes a coloured block, which trips the change
    detector and drives the full pipeline — useful for verifying the UI and for a
    fallback demo if no camera is available.
    """

    def __init__(self, width: int = 640, height: int = 480, period: float = 3.0) -> None:
        self.w, self.h, self.period = width, height, period
        self._blocks: list[tuple[int, int, int, tuple[int, int, int]]] = []
        self._t0 = time.time()
        self._last_change = 0.0
        self._palette = [
            (60, 160, 250), (80, 210, 130), (250, 200, 90),
            (230, 110, 120), (180, 130, 240),
        ]

    def read(self) -> np.ndarray | None:
        now = time.time()
        if now - self._last_change > self.period:
            self._last_change = now
            if self._blocks and (len(self._blocks) >= 4 or now % 2 < 1):
                self._blocks.pop(0)            # remove an object
            else:
                x = int(40 + (now * 53) % (self.w - 160))
                y = int(40 + (now * 31) % (self.h - 160))
                color = self._palette[len(self._blocks) % len(self._palette)]
                self._blocks.append((x, y, 90, color))
        frame = np.full((self.h, self.w, 3), 18, dtype=np.uint8)
        for (x, y, s, color) in self._blocks:
            cv2.rectangle(frame, (x, y), (x + s, y + s), color, -1)
        # mild texture so the trust-gate's blur check passes
        noise = np.random.randint(0, 12, (self.h, self.w, 3), dtype=np.uint8)
        return cv2.add(frame, noise)


def make_source(cfg) -> FrameSource:
    """Build a FrameSource from the capture config."""
    if cfg.source == "webcam":
        return WebcamFrameSource(cfg.webcam_index)
    if cfg.source == "demo":
        return DemoFrameSource()
    return StreamFrameSource()


def encode_jpeg(frame: np.ndarray, quality: int = 80) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else b""
