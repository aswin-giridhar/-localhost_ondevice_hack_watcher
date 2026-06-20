"""Cheap, always-on change detection.

This is the gate that makes Watcher run on a 4GB GPU: it inspects every frame
with near-zero cost (grayscale diff against a running background) and only fires
when something meaningful changes. The expensive VLM is woken a few times per
*minute*, not per frame.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class ChangeEvent:
    """A debounced, significant change in the scene."""
    timestamp: float
    changed_area: int          # total area of changed pixels (px)
    bbox: tuple[int, int, int, int]  # x, y, w, h of the largest changed region


class ChangeDetector:
    def __init__(
        self,
        min_area: int = 1500,
        threshold: int = 25,
        debounce_seconds: float = 2.0,
        warmup_frames: int = 15,
    ) -> None:
        self.min_area = min_area
        self.threshold = threshold
        self.debounce_seconds = debounce_seconds
        self.warmup_frames = warmup_frames

        self._bg: np.ndarray | None = None
        self._frames_seen = 0
        self._last_fire = 0.0

    @staticmethod
    def _prep(frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(gray, (21, 21), 0)

    def update(self, frame: np.ndarray) -> ChangeEvent | None:
        """Feed a frame. Returns a ChangeEvent only on a significant, debounced change."""
        self._frames_seen += 1
        prepped = self._prep(frame)

        if self._bg is None:
            self._bg = prepped.astype("float")
            return None

        # Running average background so slow lighting drift doesn't trigger.
        cv2.accumulateWeighted(prepped, self._bg, 0.1)

        if self._frames_seen < self.warmup_frames:
            return None

        delta = cv2.absdiff(prepped, cv2.convertScaleAbs(self._bg))
        thresh = cv2.threshold(delta, self.threshold, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)

        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        area = int(cv2.contourArea(largest))
        if area < self.min_area:
            return None

        now = time.time()
        if now - self._last_fire < self.debounce_seconds:
            return None
        self._last_fire = now

        x, y, w, h = cv2.boundingRect(largest)
        return ChangeEvent(timestamp=now, changed_area=area, bbox=(x, y, w, h))

    def reset(self) -> None:
        self._bg = None
        self._frames_seen = 0
        self._last_fire = 0.0
