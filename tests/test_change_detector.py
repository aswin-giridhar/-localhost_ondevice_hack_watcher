"""Tests for the cheap change detector."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from watcher.change_detector import ChangeDetector


def _blank(w=320, h=240):
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_no_fire_on_static_scene():
    cd = ChangeDetector(warmup_frames=5, debounce_seconds=0)
    fired = [cd.update(_blank()) for _ in range(30)]
    assert all(e is None for e in fired)


def test_fires_on_significant_change():
    cd = ChangeDetector(min_area=500, warmup_frames=5, debounce_seconds=0)
    for _ in range(10):
        cd.update(_blank())
    frame = _blank()
    frame[40:160, 40:200] = 255  # large bright block appears
    event = None
    for _ in range(3):  # allow background-diff to register
        event = cd.update(frame) or event
    assert event is not None
    assert event.changed_area >= 500


def test_debounce_suppresses_rapid_repeats():
    cd = ChangeDetector(min_area=500, warmup_frames=3, debounce_seconds=10)
    for _ in range(5):
        cd.update(_blank())
    frame = _blank()
    frame[40:160, 40:200] = 255
    first = None
    for _ in range(3):
        first = cd.update(frame) or first
    second = cd.update(frame)
    assert first is not None
    assert second is None  # within debounce window
