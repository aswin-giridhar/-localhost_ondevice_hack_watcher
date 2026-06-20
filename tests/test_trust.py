"""Tests for the Captur-style on-device trust gate."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from watcher.config import TrustCfg
from watcher.trust import TrustGate


def test_rejects_black_frame():
    gate = TrustGate(TrustCfg())
    black = np.zeros((240, 320, 3), dtype=np.uint8)
    res = gate.check(black)
    assert res.trusted is False


def test_rejects_blown_out_white_frame():
    gate = TrustGate(TrustCfg())
    white = np.full((240, 320, 3), 255, dtype=np.uint8)
    res = gate.check(white)
    assert res.trusted is False


def test_accepts_textured_frame():
    gate = TrustGate(TrustCfg())
    # High-detail mid-brightness frame -> high Laplacian variance, passes the gate.
    textured = np.random.randint(40, 200, (240, 320, 3), dtype=np.uint8)
    res = gate.check(textured)
    assert res.trusted is True
    assert 0.0 <= res.score <= 1.0


def test_disabled_gate_always_trusts():
    gate = TrustGate(TrustCfg(enabled=False))
    black = np.zeros((240, 320, 3), dtype=np.uint8)
    assert gate.check(black).trusted is True


def test_backend_is_local_without_sdk():
    gate = TrustGate(TrustCfg())
    assert gate.backend == "local-heuristics"
