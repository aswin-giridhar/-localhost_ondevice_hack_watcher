"""On-device photo-trust gate (Captur-style).

Before the agent reasons about a frame, it validates that the frame is a genuine,
usable on-device capture: not blank, not frozen-black, not unusably blurry. If the
Captur SDK is present it is used; otherwise local, fully-offline heuristics run.

This is the "images never leave the device" trust layer — nothing here touches the
network.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class TrustResult:
    trusted: bool
    score: float                       # 0..1 confidence the frame is a valid capture
    reasons: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)


class TrustGate:
    def __init__(self, cfg) -> None:
        self.enabled = getattr(cfg, "enabled", True)
        self.min_blur = getattr(cfg, "min_blur", 40.0)
        self.min_brightness = getattr(cfg, "min_brightness", 25)
        self.max_brightness = getattr(cfg, "max_brightness", 235)
        self._sdk = None
        # Optional: real Captur SDK if installed. Stays fully on-device.
        try:
            import captur  # type: ignore  # noqa: F401

            self._sdk = captur
        except Exception:
            self._sdk = None

    @property
    def backend(self) -> str:
        return "captur-sdk" if self._sdk is not None else "local-heuristics"

    def check(self, frame: np.ndarray) -> TrustResult:
        if not self.enabled:
            return TrustResult(True, 1.0, ["trust-gate disabled"], {})
        if self._sdk is not None:
            try:
                return self._check_sdk(frame)
            except Exception:
                pass  # fall through to heuristics
        return self._check_local(frame)

    def _check_sdk(self, frame: np.ndarray) -> TrustResult:
        verdict = self._sdk.validate(frame)  # SDK-specific; adapt as needed
        trusted = bool(getattr(verdict, "valid", verdict))
        score = float(getattr(verdict, "score", 1.0 if trusted else 0.0))
        return TrustResult(trusted, score, ["captur sdk"], {})

    def _check_local(self, frame: np.ndarray) -> TrustResult:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = float(gray.mean())
        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        reasons: list[str] = []
        if brightness < self.min_brightness:
            reasons.append(f"too dark ({brightness:.0f})")
        if brightness > self.max_brightness:
            reasons.append(f"too bright ({brightness:.0f})")
        if blur < self.min_blur:
            reasons.append(f"too blurry / no detail ({blur:.0f})")

        trusted = not reasons
        # Simple confidence: blur headroom blended with brightness sanity.
        score = max(0.0, min(1.0, (blur / (self.min_blur * 3)) if trusted else 0.2))
        return TrustResult(
            trusted=trusted,
            score=round(score, 2),
            reasons=reasons or ["valid capture"],
            metrics={"brightness": round(brightness, 1), "blur": round(blur, 1)},
        )
