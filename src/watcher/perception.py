"""Perception: turn a keyframe into a structured observation using a local VLM.

A single small VLM (e.g. qwen2-vl:2b via Ollama) both *sees* and emits structured
JSON, keeping us under 4GB VRAM. If Ollama is unavailable, a mock backend lets the
rest of the pipeline run for development.
"""
from __future__ import annotations

import base64
import json
import time
from dataclasses import dataclass, field

import numpy as np

from .capture import encode_jpeg

_PROMPT = """You are the vision module of an autonomous monitoring agent.
Look at the image and report ONLY what you can see, as strict JSON:
{
  "items": [
    {"label": "<short noun>", "position": "<left|center|right|top|bottom>", "notable": "<short note or empty>"}
  ],
  "summary": "<one short sentence describing the scene>"
}
List distinct physical objects only. No prose outside the JSON."""


@dataclass
class Observation:
    timestamp: float
    zone: str
    items: list[dict] = field(default_factory=list)
    summary: str = ""
    source: str = "vlm"        # which perception backend produced this
    camera: str = "local"      # which camera/source the frame came from

    def labels(self) -> set[str]:
        return {str(i.get("label", "")).strip().lower() for i in self.items if i.get("label")}


class Perception:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self._client = None
        if cfg.backend == "ollama":
            try:
                import ollama

                self._client = ollama.Client(host=getattr(cfg, "ollama_host", None) or None)
            except Exception:
                self._client = None

    @property
    def available(self) -> bool:
        return self._client is not None or self.cfg.backend == "mock"

    def source_name(self) -> str:
        return "ollama" if self._client is not None else "mock"

    def observe(self, frame: np.ndarray, zone: str) -> Observation:
        if self._client is None:
            return self._mock(frame, zone)
        try:
            return self._observe_ollama(frame, zone)
        except Exception as exc:  # degrade gracefully — never crash the loop
            obs = self._mock(frame, zone)
            obs.summary = f"[vlm error, mock used] {exc}"
            return obs

    def _observe_ollama(self, frame: np.ndarray, zone: str) -> Observation:
        jpeg = encode_jpeg(frame)
        b64 = base64.b64encode(jpeg).decode("ascii")
        resp = self._client.generate(
            model=self.cfg.vlm_model,
            prompt=_PROMPT,
            images=[b64],
            options={"temperature": 0},
        )
        text = resp.get("response", "") if isinstance(resp, dict) else str(resp)
        data = _extract_json(text)
        return Observation(
            timestamp=time.time(),
            zone=zone,
            items=data.get("items", []) if isinstance(data, dict) else [],
            summary=data.get("summary", "") if isinstance(data, dict) else text[:200],
            source="vlm",
        )

    def _mock(self, frame: np.ndarray, zone: str) -> Observation:
        # Honest placeholder: with no vision model we CANNOT identify contents, so
        # we report a single generic "unidentified object" instead of inventing
        # things. Connect Ollama (see README) for real, accurate perception.
        return Observation(
            timestamp=time.time(),
            zone=zone,
            items=[{"label": "unidentified object", "position": "center",
                    "notable": "no vision model connected"}],
            summary="[mock] change detected — no local VLM, contents not identified",
            source="mock",
        )


def _extract_json(text: str) -> dict:
    """Best-effort JSON extraction from a model response."""
    text = text.strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {}
