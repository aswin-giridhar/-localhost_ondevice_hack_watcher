"""Load and validate Watcher configuration from a YAML file."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ServerCfg:
    host: str = "0.0.0.0"
    port: int = 8000


@dataclass
class CaptureCfg:
    source: str = "stream"
    webcam_index: int = 0
    target_fps: int = 10
    zone: str = "desk"
    local_webcam: bool = False      # also use this laptop's built-in camera as a node
    webcam_zone: str = "laptop"     # zone label for the laptop camera


@dataclass
class ChangeDetectorCfg:
    min_area: int = 1500
    threshold: int = 25
    debounce_seconds: float = 2.0
    warmup_frames: int = 15


@dataclass
class PerceptionCfg:
    backend: str = "ollama"
    vlm_model: str = "qwen2-vl:2b"
    reasoner_model: str = "phi4-mini"
    request_timeout: int = 60


@dataclass
class MemoryCfg:
    backend: str = "networkx"
    db_path: str = "data/watcher_graph.sqlite"
    recent_window: int = 20


@dataclass
class TrustCfg:
    enabled: bool = True
    min_blur: float = 40.0
    min_brightness: int = 25
    max_brightness: int = 235


@dataclass
class NarratorCfg:
    enabled: bool = True
    tts: bool = False


@dataclass
class TracerCfg:
    path: str = "traces/decisions.jsonl"


@dataclass
class Config:
    server: ServerCfg = field(default_factory=ServerCfg)
    capture: CaptureCfg = field(default_factory=CaptureCfg)
    change_detector: ChangeDetectorCfg = field(default_factory=ChangeDetectorCfg)
    perception: PerceptionCfg = field(default_factory=PerceptionCfg)
    memory: MemoryCfg = field(default_factory=MemoryCfg)
    trust: TrustCfg = field(default_factory=TrustCfg)
    narrator: NarratorCfg = field(default_factory=NarratorCfg)
    tracer: TracerCfg = field(default_factory=TracerCfg)

    @staticmethod
    def load(path: str | Path = "config.yaml") -> "Config":
        path = Path(path)
        raw: dict[str, Any] = {}
        if path.exists():
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return Config(
            server=ServerCfg(**raw.get("server", {})),
            capture=CaptureCfg(**raw.get("capture", {})),
            change_detector=ChangeDetectorCfg(**raw.get("change_detector", {})),
            perception=PerceptionCfg(**raw.get("perception", {})),
            memory=MemoryCfg(**raw.get("memory", {})),
            trust=TrustCfg(**raw.get("trust", {})),
            narrator=NarratorCfg(**raw.get("narrator", {})),
            tracer=TracerCfg(**raw.get("tracer", {})),
        )
