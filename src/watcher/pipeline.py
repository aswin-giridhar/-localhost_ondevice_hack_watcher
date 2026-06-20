"""The sense -> think -> act loop that ties everything together.

Runs in a background thread (OpenCV + Ollama are blocking). On every frame it runs
the cheap change detector; only on a significant change does it wake the VLM, update
the graph, reason, narrate, and emit a trace. Events are pushed to the dashboard via
the supplied thread-safe callback.
"""
from __future__ import annotations

import base64
import threading
import time
from dataclasses import dataclass
from typing import Callable

from .capture import FrameSource, encode_jpeg
from .change_detector import ChangeDetector
from .config import Config
from .memory import SceneGraph
from .narrator import Narrator
from .perception import Perception
from .reasoner import Reasoner
from .tracer import Tracer
from .trust import TrustGate

EventCallback = Callable[[dict], None]


@dataclass
class Camera:
    """One eye: a frame source + its own change detector + the zone it watches."""
    id: str
    zone: str
    source: FrameSource
    detector: ChangeDetector


class Pipeline:
    def __init__(self, cfg: Config, on_event: EventCallback | None = None):
        self.cfg = cfg
        self.on_event = on_event or (lambda e: None)

        # Shared brain across all cameras.
        self.graph = SceneGraph(cfg.memory.db_path)
        self.perception = Perception(cfg.perception)
        self.reasoner = Reasoner(cfg.perception, self.graph)
        self.narrator = Narrator(cfg.narrator.enabled, cfg.narrator.tts)
        self.tracer = Tracer(cfg.tracer.path)
        self.trust = TrustGate(cfg.trust)

        self._cameras: dict[str, Camera] = {}
        self._cam_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._running = False

    # --- camera registry -------------------------------------------------
    def add_camera(self, cam_id: str, zone: str, source: FrameSource) -> Camera:
        cd = ChangeDetector(
            min_area=self.cfg.change_detector.min_area,
            threshold=self.cfg.change_detector.threshold,
            debounce_seconds=self.cfg.change_detector.debounce_seconds,
            warmup_frames=self.cfg.change_detector.warmup_frames,
        )
        cam = Camera(id=cam_id, zone=zone, source=source, detector=cd)
        with self._cam_lock:
            self._cameras[cam_id] = cam
        self._emit({"type": "camera", "event": "connected", "camera": cam_id, "zone": zone,
                    "count": len(self._cameras)})
        return cam

    def remove_camera(self, cam_id: str) -> None:
        with self._cam_lock:
            cam = self._cameras.pop(cam_id, None)
        if cam is not None:
            cam.source.release()
            self._emit({"type": "camera", "event": "disconnected", "camera": cam_id,
                        "count": len(self._cameras)})

    # --- lifecycle -------------------------------------------------------
    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        with self._cam_lock:
            cams = list(self._cameras.values())
        for cam in cams:
            cam.source.release()

    # --- main loop -------------------------------------------------------
    def _loop(self) -> None:
        interval = 1.0 / max(1, self.cfg.capture.target_fps)
        self._emit({"type": "status", "message": "watcher started",
                    "perception": self.perception.source_name(),
                    "trust": self.trust.backend})
        while self._running:
            with self._cam_lock:
                cams = list(self._cameras.values())
            if not cams:
                time.sleep(interval)
                continue
            for cam in cams:
                frame = cam.source.read()
                if frame is None:
                    continue
                try:
                    event = cam.detector.update(frame)
                except Exception as exc:  # never let the loop die
                    self._emit({"type": "error", "message": f"detector[{cam.id}]: {exc}"})
                    continue
                if event is not None:
                    self._process(cam, frame, event)
            time.sleep(interval)

    def _process(self, cam: Camera, frame, event) -> None:
        zone = cam.zone
        self._emit({"type": "change", "ts": event.timestamp, "area": event.changed_area,
                    "camera": cam.id, "zone": zone})

        # 1) Captur-style trust gate: validate the capture before acting on it.
        trust = self.trust.check(frame)
        trust_payload = {"trusted": trust.trusted, "score": trust.score,
                         "reasons": trust.reasons, "backend": self.trust.backend}
        if not trust.trusted:
            self.tracer.log("rejected", {"camera": cam.id, "zone": zone, "trust": trust_payload})
            self._emit({"type": "rejected", "ts": event.timestamp, "camera": cam.id,
                        "zone": zone, "trust": trust_payload})
            return

        # 2) Perception -> 3) graph -> 4) reasoning
        obs = self.perception.observe(frame, zone)
        obs.camera = cam.id
        diff = self.graph.ingest(obs)
        decision = self.reasoner.decide(obs, diff)

        self.narrator.say(decision["message"])
        trace = self.tracer.log(
            "decision",
            {
                "camera": cam.id,
                "zone": zone,
                "summary": obs.summary,
                "items": obs.items,
                "trust": trust_payload,
                "diff": {"appeared": diff["appeared"], "disappeared": diff["disappeared"]},
                "decision": decision,
            },
        )

        thumb = base64.b64encode(encode_jpeg(frame, quality=60)).decode("ascii")
        self._emit(
            {
                "type": "decision",
                "ts": trace["ts"],
                "camera": cam.id,
                "zone": zone,
                "summary": obs.summary,
                "trust": trust_payload,
                "diff": diff,
                "decision": decision,
                "thumb": thumb,
            }
        )

    def _emit(self, event: dict) -> None:
        try:
            self.on_event(event)
        except Exception:
            pass
