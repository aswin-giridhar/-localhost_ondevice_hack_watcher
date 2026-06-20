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
from typing import Callable

from .capture import FrameSource, encode_jpeg
from .change_detector import ChangeDetector
from .config import Config
from .memory import SceneGraph
from .narrator import Narrator
from .perception import Perception
from .reasoner import Reasoner
from .tracer import Tracer

EventCallback = Callable[[dict], None]


class Pipeline:
    def __init__(self, cfg: Config, source: FrameSource, on_event: EventCallback | None = None):
        self.cfg = cfg
        self.source = source
        self.on_event = on_event or (lambda e: None)

        self.detector = ChangeDetector(
            min_area=cfg.change_detector.min_area,
            threshold=cfg.change_detector.threshold,
            debounce_seconds=cfg.change_detector.debounce_seconds,
            warmup_frames=cfg.change_detector.warmup_frames,
        )
        self.graph = SceneGraph(cfg.memory.db_path)
        self.perception = Perception(cfg.perception)
        self.reasoner = Reasoner(cfg.perception, self.graph)
        self.narrator = Narrator(cfg.narrator.enabled, cfg.narrator.tts)
        self.tracer = Tracer(cfg.tracer.path)

        self._thread: threading.Thread | None = None
        self._running = False

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
        self.source.release()

    # --- main loop -------------------------------------------------------
    def _loop(self) -> None:
        interval = 1.0 / max(1, self.cfg.capture.target_fps)
        self._emit({"type": "status", "message": "watcher started", "perception": self.perception.source_name()})
        while self._running:
            frame = self.source.read()
            if frame is None:
                time.sleep(interval)
                continue
            try:
                event = self.detector.update(frame)
            except Exception as exc:  # never let the loop die
                self._emit({"type": "error", "message": f"detector: {exc}"})
                time.sleep(interval)
                continue
            if event is not None:
                self._process(frame, event)
            time.sleep(interval)

    def _process(self, frame, event) -> None:
        zone = self.cfg.capture.zone
        self._emit({"type": "change", "ts": event.timestamp, "area": event.changed_area})

        obs = self.perception.observe(frame, zone)
        diff = self.graph.ingest(obs)
        decision = self.reasoner.decide(obs, diff)

        self.narrator.say(decision["message"])
        trace = self.tracer.log(
            "decision",
            {
                "zone": zone,
                "summary": obs.summary,
                "items": obs.items,
                "diff": {"appeared": diff["appeared"], "disappeared": diff["disappeared"]},
                "decision": decision,
            },
        )

        thumb = base64.b64encode(encode_jpeg(frame, quality=60)).decode("ascii")
        self._emit(
            {
                "type": "decision",
                "ts": trace["ts"],
                "zone": zone,
                "summary": obs.summary,
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
