"""FastAPI server: phone ingest, dashboard websocket, and REST for graph/traces.

Layout:
  GET  /            -> dashboard (camera thumb + live knowledge graph + trace log)
  GET  /phone       -> phone capture page (streams camera frames to /ws/ingest)
  WS   /ws/ingest   -> phone pushes JPEG frames here (binary)
  WS   /ws/events   -> dashboard subscribes to agent events
  GET  /api/graph   -> current knowledge graph (nodes/edges) for visualization
  GET  /api/traces  -> recent decision traces
Everything binds to the LAN so the phone can reach it; no internet is used.
"""
from __future__ import annotations

import asyncio
import socket
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse

from .capture import DemoFrameSource, StreamFrameSource, WebcamFrameSource
from .config import Config
from .pipeline import Pipeline

WEB_DIR = Path(__file__).parent / "web"
PHONE_DIR = Path(__file__).parent.parent.parent / "phone"


def _probe_internet(timeout: float = 0.8) -> bool:
    """Best-effort connectivity probe for the kill-switch indicator.

    Returns True only if an outbound connection succeeds. When the network is cut
    (the demo flex), this fails fast and the dashboard shows OFFLINE / ON-DEVICE.
    """
    for host, port in (("8.8.8.8", 53), ("1.1.1.1", 53)):
        try:
            with socket.create_connection((host, port), timeout=timeout):
                return True
        except OSError:
            continue
    return False


class Hub:
    """Broadcasts agent events to all connected dashboard clients."""

    def __init__(self) -> None:
        self.clients: set[WebSocket] = set()
        self.queue: asyncio.Queue[dict] = asyncio.Queue()
        self.loop: asyncio.AbstractEventLoop | None = None

    def publish_threadsafe(self, event: dict) -> None:
        # Called from the pipeline thread; hop onto the asyncio loop safely.
        if self.loop is not None:
            self.loop.call_soon_threadsafe(self.queue.put_nowait, event)

    async def pump(self) -> None:
        while True:
            event = await self.queue.get()
            dead = []
            for ws in self.clients:
                try:
                    await ws.send_json(event)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                self.clients.discard(ws)


def create_app(cfg: Config) -> FastAPI:
    app = FastAPI(title="Watcher", version="0.1.0")
    hub = Hub()
    pipeline = Pipeline(cfg, on_event=hub.publish_threadsafe)
    cam_counter = {"n": 0}

    def _open_webcam(cam_id: str, zone: str) -> None:
        try:
            pipeline.add_camera(cam_id, zone, WebcamFrameSource(cfg.capture.webcam_index))
        except Exception as exc:
            hub.publish_threadsafe({"type": "error", "message": f"webcam unavailable: {exc}"})

    @app.on_event("startup")
    async def _startup() -> None:
        hub.loop = asyncio.get_running_loop()
        asyncio.create_task(hub.pump())
        # Register local camera nodes up front. Phones/tablets always join live via
        # /ws/ingest, so all three (laptop + phone + tablet) feed one shared graph.
        if cfg.capture.source == "demo":
            pipeline.add_camera("demo", cfg.capture.zone, DemoFrameSource())
        elif cfg.capture.source == "webcam":
            _open_webcam("laptop", cfg.capture.webcam_zone)
        if cfg.capture.local_webcam and cfg.capture.source != "webcam":
            _open_webcam("laptop", cfg.capture.webcam_zone)
        pipeline.start()

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        pipeline.stop()

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return (WEB_DIR / "index.html").read_text(encoding="utf-8")

    @app.get("/phone", response_class=HTMLResponse)
    async def phone() -> str:
        return (PHONE_DIR / "capture.html").read_text(encoding="utf-8")

    @app.get("/api/graph")
    async def api_graph() -> JSONResponse:
        return JSONResponse(pipeline.graph.to_vis())

    @app.get("/api/traces")
    async def api_traces() -> JSONResponse:
        return JSONResponse(pipeline.tracer.tail(50))

    @app.get("/api/netstatus")
    async def api_netstatus() -> JSONResponse:
        loop = asyncio.get_running_loop()
        online = await loop.run_in_executor(None, _probe_internet)
        return JSONResponse({"online": online})

    @app.websocket("/ws/ingest")
    async def ws_ingest(ws: WebSocket) -> None:
        await ws.accept()
        cam_counter["n"] += 1
        cam_id = ws.query_params.get("cam") or f"cam-{cam_counter['n']}"
        zone = ws.query_params.get("zone") or cfg.capture.zone
        src = StreamFrameSource()
        pipeline.add_camera(cam_id, zone, src)
        try:
            while True:
                data = await ws.receive_bytes()
                src.push_jpeg(data)
        except WebSocketDisconnect:
            pass
        finally:
            pipeline.remove_camera(cam_id)

    @app.websocket("/ws/events")
    async def ws_events(ws: WebSocket) -> None:
        await ws.accept()
        hub.clients.add(ws)
        try:
            while True:
                await ws.receive_text()  # keepalive; dashboard doesn't send much
        except WebSocketDisconnect:
            hub.clients.discard(ws)

    return app
