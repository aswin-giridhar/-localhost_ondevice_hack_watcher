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
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse

from .capture import StreamFrameSource, make_source
from .config import Config
from .pipeline import Pipeline

WEB_DIR = Path(__file__).parent / "web"
PHONE_DIR = Path(__file__).parent.parent.parent / "phone"


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
    source = make_source(cfg.capture)
    pipeline = Pipeline(cfg, source, on_event=hub.publish_threadsafe)

    @app.on_event("startup")
    async def _startup() -> None:
        hub.loop = asyncio.get_running_loop()
        asyncio.create_task(hub.pump())
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

    @app.websocket("/ws/ingest")
    async def ws_ingest(ws: WebSocket) -> None:
        await ws.accept()
        if not isinstance(source, StreamFrameSource):
            await ws.close(code=1003)
            return
        try:
            while True:
                data = await ws.receive_bytes()
                source.push_jpeg(data)
        except WebSocketDisconnect:
            pass

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
