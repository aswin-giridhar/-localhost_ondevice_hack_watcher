# Watcher

An embodied, **fully-offline** scene-monitoring agent. **Phone = eyes, laptop = brain.**
A `sense -> think -> act` loop grounded in vision, backed by a temporal-spatial
**knowledge graph**. No internet, no cloud — everything runs on-device.

> On-device AI hackathon (Dawn Office, London, 2026-06-20). See `../IMPROVED_IDEA.md`
> and `../IDEATION.md` for the full reasoning behind this build.

## What it does

1. The phone streams camera frames to the laptop over the **local network** (no internet).
2. A cheap, always-on **change detector** watches every frame for free.
3. On a significant change it wakes a small local **VLM** (via Ollama) to describe the scene.
4. The observation updates a **knowledge graph** (objects, zones, events over time).
5. A local **SLM** (Phi-4-mini) reasons over the graph + change and **decides** (normal /
   noteworthy / anomaly), then **narrates** it aloud.
6. Every decision is logged as an **Overmind-style trace** — proof it's reasoning, not scripted.

The dashboard shows the live camera keyframe, the **knowledge graph growing in real time**,
and the decision trace.

## Architecture

```
[phone camera] --JPEG/WS--> [laptop]
   change detector (every frame, ~free)
        -> on change -> VLM perception (Ollama)
        -> knowledge graph (NetworkX + SQLite)  [GraphRAG context]
        -> reasoner (Phi-4-mini, strict JSON)
        -> narrate (TTS / on-screen) + decision trace
```

Sponsors: **Captur** (vision trust gate), **Cognee** (graph memory — swappable behind
`memory.py`), **Overmind** (decision traces). Multi-camera / Exo cluster are stretch goals.

## Quickstart

```bash
pip install -r requirements.txt

# (Recommended) pull a small local VLM + reasoner via Ollama:
#   ollama pull qwen2-vl:2b
#   ollama pull phi4-mini
# Without Ollama, Watcher runs with a mock perception + rule-based reasoning.

python run.py
```

- Laptop dashboard: `http://localhost:8000`
- Phone (same Wi-Fi/LAN): `http://<laptop-lan-ip>:8000/phone` -> **Start streaming**

To test on the laptop alone (no phone), set `capture.source: webcam` in `config.yaml`.

## Configuration

All knobs live in `config.yaml` (camera source, change-detector sensitivity, model names,
graph backend, narration). It's fully local by design.

## Offline proof (the demo flex)

Watcher needs no internet. During the demo, enable airplane mode / firewall-block the
process and show it still sees, remembers, reasons, and narrates.

## Tests

```bash
pytest -q
```

`tests/` cover the change detector and the knowledge-graph memory (no models needed).

## Project layout

```
watcher/
  run.py                 entry point
  config.yaml            all settings (local-only)
  phone/capture.html     phone camera -> websocket streamer
  src/watcher/
    config.py            typed config loader
    capture.py           webcam / phone-stream frame sources
    change_detector.py   cheap always-on change gate
    perception.py        VLM keyframe -> structured observation (Ollama / mock)
    memory.py            temporal-spatial knowledge graph (NetworkX + SQLite)
    reasoner.py          decision over diff + graph (Phi-4-mini / rules)
    narrator.py          offline TTS + on-screen narration
    tracer.py            Overmind-style decision traces (JSONL)
    pipeline.py          the sense -> think -> act loop
    server.py            FastAPI: ingest, events ws, graph/trace REST
    web/index.html       dashboard (camera + live graph + trace log)
  tests/
```
