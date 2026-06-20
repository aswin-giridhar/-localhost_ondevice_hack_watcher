"""Temporal-spatial knowledge graph (the agent's memory).

Each observation is decomposed into nodes (Object, Zone, Event) and edges
(located_in, concerns, preceded_by). This turns anomaly detection into a
*structural* question ("is this object normally in this zone?") and enables
cross-session recall. NetworkX + a JSON snapshot in SQLite keeps it fully local
and reliable; Cognee can be swapped in later behind the same interface.
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

import networkx as nx

from .perception import Observation


class SceneGraph:
    def __init__(self, db_path: str = "data/watcher_graph.sqlite") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.g = nx.DiGraph()
        self._last_event_id: str | None = None
        self._init_db()
        self._load()

    # --- persistence -----------------------------------------------------
    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as con:
            con.execute(
                "CREATE TABLE IF NOT EXISTS graph (id INTEGER PRIMARY KEY, snapshot TEXT)"
            )

    def _load(self) -> None:
        with sqlite3.connect(self.db_path) as con:
            row = con.execute(
                "SELECT snapshot FROM graph ORDER BY id DESC LIMIT 1"
            ).fetchone()
        if row and row[0]:
            try:
                self.g = nx.node_link_graph(json.loads(row[0]), directed=True)
            except Exception:
                self.g = nx.DiGraph()

    def _save(self) -> None:
        snapshot = json.dumps(nx.node_link_data(self.g))
        with sqlite3.connect(self.db_path) as con:
            con.execute("INSERT INTO graph (snapshot) VALUES (?)", (snapshot,))

    # --- ingest ----------------------------------------------------------
    def ingest(self, obs: Observation) -> dict:
        """Update the graph with an observation; return a change diff."""
        zone_id = f"zone:{obs.zone}"
        if not self.g.has_node(zone_id):
            self.g.add_node(zone_id, type="zone", label=obs.zone)

        seen = obs.labels()
        present = self._present_objects(obs.zone)

        appeared = sorted(seen - present)
        disappeared = sorted(present - seen)

        for label in seen:
            self._touch_object(obs.zone, label, present=True, ts=obs.timestamp, camera=obs.camera)
        for label in disappeared:
            self._touch_object(obs.zone, label, present=False, ts=obs.timestamp, camera=obs.camera)

        events = []
        for label in appeared:
            events.append(self._add_event("appeared", obs.zone, label, obs.timestamp, obs.camera))
        for label in disappeared:
            events.append(self._add_event("disappeared", obs.zone, label, obs.timestamp, obs.camera))

        self._save()
        return {
            "zone": obs.zone,
            "camera": obs.camera,
            "appeared": appeared,
            "disappeared": disappeared,
            "present": sorted(seen),
            "events": events,
        }

    def _present_objects(self, zone: str) -> set[str]:
        out = set()
        for nid, data in self.g.nodes(data=True):
            if data.get("type") == "object" and data.get("zone") == zone and data.get("present"):
                out.add(data["label"])
        return out

    def _touch_object(self, zone: str, label: str, present: bool, ts: float, camera: str = "local") -> str:
        oid = f"obj:{zone}:{label}"
        if self.g.has_node(oid):
            self.g.nodes[oid]["present"] = present
            self.g.nodes[oid]["last_seen"] = ts
            self.g.nodes[oid]["camera"] = camera
            if present:
                self.g.nodes[oid]["seen_count"] = self.g.nodes[oid].get("seen_count", 0) + 1
        else:
            self.g.add_node(
                oid,
                type="object",
                label=label,
                zone=zone,
                camera=camera,
                present=present,
                first_seen=ts,
                last_seen=ts,
                seen_count=1 if present else 0,
            )
            self.g.add_edge(oid, f"zone:{zone}", rel="located_in")
        return oid

    def _add_event(self, etype: str, zone: str, label: str, ts: float, camera: str = "local") -> dict:
        eid = f"event:{ts:.3f}:{etype}:{label}"
        self.g.add_node(eid, type="event", etype=etype, label=label, zone=zone, camera=camera, ts=ts)
        self.g.add_edge(eid, f"obj:{zone}:{label}", rel="concerns")
        if self._last_event_id and self.g.has_node(self._last_event_id):
            self.g.add_edge(eid, self._last_event_id, rel="preceded_by")
        self._last_event_id = eid
        return {"id": eid, "etype": etype, "zone": zone, "label": label, "camera": camera, "ts": ts}

    # --- queries (GraphRAG) ---------------------------------------------
    def is_familiar(self, zone: str, label: str) -> bool:
        """Has this object been seen in this zone before? (structural anomaly check)"""
        oid = f"obj:{zone}:{label}"
        return self.g.has_node(oid) and self.g.nodes[oid].get("seen_count", 0) > 1

    def recent_events(self, n: int = 20) -> list[dict]:
        evs = [
            {"label": d["label"], "etype": d["etype"], "zone": d["zone"], "ts": d["ts"]}
            for _, d in self.g.nodes(data=True)
            if d.get("type") == "event"
        ]
        evs.sort(key=lambda e: e["ts"], reverse=True)
        return evs[:n]

    def context_for_reasoner(self, zone: str, window: int = 20) -> str:
        present = sorted(self._present_objects(zone))
        familiar = sorted(
            d["label"]
            for _, d in self.g.nodes(data=True)
            if d.get("type") == "object" and d.get("zone") == zone and d.get("seen_count", 0) > 1
        )
        recent = self.recent_events(window)
        lines = [
            f"Zone: {zone}",
            f"Currently present: {present or 'nothing'}",
            f"Historically familiar here: {familiar or 'nothing yet'}",
            "Recent events (newest first):",
        ]
        for e in recent:
            lines.append(f"  - {e['etype']} {e['label']}")
        return "\n".join(lines)

    # --- visualization ---------------------------------------------------
    def to_vis(self) -> dict:
        groups = {"zone": "zone", "object": "object", "event": "event"}
        nodes = []
        for nid, d in self.g.nodes(data=True):
            t = d.get("type", "object")
            label = d.get("label", nid)
            if t == "event":
                label = f"{d.get('etype', '?')}: {d.get('label', '')}"
            nodes.append({"id": nid, "label": label, "group": groups.get(t, t),
                          "present": d.get("present", True)})
        edges = [
            {"from": u, "to": v, "label": d.get("rel", "")}
            for u, v, d in self.g.edges(data=True)
        ]
        return {"nodes": nodes, "edges": edges, "ts": time.time()}
