"""Overmind-style decision traces.

Every decision the agent makes is appended as a JSON line: the observation that
triggered it, the subgraph context it reasoned over, its decision, and the action
taken. This is the raw material for offline failure analysis / improvement, and it
doubles as on-stage proof that the agent is reasoning, not scripted.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class Tracer:
    def __init__(self, path: str = "traces/decisions.jsonl") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, kind: str, payload: dict[str, Any]) -> dict[str, Any]:
        record = {"ts": time.time(), "kind": kind, **payload}
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
        return record

    def tail(self, n: int = 50) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        lines = self.path.read_text(encoding="utf-8").splitlines()[-n:]
        out: list[dict[str, Any]] = []
        for line in lines:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out
