"""Decision step: reason over (change diff + graph context) and decide what to do.

Uses a local SLM (Phi-4-mini via Ollama) with strict-JSON output. Falls back to a
deterministic rule-based decision so the agent stays autonomous even with no model.
"""
from __future__ import annotations

import json

from .memory import SceneGraph
from .perception import Observation

_SYS = """You are the decision module of an autonomous, offline monitoring agent.
Given the scene CONTEXT (memory) and the CHANGE just detected, decide the status and
a short spoken message. Reply ONLY as strict JSON:
{"status": "normal|noteworthy|anomaly", "message": "<one short sentence to speak>", "reason": "<brief>"}
Rules: an object appearing that was never seen here before is usually an anomaly.
A familiar object moving or leaving is noteworthy. Trivial changes are normal."""


class Reasoner:
    def __init__(self, cfg, graph: SceneGraph) -> None:
        self.cfg = cfg
        self.graph = graph
        self._client = None
        self._model = cfg.reasoner_model or cfg.vlm_model
        if cfg.backend == "ollama":
            try:
                import ollama

                self._client = ollama
            except Exception:
                self._client = None

    def decide(self, obs: Observation, diff: dict) -> dict:
        context = self.graph.context_for_reasoner(obs.zone, self.cfg.request_timeout and 20)
        if self._client is not None:
            try:
                return self._decide_llm(obs, diff, context)
            except Exception as exc:
                out = self._decide_rules(obs, diff)
                out["reason"] = f"[llm error, rules used] {exc}: {out['reason']}"
                return out
        return self._decide_rules(obs, diff)

    def _decide_llm(self, obs: Observation, diff: dict, context: str) -> dict:
        moves = "; ".join(
            f"{t['label']} {t['from_zone']}->{t['to_zone']}" for t in diff.get("transfers", [])
        ) or "none"
        user = (
            f"CONTEXT:\n{context}\n\n"
            f"CHANGE:\n appeared={diff['appeared']} disappeared={diff['disappeared']}\n"
            f" cross-camera moves={moves}\n"
            f"VLM summary: {obs.summary}\n"
            "If a cross-camera move is present, prioritise narrating it (e.g. 'the mug "
            "moved from desk to door')."
        )
        resp = self._client.chat(
            model=self._model,
            messages=[
                {"role": "system", "content": _SYS},
                {"role": "user", "content": user},
            ],
            options={"temperature": 0},
        )
        content = resp["message"]["content"] if isinstance(resp, dict) else str(resp)
        data = _extract_json(content)
        if not data:
            return self._decide_rules(obs, diff)
        return {
            "status": data.get("status", "noteworthy"),
            "message": data.get("message", "Something changed in the scene."),
            "reason": data.get("reason", ""),
            "engine": "llm",
        }

    def _decide_rules(self, obs: Observation, diff: dict) -> dict:
        appeared, disappeared = diff["appeared"], diff["disappeared"]
        transfers = diff.get("transfers", [])
        if transfers:
            moves = ", ".join(f"{t['label']} ({t['from_zone']}→{t['to_zone']})" for t in transfers)
            return {
                "status": "noteworthy",
                "message": f"Tracked a move across cameras: {moves}.",
                "reason": "object left one zone and reappeared in another within the window",
                "engine": "rules",
            }
        unfamiliar = [l for l in appeared if not self.graph.is_familiar(obs.zone, l)]
        if unfamiliar:
            return {
                "status": "anomaly",
                "message": f"New, unfamiliar item appeared: {', '.join(unfamiliar)}.",
                "reason": "object never seen in this zone before",
                "engine": "rules",
            }
        if appeared:
            return {
                "status": "noteworthy",
                "message": f"A familiar item reappeared: {', '.join(appeared)}.",
                "reason": "known object returned",
                "engine": "rules",
            }
        if disappeared:
            return {
                "status": "noteworthy",
                "message": f"Item removed from the scene: {', '.join(disappeared)}.",
                "reason": "known object left",
                "engine": "rules",
            }
        return {
            "status": "normal",
            "message": "Minor change, nothing notable.",
            "reason": "no object-level change",
            "engine": "rules",
        }


def _extract_json(text: str) -> dict:
    text = text.strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {}
