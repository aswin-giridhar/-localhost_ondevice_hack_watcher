"""Tests for the knowledge-graph memory (pure logic, no models needed)."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from watcher.memory import SceneGraph
from watcher.perception import Observation


def _obs(labels, zone="desk", ts=None):
    return Observation(
        timestamp=ts or time.time(),
        zone=zone,
        items=[{"label": l, "position": "center", "notable": ""} for l in labels],
        summary="test",
    )


def test_appeared_and_disappeared(tmp_path):
    g = SceneGraph(str(tmp_path / "g.sqlite"))
    d1 = g.ingest(_obs(["mug", "laptop"]))
    assert set(d1["appeared"]) == {"mug", "laptop"}
    assert d1["disappeared"] == []

    d2 = g.ingest(_obs(["laptop", "phone"]))
    assert d2["appeared"] == ["phone"]
    assert d2["disappeared"] == ["mug"]


def test_familiarity_tracks_repeat_sightings(tmp_path):
    g = SceneGraph(str(tmp_path / "g.sqlite"))
    g.ingest(_obs(["mug"]))
    # First sighting is not yet "familiar" (seen_count == 1).
    assert g.is_familiar("desk", "mug") is False
    g.ingest(_obs([]))          # mug leaves
    g.ingest(_obs(["mug"]))     # mug returns -> seen_count grows
    assert g.is_familiar("desk", "mug") is True
    assert g.is_familiar("desk", "never-seen") is False


def test_persists_across_instances(tmp_path):
    db = str(tmp_path / "g.sqlite")
    SceneGraph(db).ingest(_obs(["mug", "laptop"]))
    g2 = SceneGraph(db)  # reload from sqlite
    vis = g2.to_vis()
    labels = {n["label"] for n in vis["nodes"]}
    assert "mug" in labels and "laptop" in labels


def test_to_vis_shape(tmp_path):
    g = SceneGraph(str(tmp_path / "g.sqlite"))
    g.ingest(_obs(["mug"]))
    vis = g.to_vis()
    assert "nodes" in vis and "edges" in vis
    assert any(n["group"] == "zone" for n in vis["nodes"])
    assert any(n["group"] == "object" for n in vis["nodes"])
