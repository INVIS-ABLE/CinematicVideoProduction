"""End-to-end fabric run with the mock engine: storyboard → chunks →
generation → quality → repair → stitch → checkpoint → resume. This is the
'MAKE WAN RUN AFTER EVERY CHANGE' gate for the fabric side."""
import json
from pathlib import Path

from wan.cognitive_fabric.brains.director_brain import DirectorBrain
from wan.cognitive_fabric.fabric_config import FabricConfig
from wan.cognitive_fabric.fabric_runtime import FabricRuntime
from wan.cognitive_fabric.pipeline.generation_controller import MockWanEngine
from wan.cognitive_fabric.pipeline.storyboard_orchestrator import (
    StoryboardOrchestrator,
)

PROMPT = ("A knight walks through a storm into a ruined castle at night, "
          "rain, torchlight, slow push-in.")


def test_full_mock_run_and_resume(tmp_path):
    project = str(tmp_path / "proj")
    storyboard = DirectorBrain().plan_from_prompt(PROMPT, duration_seconds=8)
    orchestrator = StoryboardOrchestrator(FabricConfig())
    engine = MockWanEngine(downscale=16)  # tiny for CI speed

    report = orchestrator.run(storyboard, project, engine)
    assert report["chunks_generated"] == report["chunks_total"] >= 1

    # every promised artifact exists
    root = Path(project)
    assert (root / "run_report.json").is_file()
    assert (root / "timeline.json").is_file()
    assert (root / "memory.db").is_file()
    assert (root / "checkpoints" / "fabric_state.json").is_file()
    assert (root / "checkpoints" / "identity_matrix.json").is_file()
    chunks = list((root / "chunks").glob("*.mp4")) + \
        list((root / "chunks").glob("*.pt"))
    assert len(chunks) >= report["chunks_total"]

    timeline = json.loads((root / "timeline.json").read_text())
    assert len(timeline["timeline"]) == report["chunks_total"]

    # crash-resume: rerun regenerates nothing
    report2 = orchestrator.run(storyboard, project, engine)
    assert report2["chunks_generated"] == 0
    assert report2["chunks_skipped_resume"] == report["chunks_total"]


def test_runtime_short_dry_run(tmp_path):
    runtime = FabricRuntime()
    report = runtime.run_short(PROMPT, project_dir=str(tmp_path / "p"),
                               duration_seconds=15, dry_run=True)
    assert report["mode"] == "dry_run"
    assert (tmp_path / "p" / "storyboard.json").is_file()


def test_runtime_selects_mock_without_ckpt(tmp_path):
    runtime = FabricRuntime()
    engine = runtime.select_engine(None)
    assert engine.name == "mock"


def test_runtime_status_reports_components():
    status = FabricRuntime().status()
    assert status["attention"]["selected_backend"]
    assert status["hardware_mode"] in ("low_vram", "creator", "studio_4090",
                                       "multi_gpu")
    assert all(v["ok"] for v in status["components"].values())
