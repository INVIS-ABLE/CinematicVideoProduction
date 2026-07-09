"""Storyboard parsing, validation, director planning, prompt compilation."""
import pytest

from wan.cognitive_fabric.brains.director_brain import DirectorBrain
from wan.cognitive_fabric.brains.storyboard_brain import StoryboardBrain
from wan.cognitive_fabric.pipeline.storyboard_orchestrator import (
    StoryboardOrchestrator,
)

PROMPT = ("A damaged hero walks through neon rain in a futuristic city at "
          "night, slow camera push-in, emotional atmosphere.")


def test_director_plans_valid_storyboard():
    sb = DirectorBrain().plan_from_prompt(PROMPT, duration_seconds=30)
    problems = StoryboardBrain().validate(sb)
    assert problems == []
    assert sb["worlds"][0]["time_of_day"] == "night"
    assert sb["worlds"][0]["weather"] == "rain"
    assert "city" in sb["worlds"][0]["location_type"]
    assert len(sb["shots"]) >= 3
    assert sb["shots"][0]["transition_in"] == "fade_in"
    # character extracted from prompt
    assert any(c["name"] == "the hero" for c in sb["characters"])


def test_director_is_deterministic():
    a = DirectorBrain().plan_from_prompt(PROMPT, duration_seconds=30)
    b = DirectorBrain().plan_from_prompt(PROMPT, duration_seconds=30)
    assert a == b


def test_validation_catches_broken_storyboards():
    brain = StoryboardBrain()
    assert brain.validate({}) != []
    bad = {"project": {}, "shots": [{"shot_id": "s1", "scene_id": "x",
                                     "duration_seconds": 0, "prompt": "p"}]}
    problems = brain.validate(bad)
    assert any("duration" in p for p in problems)
    dupes = {"project": {}, "shots": [
        {"shot_id": "s1", "scene_id": "a", "duration_seconds": 2, "prompt": "p"},
        {"shot_id": "s1", "scene_id": "a", "duration_seconds": 2, "prompt": "p"},
    ]}
    assert any("duplicate" in p for p in StoryboardBrain().validate(dupes))


def test_orchestrator_rejects_invalid_storyboard():
    with pytest.raises(ValueError, match="invalid storyboard"):
        StoryboardOrchestrator().load_storyboard({"shots": []})


def test_dry_run_reports_plan():
    sb = DirectorBrain().plan_from_prompt(PROMPT, duration_seconds=20)
    report = StoryboardOrchestrator().dry_run(sb)
    assert report["mode"] == "dry_run"
    assert report["chunks"] >= report["shots"] >= 1
    assert report["total_frames"] > 0
