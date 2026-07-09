"""Long-form chunking: 4n+1 invariant, overlap, seeds, 60-minute dry plan."""
from wan.cognitive_fabric.brains.director_brain import DirectorBrain
from wan.cognitive_fabric.pipeline.chunk_scheduler import (
    plan_chunks,
    snap_frame_num,
)
from wan.cognitive_fabric.pipeline.longform_orchestrator import (
    LongformOrchestrator,
)


def _shot(duration, shot_id="shot_001", strategy="per_shot"):
    return {"shot_id": shot_id, "scene_id": "scene_001",
            "duration_seconds": duration, "prompt": "test prompt",
            "seed_strategy": strategy}


def test_snap_frame_num_invariant():
    for frames in (1, 2, 5, 80, 81, 120, 121, 300):
        snapped = snap_frame_num(frames)
        assert (snapped - 1) % 4 == 0
        assert snapped >= 5 or frames <= 5


def test_short_shot_single_chunk():
    chunks = plan_chunks(_shot(4.0), fps=24, chunk_seconds=5,
                         overlap_frames=8, width=1280, height=704)
    assert len(chunks) == 1
    assert (chunks[0].frame_num - 1) % 4 == 0
    assert chunks[0].overlap_frames == 0


def test_long_shot_splits_with_overlap_and_dependency():
    chunks = plan_chunks(_shot(18.0), fps=24, chunk_seconds=5,
                         overlap_frames=8, width=1280, height=704)
    assert len(chunks) >= 3
    assert chunks[0].overlap_frames == 0
    assert all(c.overlap_frames == 8 for c in chunks[1:])
    assert [c.index for c in chunks] == list(range(len(chunks)))
    total = sum(c.frame_num for c in chunks)
    assert total >= 18 * 24 * 0.9  # covers the requested duration


def test_seed_strategies():
    per_shot = plan_chunks(_shot(12.0), fps=24, chunk_seconds=5,
                           overlap_frames=8, width=1280, height=704)
    assert len({c.seed for c in per_shot}) == 1  # same shot → same seed
    per_chunk = plan_chunks(_shot(12.0, strategy="per_chunk"), fps=24,
                            chunk_seconds=5, overlap_frames=8,
                            width=1280, height=704)
    assert len({c.seed for c in per_chunk}) == len(per_chunk)
    fixed = plan_chunks(_shot(12.0, strategy="fixed"), fps=24,
                        chunk_seconds=5, overlap_frames=8, width=1280,
                        height=704, base_seed=7)
    assert all(c.seed == 7 for c in fixed)


def _film_storyboard(minutes: float):
    shots = []
    scene_count = max(1, int(minutes))
    scenes = []
    for s in range(scene_count):
        scene_shots = []
        for i in range(4):
            shot_id = f"shot_{s:03d}_{i}"
            shots.append({
                "shot_id": shot_id, "scene_id": f"scene_{s:03d}",
                "duration_seconds": minutes * 60 / scene_count / 4,
                "prompt": "scene prompt",
            })
            scene_shots.append(shot_id)
        scenes.append({"scene_id": f"scene_{s:03d}", "shots": scene_shots})
    return {"project": {"title": "Long Film", "frame_rate": 24},
            "scenes": scenes, "shots": shots}


def test_60_minute_dry_run_plans_all_chunks(tmp_path):
    plan = LongformOrchestrator().dry_run(_film_storyboard(60),
                                          str(tmp_path / "film"))
    assert plan["within_budget"]
    assert 55 <= plan["estimated_minutes"] <= 65
    assert plan["total_chunks"] > 600      # ~5s chunks over 60 minutes
    assert (tmp_path / "film" / "film_plan.json").is_file()


def test_30_second_and_3_minute_plans(tmp_path):
    for minutes, name in ((0.5, "half"), (3, "three")):
        plan = LongformOrchestrator().dry_run(_film_storyboard(minutes),
                                              str(tmp_path / name))
        assert abs(plan["estimated_minutes"] - minutes) / minutes < 0.2
