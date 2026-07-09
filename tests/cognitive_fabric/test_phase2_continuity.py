"""Phase 2A: terminal-frame quality selection, seam trimming, user reference
images into the engine path, per-scene assemblies, streaming finishing pass."""
import json
from pathlib import Path

import pytest
import torch

from wan.cognitive_fabric.brains.director_brain import DirectorBrain
from wan.cognitive_fabric.fabric_config import FabricConfig
from wan.cognitive_fabric.fabric_runtime import FabricRuntime
from wan.cognitive_fabric.pipeline.generation_controller import (
    MockWanEngine,
    pick_best_terminal_frame,
    save_chunk_video,
    select_terminal_and_trim,
)
from wan.cognitive_fabric.pipeline.stitcher import ffmpeg_available
from wan.cognitive_fabric.pipeline.storyboard_orchestrator import (
    StoryboardOrchestrator,
)

PROMPT = ("A damaged hero walks through neon rain in a futuristic city at "
          "night, slow camera push-in.")


def _video_with_blurred_tail(f=10, sharp_idx=7):
    """Random (sharp) frames; every terminal-window frame except sharp_idx
    is blurred, so the picker has an unambiguous winner."""
    torch.manual_seed(0)
    video = torch.rand(3, f, 32, 32) * 2 - 1
    for i in range(f - 4, f):
        if i != sharp_idx:
            blurred = torch.nn.functional.avg_pool2d(
                video[:, i].unsqueeze(0), 7, stride=1, padding=3).squeeze(0)
            video[:, i] = blurred
    return video


def test_pick_best_terminal_frame_prefers_sharp():
    video = _video_with_blurred_tail(f=10, sharp_idx=7)
    assert pick_best_terminal_frame(video, window=4) == 7


def test_select_terminal_and_trim_ends_on_chosen_frame():
    video = _video_with_blurred_tail(f=10, sharp_idx=7)
    trimmed, dropped = select_terminal_and_trim(video, window=4)
    assert dropped == 2
    assert trimmed.shape[1] == 8
    assert torch.equal(trimmed[:, -1], video[:, 7])
    # sharp last frame → nothing dropped
    video2 = _video_with_blurred_tail(f=10, sharp_idx=9)
    trimmed2, dropped2 = select_terminal_and_trim(video2, window=4)
    assert dropped2 == 0 and trimmed2.shape[1] == 10


@pytest.fixture()
def ref_image(tmp_path):
    from PIL import Image
    import numpy as np
    rng = np.random.default_rng(7)
    img = Image.fromarray(rng.integers(0, 255, (64, 96, 3), dtype="uint8"),
                          "RGB")
    path = tmp_path / "hero_ref.png"
    img.save(path)
    return str(path)


def test_run_with_reference_and_first_frame(tmp_path, ref_image):
    project = str(tmp_path / "proj")
    runtime = FabricRuntime()
    report = runtime.run_short(
        PROMPT, project_dir=project, duration_seconds=8,
        force_mock=True, first_frame_image=ref_image,
        reference_images=[ref_image])
    assert report["chunks_generated"] >= 1

    storyboard = json.loads((Path(project) / "storyboard.json").read_text())
    assert storyboard["project"]["first_frame_image"] == ref_image
    assert storyboard["reference_uploads"][0]["path"] == ref_image

    # seam hygiene recorded in the timeline: continuation chunks drop their
    # duplicated lead frame; chunks with successors may trim a blurred tail
    timeline = json.loads((Path(project) / "timeline.json").read_text())
    entries = timeline["timeline"]
    assert entries[0]["trimmed_lead_frames"] == 0
    for entry in entries[1:]:
        assert entry["trimmed_lead_frames"] == 1
        assert entry["frame_num"] <= 121


def test_missing_reference_fails_loudly(tmp_path):
    runtime = FabricRuntime()
    with pytest.raises(FileNotFoundError):
        runtime.run_short(PROMPT, project_dir=str(tmp_path / "p"),
                          duration_seconds=8, dry_run=True,
                          first_frame_image=str(tmp_path / "nope.png"))


@pytest.mark.skipif(not ffmpeg_available(), reason="ffmpeg required")
def test_scene_assemblies_written(tmp_path):
    project = str(tmp_path / "proj")
    storyboard = DirectorBrain().plan_from_prompt(PROMPT, duration_seconds=8)
    orchestrator = StoryboardOrchestrator(FabricConfig())
    report = orchestrator.run(storyboard, project, MockWanEngine(downscale=16))
    assert report["scene_exports"], "expected per-scene assemblies"
    for path in report["scene_exports"].values():
        assert Path(path).is_file()


@pytest.mark.skipif(not ffmpeg_available(), reason="ffmpeg required")
def test_finish_video_file_streaming(tmp_path):
    from wan.cognitive_fabric.pipeline.finishing_pipeline import (
        finish_video_file,
    )
    import imageio.v2 as iio

    torch.manual_seed(1)
    video = torch.rand(3, 9, 48, 64) * 2 - 1
    src = save_chunk_video(video, str(tmp_path / "src"), fps=8)
    assert src.endswith(".mp4")

    out = finish_video_file(src, str(tmp_path / "final.mp4"),
                            look="cinematic_teal_orange",
                            target_size=(128, 96), batch=4)
    reader = iio.get_reader(out)
    frames = [f for f in reader]
    reader.close()
    assert len(frames) == 9                      # frame count preserved
    assert frames[0].shape[0] == 96 and frames[0].shape[1] == 128


@pytest.mark.skipif(not ffmpeg_available(), reason="ffmpeg required")
def test_finishing_requested_produces_final_export(tmp_path):
    config = FabricConfig({"cognitive_fabric": {
        "finishing_requested": True,
        "final_resolution": "480p",   # tiny target keeps the test fast
        "base_resolution": "720p",
    }})
    storyboard = DirectorBrain().plan_from_prompt(PROMPT, duration_seconds=8)
    orchestrator = StoryboardOrchestrator(config)
    report = orchestrator.run(storyboard, str(tmp_path / "p"),
                              MockWanEngine(downscale=16))
    assert report["final_export"] and Path(report["final_export"]).is_file()
