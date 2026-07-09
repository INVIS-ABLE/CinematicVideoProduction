"""Quality brain metrics, repair decisions, honest stub validators."""
import torch

from wan.cognitive_fabric.brains.quality_brain import QualityBrain
from wan.cognitive_fabric.brains.repair_brain import RepairBrain
from wan.cognitive_fabric.validators import face_validator
from wan.cognitive_fabric.validators.quality_report import build_quality_report


def _stable_video(f=12, motion=True):
    """Smooth gradient video with gentle motion — should score well."""
    ys = torch.linspace(-1, 1, 32).view(1, 1, 32, 1)
    xs = torch.linspace(-1, 1, 32).view(1, 1, 1, 32)
    t = torch.linspace(0, 0.5 if motion else 0.0, f).view(1, f, 1, 1)
    return (xs + ys * 0.5 + t).clamp(-1, 1).expand(3, f, 32, 32).clone()


def _flickery_video(f=12):
    torch.manual_seed(0)
    video = _stable_video(f)
    video[:, ::2] += 0.4  # alternate-frame brightness jumps
    return video.clamp(-1, 1)


def test_flicker_detection_orders_correctly():
    brain = QualityBrain()
    stable = brain.measure(_stable_video())
    flickery = brain.measure(_flickery_video())
    assert stable["flicker_score"] > flickery["flicker_score"]


def test_static_video_scores_low_motion():
    brain = QualityBrain()
    static = brain.measure(_stable_video(motion=False))
    moving = brain.measure(_stable_video(motion=True))
    assert moving["motion_score"] > static["motion_score"]


def test_report_recommends_stabilise_on_flicker():
    report = build_quality_report("c1", "s1", "scene_001", _flickery_video())
    assert report.recommended_action == "stabilise"
    assert report.measured["flicker_global"] > 0


def test_report_accepts_clean_video():
    report = build_quality_report("c1", "s1", "scene_001", _stable_video())
    assert report.recommended_action in ("accept", "detail_pass")
    assert 0 <= report.overall() <= 1


def test_identity_threshold_triggers_regeneration():
    brain = QualityBrain({"identity_threshold": 0.82})
    report = brain.score_chunk("c", "s", "sc", _stable_video(),
                               identity_similarity=0.5)
    assert report.recommended_action == "regenerate"


def test_repair_brain_escalation_and_cap():
    repair = RepairBrain(max_attempts=3)
    report = build_quality_report("c1", "s1", "sc", _stable_video())
    report.recommended_action = "regenerate"
    a0 = repair.decide(report, 0)
    a1 = repair.decide(report, 1)
    a2 = repair.decide(report, 2)
    a3 = repair.decide(report, 3)
    assert a0["action"] == "use_previous_frame_conditioning"
    assert a1["action"] == "strengthen_identity_tokens"
    assert a2["action"] == "regenerate"
    assert a3["action"] == "accept_with_warning" and a3["exhausted"]


def test_stub_validators_decline_rather_than_fake():
    assert face_validator.score(_stable_video()) is None
    assert face_validator.AVAILABLE is False


def test_dry_run_report_without_frames():
    report = build_quality_report("c1", "s1", "sc", None)
    assert report.recommended_action == "accept"
    assert report.overall() == 0.0
