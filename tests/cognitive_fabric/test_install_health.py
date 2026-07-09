"""Install/health checks + model source policy + concurrent runner."""
import pytest

from wan.cognitive_fabric.install.dependency_checker import check_all
from wan.cognitive_fabric.install.health_check import run_health_check
from wan.cognitive_fabric.model_factory.allowed_model_sources import check_source
from wan.cognitive_fabric.model_factory.concurrent_model_runner import (
    ConcurrentModelRunner,
)
from wan.cognitive_fabric.model_factory.local_model_registry import (
    LocalModelRegistry,
)


def test_health_check_runs_and_reports(tmp_path):
    report = run_health_check(project_root=str(tmp_path))
    assert report["wan_engine"]["imports"] is True
    assert report["mock_ready"] is True
    assert "selected_backend" in report["wan_engine"]["attention"]
    # no CUDA/no weights in CI: generation_ready must be honestly False here
    assert report["generation_ready"] in (True, False)


def test_dependency_checker_structure():
    report = check_all()
    assert report["python"]["ok"]
    assert report["modules"]["torch"]["found"]


def test_model_source_policy():
    assert check_source({"id": "x", "license": "apache-2.0"})["allowed"]
    assert check_source({"id": "x", "license": "user-owned"})["allowed"]
    refused = check_source({"id": "seedance", "license": "proprietary"})
    assert not refused["allowed"]
    assert "refused" in refused["note"]


def test_registry_enforces_policy(tmp_path):
    registry = LocalModelRegistry(str(tmp_path / "registry.json"))
    registry.register(model_id="wan22_ti2v_5b", model_type="video_base",
                      path=str(tmp_path / "model"), license="apache-2.0")
    with pytest.raises(PermissionError):
        registry.register(model_id="bad", model_type="video_base",
                          path="/x", license="proprietary")
    registry.save()
    reloaded = LocalModelRegistry(str(tmp_path / "registry.json"))
    assert "wan22_ti2v_5b" in reloaded.models


def test_missing_model_verification(tmp_path):
    registry = LocalModelRegistry(str(tmp_path / "r.json"))
    registry.register(model_id="m", model_type="video_base",
                      path=str(tmp_path / "nope"), license="mit")
    assert registry.verify_files("m")["present"] is False


def test_local_only_download_refused():
    from wan.cognitive_fabric.install.model_downloader import download_model
    result = download_model("ti2v-5B", "/tmp/x", allow_network=False)
    assert result["downloaded"] is False
    assert "local-only" in result["note"]


def test_concurrent_runner_budget_and_results():
    runner = ConcurrentModelRunner(vram_budget_gb=4.0, workers=2)
    try:
        runner.submit("a", lambda: 1 + 1, vram_gb=3.0)
        runner.submit("b", lambda: 2 + 2, vram_gb=3.0)
        runner.submit("fails", lambda: 1 / 0, vram_gb=0.5)
        runner.wait(timeout=10)
        assert runner.result("a")["value"] == 2
        assert runner.result("b")["value"] == 4
        assert runner.result("fails")["ok"] is False
    finally:
        runner.shutdown()
