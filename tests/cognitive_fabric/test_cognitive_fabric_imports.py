"""Every fabric module must import — no dead folders, no broken stubs."""
import importlib

import pytest

CORE = [
    "wan.cognitive_fabric",
    "wan.cognitive_fabric.fabric_config",
    "wan.cognitive_fabric.fabric_state",
    "wan.cognitive_fabric.fabric_bus",
    "wan.cognitive_fabric.fabric_registry",
    "wan.cognitive_fabric.fabric_runtime",
    "wan.cognitive_fabric.fabric_logging",
    "wan.cognitive_fabric.fabric_types",
    "wan.cognitive_fabric.fabric_hooks",
]

BRAINS = [f"wan.cognitive_fabric.brains.{m}" for m in (
    "director_brain", "storyboard_brain", "cinematographer_brain",
    "character_brain", "object_brain", "world_brain", "physics_brain",
    "lighting_brain", "motion_brain", "continuity_brain", "prompt_brain",
    "reference_brain", "anime_brain", "episode_brain", "quality_brain",
    "repair_brain", "upscale_brain", "sound_brain", "resource_brain")]

MEMORY = [f"wan.cognitive_fabric.memory.{m}" for m in (
    "global_identity_matrix", "character_memory", "object_memory",
    "world_memory", "style_memory", "timeline_memory", "reference_memory",
    "temporal_kv_cache", "scene_state_cache", "memory_db")]

PIPELINE = [f"wan.cognitive_fabric.pipeline.{m}" for m in (
    "storyboard_orchestrator", "longform_orchestrator", "chunk_scheduler",
    "scene_compiler", "shot_compiler", "prompt_compiler",
    "multimodal_reference_compiler", "generation_controller",
    "repair_controller", "hierarchical_scaler", "frame_interpolator",
    "stitcher", "finishing_pipeline", "anime_episode_pipeline",
    "export_engine")]

CONDITIONING = [f"wan.cognitive_fabric.conditioning.{m}" for m in (
    "reference_token_stack", "identity_token_injector",
    "object_token_injector", "world_token_injector", "physics_guidance",
    "collision_guidance", "lighting_guidance", "camera_guidance",
    "audio_guidance", "style_guidance", "storyboard_guidance")]

VALIDATORS = [f"wan.cognitive_fabric.validators.{m}" for m in (
    "human_validator", "anatomy_validator", "hand_validator",
    "face_validator", "identity_validator", "object_validator",
    "physics_validator", "lighting_validator", "flicker_validator",
    "motion_validator", "text_render_validator", "anime_validator",
    "quality_report", "world_validator")]

FACTORY_INSTALL = [
    "wan.cognitive_fabric.model_factory.local_model_registry",
    "wan.cognitive_fabric.model_factory.allowed_model_sources",
    "wan.cognitive_fabric.model_factory.adapter_trainer",
    "wan.cognitive_fabric.model_factory.lora_manager",
    "wan.cognitive_fabric.model_factory.concurrent_model_runner",
    "wan.cognitive_fabric.model_factory.model_clone_manager",
    "wan.cognitive_fabric.install.dependency_checker",
    "wan.cognitive_fabric.install.cuda_checker",
    "wan.cognitive_fabric.install.torch_checker",
    "wan.cognitive_fabric.install.ffmpeg_checker",
    "wan.cognitive_fabric.install.model_downloader",
    "wan.cognitive_fabric.install.first_run_setup",
    "wan.cognitive_fabric.install.health_check",
]


@pytest.mark.parametrize("module", CORE + BRAINS + MEMORY + PIPELINE +
                         CONDITIONING + VALIDATORS + FACTORY_INSTALL)
def test_module_imports(module):
    importlib.import_module(module)


def test_base_wan_still_imports():
    """Refactors R-001/R-003/R-005 must keep base `import wan` working."""
    import wan
    assert hasattr(wan, "WanT2V")
    assert hasattr(wan, "WanTI2V")
    assert "WanS2V" in dir(wan)  # lazy export visible


def test_fabric_exported_lazily_from_wan():
    import wan
    assert wan.cognitive_fabric.__version__
