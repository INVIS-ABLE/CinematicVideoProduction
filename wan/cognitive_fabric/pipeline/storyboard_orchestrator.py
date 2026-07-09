"""Storyboard orchestrator (spec §9): parses a full film plan and drives Wan
chunk by chunk — memory registration, conditioning, generation, validation,
repair, disk streaming, checkpointing, resume, stitching.

Never holds more than one chunk of frames in memory; approved chunks stream
to <project>/chunks/ and the ledger + checkpoint make any crash resumable
from the last approved chunk.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from ..brains.continuity_brain import ContinuityBrain
from ..brains.character_brain import CharacterBrain
from ..brains.object_brain import ObjectBrain
from ..brains.quality_brain import QualityBrain
from ..brains.repair_brain import RepairBrain
from ..brains.storyboard_brain import StoryboardBrain
from ..brains.world_brain import WorldBrain
from ..fabric_config import FabricConfig
from ..fabric_logging import get_fabric_logger
from ..fabric_state import FabricState
from ..fabric_types import ChunkSpec
from ..memory.character_memory import CharacterMemory
from ..memory.global_identity_matrix import GlobalIdentityMatrix
from ..memory.memory_db import MemoryDB
from ..memory.object_memory import ObjectMemory
from ..memory.style_memory import StyleMemory
from ..memory.temporal_kv_cache import TemporalKVCache
from ..memory.timeline_memory import TimelineMemory
from ..memory.world_memory import WorldMemory
from ..validators.quality_report import build_quality_report
from .chunk_scheduler import plan_storyboard_chunks
from .generation_controller import (
    extract_last_frame_png,
    save_chunk_video,
)
from .prompt_compiler import PromptCompiler
from .repair_controller import apply_repair_action
from .stitcher import concat_clips, ffmpeg_available


class StoryboardOrchestrator:
    def __init__(self, config: Optional[FabricConfig] = None):
        self.config = config or FabricConfig()
        self.storyboard_brain = StoryboardBrain()

    # ------------------------------------------------------------------
    # planning (pure, no side effects) — used by dry runs and tests
    # ------------------------------------------------------------------

    def load_storyboard(self, source: Any) -> Dict[str, Any]:
        if isinstance(source, (str, Path)):
            data = json.loads(Path(source).read_text(encoding="utf-8"))
        else:
            data = source
        problems = self.storyboard_brain.validate(data)
        if problems:
            raise ValueError("invalid storyboard: " + "; ".join(problems))
        return self.storyboard_brain.normalise(data)

    def plan(self, storyboard: Dict[str, Any]) -> List[ChunkSpec]:
        cf = self.config.data["cognitive_fabric"]
        width, height = self.config.native_generation_size(
            cf["base_resolution"])
        fps = storyboard.get("project", {}).get("frame_rate",
                                                cf["frame_rate"])
        return plan_storyboard_chunks(
            storyboard, fps=fps, chunk_seconds=cf["chunk_seconds"],
            overlap_frames=cf["overlap_frames"], width=width, height=height)

    def dry_run(self, source: Any) -> Dict[str, Any]:
        """Full plan without generating a single frame."""
        storyboard = self.load_storyboard(source)
        chunks = self.plan(storyboard)
        total_frames = sum(c.frame_num for c in chunks)
        fps = storyboard.get("project", {}).get(
            "frame_rate", self.config.data["cognitive_fabric"]["frame_rate"])
        return {
            "mode": "dry_run",
            "shots": len(storyboard.get("shots", [])),
            "scenes": len(storyboard.get("scenes", [])),
            "chunks": len(chunks),
            "total_frames": total_frames,
            "estimated_minutes": round(total_frames / fps / 60, 2),
            "chunk_ids": [c.chunk_id for c in chunks],
        }

    # ------------------------------------------------------------------
    # execution
    # ------------------------------------------------------------------

    def run(self, source: Any, project_dir: str, engine: Any,
            project_id: Optional[str] = None) -> Dict[str, Any]:
        started = time.time()
        storyboard = self.load_storyboard(source)
        project_id = project_id or storyboard["project"].get(
            "title", "untitled").lower().replace(" ", "_")[:40]

        logger = get_fabric_logger("orchestrator", project_dir)
        state = FabricState.resume_or_create(project_id, project_dir)
        state.storyboard = storyboard
        db = MemoryDB(str(Path(project_dir) / "memory.db"))

        # --- memory registration -------------------------------------------------
        cf = self.config.data["cognitive_fabric"]
        matrix = GlobalIdentityMatrix(
            embed_dim=self.config.get("engine.reference_embed_dim", 256))
        worlds = WorldMemory()
        characters = CharacterMemory(matrix)
        objects = ObjectMemory(matrix)
        style = StyleMemory({"look": storyboard["project"].get("style", "")})
        WorldBrain(worlds).register_worlds(storyboard)
        CharacterBrain(characters).register_characters(storyboard)
        ObjectBrain(objects).register_objects(storyboard)

        cache = TemporalKVCache(
            max_cache_chunks=self.config.get("temporal_memory.max_cache_chunks", 12),
            decay_rate=self.config.get("temporal_memory.decay_rate", 0.82),
            scene_boundary_decay=self.config.get(
                "temporal_memory.scene_boundary_decay", 0.35))
        continuity = ContinuityBrain(worlds, cache)
        quality = QualityBrain(self.config.data.get("repair", {}))
        repair = RepairBrain(
            max_attempts=cf["max_regenerations_per_chunk"],
            thresholds=self.config.data.get("repair", {}))
        compiler = PromptCompiler(matrix=matrix, worlds=worlds,
                                  objects=objects, style=style)

        shots_by_id = {s["shot_id"]: s for s in storyboard["shots"]}
        shot_order = [s["shot_id"] for s in storyboard["shots"]]
        chunks = self.plan(storyboard)
        for chunk in chunks:
            state.register_chunk(chunk.chunk_id, chunk.shot_id, chunk.scene_id)
            db.upsert_shot(chunk.shot_id, chunk.scene_id,
                           shots_by_id[chunk.shot_id])

        timeline = TimelineMemory()
        terminal_frame_png: Optional[str] = None
        previous_shot: Optional[Dict[str, Any]] = None
        generated, repaired, skipped = 0, 0, 0

        for chunk in chunks:
            shot = shots_by_id[chunk.shot_id]
            ledger = state.ledger[chunk.chunk_id]
            if ledger.status in ("approved", "accepted_with_warning") and \
                    ledger.output_path and Path(ledger.output_path).exists():
                logger.info("resume: %s already approved, skipping",
                            chunk.chunk_id)
                timeline.append_clip(
                    chunk_id=chunk.chunk_id, shot_id=chunk.shot_id,
                    scene_id=chunk.scene_id, path=ledger.output_path,
                    frame_num=chunk.frame_num, fps=chunk.fps,
                    transition_in=shot.get("transition_in", "cut"),
                    transition_out=shot.get("transition_out", "cut"))
                skipped += 1
                previous_shot = shot
                continue

            # continuity conditioning (previous shot + world memory)
            clauses = continuity.continuity_clauses(shot, previous_shot)
            compiled = compiler.compile_chunk(chunk, shot,
                                              continuity_clauses=clauses)
            conditioning = {
                "prompt": compiled["prompt"],
                "negative_prompt": compiled["negative_prompt"],
                "pack_summary": compiled["pack"].summary(),
            }

            # first-frame continuity via predecessor terminal frame
            first_frame = None
            if chunk.index > 0 or (shot.get("continuity_from_previous")
                                   and previous_shot is not None):
                if terminal_frame_png and Path(terminal_frame_png).exists():
                    from PIL import Image
                    first_frame = Image.open(terminal_frame_png)
                    chunk.first_frame_path = terminal_frame_png

            attempt = 0
            video: Optional[torch.Tensor] = None
            report = None
            while True:
                state.update_chunk(chunk.chunk_id, status="generating",
                                   attempt=attempt)
                logger.info("generating %s (attempt %d, engine=%s)",
                            chunk.chunk_id, attempt, engine.name)
                video = engine.generate_chunk(chunk, conditioning, first_frame)

                seam = continuity.check_seam(video[:, :4])
                identity_sim = None  # per-frame identity model lands later
                report = build_quality_report(
                    chunk.chunk_id, chunk.shot_id, chunk.scene_id, video,
                    quality_brain=quality,
                    seam_similarity=seam.get("similarity"),
                    identity_similarity=identity_sim)
                db.add_quality_report(chunk.chunk_id, report.overall(),
                                      report.to_dict())

                decision = repair.decide(report, attempt)
                if decision["action"] in ("accept", "accept_with_warning"):
                    break
                db.add_repair(chunk.chunk_id, attempt, decision["action"],
                              decision["reason"])
                chunk, conditioning = apply_repair_action(
                    decision["action"], chunk, conditioning, video)
                repaired += 1
                attempt += 1

            # stream to disk, capture memory, checkpoint
            out_path = save_chunk_video(
                video, str(Path(project_dir) / "chunks" / chunk.chunk_id),
                fps=chunk.fps)
            terminal_frame_png = extract_last_frame_png(
                video, str(Path(project_dir) / "chunks" /
                           f"{chunk.chunk_id}_last.png"))
            cache.capture(
                chunk.chunk_id, chunk.scene_id,
                terminal_frames=video[:, -min(8, video.shape[1]):],
                identity_tokens=matrix.get_identity_tokens_for_shot(
                    shot.get("character_ids", []), shot.get("object_ids", [])))
            cache.decay(scene_boundary=not shot.get("continuity_to_next", True))

            status = ("accepted_with_warning"
                      if decision.get("exhausted") else "approved")
            state.update_chunk(chunk.chunk_id, status=status,
                               output_path=out_path,
                               quality_overall=report.overall())
            db.upsert_chunk(chunk.chunk_id, chunk.shot_id, status,
                            conditioning["pack_summary"], attempt, out_path)
            timeline.append_clip(
                chunk_id=chunk.chunk_id, shot_id=chunk.shot_id,
                scene_id=chunk.scene_id, path=out_path,
                frame_num=chunk.frame_num, fps=chunk.fps,
                transition_in=shot.get("transition_in", "cut"),
                transition_out=shot.get("transition_out", "cut"))
            state.save_checkpoint()
            matrix.save(str(Path(project_dir) / "checkpoints"))
            worlds.save(str(Path(project_dir) / "checkpoints"))
            generated += 1
            previous_shot = shot
            del video

        timeline.save(str(Path(project_dir) / "timeline.json"))

        # --- stitch ------------------------------------------------------------------
        export_path = None
        clip_paths = [e["path"] for e in timeline.entries
                      if e["path"].endswith(".mp4")]
        if clip_paths and ffmpeg_available():
            export_path = concat_clips(
                clip_paths,
                str(Path(project_dir) / "exports" / "draft_assembly.mp4"))
        elif clip_paths:
            get_fabric_logger("orchestrator").warning(
                "ffmpeg not found — skipping stitch; clips remain in chunks/")

        report_payload = {
            "project_id": project_id,
            "engine": engine.name,
            "chunks_total": len(chunks),
            "chunks_generated": generated,
            "chunks_skipped_resume": skipped,
            "repair_actions": repaired,
            "export": export_path,
            "timeline": str(Path(project_dir) / "timeline.json"),
            "elapsed_seconds": round(time.time() - started, 2),
        }
        (Path(project_dir) / "run_report.json").write_text(
            json.dumps(report_payload, indent=2), encoding="utf-8")
        db.close()
        return report_payload
