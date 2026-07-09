# Model Factory

Policy first (enforced in code, `model_factory/allowed_model_sources.py`):
**only open, licensed, self-trained or user-owned model sources.** Ingesting
or cloning proprietary weights (e.g. Seedance) is refused with a
`PermissionError`. "Cloning" in this project means building local functional
equivalents from allowed sources — never weight extraction.

## Working today

- `LocalModelRegistry` — JSON-backed registry of every base model, adapter,
  validator and encoder; license policy checked at registration; file
  manifest hashing for integrity (`verify_files`).
- `ConcurrentModelRunner` — priority job queue with a VRAM budget: jobs that
  don't fit wait; validators can run on CPU while the GPU generates;
  pause/resume; per-job results and errors.
- `ModelCloneManager.ingest` — policy-gated registration path.

## Interfaces pinned for Phase 8 (raise instead of pretending)

- `LoraManager.load` — general LoRA for WanModel needs adapter key-mapping
  against the DiT module names (upstream only uses peft inside Animate's
  relighting LoRA).
- `AdapterTrainer` — style/character/object/world/motion adapter training
  (`AdapterTrainingSpec` contract is final; loops land in Phase 8).
- Reward/quality model training.

Registry example: `configs/model_registry.example.json`.
