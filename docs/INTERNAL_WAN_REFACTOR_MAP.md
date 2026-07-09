# Internal Wan Refactor Map

Exact list of upstream files modified by the Cognitive Fabric, what changed,
and why it is safe. Everything else under `wan/` (outside
`wan/cognitive_fabric/`) is untouched upstream code at commit
`42bf4cfaa384bc21833865abc2f9e6c0e67233dc`.

| ID | File | Change | Behaviour with full upstream deps | Behaviour gained |
| --- | --- | --- | --- | --- |
| R-001 | `wan/__init__.py` | `WanS2V`/`WanAnimate` exported lazily (PEP 562) | identical — same attributes, same classes | `import wan` works on base installs without s2v/animate extras |
| R-002 | `wan/modules/model.py` | self/cross attention call the `attention()` dispatcher instead of `flash_attention()` directly | identical — dispatcher forwards the same args to flash-attn | DiT forward runs on CPU / non-flash GPUs via SDPA fallback (smoke-testable) |
| R-003 | `wan/modules/t5.py` | `device` default evaluated lazily instead of `torch.cuda.current_device()` at class-definition time | identical — all in-repo callers pass a device explicitly | module imports on CUDA-less machines |
| R-004 | `wan/modules/attention.py` | SDPA fallback returns the caller's dtype (matching the flash path contract) | identical — flash path unchanged | fp32 callers (CPU tests) no longer crash on dtype mismatch |
| R-005 | `wan/utils/prompt_extend.py` | `dashscope` (cloud SDK) import made optional; clear error only when DashScope expander is actually used | identical when dashscope installed | generate.py imports without the cloud SDK; local-first at import time |
| — | `generate.py` | additive: `COGNITIVE_TASKS`, fabric CLI flags, `run_cognitive()` branch before the original path | identical for classic tasks (flag default false) | cognitive-short / cognitive-film / cognitive-anime-episode |

## Exact Wan call chain the fabric drives

```
FabricRuntime.run_short/run_film
  → StoryboardOrchestrator.run
    → PromptCompiler.compile_chunk        (conditioning pack → prompt strings)
    → WanTI2VEngine.generate_chunk
      → wan.WanTI2V.generate(input_prompt, img=terminal_frame, frame_num, seed, n_prompt, …)
        → T5EncoderModel([prompt]) / ([n_prompt])
        → Wan2_2_VAE.encode (i2v continuity path)
        → FlowUniPCMultistepScheduler loop
          → WanModel.forward(...)          ← R-002 dispatcher inside
        → Wan2_2_VAE.decode → [3, F, H, W]
    → validators.build_quality_report → RepairBrain.decide → (loop)
    → save_chunk_video / TemporalKVCache.capture / FabricState.save_checkpoint
    → stitcher.concat_clips → exports/
```

## Rules enforced

1. Original Wan path must run after every change (guarded by
   `tests/cognitive_fabric/test_tensor_shapes.py` + import tests +
   the untouched-task CLI assert behaviour).
2. Every upgrade touches or controls the real Wan pipeline (see call chain).
3. No bolt-on that does not feed back into generation.
4. Local-first: DashScope optional (R-005), model downloads opt-in,
   generation never touches the network.
