# Refactor Log

Every change to upstream Wan code, in order. Additive fabric code
(`wan/cognitive_fabric/**`, `tests/cognitive_fabric/**`, configs, docs,
installers) is not listed here — see git history.

## R-001 — wan/__init__.py — lazy heavy exports
- **Problem found:** `import wan` unconditionally imported `speech2video`
  (librosa, decord) and `animate` (cv2, peft, decord) — modules whose deps
  live only in requirements_s2v.txt / requirements_animate.txt. Base installs
  could not import the package at all.
- **Change:** PEP 562 `__getattr__` lazy export for `WanS2V` / `WanAnimate`;
  identical API, cached after first access, clear ImportError naming the
  extras file if deps are missing.
- **Risk:** none identified; `dir(wan)` still lists both names (tested).

## R-002 — wan/modules/model.py — attention dispatcher
- **Problem found:** the DiT called `flash_attention()` directly, bypassing
  the SDPA fallback that already existed in `attention.py`; the model could
  not run anywhere without flash-attn.
- **Change:** self-attention and cross-attention call `attention()` — which
  forwards identical args to `flash_attention()` when flash-attn is present.
- **Risk:** none for flash users (same code path). SDPA path drops padding
  masks with upstream's own warning (upstream behaviour, documented).
- **Note:** s2v/animate model variants still import flash_attention directly;
  Phase 1 deliberately leaves them untouched.

## R-003 — wan/modules/t5.py — lazy device default
- **Problem found:** `device=torch.cuda.current_device()` as a default
  argument executes at class-definition (import) time → crash on CUDA-less
  machines.
- **Change:** `device=None` + lazy resolution in `__init__`.
- **Risk:** none; every in-repo caller passes an explicit device.

## R-004 — wan/modules/attention.py — fallback dtype contract
- **Problem found:** flash path returns the caller's dtype
  (`x.type(out_dtype)`); the SDPA fallback returned the half compute dtype,
  crashing fp32 callers (`mat1 and mat2 must have the same dtype`).
- **Change:** fallback captures `out_dtype = q.dtype` and casts the output
  back — the two paths now share one contract (dtype in == dtype out).
- **Risk:** none; flash path untouched.

## R-005 — wan/utils/prompt_extend.py — optional dashscope
- **Problem found:** `import dashscope` (paid cloud SDK) at module import →
  generate.py failed to start on base installs; local-first violated at
  import time.
- **Change:** optional import; `DashScopePromptExpander` raises a clear
  ModuleNotFoundError only when actually instantiated without the SDK,
  pointing to the local `QwenPromptExpander`.
- **Risk:** none when dashscope installed.

## generate.py — additive cognitive route
- `COGNITIVE_TASKS`, fabric CLI flags, `run_cognitive()` dispatched before
  the original path. Classic tasks: unchanged validation, unchanged flow
  (verified: missing --ckpt_dir still asserts exactly as upstream).

## Metric fix during test hardening
- QualityBrain flicker metric moved from first-order to second-order
  luminance differences: smooth exposure ramps (camera motion) no longer
  read as flicker; alternating-frame strobing still does. Caught by
  test_report_accepts_clean_video.
