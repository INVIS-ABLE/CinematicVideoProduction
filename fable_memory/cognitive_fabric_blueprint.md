# Cognitive Fabric Blueprint

Design principles locked for all phases:

1. **Engine-internal.** Everything lives in `wan/cognitive_fabric/` and is
   reachable from generate.py; nothing is an external app.
2. **Opt-in.** `--cognitive_fabric` / cognitive-* tasks; original Wan intact.
3. **Honest capability.** A feature is working code, a tested interface, or a
   documented roadmap item — never a fake score or a silent stub.
4. **Local-first.** Cloud SDKs optional; downloads opt-in; generation offline.
5. **Chunked long-form.** Film → acts → scenes → shots → 4n+1-frame chunks;
   disk streaming; checkpoint every chunk; resume from last approved.
6. **Continuity via the engine's own mechanics.** TI2V latent clamp for
   first-frame lock; identity/world/temporal memory conditioning around it;
   Phase 4 moves conditioning into cross-attention context (free-length) —
   never into the video token sequence (RoPE-coupled).
7. **Every brain replaceable.** Registry + bus + per-brain health checks;
   deterministic heuristics first, local models behind the same interfaces.

Key seams (see recon §14): task router (done) · chunk scheduler (done) ·
prompt compiler (done) · first-frame handoff (done, engine-level) ·
context-token injection (hook + tests) · attention bias (shape-safe stub) ·
temporal KV capture (interface) · MoE guide-scale shaping (done, bounded).
