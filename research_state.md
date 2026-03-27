# Research State

## Current Best
- **27a4927a**: val_bpb=1.3591, dim=320, 12 layers, 8 heads, 4 KV heads, mlp_mult=3, SwiGLU, 2500 steps, warmdown_iters=500, grad_clip_norm=1.0 (15.1M params, 15.2MB artifact)
- Full-scale baseline target: 1.2244 BPB

## Key Findings
1. **Warmdown fix was massive**: Enabling step-based warmdown (max_wallclock_seconds=0) gave -0.0445 BPB
2. **SwiGLU confirmed better than relu^2**: -0.0066 BPB at dim=320 with same params
3. Width scaling: 128→256 massive, 256→320 modest; diminishing per-param returns
4. Training duration: strongly diminishing returns. 2000→2500 gave only -0.016 BPB
5. matrix_lr=0.04 is at or near optimal
6. Artifact at 15.2MB — near 16MB limit
7. **Warmdown fraction: 25% > 15% per earlier comparison**. Currently at 20% (500/2500).
8. **LR warmup of 50 steps hardcoded in lr_mul** (config warmup_steps=5 is for compile warmup only)
9. **Gradient clipping (1.0) has negligible effect** — only -0.0004 BPB.
10. **Cosine LR schedule is WORSE than linear warmdown** by +0.0057 BPB.
11. **10L×352d at 500 steps ≈ 9L×256d at 1000 steps** (1.5438 vs 1.5428) — unfair comparison since model was severely undertrained.

## Latest Run Analysis (139aacf0)
- 10L×352d, 500 steps, BPB=1.5438
- 15.2M params, 13.1MB artifact  
- Model was severely undertrained — loss still dropping fast at step 500
- Cannot conclude architecture is worse than 12L×320d since training was insufficient
- Major gradient spikes observed especially around warmdown phases
- The comparison is uninformative due to insufficient training

## Prediction Calibration
- First run, no prediction to check

## Hypotheses (Updated)
1. **dim=336 with 12 layers** could use ~1.3M more params, getting closer to 16MB budget, potentially improving BPB by ~0.005-0.01
2. **Warmdown_iters at 25% of total** could help (currently 20%)
3. **More training steps (3000)** would give ~0.005-0.008 BPB improvement
4. **Weight decay** for Adam params might help slightly
5. **Different model shape** (10L wider) might or might not be better - inconclusive from 139aacf0

## Open Questions (Priority Order)
1. **Can we fill the 16MB budget better?** dim=336@12L would add ~1.3M params
2. **Would 10L vs 12L be better at same total param count?** Still unknown
3. **Would warmdown_iters=625 (25% of 2500) help?**
4. **Would 3000 steps help enough to justify training time?**
5. **Any code-level improvements** (weight decay, different init, etc.)?

## Experiment Queue (ranked by expected value)
1. **dim=336, 12 layers, 500 steps** — fill budget, quick directional signal
2. **warmdown_iters=625 at 2500 steps** — quick config change
3. **3000 steps with warmdown_iters=750** — more training
4. **Weight decay (0.01) for Adam params**
5. **16 heads instead of 8** — different attention granularity
6. **Longer run of 10L×352d** to fairly compare architecture

## Strategy Notes
- Operator wants bold architectural changes and breadth of exploration
- We've been incrementally tuning; need to try broader changes
- The 16MB budget allows ~16M params — currently at 15.1M
- Short runs (200-500 steps) for directional signal, then longer to confirm
