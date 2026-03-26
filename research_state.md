# Research State

## Beliefs
- Starting from OpenAI baseline (9 layers, dim=512, relu^2 MLP, int8+zlib). Baseline BPB: 1.2244 on 8xH100.
- Small-scale smoke tests (dim=128 on A6000) should directionally predict full-scale results.
- No beliefs about what will work yet — need to establish baseline and start exploring.

## Open Questions
- What is our A6000 smoke-test baseline BPB at dim=128?
- Does the ranking of changes at dim=128 transfer to dim=512?
- How much headroom does better quantization (int6, int5) give us in the 16MB budget?
- What activation function works best for this architecture at this scale?

## Queue
1. Run baseline at smoke scale (dim=128) — establish reference BPB
2. Run baseline at full scale (dim=512) on A6000 — establish full-scale reference
3. First experiment: TBD after seeing baseline results
