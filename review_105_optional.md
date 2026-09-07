# Review 105 — optional and hardware slices

## Standards

No findings. The new speculative runner path is opt-in, rejects draft models under the single-model policy, and shell syntax checks pass. The independent verifier stores only verdict/check metadata, not reasoning text. Architecture fitness is green.

## Spec

No implementation defects found. TASK-015 is implemented and tested as an optional semantic critic; no separate-model quality claim is made. TASK-065 has the current canonical ngram artifact and a quality/latency approval gate; the observed research failure is retained and promotion is rejected. TASK-102/103 have executable fitness/measurement artifacts. TASK-064 has complete ROCm/Vulkan matrix coverage for the permitted Qwen3.5-4B-Q6_K model; historical 9B/27B variants are outside the executable single-model scope.

Remaining statuses are intentional optional/resource-gated work: TASK-074 27B teacher and TASK-075 LoRA training; TASK-065 is implemented with a safe non-promoting result. No fabricated measurements were added.

Validation: `uv run python -m unittest discover -s tests -v` — 119 passed, 1 skipped; `uv run python -m compileall -q src scripts` — passed.
