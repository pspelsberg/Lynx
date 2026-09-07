# Review 106 — remaining hardware/optional slices

## Standards

No findings. The speculative launch path is shell-validated and remains opt-in; draft paths are rejected by the single-model policy. Architecture fitness is green and no direct external calls were added to the agent loop. The profiler remains callback/budget bounded.

## Spec

No defects in the newly implemented scope. TASK-015 is implemented and tested with the permitted model backend; no separate-model comparison is claimed. The 27B/9B comparison rows are historical, non-executable references because only Qwen3.5-4B-Q6_K may be used. TASK-065 has a current separate ngram artifact (`build/profile-rx9060xt-qwen35-4b-ngram.json`); its failed research request and lower verifier rate are recorded, so promotion is rejected. TASK-084 has ACP round-trip coverage. TASK-102 and TASK-103 have executable artifacts. The profiler matrix contains only the permitted Qwen3.5-4B-Q6_K model.

The only non-`[x]` statuses are TASK-074 (27B teacher) and TASK-075 (LoRA/QLoRA training). TASK-064 is complete within the executable single-model scope: its 4B Q6_K ROCm/Vulkan/context/slot/flash matrix is recorded in `build/profile-rx9060xt-qwen35-4b-q6.json`; historical 9B rows are not runtime work. `build/optional-models-not-run.json` records only out-of-scope model references without fabricated data, honoring the explicit no-extra-model-download constraint.

Validation: `uv run python -m unittest discover -s tests -v` — 119 passed, 1 skipped; `uv run python -m compileall -q src scripts` — passed; shell `bash -n` passed for all relevant scripts.
