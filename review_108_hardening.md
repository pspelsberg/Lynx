# Review 108 — contract, cancellation, and model hardening

## Standards

No new findings in the hardened seams. Contract budgets are bounded and always ledgered; risk/confirmation checks occur before tool execution. JSON parsing rejects duplicate/non-finite values, artifacts preserve immutable provenance, and tool observations remain data-plane.

## Spec

The production CLI/API now construct root contracts and enforce verified status. RAH child budgets carry web/browser/python/recursive dimensions, child artifacts require verified stores, and controller rejection/final events are emitted. Research calls and subprocess-backed adapters are deadline/cancellation bounded. Runtime launch/profile defaults are restricted to `models/Qwen3.5-4B-Q6_K.gguf`; no additional model was downloaded.

## Validation

- `uv run python -m unittest discover -s tests -v`: 119 passed, 1 skipped (opt-in live llama test)
- `uv run python -m compileall -q src scripts`: passed
- `bash -n scripts/build-llama.sh scripts/download-model.sh scripts/run-llama-rocm.sh scripts/run-llama-vulkan.sh`: passed
- `uv run lynx demo`: passed

Resource-gated benchmark/training slices remain honestly unexecuted; no fabricated 9B/27B or LoRA measurements were added.
