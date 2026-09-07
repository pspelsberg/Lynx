# Review 107 — teacher/LoRA infrastructure

## Standards

No findings. Teacher/student evaluation is bounded, uses the public `InferenceBackend.decide` seam, verifies results deterministically when a contract is supplied, and does not persist model reasoning. LoRA export accepts only `TrainingExample` records with successful status, passed verification, non-empty check IDs, event sequence and run lineage.

## Spec

No findings in the implemented infrastructure. `TeacherStudentEvaluator` infrastructure was exercised with deterministic test backends; no 9B/27B model was loaded because the runtime is restricted to `models/Qwen3.5-4B-Q6_K.gguf`. `LoRAExperimentSpec` and `write_verified_lora_dataset` provide the required structured-data gate without pretending that training occurred.

The 27B Teacher requirement and actual LoRA training remain optional/resource-gated; no extra model was downloaded. This is recorded in `build/optional-models-not-run.json`.

Validation: `uv run python -m unittest discover -s tests -v` — 119 passed, 1 skipped; `uv run python -m compileall -q src scripts` — passed.
