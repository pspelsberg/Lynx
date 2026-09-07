# Lynx runbook

## Preconditions and model integrity

1. Set an immutable `LLAMA_CPP_REF` to an exact 40-character commit SHA; branches, tags, and short SHAs are rejected.
2. Use `MODEL_ID` with `scripts/download-model.sh`; the script resolves `config/model-manifest.json`, requires a pinned revision and 64-character SHA-256, downloads to a temporary file, verifies, and atomically renames it.
3. Do not download the whole matrix. The reproducible local set used here is `Qwen3.5-4B-Q6_K.gguf` only.

```bash
MODEL_ID=qwen35-4b-q6_k bash scripts/download-model.sh
sha256sum models/Qwen3.5-4B-Q6_K.gguf
```

## Build and smoke test

```bash
LLAMA_CPP_REF=<immutable-40-character-commit-sha> LLAMA_BACKENDS=rocm bash scripts/build-llama.sh
# Development headers are build-only dependencies (not model downloads):
git clone --depth 1 --branch v1.4.313 https://github.com/KhronosGroup/Vulkan-Headers.git build/vulkan-headers
git clone --depth 1 --branch vulkan-sdk-1.4.313.0 https://github.com/KhronosGroup/SPIRV-Headers.git build/spirv-headers-src
cmake -S build/spirv-headers-src -B build/spirv-headers-build -DCMAKE_INSTALL_PREFIX=$PWD/build/spirv-headers
cmake --build build/spirv-headers-build --target install -j2
LLAMA_CPP_REF=<immutable-40-character-commit-sha> LLAMA_BACKENDS=vulkan VULKAN_INCLUDE_DIR=$PWD/build/vulkan-headers/include VULKAN_LIBRARY=/usr/lib/libvulkan.so SPIRV_HEADERS_DIR=$PWD/build/spirv-headers SPIRV_HEADERS_INCLUDE_DIR=$PWD/build/spirv-headers/include bash scripts/build-llama.sh
bash scripts/check-amd.sh
MODEL_PATH=$PWD/models/Qwen3.5-4B-Q6_K.gguf MODEL_SHA256=<manifest-sha> bash scripts/run-llama-rocm.sh
LYNX_INTEGRATION=1 uv run python -m unittest discover -s tests -v
```

The server/API bind to loopback by default. Configure `LYNX_API_TOKEN` before exposing the API behind an authenticated TLS reverse proxy.

## Reproducible profiler

The profiler never downloads models and only uses explicitly passed files. It records model hash, backend, context, slots, Flash Attention, mode outcomes, latency, driver output, task success, and verifier rate:

```bash
uv run python scripts/profile-local.py --models models/Qwen3.5-4B-Q6_K.gguf --output build/profile-4b.json
uv run python scripts/profile-local.py --models models/Qwen3.5-4B-Q6_K.gguf --output build/profile-4b-q6.json
```

Missing backends are recorded as failures rather than silently treated as CPU. The checked-in run artifacts include both ROCm and Vulkan runs; the Vulkan development headers are pinned in the local build manifest and are not model downloads.

Teacher/student holdouts use `TeacherStudentEvaluator` and store only verdicts/latencies; no reasoning text is retained. Both backends must explicitly declare the permitted `Qwen3.5-4B-Q6_K.gguf` identity; missing identities and the requested 27B teacher fail closed, so a same-model comparison is never reported as a teacher result. LoRA inputs must pass `write_verified_lora_dataset`, which rejects unverified trajectories before any external trainer is invoked. `LoRAExperimentSpec` and `evaluate_lora_ablation` accept only the canonical model identity; no training run or improvement is claimed without a real local artifact and measured holdout.

The optional Rust-boundary decision is measured with `uv run python scripts/measure-python-boundary.py`; the current artifact `build/python-boundary-measurement.json` shows no measured Python scheduler/gateway bottleneck, so no Rust implementation is enabled.

## Replay, fork and incident handling

```bash
uv run lynx replay <run-id> --branch main
uv run lynx fork <run-id> --from <complete-event-seq>
uv run lynx resume <run-id> --from <complete-event-seq>
```

Replay uses recorded decisions and observations only; it never invokes external tools. On a tool, child, MCP, browser, or kernel timeout, preserve the final timeout event, verify the owned process is terminated, inspect the budget snapshot, and resume only from a complete checkpoint. Do not treat a fork as rollback of an external side effect.

## Retention and secrets

Trajectories, flight events, and artifacts are append-only audit records. Redaction runs before persistence, including nested values, URL credentials, bearer tokens, headers, cookies, and UTF-8 byte artifacts. Apply deployment-specific rotation/retention to `data/runs`, `data/trajectories.jsonl`, and `artifacts`; never place credentials in a task, prompt, model output, or fixture.


## Reproduced RX 9060 XT profile

The Qwen3.5-4B-Q6_K-only matrix was executed with `scripts/profile-local.py` for ROCm/Vulkan, 4k/8k/16k context, 1/2/3 slots, and Flash Attention on/off. The immutable model digest is `fdedd781c9ce676ab66b018ca247ff78e8a33c98098a822c1e2d5075e7718f66`; the observed llama.cpp commit and driver are recorded per report. Reports: `build/profile-rx9060xt-qwen35-4b-q6.json`.

The opt-in `ngram-simple` speculative smoke run is recorded in `build/profile-rx9060xt-qwen35-4b-ngram.json`; `build/speculative-evaluation.json` records that it was not promoted because p50 latency was slightly worse (quality/verifier rate remained 1.0). No draft or second model was loaded.
