# Lynx — Policy-Driven Agent Harness

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Zero Runtime Dependencies](https://img.shields.io/badge/core_deps-zero-success.svg)](pyproject.toml)
[![Tests](https://img.shields.io/badge/tests-148%20passed-brightgreen.svg)](tests/)
[![Hardware](https://img.shields.io/badge/accelerated-AMD%20ROCm%20%7C%20Vulkan-red.svg)](scripts/)
[![Model](https://img.shields.io/badge/model-Qwen3.5--4B--Q6__K.gguf-orange.svg)](config/model-manifest.json)

**Lynx** is a lightweight, policy-driven agent harness designed from the ground up around small local models. Rather than relying on massive frontier LLMs, Lynx treats the model (`Qwen3.5-4B-Q6_K.gguf`) strictly as a **decision kernel**: compute budgets, tool gating, output validation, artifact management, and audit trajectories reside entirely outside the model boundary.

This repository is an engineered, hardened proof-of-concept (POC v0.1.0) demonstrating that a 4B parameter model running locally on consumer hardware can reliably perform structured coding, research, and multi-step reasoning tasks when governed by deterministic external policies.

---

## Key Documentation

- [Architecture & Security Boundaries](docs/architecture.md)
- [Operational & Incident Runbook](docs/runbook.md)
- [ADR 0001: llama.cpp as Local Inference Backend](docs/adr/0001-llama-cpp-as-local-inference-backend.md)
- [Harness Guide & Agent Rules](AGENTS.md)

---

## Core Philosophy & Design Principles

```
  ┌─────────────────────────────────────────────────────────┐
  │                      Lynx Harness                       │
  │                                                         │
  │  ┌──────────────┐   Decide   ┌───────────────────────┐  │
  │  │  Inference   │ ◄───────── │   Deterministic Loop  │  │
  │  │   Backend    │ ─────────► │ (Observe-Decide-Act)  │  │
  │  └──────────────┘            └───────────┬───────────┘  │
  │    (llama.cpp)                           │              │
  │                                          │ Enforce      │
  │                                          ▼              │
  │                              ┌───────────────────────┐  │
  │                              │      ToolGateway      │  │
  │                              │ (L0-L3 Risk Policies) │  │
  │                              └───────────┬───────────┘  │
  │                                          │ Execute      │
  │                                          ▼              │
  │                              ┌───────────────────────┐  │
  │                              │   Workspace & Tools   │  │
  │                              │  (AST, Diff, Files)   │  │
  │                              └───────────────────────┘  │
  └─────────────────────────────────────────────────────────┘
```

1. **Zero Runtime Core Dependencies:** The core harness is built entirely with Python 3.11+ standard library features (fully async). Model backends, web APIs (`fastapi`/`uvicorn`), evaluation parsers (`pyyaml`), and browser sensors (`playwright`) are strictly opt-in extras.
2. **Untrusted Observations as Data-Plane:** Tool outputs and web search responses are classified as untrusted external observations. They carry provenance and trust metadata (`can_instruct=False`, `can_authorize=False`) and cannot override system policy.
3. **Strict Tool Risk Levels & Approval Modes:** Every tool call is vetted by a `ToolGateway` before execution. Destructive actions require explicit confirmation, and arbitrary shell execution is denied.
4. **Immutable Audit Trajectories & Checkpoints:** Every state transition, action, decision, and observation is persisted as an append-only, secret-redacted JSONL flight log with SHA-256 content-addressed artifacts.
5. **Deterministic Replay Without Side Effects:** Runs can be replayed offline from recorded logs. External tools, web APIs, and model servers are never re-invoked during replays.

---

## Feature Matrix

### Deterministic Agent Loop & Reasoning
- **Loop Architecture:** Strict, deterministic `observe` → `decide` → `act` → `finish` transitions enforced by a state machine `Controller`.
- **Operating Modes:**
  - `think` *(Default)*: Full chain-of-thought allocation with external compute budgets.
  - `fast`: Disables thinking tokens for fast, low-latency utility commands.
  - `research`: Multi-step evidence collection, claim synthesis, and freshness verification.
- **KV-Cache Prompt Stabilization:** Deterministic tool catalog sorting maximizes local prefix cache hits in `llama-server`.
- **Sliding Observation Compaction:** Automatically compacts older tool outputs to avoid attention degradation and context exhaustion during long sessions.
- **Progress Monitor:** Proactively detects and breaks cyclic action-observation loops.

### Safety & Workspace Integrity
- **Granular Approval Modes:** `plan`, `ask`, `auto-edit`, `auto`, and `yolo` for fine-grained execution control.
- **Workspace Snapshots & Auto-Rollback:** Automatic filesystem snapshots with `--rollback-on-failure` and interactive CLI undo (`lynx undo [RUN_ID]`).
- **AST-Aware File Operations:**
  - `filesystem.patch`: Applies unified diffs with syntax pre-validation (AST validation for Python, JSON parsing for JSON).
  - `filesystem.read` / `filesystem.write`: Atomic writes, directory traversal guards, size boundaries.
  - `filesystem.grep` / `filesystem.find`: Fast structured code search.
  - `workspace.symbols`: AST symbol extraction for class, function, and import definitions.
  - `workspace.test`: Isolated test execution with execution timeouts.
- **Safe Network & Shell Guards:**
  - Read-only shell allowlist (`pwd`, `ls`, `find`, `rg`, `git status/log/diff/show`).
  - Web search & fetch with strict private IP address / intranet SSRF blocking.

### Flight Recorder & Replay Engine
- **Per-Run Flight Records:** Stored in `data/runs/<run_id>/` with `manifest.json`, sequential events, and execution checkpoints.
- **Offline Deterministic Replay:** `lynx replay <run_id>` re-simulates agent execution without triggering live tools or model queries.
- **Prefix Forking:** `lynx fork <run_id> --from <seq>` branches execution from a historical checkpoint for counterfactual debugging.
- **Checkpoint Resumption:** `lynx resume <run_id> --from <seq>` continues execution from an isolated checkpoint.

### Advanced Architectural Slices (Opt-In)
- **RLM (Recursive Language Model):** Extreme context handling by decoupling context into content-addressed SHA-256 artifacts with Map/Reduce and bounded inspection engines (`RlmContextEngine`).
- **RAH (Recursive Agent Harness):** Allows root agents to spawn isolated child harnesses (`LocalChildExecutor`) with dedicated budgets, recursion depth limits, and tool firewalls. Results pass through an explicit Verifier/Merge Gate.
- **Governed Promotion:** Replay- and verifier-gated promotion for dynamically acquired skills and knowledge candidates into persistent catalogs.
- **Best-of-N Branch Execution:** Parallel branch exploration with independent scoring and branch merge isolation.
- **External Integrations:** Lifecycle-managed MCP (Model Context Protocol) stdio client, persistent Python kernel worker (isolated, network-disabled by default), Playwright browser sensor, and LSP language server.
- **A2A & ACP Protocols:** Agent2Agent task/artifact interchange data models and Agent Control Protocol (ACP) translation.
- **Verified Dataset & LoRA Export:** Gates for exporting verified teacher/student trajectories for fine-tuning without prompt/thought pollution.

---

## Quickstart (Offline / CPU)

### Prerequisites
- Python 3.11+
- [uv](https://docs.astral.sh/uv/)

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/pspelsberg/Lynx.git
cd Lynx

# Install dependencies into virtual environment
uv sync
```

### 2. Run Test Suite & Demo

```bash
# Run the 148+ offline tests
make test
# Or: uv run python -m unittest discover -s tests -v

# Run the offline smoke-test demo (does not require a model server)
make demo

# Test the bounded, network-isolated Python worker
uv run lynx kernel "_=2+2"
```

### 3. Live Run with Local llama-server

When a local `llama-server` is active (see Hardware Setup below):

```bash
# Basic task execution in think mode (default)
uv run lynx run "Inspect the files in the workspace and summarize their purpose."

# Low-latency run in fast mode with automatic file edits
uv run lynx run "Refactor tests/test_harness.py imports" --mode fast --approval-mode auto-edit

# Run with live event streaming and sidecar dual-output
uv run lynx run "Analyze project symbols" --stream --dual-output build/events.jsonl

# Revert file changes if task fails
uv run lynx run "Experimental changes" --rollback-on-failure
```

### 4. Replay, Fork, and Undo

```bash
# Inspect a past run deterministically
uv run lynx replay <run_id>

# Fork a run from event sequence 4
uv run lynx fork <run_id> --from 4

# Roll back all file modifications from a previous run
uv run lynx undo <run_id>
```

---

## CLI Reference

```text
lynx run TASK [options]                     Execute a task against llama-server
  --mode fast|think|research                Thinking mode and budget profile (default: think)
  --approval-mode plan|ask|auto-edit|auto|yolo  Execution approval level (default: ask)
  --stream                                  Live-stream events, diffs, and self-repair diagnostics
  --dual-output PATH                        Emit parallel JSONL stream for IDEs and sidecars
  --rollback-on-failure                     Automatically revert workspace changes on error

lynx demo                                   Run offline deterministic smoke test
lynx undo [RUN_ID]                          Revert file modifications made during a run
lynx kernel CODE                            Execute bounded Python snippet in isolated worker
lynx replay RUN_ID [--branch BRANCH]        Deterministic offline run replay
lynx fork RUN_ID --from SEQ                 Fork an isolated run from a historical checkpoint
lynx resume RUN_ID --from SEQ               Resume execution from a checkpoint
lynx eval BENCHMARK [--variants ...]        Run ablation benchmarks across system configurations
```

---

## Hardware Acceleration: AMD Radeon RX 9060 XT (16 GB)

Lynx is optimized for local execution on AMD Radeon RX 9060 XT (RDNA 4 / `gfx1200`) under ROCm/HIP and Vulkan.

```
  ┌──────────────────────────────────────────────────────────┐
  │               Local Hardware Stack                       │
  │                                                          │
  │   Lynx Agent Harness (127.0.0.1:8090 optional API)       │
  │           │ HTTP / SSE                                   │
  │           ▼                                              │
  │   llama-server (127.0.0.1:8080)                          │
  │           │ ROCm (HIP) / Vulkan                          │
  │           ▼                                              │
  │   AMD Radeon RX 9060 XT (16 GB VRAM, gfx1200)            │
  │   Model: Qwen3.5-4B-Q6_K.gguf (Pinned SHA-256)           │
  └──────────────────────────────────────────────────────────┘
```

### 1. Verify GPU Driver & Environment

```bash
make check-amd
```

### 2. Build Pinned llama.cpp

The build script enforces a tested, pinned 40-character commit SHA to guarantee reproducibility:

```bash
export LLAMA_CPP_REF=<tested-40-character-commit-sha>

# Build ROCm baseline
make build-llama

# Or build both ROCm and Vulkan (requires Vulkan SDK / headers)
LLAMA_BACKENDS=rocm,vulkan make build-llama
```

### 3. Download Verified Model GGUF

Lynx enforces a strict single-model policy. The downloader verifies the model against `config/model-manifest.json`:

```bash
MODEL_ID=qwen35-4b-q6_k bash scripts/download-model.sh
# Check hash integrity
sha256sum models/Qwen3.5-4B-Q6_K.gguf
# Expected: fdedd781c9ce676ab66b018ca247ff78e8a33c98098a822c1e2d5075e7718f66
```

### 4. Start Local Model Server

```bash
export MODEL_PATH="$PWD/models/Qwen3.5-4B-Q6_K.gguf"
export MODEL_SHA256=fdedd781c9ce676ab66b018ca247ff78e8a33c98098a822c1e2d5075e7718f66

# Launch with ROCm/HIP backend (binds strictly to 127.0.0.1:8080)
./scripts/run-llama-rocm.sh

# Or launch with Vulkan backend
./scripts/run-llama-vulkan.sh
```

### 5. Benchmark Performance

```bash
make benchmark
```

---

## Configuration & Environment Variables

Configuration is loaded from environment variables or a local `.env` file via `Settings.from_env()`. See [`config/llama.env.example`](config/llama.env.example):

| Variable | Default | Description |
|---|---|---|
| `LLAMA_SERVER_URL` | `http://127.0.0.1:8080` | OpenAI-compatible endpoint of `llama-server` |
| `LLAMA_MODEL` | `Qwen3.5-4B-Q6_K.gguf` | Canonical model identifier |
| `LLAMA_TIMEOUT_S` | `60` | Model inference timeout in seconds |
| `MODEL_PATH` | `models/Qwen3.5-4B-Q6_K.gguf` | Local path to verified GGUF model file |
| `MODEL_SHA256` | `fdedd78...` | Expected SHA-256 hash for model verification |
| `LYNX_WORKSPACE` | `.` | Target directory boundary for local file operations |
| `LYNX_TRAJECTORY` | `data/trajectories.jsonl` | Output path for serialized agent trajectories |
| `LYNX_ARTIFACTS` | `artifacts/` | Directory for content-addressed SHA-256 artifacts |
| `LYNX_KNOWLEDGE` | `knowledge/` | Output directory for OKF knowledge candidates |
| `LYNX_FLIGHT_ROOT` | `data/runs/` | Root directory for Flight Recorder event logs |
| `LYNX_ALLOW_EXTERNAL` | `0` | Explicit toggle to permit external web searches |
| `LYNX_ALLOW_REMOTE_LLM`| `0` | Explicit toggle to permit non-loopback inference |
| `LYNX_KERNEL_ALLOW_NETWORK` | `0` | Explicit toggle to allow network in Python worker |

---

## Operational Security & Hardening

1. **Loopback Isolation:** Both `llama-server` and the optional FastAPI layer bind strictly to `127.0.0.1`. Never expose on `0.0.0.0` without an authenticated TLS reverse proxy.
2. **Path Traversal Defenses:** All file operations verify paths remain within `LYNX_WORKSPACE` using resolved canonical paths.
3. **Subprocess Isolation:** Commands are executed in sanitized environments with minimal environment variables (`PATH`, `LANG`, `LC_ALL`) and explicit timeouts.
4. **Secret Redaction:** Logs, events, and artifacts automatically scrub common credential patterns, bearer tokens, and keys before writing to disk.
5. **Durable Services:** Production deployments can use the provided systemd service templates:
   - `config/llama-server.service.example`
   - `config/lynx-api.service.example`

---

## Repository Structure

```text
Lynx/
├── src/lynx_harness/
│   ├── models.py           # TaskContract, AgentState, Decisions, Budgets, Risk types
│   ├── config.py           # Environment and .env configuration parser
│   ├── controller.py       # Deterministic state machine controller
│   ├── loop.py             # Core agent execution loop
│   ├── inference.py        # llama.cpp client and offline Mock/Scripted backends
│   ├── tools.py            # ToolGateway, registry, retrieval, policy & built-ins
│   ├── security.py         # Secret redaction, atomic writes, subprocess sanitization
│   ├── observation.py      # Untrusted observation boundary and trust metadata
│   ├── context.py          # Sliding compaction, RLM context engine, Map/Reduce
│   ├── memory.py           # Artifact store, JSONL logs, knowledge candidate sink
│   ├── flight.py           # Flight Recorder, checkpoints, replay and fork engine
│   ├── branch.py           # Branch isolation and merge proposals
│   ├── progress.py         # Progress monitor against cyclic action patterns
│   ├── verifier.py         # Schema, policy, postcondition, and evidence verifier
│   ├── rah.py              # Recursive Agent Harness & child execution firewalls
│   ├── research.py         # Fact-based research workflow with claim verification
│   ├── skills.py           # Versioned skill catalog and evaluation
│   ├── promotion.py        # Replay-gated skill and knowledge promotion
│   ├── bestofn.py          # Best-of-N branch executor
│   ├── integrations.py     # IntegrationManager for MCP, Kernel, Browser, LSP lifecycle
│   ├── mcp.py              # Model Context Protocol stdio client and adapter
│   ├── kernel.py           # Isolated, network-disabled persistent Python worker
│   ├── playwright.py       # Bounded Playwright browser sensor
│   ├── lsp.py              # Bounded Language Server Protocol sensor
│   ├── a2a.py, acp.py      # Agent2Agent interchange and ACP protocol translation
│   ├── api.py              # Optional loopback FastAPI server
│   ├── eval.py             # Ablation benchmark evaluator
│   ├── profiler.py         # Local benchmark matrix and profiling reports
│   ├── training.py         # Verified trajectory and LoRA dataset gates
│   ├── model_profiles.py   # Single-model capability policies and prompt layouts
│   ├── architecture.py     # Architecture fitness reporting
│   └── cli.py              # CLI entry point (run, demo, undo, replay, fork, etc.)
├── tests/                  # 148+ offline unit, security, and architectural tests
├── docs/                   # Architecture specs, runbook, and ADRs
├── scripts/                # ROCm/Vulkan build, run, benchmark, and download scripts
├── config/                 # Service templates, manifests, and environment examples
└── evals/                  # Ablation evaluation benchmark suites
```

---

## Verification & Testing

Lynx includes a comprehensive test suite covering safety boundaries, symlink hardening, verifiers, AST parsers, and replay mechanics:

```bash
# Run all unit tests offline
uv run python -m unittest discover -s tests -v

# Run optional integration tests with a live local llama-server
LYNX_INTEGRATION=1 uv run python -m unittest discover -s tests -v
```

---

## License

This project is open-source software licensed under the [MIT License](LICENSE).
