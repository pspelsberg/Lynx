#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# Reject speculative/draft checkpoints before any model/server setup. This
# keeps the single-model policy deterministic even on an unbuilt host.
if [[ -n "${DRAFT_MODEL_PATH:-}" || -n "${DRAFT_MODEL_SHA256:-}" ]]; then
  echo "Draft models are disabled: only models/Qwen3.5-4B-Q6_K.gguf may be loaded" >&2
  exit 2
fi
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to the canonical models/Qwen3.5-4B-Q6_K.gguf}"
CANONICAL_MODEL_PATH="$ROOT/models/Qwen3.5-4B-Q6_K.gguf"
# A filename/hash check is insufficient: MODEL_PATH must resolve to the
# repository-controlled canonical file, and a symlink is rejected to avoid a
# launcher-time path substitution.
# Require the lexical repository path, not merely bytes that resolve to it;
# this rejects symlinked model directories/ancestors and closes a path
# substitution at launcher time.
if [[ "$MODEL_PATH" != "$CANONICAL_MODEL_PATH" ]]; then
  if [[ "$PWD" != "$ROOT" || "$MODEL_PATH" != "models/Qwen3.5-4B-Q6_K.gguf" ]]; then
    echo "MODEL_PATH must be the canonical models/Qwen3.5-4B-Q6_K.gguf" >&2; exit 2
  fi
fi
[[ ! -L "$ROOT/models" && ! -L "$CANONICAL_MODEL_PATH" && ! -L "$MODEL_PATH" ]] || { echo "MODEL_PATH and its ancestors must not be symlinks" >&2; exit 2; }
PORT="${PORT:-8080}"
CONTEXT="${LLAMA_CONTEXT:-8192}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-60}"
# Speculation is opt-in and must be evaluated against the verified baseline.
# ngram-simple uses only the running model and therefore respects the
# single-model policy. Draft/MTP modes require an additional verified model or
# sidecar and remain fail-closed until that policy is explicitly extended.
SPEC_TYPE="${LLAMA_SPEC_TYPE:-none}"
case "$SPEC_TYPE" in
  none|ngram-simple) ;;
  draft-*|mtp|draft-mtp)
    echo "Draft/MTP speculation is disabled by the single-model policy; use LLAMA_SPEC_TYPE=ngram-simple or an evaluated profile" >&2
    exit 2
    ;;
  *) echo "LLAMA_SPEC_TYPE must be none or ngram-simple" >&2; exit 2 ;;
esac
SERVER="$ROOT/build/llama-vulkan/bin/llama-server"
[[ -x "$SERVER" ]] || { echo "Missing $SERVER; run scripts/build-llama.sh" >&2; exit 2; }
[[ -f "$MODEL_PATH" ]] || { echo "Model not found: $MODEL_PATH" >&2; exit 2; }
[[ "$(basename "$MODEL_PATH")" == "Qwen3.5-4B-Q6_K.gguf" ]] || { echo "Only models/Qwen3.5-4B-Q6_K.gguf is allowed" >&2; exit 2; }
EXPECTED_MODEL_SHA256="fdedd781c9ce676ab66b018ca247ff78e8a33c98098a822c1e2d5075e7718f66"
SHA="$(sha256sum "$MODEL_PATH" | awk '{print $1}')"
[[ "$SHA" == "$EXPECTED_MODEL_SHA256" ]] || { echo "Model SHA-256 mismatch or unpinned model: $SHA" >&2; exit 2; }
[[ -z "${MODEL_SHA256:-}" || "$SHA" == "$MODEL_SHA256" ]] || { echo "Model SHA-256 mismatch: $SHA" >&2; exit 2; }
export LLAMA_BACKEND="${LLAMA_BACKEND:-vulkan}"
export LLAMA_CPP_COMMIT="${LLAMA_CPP_COMMIT:-$(awk -F': ' '/llama.cpp commit:/{print $2}' "$ROOT/build/llama-build-manifest.txt" 2>/dev/null || true)}"
echo "model=$MODEL_PATH sha256=$SHA backend=vulkan bind=127.0.0.1:$PORT"
HELP="$("$SERVER" --help 2>&1)"
ARGS=(-m "$MODEL_PATH" --host 127.0.0.1 --port "$PORT" -c "$CONTEXT" -ngl 999)
if [[ "$SPEC_TYPE" != "none" ]]; then
  grep -q -- '--spec-type' <<<"$HELP" || { echo "llama-server does not support speculative decoding" >&2; exit 2; }
  ARGS+=(--spec-type "$SPEC_TYPE")
fi
if grep -q -- '--reasoning' <<<"$HELP"; then ARGS+=(--reasoning "${LLAMA_REASONING:-on}"); fi
grep -q -- '--jinja' <<<"$HELP" && ARGS+=(--jinja)
grep -q -- '-np' <<<"$HELP" && ARGS+=(-np 1)
cleanup() {
  [[ -n "${SERVER_PID:-}" ]] || return 0
  kill "$SERVER_PID" 2>/dev/null || true
  # Wait for graceful termination so the launcher cannot leave an orphaned
  # llama-server behind when interrupted or when readiness fails.
  if ! wait "$SERVER_PID" 2>/dev/null; then
    kill -KILL "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM
 "$SERVER" "${ARGS[@]}" & SERVER_PID=$!
for ((i=0;i<STARTUP_TIMEOUT;i++)); do
  if curl -fsS --max-time 2 "http://127.0.0.1:$PORT/health" >/dev/null; then echo "llama-server ready"; wait "$SERVER_PID"; exit $?; fi
  kill -0 "$SERVER_PID" 2>/dev/null || { echo "llama-server exited" >&2; wait "$SERVER_PID"; exit 1; }
  sleep 1
done
echo "llama-server readiness timeout" >&2; exit 1
