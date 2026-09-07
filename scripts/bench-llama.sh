#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BASE="${LLAMA_SERVER_URL:-http://127.0.0.1:8080}"
OUT="${BENCH_OUTPUT:-build/benchmark-$(date +%Y%m%dT%H%M%S).json}"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to the canonical models/Qwen3.5-4B-Q6_K.gguf}"
CANONICAL_MODEL_PATH="$ROOT/models/Qwen3.5-4B-Q6_K.gguf"
# Require the lexical repository path, not merely bytes that resolve to it;
# this rejects symlinked model directories/ancestors and closes a path
# substitution at launcher time.
if [[ "$MODEL_PATH" != "$CANONICAL_MODEL_PATH" ]]; then
  if [[ "$PWD" != "$ROOT" || "$MODEL_PATH" != "models/Qwen3.5-4B-Q6_K.gguf" ]]; then
    echo "MODEL_PATH must be the canonical models/Qwen3.5-4B-Q6_K.gguf" >&2; exit 2
  fi
fi
[[ ! -L "$ROOT/models" && ! -L "$CANONICAL_MODEL_PATH" && ! -L "$MODEL_PATH" ]] || { echo "MODEL_PATH and its ancestors must not be symlinks" >&2; exit 2; }
[[ -f "$MODEL_PATH" ]] || { echo "Model not found: $MODEL_PATH" >&2; exit 2; }
MODEL_SHA="$(sha256sum "$MODEL_PATH" | awk '{print $1}')"
EXPECTED_MODEL_SHA256="fdedd781c9ce676ab66b018ca247ff78e8a33c98098a822c1e2d5075e7718f66"
[[ "$MODEL_SHA" == "$EXPECTED_MODEL_SHA256" ]] || { echo "Model SHA-256 mismatch or unpinned model: $MODEL_SHA" >&2; exit 2; }
# Benchmarking is local-only; this also prevents credentials from entering
# the report's server field.
python3 - "$BASE" <<'PY'
import sys
from urllib.parse import urlparse
parsed = urlparse(sys.argv[1])
if parsed.scheme not in {"http", "https"} or parsed.username or parsed.password or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
    raise SystemExit("benchmark server must be a credential-free loopback URL")
PY
BACKEND="${LLAMA_BACKEND:-unknown}"
COMMIT="${LLAMA_CPP_COMMIT:-unknown}"
mkdir -p "$(dirname "$OUT")"
curl -fsS "$BASE/health" >/dev/null || { echo "server not ready: $BASE" >&2; exit 2; }
python3 - "$BASE" "$OUT" "$MODEL_SHA" "$BACKEND" "$COMMIT" <<'PY'
import json,sys,time,urllib.request
base,out,model_sha,backend,commit=sys.argv[1:]
prompts=["Reply with exactly OK.", "Return a JSON object: {\"type\":\"finish\",\"answer\":\"2+2=4\"}."]
rows=[]
for prompt in prompts:
 payload=json.dumps({"model":"Qwen3.5-4B-Q6_K.gguf","messages":[{"role":"user","content":prompt}],"temperature":0,"top_p":1,"seed":7,"max_tokens":64}).encode()
 req=urllib.request.Request(base+"/v1/chat/completions",data=payload,headers={"Content-Type":"application/json"})
 started=time.perf_counter()
 try:
  with urllib.request.urlopen(req,timeout=60) as r:
   body=json.loads(r.read(2_000_000))
  choices=body.get("choices", []) if isinstance(body, dict) else []
  message=choices[0].get("message", {}) if choices and isinstance(choices[0], dict) else {}
  # Never persist model content: Qwen thinking text and server payloads may
  # contain sensitive task context. Keep only structured smoke-test metadata.
  rows.append({"prompt":prompt,"elapsed_ms":round((time.perf_counter()-started)*1000,1),"response_valid":isinstance(message, dict) and isinstance(message.get("content"), (str,list)),"usage":body.get("usage", {}) if isinstance(body, dict) and isinstance(body.get("usage", {}), dict) else {}})
 except Exception as exc: rows.append({"prompt":prompt,"error":type(exc).__name__})
json.dump({"server":base,"created_at":time.time(),"model":"Qwen3.5-4B-Q6_K.gguf","model_sha256":model_sha,"backend":backend,"llama_cpp_commit":commit,"context":8192,"temperature":0,"top_p":1,"seed":7,"rows":rows},open(out,"w"),indent=2)
print(out)
PY
