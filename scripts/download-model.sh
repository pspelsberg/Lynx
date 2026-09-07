#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_DIR="${MODEL_DIR:-$ROOT/models}"
MODEL_ID="${MODEL_ID:-qwen35-4b-q6_k}"
MODEL_FILE="${MODEL_FILE:-Qwen3.5-4B-Q6_K.gguf}"
REVISION="${MODEL_REVISION:-e87f176479d0855a907a41277aca2f8ee7a09523}"
EXPECTED_SHA256="${MODEL_SHA256:-fdedd781c9ce676ab66b018ca247ff78e8a33c98098a822c1e2d5075e7718f66}"
if [[ "${MODEL_USE_MANIFEST:-1}" == "1" && -f "$ROOT/config/model-manifest.json" ]]; then
  read -r MANIFEST_REPO MANIFEST_REV MANIFEST_FILE MANIFEST_SHA MANIFEST_DOWNLOADABLE < <(python3 - "$ROOT/config/model-manifest.json" "$MODEL_ID" <<'PY'
import json, sys

def reject_constant(value):
    raise ValueError(f"non-finite manifest value: {value}")

def reject_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate manifest key: {key}")
        result[key] = value
    return result

with open(sys.argv[1], encoding="utf-8") as stream:
    raw=json.load(stream, parse_constant=reject_constant,
                  object_pairs_hook=reject_duplicates)
entries=raw.get("models", [raw])
for entry in entries:
    if entry.get("id", entry.get("model")) == sys.argv[2]:
        print(entry.get("repository", ""), entry.get("revision", ""), entry.get("filename", entry.get("model", "")), entry.get("sha256") or "", str(entry.get("downloadable", False)).lower())
        break
else: raise SystemExit("unknown model id")
PY
  )
  [[ "$MANIFEST_DOWNLOADABLE" == "true" && -n "$MANIFEST_REPO" && -n "$MANIFEST_REV" && -n "$MANIFEST_FILE" && -n "$MANIFEST_SHA" ]] || { echo "Manifest entry is not downloadable" >&2; exit 2; }
  REPOSITORY="$MANIFEST_REPO"; REVISION="$MANIFEST_REV"; MODEL_FILE="$MANIFEST_FILE"; EXPECTED_SHA256="$MANIFEST_SHA"
else
  REPOSITORY="unsloth/Qwen3.5-4B-GGUF"
fi
# This project deliberately has one permitted checkpoint.  Reject all
# environment/manifest attempts to turn this helper into a generic downloader.
[[ "$MODEL_ID" == "qwen35-4b-q6_k" && "$MODEL_FILE" == "Qwen3.5-4B-Q6_K.gguf" ]] || { echo "Only Qwen3.5-4B-Q6_K.gguf is permitted" >&2; exit 2; }
[[ "$REPOSITORY" == "unsloth/Qwen3.5-4B-GGUF" ]] || { echo "Unexpected model repository" >&2; exit 2; }
[[ "$REVISION" == "e87f176479d0855a907a41277aca2f8ee7a09523" ]] || { echo "Unexpected model revision" >&2; exit 2; }
[[ "$EXPECTED_SHA256" == "fdedd781c9ce676ab66b018ca247ff78e8a33c98098a822c1e2d5075e7718f66" ]] || { echo "Unexpected model digest" >&2; exit 2; }
case "$REVISION" in main|master|HEAD|latest|UNPINNED|reference-only) echo "Refusing unpinned model revision: $REVISION" >&2; exit 2;; esac
[[ "$REVISION" =~ ^[A-Za-z0-9._-]{7,128}$ ]] || { echo "Invalid model revision" >&2; exit 2; }
[[ "$EXPECTED_SHA256" =~ ^[0-9a-fA-F]{64}$ ]] || { echo "MODEL_SHA256 must be a 64-character hex digest" >&2; exit 2; }
[[ "$MODEL_FILE" != /* && "$MODEL_FILE" != *..* ]] || { echo "Invalid model filename" >&2; exit 2; }
URL="https://huggingface.co/$REPOSITORY/resolve/$REVISION/$MODEL_FILE?download=true"
mkdir -p "$MODEL_DIR"
DEST="$MODEL_DIR/$MODEL_FILE"
echo "Downloading pinned model revision $REVISION"
curl -L --fail --retry 3 --progress-bar -o "$DEST.tmp" "$URL"
ACTUAL="$(sha256sum "$DEST.tmp" | awk '{print $1}')"
if [[ "$ACTUAL" != "$EXPECTED_SHA256" ]]; then rm -f "$DEST.tmp"; echo "SHA-256 mismatch: $ACTUAL" >&2; exit 1; fi
mv "$DEST.tmp" "$DEST"
echo "Installed $DEST ($ACTUAL)"
