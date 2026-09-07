#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REF="${LLAMA_CPP_REF:?Set LLAMA_CPP_REF to a tested llama.cpp commit (full 40-character SHA required)}"
# A branch or short SHA is mutable/ambiguous and cannot reproduce the
# recorded backend evidence. Fetch only an exact commit object.
[[ "$REF" =~ ^[0-9a-fA-F]{40}$ ]] || { echo "LLAMA_CPP_REF must be an exact 40-character commit SHA" >&2; exit 2; }
SRC="$ROOT/third_party/llama.cpp"
JOBS="${JOBS:-$(nproc)}"
BACKENDS="${LLAMA_BACKENDS:-rocm}" # Vulkan requires Vulkan development headers; opt in with rocm,vulkan
mkdir -p "$ROOT/third_party"
if [[ ! -d "$SRC/.git" ]]; then git clone https://github.com/ggerganov/llama.cpp "$SRC"; fi
git -C "$SRC" fetch --tags --force origin "$REF"
git -C "$SRC" checkout --detach "$REF"
COMMIT="$(git -C "$SRC" rev-parse HEAD)"
for backend in ${BACKENDS//,/ }; do
  [[ "$backend" == "rocm" || "$backend" == "vulkan" ]] || { echo "Unknown backend: $backend" >&2; exit 2; }
  build="$ROOT/build/llama-$backend"; mkdir -p "$build"
  if [[ "$backend" == rocm ]]; then
    cmake -S "$SRC" -B "$build" -DCMAKE_BUILD_TYPE=Release -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1200
  else
    VULKAN_ARGS=()
    if [[ -n "${VULKAN_INCLUDE_DIR:-}" ]]; then VULKAN_ARGS+=("-DVulkan_INCLUDE_DIR=${VULKAN_INCLUDE_DIR}"); fi
    if [[ -n "${VULKAN_LIBRARY:-}" ]]; then VULKAN_ARGS+=("-DVulkan_LIBRARY=${VULKAN_LIBRARY}"); fi
    if [[ -n "${SPIRV_HEADERS_DIR:-}" ]]; then VULKAN_ARGS+=("-DCMAKE_PREFIX_PATH=${SPIRV_HEADERS_DIR}"); fi
    if [[ -n "${SPIRV_HEADERS_INCLUDE_DIR:-}" ]]; then VULKAN_ARGS+=("-DCMAKE_CXX_FLAGS=-I${SPIRV_HEADERS_INCLUDE_DIR}"); fi
    cmake -S "$SRC" -B "$build" -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON "${VULKAN_ARGS[@]}"
  fi
  cmake --build "$build" --config Release --target llama-server -j"$JOBS"
  test -x "$build/bin/llama-server" || { echo "llama-server missing in $build/bin" >&2; exit 1; }
  "$build/bin/llama-server" --help > "$ROOT/build/llama-$backend-help.txt"
done
cat > "$ROOT/build/llama-build-manifest.txt" <<EOF
llama.cpp commit: $COMMIT
requested ref: $REF
built at: $(date -Is)
compiler: $(c++ --version | head -1)
rocm: $(command -v rocminfo || echo unavailable)
vulkan: $(command -v vulkaninfo || echo unavailable)
backends: $BACKENDS
vulkan_headers: ${VULKAN_INCLUDE_DIR:-unconfigured}
vulkan_headers_commit: $(git -C "${VULKAN_HEADERS_SOURCE:-${VULKAN_INCLUDE_DIR:-/nonexistent}/..}" rev-parse HEAD 2>/dev/null || echo unknown)
spirv_headers: ${SPIRV_HEADERS_INCLUDE_DIR:-unconfigured}
spirv_headers_commit: $(git -C "${SPIRV_HEADERS_SOURCE:-/nonexistent}" rev-parse HEAD 2>/dev/null || echo unknown)
EOF
echo "Built $BACKENDS. Manifest: build/llama-build-manifest.txt"
