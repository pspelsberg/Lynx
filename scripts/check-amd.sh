#!/usr/bin/env bash
set -Eeuo pipefail
command -v rocminfo >/dev/null || { echo "rocminfo fehlt: ROCm installieren" >&2; exit 2; }
echo "== ROCm device =="
rocminfo | grep -E 'Name:|Marketing Name:|gfx[0-9]+' | head -30 || true
if command -v clinfo >/dev/null; then
  echo "== OpenCL/Vulkan-ish device =="; clinfo 2>/dev/null | grep -E 'Device Name|Device Version' | head -10 || true
fi
echo "Expected for RX 9060 XT: gfx1200 (diagnostic only; driver output is authoritative)."
