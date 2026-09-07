from __future__ import annotations

import asyncio
import os
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from lynx_harness.inference import LlamaCppBackend


@unittest.skipUnless(os.getenv("LYNX_INTEGRATION") == "1", "set LYNX_INTEGRATION=1 for a live llama-server test")
class LiveLlamaTests(unittest.TestCase):
    def test_health(self):
        backend = LlamaCppBackend(os.getenv("LLAMA_SERVER_URL", "http://127.0.0.1:8080"), os.getenv("LLAMA_MODEL", "Qwen3.5-4B-Q6_K.gguf"))
        self.assertTrue(asyncio.run(backend.health()))
