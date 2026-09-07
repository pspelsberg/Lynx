from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path



@dataclass(frozen=True)
class Settings:
    server_url: str = "http://127.0.0.1:8080"
    model: str = "Qwen3.5-4B-Q6_K.gguf"
    timeout_s: float = 60.0
    workspace: Path = Path(".")
    trajectory_path: Path = Path("data/trajectories.jsonl")
    artifacts_path: Path = Path("artifacts")
    knowledge_path: Path = Path("knowledge")
    flight_root: Path = Path("data/runs")
    model_path: Path | None = None
    backend: str = "unknown"
    llama_commit: str = "unknown"

    @classmethod
    def from_env(cls) -> "Settings":
        # Load only simple KEY=VALUE pairs; never execute a shell config file.
        dotenv = Path(".env")
        if dotenv.is_file():
            for line in dotenv.read_text(encoding="utf-8", errors="replace").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key, value = key.strip(), value.strip().strip('"').strip("'")
                if key and key.replace("_", "").isalnum():
                    os.environ.setdefault(key, value)
        raw_model_path = os.getenv("MODEL_PATH")
        return cls(server_url=os.getenv("LLAMA_SERVER_URL", cls.server_url), model=os.getenv("LLAMA_MODEL", cls.model), timeout_s=float(os.getenv("LLAMA_TIMEOUT_S", cls.timeout_s)), workspace=Path(os.getenv("LYNX_WORKSPACE", ".")).resolve(), trajectory_path=Path(os.getenv("LYNX_TRAJECTORY", cls.trajectory_path)), artifacts_path=Path(os.getenv("LYNX_ARTIFACTS", cls.artifacts_path)), knowledge_path=Path(os.getenv("LYNX_KNOWLEDGE", cls.knowledge_path)), flight_root=Path(os.getenv("LYNX_FLIGHT_ROOT", cls.flight_root)), model_path=Path(raw_model_path).absolute() if raw_model_path else None, backend=os.getenv("LLAMA_BACKEND", "unknown"), llama_commit=os.getenv("LLAMA_CPP_COMMIT", "unknown"))
