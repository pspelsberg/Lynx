from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .models import Observation


class ObservationNormalizer:
    """Convert tool output into data-plane records; it never promotes instructions to control."""
    def __init__(self, max_chars: int = 10_000): self.max_chars = max_chars

    def normalize(self, observation: Observation) -> dict[str, Any]:
        data = asdict(observation)
        output = data.get("output")
        if isinstance(output, str): data["output"] = output[:self.max_chars]
        elif output is not None: data["output"] = str(output)[:self.max_chars]
        data["boundary"] = "untrusted_data"
        data["instructions_allowed"] = False
        data["can_authorize"] = False
        return data
