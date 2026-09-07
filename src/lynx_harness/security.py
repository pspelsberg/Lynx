from __future__ import annotations
import os
import re
import tempfile
from pathlib import Path
from typing import Any

_SECRET_KEY = re.compile(r"(?:secret|password|passwd|token|authorization|cookie|credential|apikey|privatekey|accesskey)", re.I)

def _is_secret_key(key: Any) -> bool:
    # Headers and JSON payloads commonly use camelCase or hyphens
    # (accessToken, x-api-key); normalize separators before matching.
    normalized = re.sub(r"[^a-z0-9]", "", str(key).lower())
    sensitive = ("secret", "password", "passwd", "token", "authorization", "cookie", "credential", "apikey", "privatekey", "accesskey")
    return normalized in sensitive or any(normalized.endswith(suffix) for suffix in sensitive)
# Redact bearer/basic credentials before generic ``authorization: value``
# matching.  Otherwise ``authorization: Bearer secret`` would redact only
# the word ``Bearer`` and leak the actual credential.
_BEARER_TEXT = re.compile(r"(?i)(\b(?:bearer|basic)\s+)[A-Za-z0-9._~+/=-]+")
_SECRET_TEXT = re.compile(r"(?i)((?:password|passwd|token|api[_-]?key|secret|authorization|cookie)\s*[:=]\s*)[^\s,;&]+|((?<![A-Za-z0-9_])(?!llm_tokens\b|used_llm_tokens\b)(?:[A-Z][A-Z0-9_]*_)?(?:SECRET|PASSWORD|TOKEN|API_KEY|PRIVATE_KEY)(?:[A-Z0-9_]*))\s*=\s*[^\s,;&]+")
_URL_AUTH = re.compile(r"(https?://)([^/@\s]+):([^/@\s]+)@", re.I)

def redact_text(value: str, *, max_chars: int | None = 10_000) -> str:
    value = _URL_AUTH.sub(r"\1[redacted]@", value)
    value = _BEARER_TEXT.sub(r"\1[redacted]", value)
    def replacement(match: re.Match[str]) -> str:
        if match.group(1): return match.group(1) + "[redacted]"
        return match.group(2) + "=[redacted]"
    value = _SECRET_TEXT.sub(replacement, value)
    return value if max_chars is None else value[:max_chars]

def redact(value: Any, depth: int = 0) -> Any:
    if depth > 8: return "[redacted-depth]"
    if isinstance(value, dict):
        result={}
        for key, item in value.items():
            result[key] = "[redacted]" if _is_secret_key(key) else redact(item, depth+1)
        return result
    if isinstance(value, (list, tuple)): return [redact(item, depth+1) for item in list(value)[:100]]
    if isinstance(value, str): return redact_text(value)
    return value


def redact_bytes(value: bytes, *, max_chars: int | None = 10_000) -> bytes:
    if not isinstance(value, bytes): raise TypeError("value must be bytes")
    try:
        text = value.decode("utf-8")
        return redact_text(text, max_chars=max_chars).encode("utf-8")
    except UnicodeDecodeError:
        # Binary artifacts can still contain ASCII headers/tokens.  Preserve
        # all non-ASCII bytes while applying the same redaction patterns to a
        # one-byte view instead of silently bypassing trajectory hygiene.
        text = value.decode("latin-1")
        return redact_text(text, max_chars=max_chars).encode("latin-1")


def sanitized_subprocess_env() -> dict[str, str]:
    """Return the minimal environment allowed for configured child tools."""
    return {
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
    }


def atomic_write_text(path: Path, text: str, *, max_bytes: int = 50_000_000) -> Path:
    """Write a report without following a pre-existing symlink."""
    if not isinstance(path, Path) or not isinstance(text, str):
        raise ValueError("path and text are required")
    encoded = text.encode("utf-8")
    if len(encoded) > max_bytes:
        raise ValueError("output exceeds configured limit")
    absolute = path.absolute()
    if absolute.is_symlink() or any(parent.is_symlink() for parent in absolute.parents):
        raise ValueError("output path and ancestors must not be symlinks")
    absolute.parent.mkdir(parents=True, exist_ok=True)
    if absolute.parent.is_symlink() or not absolute.parent.is_dir():
        raise ValueError("output parent must be a regular directory")
    fd, temporary_name = tempfile.mkstemp(prefix=".lynx-output-", suffix=".tmp", dir=str(absolute.parent))
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(absolute)
    except BaseException:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return path
