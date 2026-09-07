from __future__ import annotations

import asyncio
import ast
import operator
import ipaddress
import json
import os
import math
import fnmatch
import sys
from pathlib import Path
import re
import shlex
import socket
import subprocess
import time
import shutil
import tempfile
import difflib
import inspect
from dataclasses import dataclass
from typing import Any, Awaitable, Callable
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .models import Observation, RiskLevel, ToolSpec

ToolHandler = Callable[[dict[str, Any]], Awaitable[Any]]


class ToolError(RuntimeError):
    pass


def _safe_url(url: str, allow_hosts: set[str]) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username is not None or parsed.password is not None:
        raise ToolError("only credential-free http(s) URLs are allowed")
    host = parsed.hostname.lower().rstrip(".")
    if allow_hosts and host not in allow_hosts:
        raise ToolError(f"host is not in web allowlist: {host}")
    try:
        addresses = {item[4][0] for item in socket.getaddrinfo(host, parsed.port or (443 if parsed.scheme == "https" else 80), type=socket.SOCK_STREAM)}
        for address in addresses:
            ip = ipaddress.ip_address(address)
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
                raise ToolError("private or special network targets are blocked")
    except socket.gaierror as exc:
        raise ToolError(f"DNS lookup failed: {host}") from exc
    return url


class _NoRedirectHandler(HTTPRedirectHandler):
    """Fail closed on redirects so the validated destination cannot change."""
    def _blocked(self, req, fp, code, msg, headers):
        raise ToolError("redirects are disabled for outbound tool requests")
    http_error_301 = _blocked
    http_error_302 = _blocked
    http_error_303 = _blocked
    http_error_307 = _blocked
    http_error_308 = _blocked


def _open_public(request: Request, allow_hosts: set[str], max_bytes: int):
    _safe_url(request.full_url, allow_hosts)
    response = build_opener(_NoRedirectHandler()).open(request, timeout=10)
    try:
        # DNS is resolved once for admission and again by the socket layer.
        # Check the connected peer as well, closing the rebinding gap.
        raw = getattr(getattr(response, "fp", None), "raw", None)
        sock = getattr(raw, "_sock", None)
        if sock is None:
            raise ToolError("unable to validate outbound peer")
        peer = sock.getpeername()[0]
        address = ipaddress.ip_address(peer)
        if (address.is_private or address.is_loopback or address.is_link_local
                or address.is_reserved or address.is_multicast):
            raise ToolError("private or special network peer is blocked")
        return response, response.read(max_bytes).decode("utf-8", "replace")
    except BaseException:
        response.close()
        raise


class ToolRegistry:
    def __init__(self) -> None:
        self._entries: dict[str, tuple[ToolSpec, ToolHandler]] = {}
        self._frozen = False

    def register(self, spec: ToolSpec, handler: ToolHandler) -> None:
        if self._frozen: raise ToolError("tool registry is frozen")
        if spec.name in self._entries:
            raise ValueError(f"duplicate tool: {spec.name}")
        self._entries[spec.name] = (spec, handler)

    def freeze(self) -> None: self._frozen = True
    @property
    def frozen(self) -> bool: return self._frozen
    def specs(self) -> list[ToolSpec]:
        return [entry[0] for entry in self._entries.values()]

    def get(self, name: str) -> tuple[ToolSpec, ToolHandler]:
        try:
            return self._entries[name]
        except KeyError as exc:
            raise ToolError(f"unknown tool: {name}") from exc

    def retrieve_ranked(self, task: str, limit: int = 6, *, min_score: int = 0) -> list[tuple[ToolSpec, int]]:
        if not isinstance(task,str) or len(task)>20_000 or not 1<=limit<=100: raise ValueError("invalid tool retrieval request")
        terms = set(re.findall(r"[a-z0-9_.-]+", task.lower())); ranked=[]
        for spec in self.specs():
            haystack=f"{spec.name} {spec.description} {spec.family}".lower(); score=sum(1 for term in terms if term in haystack)
            if score >= min_score: ranked.append((spec,score))
        return sorted(ranked,key=lambda row:(-row[1],row[0].name))[:limit]

    def retrieve(self, task: str, limit: int = 6) -> list[ToolSpec]:
        return [spec for spec, _ in self.retrieve_ranked(task, limit)]


def _validate_schema(value: Any, schema: dict[str, Any], path: str = "arguments", depth: int = 0) -> None:
    if depth > 20: raise ToolError(f"schema nesting too deep at {path}")
    if not isinstance(schema, dict): raise ToolError(f"invalid schema at {path}")
    if "enum" in schema and value not in schema["enum"]: raise ToolError(f"invalid enum at {path}")
    expected = schema.get("type")
    valid = (expected is None or (expected == "object" and isinstance(value, dict)) or
             (expected == "array" and isinstance(value, list)) or
             (expected == "string" and isinstance(value, str)) or
             (expected == "number" and isinstance(value, (int, float)) and not isinstance(value, bool)) or
             (expected == "integer" and isinstance(value, int) and not isinstance(value, bool)) or
             (expected == "boolean" and isinstance(value, bool)))
    if not valid: raise ToolError(f"invalid type at {path}")
    if isinstance(value, str):
        if len(value) > int(schema.get("maxLength", 100_000)): raise ToolError(f"string too long at {path}")
        if schema.get("format") == "uri" and urlparse(value).scheme not in {"http", "https"}: raise ToolError(f"invalid URI at {path}")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, float) and not math.isfinite(value): raise ToolError(f"non-finite number at {path}")
        if "minimum" in schema and value < schema["minimum"]: raise ToolError(f"below minimum at {path}")
        if "maximum" in schema and value > schema["maximum"]: raise ToolError(f"above maximum at {path}")
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        missing = set(schema.get("required", [])) - value.keys()
        unknown = value.keys() - properties.keys()
        if missing or unknown: raise ToolError(f"invalid object at {path} (missing={sorted(missing)}, unknown={sorted(unknown)})")
        for key, child in properties.items():
            if key in value: _validate_schema(value[key], child, f"{path}.{key}", depth + 1)
    if isinstance(value, list):
        if len(value) > int(schema.get("maxItems", 100)): raise ToolError(f"too many items at {path}")
        if isinstance(schema.get("items"), dict):
            for index, item in enumerate(value): _validate_schema(item, schema["items"], f"{path}[{index}]", depth + 1)


@dataclass
class ToolPolicy:
    max_risk: RiskLevel = RiskLevel.LOCAL_MUTATION
    confirmed: bool = False
    workspace: Path = Path(".")
    max_output_bytes: int = 1_000_000
    max_concurrency: int = 8
    approval_mode: str = "auto-edit"
    confirm_callback: Callable[[str, dict[str, Any], ToolSpec], Any] | None = None

    def __post_init__(self):
        self.max_risk = self.max_risk if isinstance(self.max_risk, RiskLevel) else (RiskLevel[self.max_risk.upper()] if isinstance(self.max_risk, str) else RiskLevel(self.max_risk))
        if not 1 <= self.max_output_bytes <= 10_000_000 or not 1 <= self.max_concurrency <= 128: raise ValueError("tool policy limits out of bounds")
        if self.approval_mode not in {"plan", "ask", "auto-edit", "auto", "yolo"}:
            raise ValueError(f"invalid approval_mode: {self.approval_mode}")

    def allows(self, spec: ToolSpec) -> bool:
        if spec.risk > self.max_risk:
            return False
        if self.approval_mode == "plan":
            return spec.risk <= RiskLevel.READ_ONLY
        if spec.risk >= RiskLevel.EXTERNAL_MUTATION and not self.confirmed and self.approval_mode != "yolo":
            if self.confirm_callback is None:
                return False
        return True

    def requires_confirmation(self, spec: ToolSpec) -> bool:
        if self.approval_mode == "yolo" or self.approval_mode == "plan":
            return False
        if self.approval_mode == "ask":
            return spec.risk >= RiskLevel.LOCAL_MUTATION
        if self.approval_mode == "auto-edit":
            return spec.risk >= RiskLevel.EXTERNAL_MUTATION
        if self.approval_mode == "auto":
            return spec.risk >= RiskLevel.EXTERNAL_MUTATION or spec.family not in {"filesystem", "general", "lsp"}
        return False


class ToolGateway:
    def __init__(self, registry: ToolRegistry, policy: ToolPolicy | None = None) -> None:
        self.registry = registry
        self.policy = policy or ToolPolicy()
        self._semaphore = asyncio.Semaphore(self.policy.max_concurrency)

    async def execute(self, name: str, arguments: dict[str, Any]) -> Observation:
        started = time.monotonic()
        try:
            spec, handler = self.registry.get(name)
            if not isinstance(arguments, dict):
                raise ToolError("tool arguments must be an object")
            _validate_schema(arguments, spec.input_schema)
            if not self.policy.allows(spec):
                raise ToolError(f"policy denied {name} (risk={spec.risk.name})")
            if self.policy.requires_confirmation(spec):
                confirmed = False
                if self.policy.confirmed:
                    confirmed = True
                elif self.policy.confirm_callback is not None:
                    res = self.policy.confirm_callback(name, arguments, spec)
                    if inspect.isawaitable(res):
                        res = await res
                    confirmed = bool(res)
                if not confirmed:
                    raise ToolError(f"policy denied {name}: confirmation declined or required ({self.policy.approval_mode} mode)")
            async with self._semaphore:
                output = await handler(arguments)
            try: output_size = len(json.dumps(output, ensure_ascii=False, allow_nan=False).encode("utf-8"))
            except (TypeError, ValueError) as exc: raise ToolError("tool output is not valid JSON") from exc
            if output_size > self.policy.max_output_bytes:
                raise ToolError("tool output exceeds configured limit")
            # Tool output is data-plane by default; a registry declaration
            # alone cannot attest that an observation is safe instructions.
            return Observation(name, True, output=output, elapsed_ms=int((time.monotonic() - started) * 1000), origin=spec.origin, trust="untrusted", provenance=spec.provenance or spec.name)
        except Exception as exc:
            # Unknown tools have no spec; all other errors retain origin/trust metadata.
            try:
                spec = self.registry.get(name)[0]
                origin, trust = spec.origin, spec.trust
            except ToolError:
                origin, trust = "gateway", "trusted"
            return Observation(name, False, error=str(exc), elapsed_ms=int((time.monotonic() - started) * 1000), origin=origin, trust=trust, provenance=(spec.provenance or spec.name) if "spec" in locals() else "gateway")


def register_builtin_tools(registry: ToolRegistry, workspace: Path, web_allow_hosts: set[str] | None = None, snapshot_manager: Any | None = None) -> None:
    workspace = workspace.resolve()
    allow_hosts = {host.lower() for host in (web_allow_hosts or set())}

    async def calculator(args: dict[str, Any]) -> Any:
        expression = args.get("expression")
        if not isinstance(expression, str) or len(expression) > 200:
            raise ToolError("expression must be a bounded string")
        operators = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg}
        def evaluate(node: ast.AST) -> float:
            if isinstance(node, ast.Expression): return evaluate(node.body)
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool): return node.value
            if isinstance(node, ast.UnaryOp) and type(node.op) in operators: return operators[type(node.op)](evaluate(node.operand))
            if isinstance(node, ast.BinOp) and type(node.op) in operators:
                left, right = evaluate(node.left), evaluate(node.right)
                if abs(left) > 1e12 or abs(right) > 1e12: raise ToolError("number too large")
                return operators[type(node.op)](left, right)
            raise ToolError("only basic arithmetic is allowed")
        try:
            tree = ast.parse(expression, mode="eval")
            nodes = list(ast.walk(tree))
            if len(nodes) > 100: raise ToolError("expression is too complex")
            for node in nodes:
                if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and abs(node.value) > 1e12: raise ToolError("number too large")
                if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
                    if not isinstance(node.right, ast.Constant) or not isinstance(node.right.value, int) or node.right.value < 0 or node.right.value > 1024: raise ToolError("exponent is too large")
            value = evaluate(tree)
            if isinstance(value, int) and value.bit_length() > 4096: raise ToolError("result is too large")
        except (SyntaxError, ValueError, ZeroDivisionError, OverflowError) as exc:
            raise ToolError("invalid arithmetic expression") from exc
        return value

    async def filesystem_list(args: dict[str, Any]) -> Any:
        relative = str(args.get("path", "."))
        target = (workspace / relative).resolve()
        if workspace not in target.parents and target != workspace:
            raise ToolError("path escapes workspace")
        if not target.exists():
            raise ToolError("path does not exist")
        return [{"name": p.name, "directory": p.is_dir()} for p in sorted(target.iterdir())[:200]]

    async def filesystem_read(args: dict[str, Any]) -> Any:
        relative = str(args.get("path", ""))
        target = (workspace / relative).resolve()
        if workspace not in target.parents or not target.is_file():
            raise ToolError("file is outside workspace or not a regular file")
        if target.stat().st_size > 1_000_000:
            raise ToolError("file exceeds 1 MB observation limit")
        return target.read_text(encoding="utf-8", errors="replace")

    async def filesystem_write(args: dict[str, Any]) -> Any:
        relative = str(args.get("path", "")).strip()
        content = args.get("content")
        if not relative:
            raise ToolError("path is required")
        if not isinstance(content, str):
            raise ToolError("content must be a string")
        if len(content.encode("utf-8")) > 1_000_000:
            raise ToolError("content exceeds 1 MB limit")
        target = (workspace / relative).resolve()
        if workspace not in target.parents:
            raise ToolError("path escapes workspace")
        if target.is_symlink() or any(p.is_symlink() for p in target.parents if workspace in p.parents or p == workspace):
            raise ToolError("symlinks are not permitted in workspace path")
        if snapshot_manager is not None:
            snapshot_manager.record_before_mutation(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        existed = target.exists()
        fd, tmp_name = tempfile.mkstemp(prefix=".lynx_write_", suffix=".tmp", dir=str(target.parent))
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(content)
                f.flush()
                os.fsync(f.fileno())
            tmp_path.replace(target)
        except BaseException:
            try: tmp_path.unlink(missing_ok=True)
            except OSError: pass
            raise
        diagnostics = []
        if target.suffix == ".py":
            try:
                ast.parse(content, filename=relative)
            except SyntaxError as exc:
                diagnostics.append({"severity": "error", "line": exc.lineno, "offset": exc.offset, "message": str(exc.msg)})
        elif target.suffix == ".json":
            try:
                json.loads(content)
            except json.JSONDecodeError as exc:
                diagnostics.append({"severity": "error", "line": exc.lineno, "message": exc.msg})
        return {
            "path": relative,
            "bytes": len(content.encode("utf-8")),
            "created": not existed,
            "diagnostics": diagnostics,
        }

    async def filesystem_patch(args: dict[str, Any]) -> Any:
        relative = str(args.get("path", "")).strip()
        target_content = args.get("target_content")
        replacement_content = args.get("replacement_content")
        allow_multiple = bool(args.get("allow_multiple", False))
        if not relative:
            raise ToolError("path is required")
        if not isinstance(target_content, str) or not target_content:
            raise ToolError("target_content is required and must not be empty")
        if not isinstance(replacement_content, str):
            raise ToolError("replacement_content must be a string")
        if len(target_content) > 100_000 or len(replacement_content) > 100_000:
            raise ToolError("patch chunks must not exceed 100,000 characters")
        target = (workspace / relative).resolve()
        if workspace not in target.parents:
            raise ToolError("path escapes workspace")
        if target.is_symlink() or not target.is_file():
            raise ToolError("file does not exist or is not a regular file")
        if target.stat().st_size > 1_000_000:
            raise ToolError("file exceeds 1 MB limit")
        if snapshot_manager is not None:
            snapshot_manager.record_before_mutation(target)
        original = target.read_text(encoding="utf-8", errors="replace")
        count = original.count(target_content)
        if count == 0:
            raise ToolError(f"target_content not found in {relative}")
        if count > 1 and not allow_multiple:
            raise ToolError(f"target_content found {count} times in {relative}; provide more unique context or set allow_multiple=true")
        new_content = original.replace(target_content, replacement_content) if allow_multiple else original.replace(target_content, replacement_content, 1)
        if len(new_content.encode("utf-8")) > 1_000_000:
            raise ToolError("resulting file exceeds 1 MB limit")
        diff_lines = list(difflib.unified_diff(
            original.splitlines(keepends=True),
            new_content.splitlines(keepends=True),
            fromfile=f"a/{relative}",
            tofile=f"b/{relative}",
            n=3
        ))
        diff_text = "".join(diff_lines)
        fd, tmp_name = tempfile.mkstemp(prefix=".lynx_patch_", suffix=".tmp", dir=str(target.parent))
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(new_content)
                f.flush()
                os.fsync(f.fileno())
            tmp_path.replace(target)
        except BaseException:
            try: tmp_path.unlink(missing_ok=True)
            except OSError: pass
            raise
        diagnostics = []
        if target.suffix == ".py":
            try:
                ast.parse(new_content, filename=relative)
            except SyntaxError as exc:
                diagnostics.append({"severity": "error", "line": exc.lineno, "offset": exc.offset, "message": str(exc.msg)})
        elif target.suffix == ".json":
            try:
                json.loads(new_content)
            except json.JSONDecodeError as exc:
                diagnostics.append({"severity": "error", "line": exc.lineno, "message": exc.msg})
        return {
            "path": relative,
            "occurrences": count,
            "diff": diff_text,
            "diagnostics": diagnostics,
        }

    async def shell_exec(args: dict[str, Any]) -> Any:
        command = args.get("command")
        if not isinstance(command, str) or not command.strip() or len(command) > 4_096:
            raise ToolError("command is required and must be at most 4096 characters")
        parts = shlex.split(command)
        allowed = {"pwd", "ls", "find", "rg", "git"}
        if not parts or parts[0] not in allowed:
            raise ToolError(f"command is not allowlisted: {parts[0] if parts else ''}")
        if parts[0] == "git" and (len(parts) < 2 or parts[1] not in {"status", "log", "diff", "show"}):
            raise ToolError("only read-only git subcommands are allowed")
        # Do not pass arbitrary git flags: several read-looking commands have
        # path/configuration options that write files or read outside the
        # workspace. Plain subcommands are sufficient for the safe baseline.
        if parts[0] == "git" and any(token.startswith("-") for token in parts[2:]):
            raise ToolError("git options are not allowed by the read-only policy")
        forbidden_args = {"-exec", "-execdir", "-delete", "-ok", "-okdir", "--pre", "--hostname-command", "--textconv", "--ext-diff", "--no-ext-diff", "--output", "--git-dir", "--work-tree", "--config", "--exec-path", "-C", "-L", "--follow", "--dereference", "-R", "--recursive", "-fprint", "-fprint0", "-fprintf", "-fls"}
        # GNU find accepts output actions with an attached filename (e.g.
        # -fprintf=/tmp/file), so reject those prefixes as well as the bare
        # spellings.  This command is intended to be observational only.
        forbidden_prefixes = ("--output=", "--git-dir=", "--work-tree=", "--config=", "--exec-path=", "-fprint=", "-fprint0=", "-fprintf=", "-fls=")
        if any(token in forbidden_args or any(token.startswith(prefix) for prefix in forbidden_prefixes) for token in parts[1:]):
            raise ToolError("unsafe command option is not allowed")
        for index, token in enumerate(parts[1:], 1):
            if token.startswith("/"):
                candidate = Path(token).resolve()
                if workspace not in candidate.parents and candidate != workspace:
                    raise ToolError("command path escapes workspace")
            elif not token.startswith("-") and not (parts[0] == "git" and index == 1):
                # Existing relative paths (including symlinks) are resolved
                # against the workspace before being passed to a subprocess.
                raw_candidate = workspace / token
                if raw_candidate.exists():
                    candidate = raw_candidate.resolve()
                    if workspace not in candidate.parents and candidate != workspace:
                        raise ToolError("command path escapes workspace")
            if ".." in Path(token).parts:
                raise ToolError("parent traversal is not allowed")
        if any(token in command for token in (";", "&&", "||", "|", ">", "<", "`", "$((", "$(")):
            raise ToolError("shell operators are not allowed")
        env = {**os.environ, "GIT_CEILING_DIRECTORIES": str(workspace.parent), "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull, "GIT_ATTR_NOSYSTEM": "1", "GIT_EXTERNAL_DIFF": "", "GIT_PAGER": "cat", "GIT_CONFIG_COUNT": "1", "GIT_CONFIG_KEY_0": "diff.external", "GIT_CONFIG_VALUE_0": "", "GIT_OPTIONAL_LOCKS": "0"}
        exec_parts = parts
        if parts[0] == "git":
            exec_parts = ["git", "-c", "core.fsmonitor=false", "-c", "diff.external=", "-c", "diff.trustExitCode=false"] + parts[1:]
            if parts[1] in {"diff", "show"}: exec_parts += ["--no-ext-diff", "--no-textconv"]
        proc = await asyncio.create_subprocess_exec(*exec_parts, cwd=workspace, env=env, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, start_new_session=True)
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=10)
        except asyncio.TimeoutError:
            proc.kill(); await proc.wait(); raise ToolError("command timed out")
        except asyncio.CancelledError:
            proc.kill(); await proc.wait(); raise
        output = (stdout + stderr).decode("utf-8", "replace")[:50_000]
        return {"returncode": proc.returncode, "output": output}

    async def web_search(args: dict[str, Any]) -> Any:
        query = str(args.get("query", "")).strip()
        if not query or len(query) > 500:
            raise ToolError("query must be 1..500 characters")
        # A configured search endpoint keeps credentials and provider choice outside the model.
        endpoint = os.environ.get("LYNX_SEARCH_ENDPOINT", "https://html.duckduckgo.com/html/")
        parsed = urlparse(endpoint)
        host = parsed.hostname or ""
        if not allow_hosts:
            configured_hosts = {host.lower()} if host else set()
        else:
            configured_hosts = allow_hosts
        _safe_url(endpoint, configured_hosts)
        from urllib.parse import urlencode
        url = endpoint + ("&" if "?" in endpoint else "?") + urlencode({"q": query})
        def fetch() -> str:
            req = Request(url, headers={"User-Agent": "lynx-harness/0.1"})
            response, body = _open_public(req, configured_hosts, 500_000)
            response.close()
            return body
        html = await asyncio.to_thread(fetch)
        # Keep this dependency-free; callers receive bounded snippets, not arbitrary HTML.
        from html.parser import HTMLParser
        class Parser(HTMLParser):
            def __init__(self): super().__init__(); self.items=[]; self.current=None
            def handle_starttag(self, tag, attrs):
                if tag == "a" and any(k == "class" and v and "result__a" in v for k,v in attrs):
                    self.current = {"title":"", "url": next((v for k,v in attrs if k == "href"), "")}
            def handle_data(self, data):
                if self.current is not None and data.strip(): self.current["title"] += data.strip()
            def handle_endtag(self, tag):
                if tag == "a" and self.current is not None:
                    self.items.append(self.current); self.current=None
        parser=Parser(); parser.feed(html)
        return {"query": query, "results": parser.items[:10]}

    async def web_fetch(args: dict[str, Any]) -> Any:
        url = str(args.get("url", ""))
        host = (urlparse(url).hostname or "").lower()
        _safe_url(url, allow_hosts or {host})
        def fetch() -> str:
            req = Request(url, headers={"User-Agent": "lynx-harness/0.1"})
            response, body = _open_public(req, allow_hosts, 1_000_000)
            response.close()
            return body
        return (await asyncio.to_thread(fetch))[:1_000_000]

    async def filesystem_grep(args: dict[str, Any]) -> Any:
        query = str(args.get("query", "")).strip()
        rel_path = str(args.get("path", ".")).strip()
        pattern = args.get("pattern")
        case_sensitive = bool(args.get("case_sensitive", True))
        max_results = min(200, max(1, int(args.get("max_results", 50))))
        if not query:
            raise ToolError("query is required and must not be empty")
        target = (workspace / rel_path).resolve()
        if workspace not in target.parents and target != workspace:
            raise ToolError("path escapes workspace")
        if not target.exists():
            raise ToolError("path does not exist")
        flags = 0 if case_sensitive else re.IGNORECASE
        try:
            regex = re.compile(query, flags)
        except re.error:
            regex = re.compile(re.escape(query), flags)
        skip_dirs = {".git", ".venv", "venv", "__pycache__", "build", "dist", ".pytest_cache", ".ruff_cache", "node_modules", "third_party"}
        matches = []
        def search_file(fpath: Path) -> None:
            nonlocal matches
            if len(matches) >= max_results: return
            try: rel = str(fpath.relative_to(workspace))
            except ValueError: rel = str(fpath)
            try:
                with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                    for lineno, line in enumerate(f, 1):
                        if regex.search(line):
                            matches.append({"path": rel, "line": lineno, "content": line.rstrip()[:200]})
                            if len(matches) >= max_results: break
            except OSError: pass

        if target.is_file():
            search_file(target)
        else:
            for root, dirs, files in os.walk(target):
                dirs[:] = [d for d in dirs if d not in skip_dirs and not d.startswith(".")]
                for fname in sorted(files):
                    if pattern and not fnmatch.fnmatch(fname, pattern):
                        continue
                    fpath = Path(root) / fname
                    search_file(fpath)
                    if len(matches) >= max_results: break
                if len(matches) >= max_results: break
        return {"query": query, "matches": matches, "count": len(matches), "truncated": len(matches) >= max_results}

    async def filesystem_find(args: dict[str, Any]) -> Any:
        pattern = str(args.get("pattern", "")).strip()
        rel_path = str(args.get("path", ".")).strip()
        max_depth = min(10, max(1, int(args.get("max_depth", 5))))
        max_results = min(200, max(1, int(args.get("max_results", 50))))
        if not pattern:
            raise ToolError("pattern is required")
        target = (workspace / rel_path).resolve()
        if workspace not in target.parents and target != workspace:
            raise ToolError("path escapes workspace")
        if not target.exists():
            raise ToolError("path does not exist")
        skip_dirs = {".git", ".venv", "venv", "__pycache__", "build", "dist", ".pytest_cache", ".ruff_cache", "node_modules", "third_party"}
        results = []
        for root, dirs, files in os.walk(target):
            dirs[:] = [d for d in dirs if d not in skip_dirs and not d.startswith(".")]
            depth = len(Path(root).relative_to(target).parts)
            if depth > max_depth:
                dirs.clear()
                continue
            for dname in sorted(dirs):
                if fnmatch.fnmatch(dname, pattern):
                    dpath = Path(root) / dname
                    try: rel = str(dpath.relative_to(workspace))
                    except ValueError: rel = str(dpath)
                    results.append({"path": rel, "directory": True})
                    if len(results) >= max_results: break
            if len(results) >= max_results: break
            for fname in sorted(files):
                if fnmatch.fnmatch(fname, pattern):
                    fpath = Path(root) / fname
                    try: rel = str(fpath.relative_to(workspace))
                    except ValueError: rel = str(fpath)
                    results.append({"path": rel, "directory": False})
                    if len(results) >= max_results: break
            if len(results) >= max_results: break
        return {"pattern": pattern, "results": results, "count": len(results)}

    async def workspace_symbols(args: dict[str, Any]) -> Any:
        rel_path = str(args.get("path", ".")).strip()
        max_depth = min(10, max(1, int(args.get("max_depth", 4))))
        target_dir = (workspace / rel_path).resolve()
        if workspace not in target_dir.parents and target_dir != workspace:
            raise ToolError("path escapes workspace")
        if not target_dir.exists():
            raise ToolError("path does not exist")
        symbols_by_file = []
        skip_dirs = {".git", ".venv", "venv", "__pycache__", "build", "dist", ".pytest_cache", ".ruff_cache", "node_modules", "third_party"}
        total_symbols = 0
        total_files = 0
        for root, dirs, files in os.walk(target_dir):
            dirs[:] = [d for d in dirs if d not in skip_dirs and not d.startswith(".")]
            depth = len(Path(root).relative_to(target_dir).parts)
            if depth > max_depth:
                dirs.clear()
                continue
            for fname in sorted(files):
                if not fname.endswith(".py"): continue
                if total_files >= 100: break
                fpath = Path(root) / fname
                try: rel = str(fpath.relative_to(workspace))
                except ValueError: rel = str(fpath)
                try:
                    code = fpath.read_text(encoding="utf-8", errors="replace")[:200_000]
                    tree = ast.parse(code, filename=rel)
                except Exception: continue
                file_symbols = []
                for node in tree.body:
                    if isinstance(node, ast.ClassDef):
                        file_symbols.append({"type": "class", "name": node.name, "line": node.lineno})
                        for item in node.body:
                            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                                args_list = [a.arg for a in item.args.args]
                                file_symbols.append({"type": "method", "name": f"{node.name}.{item.name}", "line": item.lineno, "args": args_list[:6]})
                    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        args_list = [a.arg for a in node.args.args]
                        file_symbols.append({"type": "function", "name": node.name, "line": node.lineno, "args": args_list[:6]})
                if file_symbols:
                    symbols_by_file.append({"path": rel, "symbols": file_symbols[:30]})
                    total_symbols += len(file_symbols)
                    total_files += 1
        return {"files": symbols_by_file, "total_files": total_files, "total_symbols": total_symbols}

    async def workspace_test(args: dict[str, Any]) -> Any:
        runner = str(args.get("runner", "auto")).strip().lower()
        target = str(args.get("target", "")).strip()
        extra_args = args.get("extra_args") or []
        if not isinstance(extra_args, list):
            raise ToolError("extra_args must be a list of strings")
        if target:
            if target.startswith("/") or ".." in Path(target).parts:
                raise ToolError("target path escapes workspace or uses parent traversal")
            target_path = (workspace / target).resolve()
            if workspace not in target_path.parents and target_path != workspace:
                raise ToolError("target path escapes workspace")
        cmd = [sys.executable, "-m"]
        if runner == "pytest":
            cmd.append("pytest")
            if target: cmd.append(target)
        elif runner == "unittest":
            cmd.append("unittest")
            if target: cmd.append(target)
            else:
                tests_dir = "tests" if (workspace / "tests").is_dir() else ("test" if (workspace / "test").is_dir() else ".")
                cmd.extend(["discover", "-s", tests_dir, "-v"])
        elif runner == "auto":
            is_pytest = (workspace / "pytest.ini").exists() or ((workspace / "pyproject.toml").exists() and "pytest" in (workspace / "pyproject.toml").read_text(encoding="utf-8", errors="ignore"))
            if is_pytest:
                cmd.append("pytest")
                if target: cmd.append(target)
            else:
                cmd.append("unittest")
                if target: cmd.append(target)
                else:
                    tests_dir = "tests" if (workspace / "tests").is_dir() else ("test" if (workspace / "test").is_dir() else ".")
                    cmd.extend(["discover", "-s", tests_dir, "-v"])
        else:
            raise ToolError(f"unsupported test runner: {runner}")

        safe_arg_pattern = re.compile(r"^[a-zA-Z0-9_./:=-]+$")
        for arg in extra_args:
            if not isinstance(arg, str) or not safe_arg_pattern.match(arg) or any(c in arg for c in (";", "&", "|", ">", "<", "`", "$")):
                raise ToolError(f"invalid or unsafe extra argument: {arg}")
            cmd.append(arg)

        env = {**os.environ, "PYTHONPATH": str(workspace), "PYTHONDONTWRITEBYTECODE": "1"}
        proc = await asyncio.create_subprocess_exec(*cmd, cwd=workspace, env=env, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, start_new_session=True)
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
        except asyncio.TimeoutError:
            proc.kill(); await proc.wait(); raise ToolError("test execution timed out (30s limit)")
        except asyncio.CancelledError:
            proc.kill(); await proc.wait(); raise
        combined_output = (stdout + stderr).decode("utf-8", "replace")[:50_000]
        return {
            "runner": runner,
            "command": " ".join(cmd),
            "returncode": proc.returncode,
            "passed": proc.returncode == 0,
            "output": combined_output,
        }

    registry.register(ToolSpec("calculator", "Evaluate basic arithmetic without code execution", {"type":"object","required":["expression"],"properties":{"expression":{"type":"string"}}}, family="general"), calculator)
    registry.register(ToolSpec("filesystem.list", "List files in the workspace", {"type":"object","properties":{"path":{"type":"string","maxLength":4096}}}, family="filesystem"), filesystem_list)
    registry.register(ToolSpec("filesystem.read", "Read a bounded text file in the workspace", {"type":"object","required":["path"],"properties":{"path":{"type":"string","maxLength":4096}}}, family="filesystem"), filesystem_read)
    registry.register(ToolSpec("filesystem.write", "Write or overwrite a bounded text file in the workspace", {"type":"object","required":["path","content"],"properties":{"path":{"type":"string","maxLength":4096},"content":{"type":"string","maxLength":1_000_000}}}, risk=RiskLevel.LOCAL_MUTATION, family="filesystem"), filesystem_write)
    registry.register(ToolSpec("filesystem.patch", "Surgically patch a text file by replacing target_content with replacement_content", {"type":"object","required":["path","target_content","replacement_content"],"properties":{"path":{"type":"string","maxLength":4096},"target_content":{"type":"string","maxLength":100_000},"replacement_content":{"type":"string","maxLength":100_000},"allow_multiple":{"type":"boolean"}}}, risk=RiskLevel.LOCAL_MUTATION, family="filesystem"), filesystem_patch)
    registry.register(ToolSpec("filesystem.grep", "Search for text or regex pattern across workspace files without shell quoting issues", {"type":"object","required":["query"],"properties":{"query":{"type":"string","maxLength":1000},"path":{"type":"string","maxLength":4096},"pattern":{"type":"string","maxLength":200},"case_sensitive":{"type":"boolean"},"max_results":{"type":"integer","minimum":1,"maximum":200}}}, risk=RiskLevel.READ_ONLY, family="filesystem"), filesystem_grep)
    registry.register(ToolSpec("filesystem.find", "Search for files and directories matching a glob pattern within the workspace", {"type":"object","required":["pattern"],"properties":{"pattern":{"type":"string","maxLength":200},"path":{"type":"string","maxLength":4096},"max_depth":{"type":"integer","minimum":1,"maximum":10},"max_results":{"type":"integer","minimum":1,"maximum":200}}}, risk=RiskLevel.READ_ONLY, family="filesystem"), filesystem_find)
    registry.register(ToolSpec("workspace.symbols", "Extract a compact AST outline of classes, functions, and methods across workspace Python files", {"type":"object","properties":{"path":{"type":"string","maxLength":4096},"max_depth":{"type":"integer","minimum":1,"maximum":10}}}, risk=RiskLevel.READ_ONLY, family="workspace"), workspace_symbols)
    registry.register(ToolSpec("workspace.test", "Run project test suites (unittest, pytest) within timeout and sandbox limits to verify code changes", {"type":"object","properties":{"runner":{"type":"string","enum":["auto","unittest","pytest"]},"target":{"type":"string","maxLength":500},"extra_args":{"type":"array","items":{"type":"string"}}}}, risk=RiskLevel.LOCAL_MUTATION, family="workspace"), workspace_test)
    registry.register(ToolSpec("shell.exec", "Run one read-only allowlisted command in the workspace", {"type":"object","required":["command"],"properties":{"command":{"type":"string","maxLength":4096}}}, risk=RiskLevel.READ_ONLY, family="shell"), shell_exec)
    registry.register(ToolSpec("web.search", "Search the web for current information (external request)", {"type":"object","required":["query"],"properties":{"query":{"type":"string"}}}, risk=RiskLevel.EXTERNAL_MUTATION, family="research", origin="web", trust="untrusted", provenance="configured-web-endpoint"), web_search)
    registry.register(ToolSpec("web.fetch", "Fetch a public web page with a bounded response (external request)", {"type":"object","required":["url"],"properties":{"url":{"type":"string","format":"uri"}}}, risk=RiskLevel.EXTERNAL_MUTATION, family="research", origin="web", trust="untrusted", provenance="configured-web-endpoint"), web_fetch)
