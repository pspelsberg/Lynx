from __future__ import annotations
import ast
import re
from pathlib import Path
from urllib.parse import urlparse
from typing import Any

_EXTERNAL_MODULES={"subprocess","urllib","requests","httpx","aiohttp"}

def _imports(path: Path) -> list[str]:
    try: tree=ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError,SyntaxError): return [f"unparseable:{path}"]
    result=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Import): result.extend(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node,ast.ImportFrom) and node.module: result.append(node.module.split(".")[0])
    return result

def fitness_report(root: Path) -> dict[str, Any]:
    root=Path(root).resolve(); src=(root/"src"/"lynx_harness") if (root/"src").is_dir() else root
    violations=[]; evidence={}
    loop=src/"loop.py"
    loop_imports=_imports(loop) if loop.is_file() else ["missing:loop.py"]
    if any(item in _EXTERNAL_MODULES for item in loop_imports): violations.append("agent loop imports external runtime module")
    evidence["loop_imports"]=[item for item in loop_imports if item in _EXTERNAL_MODULES]
    pyproject=root/"pyproject.toml"; dependencies=[]
    if pyproject.is_file():
        text=pyproject.read_text(encoding="utf-8"); dependencies=re.findall(r"^dependencies\s*=\s*\[([^]]*)\]",text,re.M)
    core_dependency_free=not dependencies or dependencies[0].strip()==""
    if not core_dependency_free: violations.append("core project declares mandatory dependencies")
    config=src/"config.py"; config_text=config.read_text(encoding="utf-8") if config.is_file() else ""
    try:
        from .config import Settings
        default_url=Settings().server_url; parsed=urlparse(default_url)
        remote_binding_default=parsed.hostname in {"127.0.0.1","localhost","::1"}
    except Exception: remote_binding_default=False
    if not remote_binding_default: violations.append("remote binding is not provably loopback by default")
    try:
        from .models import ComputeBudget, Mode, RiskLevel, SecurityPolicy, TaskContract
        from .rah import ChildTask, ChildBudget
        parent=TaskContract("fitness-parent","fitness",budget=ComputeBudget.for_mode(Mode.THINK),security=SecurityPolicy(max_risk=RiskLevel.READ_ONLY))
        child=ChildTask("fitness-child",parent_task_id=parent.id,depth=1,budget=ChildBudget(depth=1),allowed_tools=(),max_risk=RiskLevel.READ_ONLY)
        child.context_firewall(); child_policy_non_escalation=child.max_risk <= parent.security.max_risk and child.depth <= parent.depth+1
    except Exception: child_policy_non_escalation=False
    if not child_policy_non_escalation: violations.append("child policy can escalate")
    return {"core_dependency_free":core_dependency_free,"loop_gateway_boundary":not evidence["loop_imports"],"violations":violations,"remote_binding_default":remote_binding_default,"child_policy_non_escalation":child_policy_non_escalation,"evidence":evidence}

def evaluate_rust_boundary(*, python_p95_ms: float, target_p95_ms: float, rust_available: bool = False) -> dict[str, Any]:
    if min(python_p95_ms,target_p95_ms)<0: raise ValueError("latency must be non-negative")
    bottleneck=python_p95_ms>target_p95_ms
    return {"rust_required": bool(bottleneck and rust_available), "bottleneck": bottleneck, "reason":"measured scheduler/gateway latency exceeds target" if bottleneck else "no measured bottleneck; retain Python core"}
