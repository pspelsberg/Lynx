#!/usr/bin/env python3
"""Run a bounded local llama.cpp profile matrix without downloading models.

Only files explicitly passed via --models are used. Every row records model hash,
llama commit, backend, context/slot/flash settings and structured-output results.
"""
from __future__ import annotations
import argparse, asyncio, hashlib, json, os, signal, subprocess, time, re
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from lynx_harness.inference import LlamaCppBackend
from lynx_harness.models import Mode, TaskContract, RiskLevel, DecisionType
from lynx_harness.model_profiles import CANONICAL_MODEL_FILENAME, CANONICAL_MODEL_ID, require_single_model
from lynx_harness.verifier import Verifier
from lynx_harness.profiler import BenchmarkProfile, BenchmarkResult, write_benchmark_report
from lynx_harness.security import atomic_write_text

async def health(backend: LlamaCppBackend, timeout: float = 90) -> bool:
    end=time.monotonic()+timeout
    while time.monotonic()<end:
        if await backend.health(): return True
        await asyncio.sleep(1)
    return False

def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024), b""): h.update(chunk)
    return h.hexdigest()

def report_model_path(root: Path, model: Path) -> str:
    """Return a portable, non-host-leaking model path for report metadata."""
    try:
        return model.relative_to(root).as_posix()
    except ValueError:
        # ``main`` currently restricts models to root/models, but retain a
        # bounded fallback if this helper is reused by another caller.
        return model.name

def llama_commit(root: Path) -> str:
    candidate = os.getenv("LLAMA_CPP_COMMIT", "")
    manifest=root/"build/llama-build-manifest.txt"
    if not candidate and manifest.is_file():
        for line in manifest.read_text().splitlines():
            if line.startswith("llama.cpp commit:"):
                candidate = line.split(":",1)[1].strip()
                break
    # Keep runtime provenance bounded and commit-shaped; arbitrary environment
    # text must not become a misleading or oversized report field.
    return candidate if re.fullmatch(r"[0-9a-fA-F]{7,128}", candidate) else "unknown"

def vram() -> str:
    try: return subprocess.run(["rocm-smi", "--showmeminfo", "vram", "--json"], capture_output=True, text=True, timeout=5).stdout[:20_000] or "unavailable"
    except Exception: return "unavailable"

def driver() -> str:
    try:
        result=subprocess.run(["rocm-smi", "--showdriverversion", "--json"], capture_output=True, text=True, timeout=5)
        return result.stdout[:20_000] or "unavailable"
    except Exception: return "unavailable"

def timing_value(timings, *names):
    for name in names:
        value=timings.get(name) if isinstance(timings,dict) else None
        if isinstance(value,(int,float)) and not isinstance(value,bool) and value >= 0: return float(value)
    return None

async def one(args, model: Path, model_sha256: str, backend_name: str, context: int, slots: int, flash: str) -> dict:
    binary=Path(args.root)/f"build/llama-{backend_name}/bin/llama-server"
    profile=BenchmarkProfile(f"{model.name}:{backend_name}:ctx{context}:slots{slots}:flash{flash}:spec{args.spec_type}",backend=backend_name,model=model.name,context_size=context,batch_size=slots,options=(("flash_attention",flash),("spec_type",args.spec_type)))
    model_label = report_model_path(Path(args.root).resolve(), model)
    row={"profile":profile.name,"model":model_label,"model_sha256":model_sha256,"llama_cpp_commit":llama_commit(Path(args.root)),"backend":backend_name,"context":context,"slots":slots,"flash_attention":flash,"spec_type":args.spec_type,"vram":vram(),"driver":driver(),"results":[],"task_success":False,"verifier_rate":0.0,"latency_p50_ms":None,"latency_p95_ms":None,"prompt_tokens_per_second":None,"decode_tokens_per_second":None}
    if not binary.is_file(): row["error"]=f"missing server binary: {binary}"; return row
    port=18081
    cmd=[str(binary),"-m",str(model),"-c",str(context),"-ngl","999","--host","127.0.0.1","--port",str(port),"-np",str(slots),"--jinja","--reasoning","on","--flash-attn",flash]
    if args.spec_type != "none":
        cmd += ["--spec-type",args.spec_type]
    proc=subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True, env={**os.environ,"HIP_VISIBLE_DEVICES":os.getenv("HIP_VISIBLE_DEVICES","0"),"ROCR_VISIBLE_DEVICES":os.getenv("ROCR_VISIBLE_DEVICES","0")})
    try:
        backend=LlamaCppBackend(f"http://127.0.0.1:{port}",model.name,args.timeout)
        if not await health(backend, args.startup_timeout): row["error"]="server readiness timeout"; return row
        verifier=Verifier()
        for mode in (Mode.FAST,Mode.THINK,Mode.RESEARCH):
            started=time.perf_counter(); item={"mode":mode.value}
            try:
                decision=await backend.decide({"mode":mode.value,"goal":"Return exactly a finish decision with answer OK.","untrusted_observations":[]},[],512)
                if decision.type != DecisionType.FINISH or decision.answer != "OK":
                    raise ValueError("profile task did not return the required finish answer")
                contract=TaskContract(id=f"profile-{mode.value}",goal="Return exactly a finish decision with answer OK.",mode=mode)
                verification=verifier.check(decision,contract,{"tools_used":(),"max_risk":RiskLevel.READ_ONLY})
                timings=dict(backend.last_timings)
                item.update({"valid":True,"decision_type":decision.type.value,"answer":decision.answer[:200],"usage":dict(backend.last_usage),"timings":timings,"verification_status":verification.status.value,"verified":verification.passed,"verification_checks":[{"id":check.id,"status":check.status.value} for check in verification.checks],"prompt_tokens_per_second":timing_value(timings,"prompt_per_second","prompt_tokens_per_second"),"decode_tokens_per_second":timing_value(timings,"predicted_per_second","decode_tokens_per_second")})
            except Exception as exc: item.update({"valid":False,"verified":False,"error":str(exc)[:500]})
            item["elapsed_ms"]=round((time.perf_counter()-started)*1000,2); row["results"].append(item)
        row["task_success"]=all(item.get("verified",False) for item in row["results"]); row["verifier_rate"]=sum(item.get("verified",False) for item in row["results"])/len(row["results"])
        latencies=sorted(item["elapsed_ms"] for item in row["results"])
        row["latency_p50_ms"]=latencies[len(latencies)//2]; row["latency_p95_ms"]=latencies[min(len(latencies)-1, int(len(latencies)*.95))]
        for metric in ("prompt_tokens_per_second","decode_tokens_per_second"):
            values=sorted(item[metric] for item in row["results"] if isinstance(item.get(metric),(int,float)))
            row[metric]=values[len(values)//2] if values else None
    finally:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=10)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try: os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError: pass
            proc.wait()
    return row

async def main(args):
    if not args.backends or any(backend not in {"rocm", "vulkan"} for backend in args.backends):
        raise SystemExit("backends must be rocm or vulkan")
    if not args.contexts or any(type(value) is not int or not 1 <= value <= 1_000_000 for value in args.contexts):
        raise SystemExit("contexts must be integers in 1..1000000")
    if not args.slots or any(type(value) is not int or not 1 <= value <= 65_536 for value in args.slots):
        raise SystemExit("slots must be integers in 1..65536")
    if type(args.max_profiles) is not int or not 1 <= args.max_profiles <= 128:
        raise SystemExit("max-profiles must be in 1..128")
    if not isinstance(args.timeout, (int, float)) or not 0.01 <= args.timeout <= 3_600:
        raise SystemExit("timeout must be in 0.01..3600")
    if not isinstance(args.startup_timeout, (int, float)) or not 0.01 <= args.startup_timeout <= 3_600:
        raise SystemExit("startup-timeout must be in 0.01..3600")
    project_root=Path(__file__).parents[1].resolve()
    supplied_root=Path(args.root).resolve()
    if supplied_root != project_root:
        raise SystemExit("--root cannot redirect profiling outside this repository")
    raw_models=[Path(item) for item in args.models]
    # Reject symlinked ancestors as well as a symlink final component.  A
    # path such as ``alias/models/model.gguf`` can otherwise resolve to the
    # canonical bytes while allowing the caller to redirect the model root.
    for path in raw_models:
        absolute=path.absolute()
        if path.is_symlink() or any(parent.is_symlink() for parent in absolute.parents):
            raise SystemExit("model paths and ancestors must not be symlinks")
    models=[path.resolve() for path in raw_models]
    canonical_path=project_root/"models"/"Qwen3.5-4B-Q6_K.gguf"
    if canonical_path.is_symlink() or canonical_path.parent.is_symlink():
        raise SystemExit("canonical model path must not be a symlink")
    canonical=canonical_path.resolve()
    if any(path != canonical for path in models): raise SystemExit("only models/Qwen3.5-4B-Q6_K.gguf is permitted")
    if args.spec_type.startswith("draft-"): raise SystemExit("draft-model/MTP variants are disabled by the single-model runtime policy")
    if args.draft_model is not None or args.draft_sha256: raise SystemExit("draft models are disabled by the single-model runtime policy")
    digests: dict[Path, str] = {}
    for path in models:
        if not path.is_file(): raise SystemExit(f"model not found: {path}")
        digest = sha(path)
        if digest != "fdedd781c9ce676ab66b018ca247ff78e8a33c98098a822c1e2d5075e7718f66": raise SystemExit("canonical model SHA-256 mismatch")
        digests[path] = digest
    combos=[(model,digests[model],backend,context,slots,flash) for model in models for backend in args.backends for context in args.contexts for slots in args.slots for flash in args.flash]
    if len(combos)>args.max_profiles: raise SystemExit(f"matrix has {len(combos)} profiles; pass a smaller explicit batch or --max-profiles")
    rows=[]
    for combo in combos: rows.append(await one(args,*combo))
    output={"schema":"lynx.local-profile.v1","model_policy":{"model_id":CANONICAL_MODEL_ID,"model_filename":CANONICAL_MODEL_FILENAME},"matrix":{"models":[f"models/{CANONICAL_MODEL_FILENAME}" for _ in models],"backends":args.backends,"contexts":args.contexts,"slots":args.slots,"flash":args.flash},"llama_cpp_commit":llama_commit(Path(args.root)),"results":rows}
    atomic_write_text(args.output, json.dumps(output, ensure_ascii=False, indent=2) + "\n", max_bytes=50_000_000)
    print(args.output)

if __name__=="__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--root",default=Path(__file__).parents[1]); parser.add_argument("--spec-type",choices=["none","ngram-simple","draft-simple","draft-mtp"],default="none"); parser.add_argument("--draft-model",type=Path); parser.add_argument("--draft-sha256"); parser.add_argument("--models",nargs="+",type=Path,default=[Path("models/Qwen3.5-4B-Q6_K.gguf")]); parser.add_argument("--backends",nargs="+",default=["rocm","vulkan"]); parser.add_argument("--contexts",nargs="+",type=int,default=[4096,8192,16384]); parser.add_argument("--slots",nargs="+",type=int,default=[1,2,3]); parser.add_argument("--flash",nargs="+",choices=["on","off","auto"],default=["on","off"]); parser.add_argument("--max-profiles",type=int,default=64); parser.add_argument("--timeout",type=float,default=120); parser.add_argument("--startup-timeout",type=float,default=180); parser.add_argument("--output",type=Path,default=Path("build/profile-local.json")); asyncio.run(main(parser.parse_args()))
