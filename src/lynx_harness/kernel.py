from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import resource
import sys
import signal
from typing import Any

from .tools import ToolError

_WORKER = r"""import contextlib, io, json, os, sys, traceback
workspace=os.path.realpath(os.environ.get("LYNX_KERNEL_WORKSPACE", os.getcwd()))
output_limit=max(1000, min(1_000_000, int(os.environ.get("LYNX_KERNEL_MAX_OUTPUT", "100000"))))
# Capture the host decision once.  Generated code can mutate ``os.environ``;
# consulting it from the audit hook would let a task self-enable networking.
network_enabled = os.environ.get("LYNX_KERNEL_NETWORK") == "1"
def deny(event, args):
 if event.startswith("socket.") and not network_enabled: raise PermissionError("kernel network disabled")
 if event in {"subprocess.Popen", "os.system", "pty.spawn", "os.fork", "os.vfork", "os.posix_spawn"} or event.startswith("os.exec"):
  raise PermissionError("kernel process spawning disabled")
 if event == "import" and args and str(args[0]).split(".")[0] in {"ctypes", "subprocess", "multiprocessing"}: raise PermissionError("kernel module is disabled")
 # Filesystem metadata APIs are capabilities too. Checking only ``open``
 # allowed generated code to enumerate/stat paths outside the workspace.
 if event == "open" or event.startswith("os."):
  for candidate in args[:2]:
   if isinstance(candidate, (str, bytes)):
    path=os.path.realpath(os.fsdecode(candidate) if isinstance(candidate,bytes) else candidate)
    if not (path == workspace or path.startswith(workspace + os.sep)): raise PermissionError("kernel path escapes workspace")
sys.addaudithook(deny)
wire=sys.stdout
ns={"__name__":"__main__"}
for line in sys.stdin:
 try:
  request=json.loads(line)
  code=request.get("code","")
  if not isinstance(code,str) or len(code)>100000: raise ValueError("code too long")
  class CappedOutput(io.StringIO):
   def write(self, value):
    remaining=output_limit-len(self.getvalue())
    if remaining <= 0: return len(value)
    return super().write(value[:remaining])
  capture=CappedOutput()
  with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
   exec(compile(code,"<kernel>","exec"),ns,ns)
  result={"ok":True,"output":capture.getvalue()[:100000],"value":repr(ns.get("_",""))[:100000]}
 except Exception as exc:
  result={"ok":False,"error":f"{type(exc).__name__}: {exc}"[:2000],"traceback":traceback.format_exc(limit=3)[:4000]}
 wire.write(json.dumps(result,ensure_ascii=False)+"\n"); wire.flush()
"""

class PythonKernel:
    """Persistent stateful Python worker; this is a process boundary, not a security sandbox."""
    def __init__(self, workspace: Path, timeout_s: float = 30, max_output: int = 100_000, allow_network: bool = False, max_memory_mb: int = 1024, max_fds: int = 256, max_processes: int = 1):
        self.workspace=workspace.resolve()
        if not self.workspace.is_dir(): raise ValueError("kernel workspace must be a directory")
        if not 1 <= timeout_s <= 300 or not 1_000 <= max_output <= 1_000_000 or not 16 <= max_memory_mb <= 65_536 or not 16 <= max_fds <= 65_536 or not 1 <= max_processes <= 128: raise ValueError("kernel limits out of bounds")
        self.timeout_s,self.max_output=timeout_s,max_output; self.allow_network=allow_network; self.max_memory_mb=max_memory_mb; self.max_fds=max_fds; self.max_processes=max_processes; self.process: asyncio.subprocess.Process | None=None; self._lock=asyncio.Lock()
    async def start(self) -> None:
        if self.process and self.process.returncode is None: return
        def limits():
            cpu=max(1,int(self.timeout_s)+1); resource.setrlimit(resource.RLIMIT_CPU,(cpu,cpu+1)); resource.setrlimit(resource.RLIMIT_AS,(self.max_memory_mb*1024*1024,self.max_memory_mb*1024*1024)); resource.setrlimit(resource.RLIMIT_NOFILE,(self.max_fds,self.max_fds))
            if hasattr(resource, "RLIMIT_NPROC"): resource.setrlimit(resource.RLIMIT_NPROC,(self.max_processes,self.max_processes))
        env={"PATH":os.environ.get("PATH",""), "LYNX_KERNEL_WORKSPACE":str(self.workspace)}
        if self.allow_network: env["LYNX_KERNEL_NETWORK"]="1"
        env["LYNX_KERNEL_MAX_OUTPUT"] = str(self.max_output)
        self.process=await asyncio.create_subprocess_exec(sys.executable,"-I","-u","-c",_WORKER,cwd=self.workspace,env=env,stdin=asyncio.subprocess.PIPE,stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.PIPE,limit=1_100_000,preexec_fn=limits,start_new_session=True)
    async def execute(self, code: str) -> dict[str,Any]:
        if not isinstance(code,str) or not code.strip() or len(code)>100_000 or "\x00" in code: raise ToolError("invalid kernel code")
        async with self._lock:
            await self.start(); assert self.process and self.process.stdin and self.process.stdout
            self.process.stdin.write((json.dumps({"code":code})+"\n").encode()); await self.process.stdin.drain()
            try:
                line=await asyncio.wait_for(self.process.stdout.readline(),timeout=self.timeout_s)
            except asyncio.TimeoutError:
                await self.close(); raise ToolError("kernel execution timed out")
            except asyncio.CancelledError:
                # The outer runner may cancel a gateway call while generated
                # code is still executing.  Terminate the worker as part of
                # cancellation so a persistent kernel cannot outlive its task.
                await self.close(); raise
            if not line: raise ToolError("kernel worker closed")
            # The worker limit is in characters while the pipe is bytes; allow
            # JSON/envelope overhead and UTF-8 expansion without unbounding the
            # response (the subprocess stream itself has a fixed limit).
            if len(line) > self.max_output * 4 + 8_192: raise ToolError("kernel response too large")
            response=json.loads(line); response["output"]=str(response.get("output",""))[:self.max_output]; return response
    async def close(self) -> None:
        if self.process:
            process = self.process
            try: os.killpg(process.pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError): process.terminate()
            try: await asyncio.wait_for(process.wait(),timeout=3)
            except asyncio.TimeoutError:
                try: os.killpg(process.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError): process.kill()
                await process.wait()
            self.process=None


def register_kernel_tool(registry, kernel: PythonKernel) -> None:
    """Opt-in model seam; callers decide whether this high-risk tool is exposed."""
    from .models import RiskLevel, ToolSpec
    async def execute(args): return await kernel.execute(args["code"])
    registry.register(ToolSpec("python.exec","Execute code in the explicitly configured kernel worker", {"type":"object","required":["code"],"properties":{"code":{"type":"string","maxLength":100000}}}, risk=RiskLevel.DANGEROUS, family="python", origin="kernel", trust="untrusted"), execute)
