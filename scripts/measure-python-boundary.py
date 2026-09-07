#!/usr/bin/env python3
"""Measure Python scheduler/gateway overhead before considering a Rust boundary."""
from __future__ import annotations
import asyncio, json, statistics, time
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parents[1]/"src"))
from lynx_harness.agent_bus import AgentBus
from lynx_harness.models import AgentTask, Mode
from lynx_harness.tools import ToolGateway, ToolPolicy, ToolRegistry, register_builtin_tools
from lynx_harness.security import atomic_write_text

async def main():
    async def handler(task): return task.id
    bus=AgentBus(handler,max_queue=256); await bus.start(); scheduler=[]
    for i in range(100):
        started=time.perf_counter(); task=AgentTask(f"measure-{i}","measure",Mode.FAST)
        await bus.submit(task)
        while bus.status(task.id).status not in {"completed","failed","cancelled"}: await asyncio.sleep(0)
        scheduler.append((time.perf_counter()-started)*1000)
    await bus.close()
    workspace=Path("build/boundary-workspace"); workspace.mkdir(parents=True,exist_ok=True)
    registry=ToolRegistry(); register_builtin_tools(registry,workspace); gateway=ToolGateway(registry,ToolPolicy(workspace=workspace)); gateway_ms=[]
    for _ in range(100):
        started=time.perf_counter(); await gateway.execute("calculator",{"expression":"6*7"}); gateway_ms.append((time.perf_counter()-started)*1000)
    def stats(values): return {"samples":len(values),"p50_ms":statistics.median(values),"p95_ms":sorted(values)[int(len(values)*.95)-1],"max_ms":max(values)}
    result={"schema":"lynx.python-boundary.v1","scheduler":stats(scheduler),"tool_gateway":stats(gateway_ms),"decision":"retain-python-until-a-measured-production-bottleneck-and-explicit-target-exist","rust_implemented":False}
    atomic_write_text(Path("build/python-boundary-measurement.json"), json.dumps(result, indent=2) + "\n"); print(json.dumps(result, indent=2))
asyncio.run(main())
