from __future__ import annotations
from typing import Any
from .models import AgentTask

def acp_to_agent_task(message: dict[str, Any]) -> AgentTask:
    if not isinstance(message,dict): raise ValueError("ACP message must be object")
    allowed={"id","task_id","goal","artifacts","input_artifacts","mode","allowed_tools","budget","expected_output","postconditions","security","parent_task_id","branch_id","depth","max_depth","max_children"}
    if set(message)-allowed: raise ValueError("unknown ACP fields")
    # Reject ambiguous aliases instead of silently selecting one.  ACP is an
    # interoperability boundary; conflicting duplicate fields must not let a
    # caller validate one task while executing another.
    if "id" in message and "task_id" in message and message["id"] != message["task_id"]:
        raise ValueError("conflicting ACP task identifiers")
    if "input_artifacts" in message and "artifacts" in message and message["input_artifacts"] != message["artifacts"]:
        raise ValueError("conflicting ACP artifact references")
    data=dict(message); data["id"]=data.get("id",data.get("task_id")); data["input_artifacts"]=data.get("input_artifacts",data.get("artifacts",()))
    data.pop("task_id",None); data.pop("artifacts",None)
    return AgentTask.from_dict(data)

def agent_task_to_acp(task: AgentTask) -> dict[str, Any]:
    if not isinstance(task,AgentTask): raise TypeError("expected AgentTask")
    return {"task_id":task.id,"goal":task.goal,"mode":task.mode.value,"allowed_tools":list(task.allowed_tools),"artifacts":list(task.input_artifacts),"budget":task.to_dict()["budget"],"expected_output":task.to_dict()["expected_output"],"postconditions":[{"name":pc.name,"arguments":pc.arguments,"required":pc.required} for pc in task.postconditions],"security":task.to_dict()["security"],"parent_task_id":task.parent_task_id,"branch_id":task.branch_id,"depth":task.depth,"max_depth":task.max_depth,"max_children":task.max_children}
