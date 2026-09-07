# Lynx architecture and boundaries

```text
 Northbound (authenticated loopback API / CLI)
                 |
          TaskContract + Controller
          /        |        \
      Inference   RLM       RAH/AgentBus
          |        |           |
          +--------+-----------+
                   |
             ToolGateway + Ledger
        /       |       |       |       \
     Web/CLI  MCP   Browser  Kernel   Filesystem
                   (southbound adapters)
                   |
     ArtifactStore / OKF candidate / Flight Recorder / Eval

 External A2A (HTTPS + host allowlist + auth)
          ^                         |
          | untrusted result        v
                Verifier / Merge Gate
```

## Security boundaries

- The model is a proposer, never an authority. The Controller, immutable contract, risk policy, and global `BudgetLedger` gate every tool, delegate, repair, and merge action.
- Web, MCP, browser, remote A2A, and artifact text are data-plane observations. They cannot instruct or authorize the model/controller.
- Child harnesses have independent state, artifact-only handoff, explicit allowlists, intersected risk limits, bounded depth/count, and a shared root ledger.
- Kernel/browser/MCP processes are owned and cancelled by their adapter. External mutations are confirmation-gated.
- Stable OKF knowledge and stable skills cannot be overwritten by candidate data; promotion requires verification/evaluation evidence.
- Flight and trajectory sinks redact secrets before persistence and include task, parent, branch, model/backend, policy, and budget lineage.
