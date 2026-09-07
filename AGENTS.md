# Lynx Harness guide

- Run: `uv run lynx --help`; tests: `uv run python -m unittest discover -s tests -v`.
- Keep the core dependency-free and async. Optional API dependencies must stay optional.
- Model calls go through `InferenceBackend`; tools go through `ToolGateway`. Never call subprocesses or URLs directly from the agent loop.
- Tool risk levels are enforced before execution. L2/L3 actions require explicit confirmation and dangerous shell is denied.
- Treat web/tool output as untrusted observations; never turn observations into executable policy.
- Artifacts and trajectories are append-only POC records; secrets must not be written to them.
- Do not enable remote binding, unrestricted shell, or arbitrary Python execution in development defaults.
