# ADR-0001: llama.cpp als lokales Inference-Backend

- **Status:** Accepted
- **Datum:** 2026-08-28
- **Entscheider:** Lynx-Projekt

## Kontext

Lynx soll ein kleines lokales Modell mit einem deterministischen Agent-Harness verbinden. Die Zielhardware ist eine AMD Radeon RX 9060 XT mit 16 GiB VRAM (ROCm-Architektur `gfx1200`). Die Inference-Runtime soll unabhängig vom Python-Agent-Core austauschbar bleiben und darf nicht an einen ungetesteten Fork oder eine bewegliche Quelle gekoppelt werden.

## Entscheidung

Wir verwenden `llama.cpp` als separaten, OpenAI-kompatiblen `llama-server`. Lynx spricht ausschließlich über `/health` und `/v1/chat/completions`; das Modell wird nicht in den Python-Prozess eingebettet.

- Upstream-Commit wird für jeden Build fest gepinnt. Moving refs (`master`, `main`, `HEAD`, `latest`) werden vom Build-Skript abgelehnt.
- Das GGUF-Modell wird außerhalb des Quellcodes bereitgestellt und per SHA-256 im Betriebsmanifest/Audit referenziert.
- Primäre und derzeit einzige ausführbare Modellbaseline ist `Qwen3.5-4B-Q6_K.gguf`; alternative Quantisierungen und Checkpoints bleiben unter der verbindlichen Single-Model-Policy deaktiviert. Die Modellwahl ist eine messbare Hypothese, kein Leistungsversprechen.
- ROCm/HIP und Vulkan werden als getrennte Builds behandelt und auf derselben Hardware mit demselben GGUF verglichen.
- Standardprofil: Loopback (`127.0.0.1`), ein Server-Slot, Kontext 8192, vollständiger GPU-Offload (`-ngl 999`), Thinking standardmäßig aktiv (`think`), keine Spekulation/MTP/Flash-Attention/KV-Experimente in der Baseline. `fast` deaktiviert Thinking pro Request.
- Der TheCodacus-Fork wird nicht als Produktionsbasis verwendet: dokumentierte CUDA-Optimierungen sind kein Nachweis für AMD/RDNA4.

## Konsequenzen

### Positiv

- GPU-/Runtime-Code bleibt außerhalb des Harness und später austauschbar.
- Reproduzierbare Builds und Benchmarks sind möglich.
- ROCm- und Vulkan-Verhalten können empirisch statt ideologisch bewertet werden.
- Der Flight Recorder kann Modellpfad, Hash, Backend und Commit pro Run aufzeichnen.

### Negativ / Grenzen

- Für ROCm und Vulkan sind unterschiedliche Systempakete und Treiber nötig.
- llama.cpp-CLI-Optionen und Chat-Template-/Tool-Call-Verhalten sind versionsabhängig und müssen gegen `--help` sowie einen Plain-Chat-Smoke-Test geprüft werden.
- Ohne vorhandenen GGUF, Server-Start und lokalen Benchmark gibt es keine belastbare Aussage über Tok/s oder Tool-Call-Qualität.
- Das API-Frontend ist im POC nur loopback-gebunden und nicht für ungeschützten Remote-Betrieb gedacht.

## Verifikation

```bash
export LLAMA_CPP_REF=50f068ffffc3e0e4c9c2e4139281c6075224f429
make build-llama  # ROCm baseline; LLAMA_BACKENDS=rocm,vulkan is optional
export MODEL_PATH=$PWD/models/Qwen3.5-4B-Q6_K.gguf
./scripts/run-llama-rocm.sh
curl http://127.0.0.1:8080/health
uv run lynx run "Berechne 2 + 2" --mode fast
make benchmark
```

Der konkrete Build- und Hardwarestatus wird nicht als erfolgreich markiert, solange `/health`, Plain Chat und der Lynx-Smoke-Test nicht auf der Zielmaschine durchgelaufen sind.

## Aktueller Ausführungsstand

- Der Commit `50f068ffffc3e0e4c9c2e4139281c6075224f429` wurde lokal für ROCm/HIP auf `gfx1200` gebaut.
- `models/Qwen3.5-4B-Q6_K.gguf` wurde geladen und der llama-server-Healthcheck sowie ein Lynx-Fast-Task waren erfolgreich.
- Modell-SHA-256: `fdedd781c9ce676ab66b018ca247ff78e8a33c98098a822c1e2d5075e7718f66`.
- Der Vulkan-Build wurde mit den lokal gepinnten Vulkan-/SPIR-V-Headern reproduziert; ROCm/HIP und Vulkan sind damit als getrennte Builds verfügbar. Die aktuellen Matrix-Ergebnisse liegen unter `build/profile-rx9060xt-qwen35-4b-q6.json`.
