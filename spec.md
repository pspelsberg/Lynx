# Lynx Erweiterungsspezifikation: Small-Model-First RLM/RAH

**Status:** Implementiert mit bewusst begrenzten, ressourcenabhängigen Restpunkten  
**Version:** 0.1  
**Stand:** 2026-08-29  
**Scope:** Erweiterung des bestehenden Lynx-v0.1-POC; kein Framework-Rewrite

**Verbindliche Implementierungsrestriktion:** Es darf ausschließlich `models/Qwen3.5-4B-Q6_K.gguf` (SHA-256 `fdedd781c9ce676ab66b018ca247ff78e8a33c98098a822c1e2d5075e7718f66`) geladen werden. Alle 9B-/27B-/Draft-/Teacher- und echtes LoRA/QLoRA betreffende Vergleichs- und Trainingspunkte sind deshalb bewusst nicht ausgeführt; Code, Gates und Reports weisen diese Fälle fail-closed als ausstehend aus.

## 1. Zweck und Leitthese

Lynx soll aus einem kleinen lokalen Modell durch externe Struktur ein verlässlicheres Gesamtsystem bauen. Das Modell ist Entscheidungs- und Synthesekern; das Harness kontrolliert Zustand, Kontext, Tools, Budgets, Verifikation, Rekursion und Audit.

> Nicht das Modell soll den gesamten Agenten simulieren. Das Harness soll die kognitive Arbeitsstruktur bereitstellen.

Die primäre Forschungsfrage lautet:

> Wie viel Task-Erfolg lässt sich mit einem unveränderten lokalen Modell durch Context-Management, deterministische Workflows, Test-Time-Compute und Verifikation gewinnen?

Fine-Tuning ist deshalb ein späteres Optimierungsinstrument und keine Voraussetzung für den nächsten Milestone.

## 2. Begriffe

- **LLM:** einzelner Modellaufruf.
- **RLM:** begrenzte, rekursive Bearbeitung großer externer Artefakte. Das Modell erhält nur relevante Ausschnitte; Artefakte bleiben außerhalb seines Prompts.
- **Agent Harness:** Loop aus Zustand, Inference, Tool-Gateway, Policies und Audit.
- **RAH (Recursive Agent Harness):** ein Harness erzeugt begrenzte Child-Harnesses mit eigenem Zustand, Budget, Tool-Allowlist und Ergebnisvertrag.
- **Skill:** versionierte, getestete prozedurale Fähigkeit; kein ungeprüfter ausführbarer Modelltext.
- **AgentTask:** transportneutrales Delegationsobjekt für interne Children und später A2A.
- **Verifier:** deterministische oder optionale unabhängige Prüfung eines Ergebnisses. Selbstbewertung des Modells gilt nicht als Verifikation.

RAH und RLM sind orthogonal: Ein RAH kann normale LLM-Aufrufe verwenden; ein RLM muss keinen vollständigen Child-Agenten erzeugen.

## 3. Bestandaufnahme des POC

Die Prüfung von `POC_Idea.txt`, `README.md`, `src/lynx_harness` und den Tests ergibt:

### Bereits vorhanden

- `InferenceBackend` und OpenAI-kompatibles `LlamaCppBackend` für `llama-server`.
- `fast`, `think`, `research` mit externen Compute-Budgets.
- deterministischer Agent-Loop und progressive Tool-Auswahl.
- Tool-Gateway mit JSON-Schema-Grundprüfung, Risk-Leveln, Workspace-Grenze und Netzwerk-Schutz.
- Web-, Filesystem-, Calculator- und read-only-Shell-Slices.
- Artefakte, JSONL-Trajectory, Flight Recorder, Checkpoints, Replay und Fork.
- Trust-/Provenance-Metadaten und Data-Plane-Grenze für untrusted Observations.
- `RlmContextEngine` und begrenzter `RlmExecutor`.
- `LocalChildExecutor` als RAH-Slice, Child-Budgets und Artefakt-Firewall.
- Branch- und Best-of-N-Proposals.
- `ResearchWorkflow` mit Sources, Evidence, Contradictions und Synthesis.
- versionierter Skill-Candidate-/Stable-Lifecycle.
- opt-in MCP-, Playwright-, LSP- und Python-Kernel-Slices.
- A2A-nahe Task-/Artifact-Datenmodelle sowie HTTPS-Allowlist-Client/Server-Slice.
- Hardware-/Inference-Profiler und verifizierter Trajectory-Export.
- 120 Offline-Tests; ein Live-llama-Test wird explizit übersprungen, sofern nicht aktiviert.

### Bewusst ausstehend bzw. optional

1. 9B-/27B-Profile, Teacher-Vergleich und Draft-Modelle werden wegen der verbindlichen Single-Model-Policy nicht geladen; die vorhandenen Gates lehnen sie ab.
2. Ein echter LoRA/QLoRA-Trainingslauf ist ohne zulässiges Trainingsmodell bewusst nicht ausgeführt; verifizierter Trajectory-Export und Dataset-Gates sind vorhanden.
3. Offizielle MCP-/Playwright-Transporte und Remote-A2A-Interoperabilität bleiben opt-in; lokale Policy-, Lifecycle- und Payload-Gates sind implementiert.

Alle übrigen in der ursprünglichen Bestandsaufnahme genannten Harness-, Budget-, RLM/RAH-, Research-, Verifier-, Artifact- und Replay-Slices sind implementiert und durch Offline-Tests abgedeckt.

## 4. Ziele und Nichtziele

### Ziele

- Single-Model-Betrieb mit der verifizierten Qwen3.5-4B-Q6_K-Baseline; alternative 9B-/27B-Profile bleiben als nicht ausgeführte Forschungsoption dokumentiert.
- Verlässliche strukturierte Actions statt Vertrauen in freie Thinking-Texte.
- Bounded RLM und RAH mit globalen Budgets, Isolation und reproduzierbaren Ergebnissen.
- Explizite Verifikation, Reparatur und Ergebnis-Promotion.
- Deterministische Deep-Research-State-Machine mit Claim/Evidence-Provenance.
- Messbare Vergleiche von Modell, Quantisierung, Backend, Rekursion und Verifikation.
- llama.cpp/`llama-server` als primärer lokaler Inference-Pfad, ohne Agent-Core an eine Runtime zu koppeln.
- spätere A2A-Kompatibilität ohne A2A als unnötigen internen Overhead.

### Nichtziele für diese Version

- unbeschränkte Rekursion oder autonomes Self-Modification.
- automatische Promotion von Wissen, Skills oder Prompts.
- rohe Chain-of-Thought-Speicherung oder Qualitätsbewertung anhand privater Thinking-Texte.
- direkte Modell-/GPU-Implementierung im Python-Agent-Core.
- sofortiger Rust-Rewrite.
- öffentliche Netzwerkbindung oder unbestätigte externe Mutationen.
- Behauptungen über Geschwindigkeit ohne reproduzierten Hardware-/Commit-/Quant-Benchmark.

## 5. Zielarchitektur

```text
User / CLI / API
       |
Session + Task Contract
       |
Policy Controller / Scheduler
   |       |        |        |
LLM     RLM      RAH      Skills
   |       |        |        |
Inference  Artifact  Child   Workflow
Backend    Context   Executor State
       \    |       |       /
        ---- Tool Gateway ----
          MCP | CLI | Browser | Kernel | Web
                    |
             Verifier / Evidence
                    |
        Merge / Repair / Promotion Gate
                    |
       Flight Recorder + Artifacts + Evals
```

Ein logischer Agent ist:

```text
Agent = State + Policy + Tools + Artifacts + Budget + Verifier
```

Mehrere logische Agents teilen sich nach Möglichkeit einen `llama-server`; Children sind keine zwingend getrennten Modellprozesse.

## 6. Task Contract und Lifecycle

Jeder Root-Task und jedes Child erhält einen unveränderlichen Vertrag:

```yaml
id: task-...
goal: "..."
mode: think
allowed_tools: [filesystem.read]
input_artifacts: [artifact://...]
expected_output:
  type: review
  schema: review-v1
postconditions:
  - no_external_mutation
  - output_is_bounded
budget:
  llm_tokens: 2000
  steps: 6
  tool_calls: 4
  recursive_calls: 0
  max_wall_time_s: 30
security:
  max_risk: READ_ONLY
  confirmation_required: false
```

Lifecycle:

```text
created -> planned -> executing -> checking
                         |             |
                         |             +-> repair
                         +-> delegated -> awaiting-child
                                           |
                                verified -> merged/promoted
                                rejected -> failed/incomplete
```

Das Modell darf lediglich die nächste erlaubte Action vorschlagen. Der Controller validiert Zustand, Transition, Budget und Policy vor jeder Ausführung.

### Neue bzw. zu schärfende Actions

- `tool`: normaler Tool-Aufruf.
- `reflect`: kurze strukturierte Zustandsnotiz, kein Anspruch auf Wahrheit.
- `delegate`: `AgentTask` mit Artifact-Inputs und expliziten Limits.
- `finish`: nur nach erfolgreicher Mindestprüfung oder mit explizitem `unverified`-Status.
- intern: `verify`, `repair`, `merge`; diese Aktionen werden nicht als frei wählbare Tools exponiert.

## 7. Globaler Budget-Ledger

Lokale Child-Budgets dürfen kein Ausgabemittel für unbegrenzte Gesamtarbeit sein. Der Root erhält einen Ledger, von dem jede Descendant-Ausführung atomar reserviert:

```yaml
llm_tokens: 20000
recursive_calls: 8
parallel_branches: 3
tool_calls: 30
web_calls: 12
browser_actions: 20
python_seconds: 30
max_wall_time_s: 300
```

Akzeptanzkriterien:

- Parent plus alle Children überschreiten niemals den Root-Ledger.
- Depth, Anzahl Children und parallele Branches sind separat begrenzt.
- Abbruch gibt `budget_exhausted` statt stiller Teilantwort zurück.
- Jeder Event enthält Budget vorher/nachher und `parent_task_id`.
- Cancellation propagiert an Children, Kernel und externe Prozesse.

## 8. RAH: Child-Verträge, Isolation und Merge

### Child-Ausführung

Der Child-Executor muss:

1. nur referenzierte `artifact://`-Inputs akzeptieren;
2. eine eigene `AgentState`, Registry und Policy verwenden;
3. nur explizit erlaubte Tools exponieren;
4. globales Budget und Depth-Limit prüfen;
5. kein Parent-Memory implizit erben;
6. ein begrenztes strukturiertes Ergebnis und optionale Artefakte liefern;
7. Children nur über eine explizite Controller-Policy weiterdelegieren lassen.

### ChildResult

```json
{
  "task_id": "child-1",
  "status": "complete",
  "answer": "...",
  "artifacts": ["artifact://..."],
  "claims": [],
  "verification": {"status": "passed", "checks": []},
  "usage": {"llm_tokens": 800, "tool_calls": 2, "elapsed_ms": 1200}
}
```

`expected_schema` wird vor Rückgabe validiert. Ungültige Ergebnisse werden nicht als erfolgreiche Antwort behandelt.

### Merge-Gate

Der Parent erhält zunächst eine `MergeProposal`. Der Root-State wird nur geändert, wenn:

- Child-Status erfolgreich oder ausdrücklich zulässig unvollständig ist;
- Schema, Tool- und Budget-Checks bestanden sind;
- erforderliche Postconditions erfüllt sind;
- Artefakte existieren und Provenance tragen;
- der Verifier die Promotion akzeptiert.

Bei konkurrierenden Branches werden alle Ergebnisse als Vorschläge behandelt. Best-of-N ohne Verifier ist keine Qualitätssteigerung, sondern nur Mehrfachausführung.

## 9. RLM und Context Engine

Artefakte bleiben außerhalb des Modellkontexts. Der Standardpfad für große Inputs ist:

```text
register artifact -> inspect metadata -> retrieve chunks
-> optional map calls -> aggregate -> verify citations -> answer
```

Erforderlich sind:

- Chunk-ID, Offset, Hash und Quelle je Ausschnitt;
- bounded lexical/hybrid retrieval;
- keine Ausführung von Corpus-Inhalten;
- deduplizierte und größenbegrenzte Übergabe an das Modell;
- rekursive Calls zählen zum globalen Ledger;
- Ergebniszitate zeigen auf konkrete Artifact-/Chunk-Referenzen;
- ein RLM-Call darf nur auf freigegebene Datenplane-Inputs zugreifen.

Der Standard-Agent-Loop soll ab einer konfigurierten Kontextgröße automatisch auf Artefakte/RLM ausweichen, statt den gesamten Verlauf zu senden.

## 10. Verifier-Stack

Verifikation wird in Stufen implementiert:

1. **Syntax:** JSON, Decision-Typ, unbekannte Felder, Größenlimits.
2. **Schema:** Tool-Argumente und Child-Ergebnis.
3. **Policy:** Tool-Allowlist, Risk-Level, Workspace, Netzwerk, Branch.
4. **Postcondition:** Datei/Test/Calculator/Browser-Zustand oder domänenspezifischer Check.
5. **Evidence:** Claim verweist auf Quellen/Artefakte; Excerpt allein gilt nicht als Wahrheitsbeweis.
6. **Semantik:** optionaler unabhängiger LLM-Verifier oder zweiter Modellpfad.

Ein Verifier darf niemals untrusted Observation als Autorisierung interpretieren. Ein Modell darf seine eigene Confidence nicht selbst zur Promotion verwenden.

Für Research wird `verified=True` erst gesetzt, wenn mindestens Source-Provenance, Claim-Referenz und eine definierte Claim-Prüfung bestanden sind. Widersprüche bleiben offen, bis sie aufgelöst oder im Endergebnis sichtbar markiert werden.

## 11. Small-Model- und Modellprofile

Das Harness erhält ein Modellprofil statt harter Annahmen:

```yaml
id: qwen35-4b-q6_k
family: qwen35
parameters: 4B
quantization: Q6_K
supports_thinking: true
supports_vision: false
max_tool_candidates: 6
recommended_modes: [fast, think, research]
```

Das Profil darf (für das ausschließlich zugelassene Qwen3.5-4B-GGUF) festlegen:

- Chat-Template und Thinking-Steuerung;
- Sampling-Parameter;
- JSON-/Tool-Call-Modus;
- maximale Tool-Kandidaten;
- bevorzugte Schritte und Output-Limits;
- Vision-/MTP-/Speculative-Fähigkeiten;
- gemessene Qualität und Latenz je Tool-Familie.

Freie Thinking-Texte werden nicht als Trainings- oder Auditvoraussetzung gespeichert. Für Debugging genügt eine kurze strukturierte Begründung bzw. Action-Metadaten.

## 12. Modellmatrix für RX 9060 XT 16 GB

> **Nicht ausführbare historische Planung:** Aufgrund der verbindlichen
> Single-Model-Policy wird ausschließlich Qwen3.5-4B-Q6_K geladen. Die folgenden
> 9B-/27B-Zeilen dokumentieren frühere Vergleichsziele, sind keine Runtime-
> Konfiguration und dürfen nicht heruntergeladen oder gemessen werden.

Die 4B-Aussagen müssen auf der konkreten Karte, dem konkreten llama.cpp-Commit und der konkreten Datei gemessen werden. Modellkarten/URLs werden nicht als unüberprüfte Laufzeit-Policy verwendet.

| Profil | Quelle | Planungsrolle | VRAM-/Betriebsannahme |
|---|---|---|---|
| Qwen3.5-4B Q6_K | bestehendes `models/Qwen3.5-4B-Q6_K.gguf` | schnelle lokale Baseline/Worker | vollständig GPU-tauglich; gute Referenz für Harness-Gewinn |
| Qwen3.8-9B-Distill Q4_K_M | [empero-ai](https://huggingface.co/empero-ai/Qwen3.8-9B-Distill-GGUF) | primärer Qualitäts-/Speed-Kandidat | 5.780 GB Datei laut Model Card; komfortabler Spielraum für KV/Slots |
| Qwen3.8-9B-Distill Q5/Q6/Q8 | ebenda | Qualitätsablation | 6.643/7.559/9.786 GB laut Model Card; Kontext, KV und Parallelität separat messen |
| Qwen3.8-27B UD-Q4_K_M | [Unsloth](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) | Offline-Teacher oder Hybrid-/Qualitätsreferenz | 16.464 GB Datei laut HF-Dateiabruf; für 16-GB-VRAM nicht als komfortabler Voll-GPU-Betrieb einplanen |
| Qwen3.8-27B Q8/hohe Quants | ebenda | nicht primärer lokaler Betrieb | deutlich außerhalb der 16-GB-Klasse; nur CPU-/Hybrid-/externe Evaluation |

Die 27B- und 9B-Dateien sind unter der verbindlichen Single-Model-Policy keine
lokalen oder Remote-Laufzeitoptionen. Frühere Überlegungen zu Teacher-
Trajectories, Offline-Referenzen oder Hybrid-Offload bleiben zurückgestellt; es
werden dafür keine Qualitäts- oder Geschwindigkeitsgewinne behauptet.

### Inference-Benchmark-Matrix

Für die zulässige Baseline testen:

- llama.cpp upstream, ROCm/HIP und Vulkan;
- Qwen3.5-4B Q6_K;
- Kontext 4k/8k/16k;
- 1, 2 und 3 parallele Slots;
- Thinking an/aus;
- Flash Attention an/aus, sofern Build unterstützt;
- keine externe Draft-Modell-Variante; MTP bleibt wegen der Single-Model-Policy deaktiviert. Same-model `ngram-simple` ist ausschließlich als gemessene, opt-in Benchmark-Variante verfügbar; die Baseline bleibt ohne Spekulation.

Zu speichern sind: llama.cpp-Commit, GGUF-Revision und SHA-256, Backend, GPU/Driver/ROCm, Parameter, Prompt-/Generation-Tok/s, p50/p95-Latenz, VRAM, Fehler, Tool-/Task-Erfolg und Reproduzierbarkeit. Das beste Profil wird getrennt für `fast`, `think`, `research` und parallele RAH-Branches gewählt.

## 13. Research-Workflow

`ResearchWorkflow` wird zu einer echten State Machine ausgebaut:

```text
clarify -> questions -> broad_search -> source_rank
-> fetch -> extract_claims -> gap_analysis
-> contradiction_check -> targeted_search
-> synthesize -> verify_claims -> cite -> complete
```

Der Zustand bleibt außerhalb des Modells. Jede Quelle und jeder Claim erhält:

```yaml
id: claim-...
text: "..."
sources: [source-...]
evidence: [artifact://...]
relation: supports|contradicts
verification: pending|passed|failed
retrieved_at: ...
```

Research darf keine stabile Knowledge-/OKF-Notiz automatisch überschreiben. Der Lifecycle ist `candidate -> verify -> staging -> stable` bzw. `rejected/deprecated`.

## 14. Tools, Skills, MCP und A2A

### Tool Fabric

MCP, CLI, Playwright, Web und Kernel werden intern über `ToolSpec`/`ToolGateway` normalisiert. Das Modell sieht nur wenige, geroutete Kandidaten; vollständige Schemas werden erst nach Auswahl nachgeladen, falls möglich.

### Skills

Skills bleiben progressive, versionierte Workflows:

```text
primitive tool -> skill -> workflow
```

Neue Skills durchlaufen `candidate -> replay evaluation -> A/B comparison -> stable`. Skills dürfen nicht allein aufgrund einer Modellbehauptung promoted werden.

### A2A

A2A wird als externe Agent-zu-Agent-Grenze berücksichtigt. Intern bleibt eine schlanke `AgentTask`-Queue effizienter. Required:

- A2A-/Agent-Card-Mapping aus `AgentDescriptor`;
- Artifact- statt Vollkontext-Übergabe;
- HTTPS, Host-Allowlist, Authentisierung, Größen-/Zeitlimits;
- Remote-Ergebnisse sind untrusted und gehen durch denselben Verifier;
- keine geheimen Prompts, Tokens oder Parent-Memory in Payloads.

ACP wird nicht als neuer Core implementiert. Falls Legacy-Kompatibilität nötig wird, entsteht ein Adapter auf `AgentTask`.

## 15. Fine-Tuning- und Trajectory-Strategie

Vor Fine-Tuning muss eine verlässliche Baseline für das ausschließlich zugelassene Qwen3.5-4B-Q6_K-GGUF vorliegen. Exportiert werden nur bounded, erfolgreich und verifier-approved Trajectories; 9B-Baselines sind unter der Single-Model-Policy nicht vorgesehen.

Trainingsziel ist nicht Weltwissen, sondern:

- Tool benötigt / benötigt kein Tool;
- Auswahl aus kleiner Kandidatenmenge;
- korrekte Argumente;
- Observation klassifizieren: Erfolg, Retry, Alternative, Finish;
- State-Machine-Action wählen;
- kompakte strukturierte Ausgabe.

Ein 27B-Teacher ist unter der verbindlichen Single-Model-Policy nicht aktiv; die vorhandenen Gates akzeptieren ausschließlich verifizierte Qwen3.5-4B-Daten und speichern kein Reasoning.

## 16. Evaluation

Die Eval-Suite wird in folgende Klassen geteilt:

- no-tool und einfache Antworten;
- Single-Tool und Multi-Tool;
- Fehler-/Retry-Fälle;
- RLM-Artefaktfragen;
- RAH-Delegation und Merge;
- Research mit Quellen und Widersprüchen;
- Browser/MCP/CLI (opt-in, sichere Fixtures);
- Prompt-Injection und Policy-Adversarial-Fälle;
- Backend-/Quantisierungsvergleich.

Pflichtmetriken:

```text
task_success
valid_decision_rate
tool_selection_accuracy
argument_validity
unnecessary_tool_calls
retry_success_rate
verification_pass_rate
citation_precision
research_coverage
rah_child_success_rate
merge_rejection_rate
tokens / tool_calls / wall_time / VRAM
```

Jede neue RLM-/RAH- oder Inference-Optimierung muss gegen `bare_llm`, normalem Harness, Harness+Verifier und RAH+Verifier laufen. Geschwindigkeit ohne Task-Korrektheit ist kein Erfolg.

## 17. Security und Governance

- Default bleibt loopback-only und externe Kommunikation deaktiviert.
- L2/L3-Aktionen brauchen explizite Confirmation; gefährliche Shell bleibt denied.
- Child-Policies dürfen Parent-Rechte nicht erweitern.
- Untrusted Observation ist Data Plane: `can_instruct=false`, `can_authorize=false`.
- Artifact-, Trajectory- und A2A-Payloads sind größenbegrenzt; Secrets werden vor Persistenz redigiert.
- Kernel, Browser, MCP und Remote-Agenten laufen hinter Prozess-/Zeit-/Ressourcenlimits.
- Verifier, Skill-Promotion und Knowledge-Promotion werden auditierbar aufgezeichnet.
- Kein automatischer Download oder Aktivieren eines Modell-/Backend-Forks aus Modell- oder Webtexten.

## 18. Implementierungsphasen und Abnahmekriterien

### Phase 1 — Contract/Verifier Foundation

- `TaskContract`, `AgentTask`, `BudgetLedger`, `Verifier`-Interfaces.
- Schema-Prüfung für ChildResult.
- globale Budgettests, Cancellation und Event-Verknüpfung.
- `DecisionType.DELEGATE` oder äquivalente Controller-Action.

**Abnahme:** Ein Child kann sicher gestartet, abgebrochen, geprüft und als Proposal zurückgegeben werden; Root-Budget wird nie überschritten.

### Phase 2 — RAH v1

- bounded rekursive Delegation;
- Artifact-only Handoff;
- Child-Tool-Firewall;
- Merge-/Repair-Gate;
- isolierter Replay-Test für Parent und Children.

**Abnahme:** mindestens zwei Ebenen funktionieren; Depth-, Tool-, Zeit- und Gesamtbudgetverletzungen werden abgewiesen.

### Phase 3 — RLM im Standardpfad

- automatische Artefaktablage ab Kontextschwelle;
- Chunk-Referenzen und map/reduce-artige Calls;
- Claim-/Citation-Referenzen;
- RLM-Ablation in Evals.

**Abnahme:** ein Input oberhalb des normalen Prompt-Limits wird mit bounded Chunks korrekt und reproduzierbar bearbeitet.

### Phase 4 — Research und Skills

- Claim-Level-Verifier;
- Contradiction-/Freshness-Checks;
- Skill-Testfälle und A/B-Promotion;
- OKF-Candidate-Lifecycle.

**Abnahme:** nicht belegte Claims werden nicht als verified/stable gespeichert.

### Phase 5 — Modell-/GPU-Profile

- Model Manifest für alle getesteten GGUFs;
- RX-9060-XT ROCm/Vulkan-Matrix;
- Qwen3.5-4B-Q6_K gegen die jeweils gemessene Harness-/Verifier-Baseline (9B/27B bleiben dokumentierte, nicht ausgeführte Vergleichsziele);
- workloadabhängige Profilwahl.

**Abnahme:** ein reproduzierbares Profil je Modus mit Qualitäts- und Performance-Messung, nicht nur Tok/s.

### Phase 6 — Optionales A2A und Fine-Tuning

- A2A nur nach lokalem Internal-Bus-Nachweis;
- optionaler Teacher-Run ist unter der Single-Model-Policy zurückgestellt und wird nicht behauptet;
- verifizierter Trajectory-Datensatz;
- LoRA/QLoRA-Ablation.

**Abnahme:** Fine-Tuning verbessert ein festgelegtes Holdout-Set, ohne Tool-Safety oder Verifier-Rate zu verschlechtern.

## 19. Erste konkrete Arbeitsliste

1. `models.py`: Ledger-/Task-/Usage-Modelle und Delegate-Action.
2. `rah.py`: Schema-Prüfung, globale Reservierung, echte Child-Verkettung.
3. neuer `verifier.py`: zusammensetzbare Checks und Merge-Gate.
4. `loop.py`: Controller-Transitions und Verifier vor Finish/Merge.
5. `research.py`: Claim-Level-Status statt „non-empty excerpt = verified“.
6. `context.py`: Artifact-/Chunk-Provenance und Integration in den Loop.
7. `eval.py`: RAH/RLM/Verifier-Varianten und Kosten-/Qualitätsmetriken.
8. `config/model-manifest.json`: der kanonische Qwen3.5-4B-Q6_K-Eintrag ist mit gepinnter Revision und SHA-256 ausführbar; die fünf erweiterten 9B-/27B-Einträge sind reine Metadaten-Referenzen und bleiben außerhalb der Runtime. Der Loader und Download-Gate erlauben ausschließlich den kanonischen Eintrag.
9. `profiler.py`/Scripts: ROCm/Vulkan/Slot-/Thinking-Matrix für RX 9060 XT.
10. Tests zuerst: Contract, Budget, Injection, Merge-Rejection, Replay und Modell-Output-Parsing.

## 20. Erfolgsdefinition

Lynx ist für diese Spezifikation erfolgreich, wenn das kanonische 4B-Modell mit begrenztem Tool-Kontext:

- valide Actions zuverlässig erzeugt,
- externe Tools nicht unkontrolliert verwenden kann,
- große Inputs über Artefakte/RLM verarbeitet,
- komplexere Aufgaben über bounded RAH zerlegt,
- Ergebnisse nicht ungeprüft merged,
- Research-Claims nachvollziehbar belegt,
- und gegenüber einem einfachen Single-Agent-Loop einen reproduzierbaren Qualitätsgewinn bei bekanntem Compute-/Latenzbudget erzielt.

Der Gewinn muss vom Harness kommen und in Evals sichtbar sein — nicht nur aus einem größeren Modell oder einer optimistischeren Antwort.
