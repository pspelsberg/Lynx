# Lynx Task Backlog: offene Slices und Tasks

**Quelle:** `POC_Idea.txt`, `spec.md`, aktueller Stand unter `src/lynx_harness/`  
**Stand:** 2026-08-29  
**Ziel:** aus dem bestehenden v0.1-POC einen messbaren, bounded Small-Model-First-RLM/RAH-Harness machen.

## 0. Status und Arbeitsregeln

- `[ ]` offen
- `[~]` teilweise vorhanden / Integration fehlt oder durch verbindliche Ressourcen-/Modell-Policy blockiert
- `[x]` bereits vorhanden; kein neuer Implementierungsauftrag
- `[?]` optional bzw. erst nach Messung

**Definition of Done für Code-Tasks:** Implementierung, Unit-/Integration-Test, Audit-/Fehlerpfad, Dokumentation und `uv run python -m unittest discover -s tests -v` erfolgreich. Externe oder GPU-abhängige Tasks benötigen zusätzlich reproduzierbare Run-Artefakte.

**Prioritäten:**

- **P0:** Voraussetzung für sicheren RAH-/Verifier-Betrieb
- **P1:** Kernfunktionalität und Forschungshypothese
- **P2:** produktionsnahe Integrationen
- **P3:** optionale Optimierung / Zukunft

## 1. Bereits als Slice vorhanden

Diese Punkte werden nicht erneut gebaut, sondern müssen bei den offenen Tasks integriert und getestet werden:

- `[x]` `InferenceBackend` und OpenAI-kompatibles `LlamaCppBackend`
- `[x]` `fast`/`think`/`research`, Agent-Loop, Tool-Router
- `[x]` Tool-Gateway, JSON-Schema-Grundvalidierung, Risk-/Network-Gates
- `[x]` Calculator, Workspace-Files, read-only Shell, Web Search/Fetch
- `[x]` ArtifactStore, Trajectory, Flight Recorder, Replay, Fork/Resume
- `[x]` Trust-/Provenance-Metadaten und untrusted-Observation-Grenze
- `[x]` `RlmContextEngine`/`RlmExecutor`
- `[x]` `LocalChildExecutor`, ChildBudget, Artifact-Firewall
- `[x]` Branch-/Best-of-N-Proposals
- `[x]` Research-State-Grundmodell
- `[x]` Skill-Katalog und Candidate-/Stable-Lifecycle
- `[x]` opt-in MCP, Playwright, LSP, Kernel und A2A-Slices
- `[x]` Profiler und verifizierter Trajectory-Export

## 2. P0 — Contract-, Controller- und Budget-Fundament

### TASK-001 — Task- und Usage-Datenmodelle `[x]` `[P0]`

**Ziel:** Root- und Child-Aufgaben transportneutral und unveränderlich beschreiben.

- `TaskContract`, `AgentTask`, `ExpectedOutput`, `Postcondition`, `Usage` ergänzen.
- IDs, Parent-ID, Branch-ID, Depth und Artifact-Referenzen standardisieren.
- strikte Größen-, Zeichen- und Enum-Validierung.

**Abnahme:** Verträge lassen sich aus JSON/YAML laden, validieren und ohne geheime Felder serialisieren; ungültige Verträge werden vor Inference abgewiesen.

### TASK-002 — Globaler `BudgetLedger` `[x]` `[P0]`

- Root-Budget für Tokens, Steps, Tool-/Web-/Browser-Calls, Python-Zeit, Children, Parallelität und Wall-Time.
- atomare Reservierung/Freigabe bei parallel laufenden Children.
- Child-Budget darf Root-Budget nicht erweitern.
- Budget-Events vorher/nachher in Flight Recorder schreiben.

**Abnahme:** adversarialer Paralleltest kann den Root-Ledger nie überschreiten; `recursive_calls` wird tatsächlich verbraucht.

### TASK-003 — Controller-State-Machine `[x]` `[P0]`

Transitions formalisieren:

```text
created -> planned -> executing -> checking
executing -> delegated -> awaiting-child
checking -> repair | verified | rejected
verified -> merged | promoted
```

- unzulässige Transitionen ablehnen;
- `finish`, `delegate`, `verify`, `repair`, `merge` kontrollieren;
- Modell darf nur erlaubte nächste Action vorschlagen.

**Abnahme:** State-Machine-Tests decken alle erlaubten und verbotenen Transitionen ab.

### TASK-004 — `delegate`-Action und Parser `[x]` `[P0]`

- neues strukturiertes Decision-Format für Child-Tasks;
- nur Artifact-Inputs und beschränkte Tool-Allowlist;
- keine frei gesetzten Policy-/Budgetüberschreibungen aus Model-Output;
- Parser- und Prompt-Kompatibilität für das zugelassene Qwen3.5-4B-GGUF (Qwen3.8-Kompatibilität bleibt außerhalb des Runtime-Scopes).

**Abnahme:** unbekannte Felder, Pfade, Tools, Depth und übergroße Payloads werden abgewiesen.

### TASK-005 — Cancellation und Deadline Propagation `[x]` `[P0]`

- Cancellation vom Root zu Children, MCP, Browser, Kernel und Subprocesses.
- `asyncio.wait_for`-Timeouts in einheitliche Statuswerte überführen.
- keine verwaisten Child-/Worker-Prozesse.

**Abnahme:** Timeout-Test beendet alle gestarteten Ressourcen und schreibt ein vollständiges Abschlussereignis.

### TASK-006 — Controller-Integration in `loop.py` `[x]` `[P0]`

- bestehende freie Loop-Entscheidungen an Controller, Ledger und Verifier anschließen;
- Finish nur mit zulässigem Status;
- Parent-/Child-Lineage in jedem Event.

**Abnahme:** bestehende Demo- und Replay-Tests bleiben erfolgreich; Root und Child sind anhand der Events eindeutig rekonstruierbar.

## 3. P0 — Verifier und Ergebnis-Gates

### TASK-010 — Verifier-Interface und Check-Registry `[x]` `[P0]`

Neues composable Interface, z. B.:

```python
Verifier.check(candidate, contract, context) -> VerificationResult
```

Checks: Syntax, Schema, Policy, Postcondition, Evidence, Semantik.

**Abnahme:** Checks liefern stabile IDs, Status, Begründung, betroffene Artefakte und keine ungeprüfte Modell-Confidence.

### TASK-011 — `ChildResult`-Schema-Verifikation `[x]` `[P0]`

- `expected_schema` tatsächlich erzwingen;
- Antwortgröße, Status, Artifact-Refs, Claims und Usage validieren;
- ungültiges ChildResult als `rejected`/`incomplete` behandeln.

**Abnahme:** positive und negative JSON-Schema-Fälle inklusive Injection-/Oversize-Fällen sind getestet.

### TASK-012 — Postcondition-Framework `[x]` `[P0]`

Implementieren:

- `artifact_exists`, `output_bounded`, `no_external_mutation`;
- Datei-/Workspace-Prüfungen;
- test command nur über explizit freigegebene sichere Tool-Policy;
- domänenspezifische Prüfer als Plugin.

**Abnahme:** Finish/Merge kann an eine Postcondition gekoppelt werden; Fehlschlag erzeugt Repair oder Reject.

### TASK-013 — Finish-Gate `[x]` `[P0]`

- Mindestchecks abhängig vom Task Contract;
- `verified`, `unverified`, `incomplete` explizit unterscheiden;
- keine stille Erfolgsantwort bei Budgetabbruch oder fehlender Prüfung.

**Abnahme:** unverified Antworten werden nicht als erfolgreiche Trainings-/Knowledge-Daten exportiert.

### TASK-014 — Merge-Gate für Branches und Children `[x]` `[P0]`

- `BranchManager.merge_proposal()` an Verifier anschließen;
- Root-State erst nach bestandenen Checks verändern;
- konkurrierende Antworten mit Herkunft und Konflikten erhalten;
- Reject-/Repair-Pfad implementieren.

**Abnahme:** ein absichtlich falsches Child wird nicht gemerged; Parent bleibt unverändert.

### TASK-015 — Optionaler unabhängiger Modell-Verifier `[x]` `[P2]`

- zweites `InferenceBackend`-Profil für Kritik/Semantik;
- nie als alleinige Sicherheitsgrenze;
- Kosten und Korrelation messen;
- rohe Chain-of-Thought nicht speichern.

**Abnahme:** Verifier verbessert Holdout-Task-Erfolg oder Fehlererkennung gegenüber deterministischen Checks nachweisbar.

## 4. P0/P1 — RAH v1

### TASK-020 — Rekursive Child-Kette `[x]` `[P0]`

- `delegate` über Controller ausführen;
- Child darf abhängig von Contract/Policy erneut delegieren;
- maximaler Depth- und Child-Count-Limit;
- Child erhält eigenen State, Runner, Registry, Gateway und Budget-Sicht.

**Abnahme:** zweistufige Kette funktioniert; dritte Ebene wird bei `max_depth=2` sauber blockiert.

### TASK-021 — Child-Tool-Firewall härten `[x]` `[P0]`

- Allowlist vor Runner-Erzeugung prüfen;
- keine Parent-Tools, Credentials, Memory oder Policies implizit vererben;
- dynamische Tool-Registrierung nach Start verhindern oder erneut prüfen;
- Child-Risk-Level darf Parent nicht überschreiten.

**Abnahme:** Tests für Registry-Überschreibung, unbekannte Tools, Parent-State-Leakage und Risk-Eskalation.

### TASK-022 — Artifact-only Handoff `[x]` `[P0]`

- Child-Eingabe nur über existierende, hashbare `artifact://`-Refs;
- Input-Manifest mit Ursprung, Größe, Hash, Trust und Branch;
- keine unbounded Inline-Kontexte.

**Abnahme:** Pfad-, URL-, Inline- und fremde Branch-Referenzen werden abgewiesen.

### TASK-023 — Child-Ergebnisartefakte `[x]` `[P1]`

- Child-Output bei Größe/Struktur automatisch als Artefakt schreiben;
- `ChildResult.artifacts` und Provenance vervollständigen;
- Parent erhält Zusammenfassung plus gezielte Refs.

**Abnahme:** große Child-Antworten sind bounded, abrufbar und replay-fähig.

### TASK-024 — Repair-Loop `[x]` `[P1]`

- bei Verifier-Fehler kontrollierte Reparatur anfordern;
- maximale Reparaturversuche und eigenes Budget;
- Repair darf keine neue Tool-Rechte erhalten.

**Abnahme:** Schema-/Postcondition-Fehler führen zu maximal N Reparaturversuchen und anschließendem Reject.

### TASK-025 — RAH Replay/Fork-Lineage `[x]` `[P1]`

- Parent-/Child-Events gemeinsam, aber branch-isoliert aufzeichnen;
- Replay ruft keine externen Tools auf;
- Fork kann ab vollständigem Child-Event fortsetzen.

**Abnahme:** ein kompletter Parent-Child-Lauf ist deterministisch replaybar und Fork verändert den Parent nicht.

### TASK-026 — Parallel-Child-Scheduler `[x]` `[P1]`

- begrenzte Parallelität und Fairness;
- gemeinsamer llama-server statt Modellprozess pro Agent;
- Fehler-/Cancellation-Isolation;
- Batch-/Slot-Nutzung messen.

**Abnahme:** 1/2/3 parallele Children sind reproduzierbar begrenzt; kein Budget- oder Prozess-Leak.

## 5. P1 — RLM und Context Management

### TASK-030 — Automatische Artifact-Schwelle `[x]` `[P1]`

- Input-/History-Größe messen;
- ab konfigurierbarer Schwelle in Artefakt auslagern;
- dem Modell nur Summary und relevante Chunks zeigen.

**Abnahme:** großer Testinput wird nicht komplett in `decide()` übergeben.

### TASK-031 — Chunk-Provenance `[x]` `[P1]`

- stabile Chunk-ID, Offset, Hash, Artifact-ID, Query und Score;
- Parent-/Child-/RLM-Call-Referenz;
- keine Prompt-Injection-Aufwertung durch Chunk-Inhalt.

**Abnahme:** jede Antwortzitation kann auf konkrete Chunks zurückgeführt werden.

### TASK-032 — Retrieval verbessern `[x]` `[P1]`

- vorhandenen Lexical Retriever evaluieren;
- Deduplizierung, Begrenzung und Ranking;
- optional Hybrid/Embedding als austauschbarer Adapter;
- keine neue harte Dependency im Core.

**Abnahme:** Retrieval-Eval mit Recall@k, Promptgröße und Laufzeit; lexical bleibt funktionierender Fallback.

### TASK-033 — RLM Map/Reduce-Workflow `[x]` `[P1]`

- Chunk-Aufteilung;
- bounded Subcalls;
- Aggregation in externem State/Artefakt;
- Verifikation und Quellenzuordnung.

**Abnahme:** ein Dokument oberhalb des normalen Arbeitskontexts wird mit bounded Calls reproduzierbar zusammengefasst/analysiert.

### TASK-034 — RLM-Integration in den Agent-Loop `[x]` `[P1]`

- RLM nicht nur als isolierten Seam, sondern als Controller-Strategie verwenden;
- RLM-Calls in globalen Ledger einbeziehen;
- Fehler, Tiefe und Rekursion sauber terminieren.

**Abnahme:** Ablation `normal_context` vs. `artifact/RLM_context` ist im Evaluator sichtbar.

### TASK-035 — Context Compaction `[x]` `[P1]`

- Working State, relevante Observations, Notes und offene Fragen komprimieren;
- alte Rohdaten nur als Artefakt behalten;
- Kompression darf Provenance nicht verlieren.

**Abnahme:** gleicher Task-Erfolg bei kleinerem Prompt-/KV-Fußabdruck auf festem Testset.

## 6. P1 — Deep Research und Evidence

### TASK-040 — Research-State-Machine vervollständigen `[x]` `[P1]`

Phasen implementieren:

```text
clarify -> questions -> search -> rank -> fetch
-> extract_claims -> gap_analysis -> contradictions
-> targeted_search -> synthesize -> verify -> cite
```

**Abnahme:** jeder Status und jede erlaubte Transition ist getestet; Budgetabbruch ist sichtbar.

### TASK-041 — Subquestion- und Gap-Handling `[x]` `[P1]`

- Fragenliste extern verwalten;
- offene Fragen aus fehlender Evidenz ableiten;
- begrenzte Nachsuche statt unbounded Loop.

**Abnahme:** Research kann mindestens eine Evidenzlücke erkennen und eine begrenzte gezielte Suche ausführen.

### TASK-042 — Claim-Level-Evidence `[x]` `[P0]`

- Claims als eigene IDs speichern;
- `supports`/`contradicts`, Source-Refs, Excerpts, Retrieval-Zeit;
- Status `pending/passed/failed` statt pauschalem `verified=True`.

**Abnahme:** nicht belegte Claims bleiben pending/failed und werden nicht promoted.

### TASK-043 — Source Ranking und Primary-Source-Regeln `[x]` `[P1]`

- Quellenqualität als nachvollziehbare Signale, kein blindes universelles Confidence-Score;
- URL, Host, Titel, Timestamp, Content-Hash;
- Dubletten erkennen;
- untrusted Web-Text bleibt Data Plane.

**Abnahme:** Ranking ist reproduzierbar und im Audit begründet.

### TASK-044 — Contradiction-/Freshness-Checks `[x]` `[P1]`

- Claim-Konflikte explizit erfassen;
- zeitabhängige Claims als stale markieren;
- keine automatische Auflösung ohne Evidenz.

**Abnahme:** absichtlich widersprüchliche Fixtures erzeugen sichtbaren Konflikt und verhindern stable Promotion.

### TASK-045 — Citation-preserving Synthesis `[x]` `[P1]`

- Synthesis referenziert Claim-/Source-/Artifact-IDs;
- fehlende Quellen sichtbar kennzeichnen;
- Output-Limit und Citation-Format definieren.

**Abnahme:** Citation-Precision und nicht belegte Aussagequote im Evaluator messbar.

### TASK-046 — OKF-Promotion `[x]` `[P2]`

- Candidate -> verify -> staging -> stable -> deprecated;
- Provenance und Verification-Events im OKF speichern;
- stable Knowledge nie ungeprüft überschreiben.

**Abnahme:** Knowledge-Promotion ist nur über Verifier-Ergebnis möglich und replaybar.

## 7. P1/P2 — Tools, Skills und Tool Routing

### TASK-050 — Progressive Tool Disclosure `[x]` `[P1]`

- Familienrouting plus Top-k-Retrieval;
- vollständiges Schema erst bei Bedarf;
- Tool-Anzahl modellabhängig konfigurieren;
- Fallback bei schlechter Retrieval-Qualität.

**Abnahme:** Promptgröße und Tool-Selection-Accuracy werden gegen „alle Tools“ verglichen.

### TASK-051 — Einheitliche Tool-Adapter `[x]` `[P1]`

- MCP, CLI, Playwright, Kernel und Web auf `ToolSpec`/`Observation` abbilden;
- identische Schema-, Risk-, Trust- und Provenance-Regeln;
- Fehler-/Retry-Klassen normalisieren.

**Abnahme:** jeder aktivierte Adapter erzeugt dieselben Auditfelder und durchläuft dasselbe Gateway.

### TASK-052 — MCP Production Slice `[~]` `[P2]`

- explizite Server-Konfiguration statt automatischer Discovery;
- Capability-/Version-Handshake persistieren;
- Tool-Allowlist, Timeouts, Output-Limits, Process-Cleanup;
- per-Server Trust-/Risk-Policy und Confirmation.

**Abnahme:** untrusted MCP-Observation kann keine Policy ändern; Server-Abbruch erzeugt keinen hängenden Run.

### TASK-053 — Playwright Production Slice `[~]` `[P2]`

- Browser-Session-Lifecycle und Cleanup;
- Accessibility-Snapshot als Standard, Screenshots nur opt-in;
- Navigation/Domain-Allowlist, Download-/Upload-Limits;
- L2/L3-Aktionen confirmation-gated.

**Abnahme:** Browser-Fixtures bestehen; externe Mutationen bleiben ohne Bestätigung blockiert.

### TASK-054 — Python-Kernel Sandbox `[x]` `[P2]`

- Worker-Prozessgrenze und Ressourcenlimits vervollständigen;
- Workspace-/Artifact-Zugriff explizit; Netzwerk default off;
- Kernel-State je Task/Branch isolieren;
- keine Selbsteskalation aus generiertem Code.

**Abnahme:** Timeout, Speicherlimit, Netzwerkversuch und Path Escape werden sicher beendet/abgewiesen.

### TASK-055 — CLI-/Shell-Tool Ausbau `[x]` `[P1]`

- read-only Baseline beibehalten;
- sichere Test-/Lint-Kommandos als explizite, policy-geprüfte Tools ergänzen;
- keine generische Shell-Freigabe;
- stdout/stderr/Exit-Code als strukturierte Observation.

**Abnahme:** erlaubte Commands funktionieren; Operator Injection, Traversal und gefährliche Optionen bleiben blockiert.

### TASK-056 — Skill Execution `[x]` `[P1]`

- Skill-Manifest mit Tools, Inputs, Outputs, Risiko und Version;
- Workflow-Ausführung nur über registrierte Host-Implementierung;
- Modell sieht zunächst nur Skill-Summary;
- kein direkt ausführbarer Skill-Code aus Modell-/Web-Text.

**Abnahme:** Skill kann candidate/stable ausgeführt werden; unregistrierte oder riskantere Skills werden abgewiesen.

### TASK-057 — Skill Evaluation und Promotion `[x]` `[P2]`

- Replay-basierte Testfälle;
- Baseline-vs.-Candidate A/B;
- Task-Erfolg, Safety, Tokens, Latenz und Verifier-Rate;
- Promotion/Reject auditieren.

**Abnahme:** schlechter Skill wird nie promoted, auch wenn einzelne Läufe besser aussehen.

## 8. P1/P2 — Inference und Modellprofile

### TASK-060 — Model Manifest erweitern `[x]` `[P1]`

> **Single-Model-Scope:** Die folgenden 9B-/27B-Einträge sind historische
> Planungsreferenzen. Runtime, Download, Profiler und Tests dürfen ausschließlich
> `Qwen3.5-4B-Q6_K.gguf` verwenden; für Alternativen werden keine Messungen oder
> Ausführungen behauptet.

Profile und Download-Metadaten für:

- bestehendes Qwen3.5-4B Q6_K;
- Qwen3.8-9B-Distill Q4_K_M/Q5_K_M/Q6_K/Q8_0;
- optional Qwen3.8-27B UD-Q4_K_M als Hybrid-/Teacher-Profil.

Je Eintrag: Repository, gepinnte Revision, Datei, SHA-256, Lizenz, Quantisierung, erwartete Fähigkeiten, VRAM-Hinweis.

**Abnahme:** kein Model-Download ohne Revision/Hash; falsche Datei wird vor Start abgewiesen.

**Implementierung:** Die sechs Manifest-Einträge enthalten Repository, gepinnte Revision, Dateiname, SHA-256, Lizenz, Quantisierung, Fähigkeiten und VRAM-Hinweis. Nur der kanonische 4B-Eintrag ist `downloadable`; alle Alternativen sind reine, nicht ausführbare Referenzen.

### TASK-061 — Qwen3.8-Chat-/Thinking-Kompatibilität `[x]` `[P1]`

- aktuelle Gated-DeltaNet-Unterstützung in llama.cpp verifizieren;
- Chat-Template, `<think>`-Handling, JSON-Modus und `response_format` testen;
- Sampling/Thinking je Modellprofil konfigurieren;
- keine Abhängigkeit von rohem CoT im Agent-State.

**Abnahme im Single-Model-Scope:** fast/think/research erzeugen valide Lynx-Decisions für Qwen3.5-4B-Q6_K; 9B/27B-Konfigurationen sind ausdrücklich nicht ausführbar.

**Implementierung:** 4B-Chat-/Thinking-/JSON-Smoke-Tests sind ausgeführt; 9B/27B-Konfigurationen bleiben unter der Single-Model-Policy bewusst deaktiviert.

### TASK-062 — Modellabhängige Capability Profiles `[x]` `[P1]`

- Tool-Kandidatenlimit, Schritte, Sampling, Vision, MTP und Speculative als Profilwerte;
- gemessene Qualität je Tool-Familie;
- Fallback auf konservatives Profil.

**Abnahme:** Modellwechsel verändert nur Profil/Backend-Konfiguration, nicht Agent-Core.

**Implementierung:** Capability-/Manifest-Code und der konservative Fallback sind vorhanden; alternative Qwen3.8-Modelle werden wegen der verbindlichen Single-Model-Policy nicht geladen oder ausgeführt.

### TASK-063 — llama.cpp Pin und Backend Matrix `[x]` `[P1]`

- getesteten upstream Commit pinnen;
- ROCm/HIP und Vulkan getrennt bauen/testen;
- Build-/Runtime-Metadaten im Profil speichern;
- Forks nur als explizite Benchmark-Variante.

**Abnahme:** gleicher Commit plus Manifest reproduziert Load-/Health-/Structured-Output-Smoke-Test.

### TASK-064 — RX-9060-XT-Profiler `[x]` `[P1]`

Auf AMD RX 9060 XT 16 GB messen:

- 4B Q6_K;
- 9B Q4/Q5/Q6;
- ROCm vs. Vulkan;
- 4k/8k/16k Context;
- 1/2/3 Slots;
- fast/think/research;
- Flash Attention on/off, falls verfügbar.

**Abnahme im Single-Model-Scope:** Profil enthält diese Messwerte für das ausschließlich zugelassene 4B-Q6_K-Modell; die zusätzlich genannten 9B-Varianten werden nicht ausgeführt.

**Implementierung:** `scripts/profile-local.py` und `LynxBenchmarkMatrix` erfassen diese Felder reproduzierbar, prüfen den kanonischen Qwen-Hash und bleiben callback-/hardware-getrieben. Die zulässige 4B-Q6_K-Matrix wurde auf der vorhandenen RX 9060 XT mit dem verifizierten Modell ausgeführt; Reports liegen unter `build/profile-rx9060xt-qwen35-4b-q6.json` und enthalten die reproduzierte Umgebung, Hashes, Commit, Driver, VRAM, Prompt-/Decode-Tok/s, p50/p95, Fehler und Task-Erfolg. Die im ursprünglichen Task zusätzlich geforderten 9B-Varianten werden wegen der verbindlichen Single-Model-Policy nicht ausgeführt.

### TASK-065 — MTP/Speculative Decoding `[~]` `[P3]`

- same-model `ngram-simple` nur als separate, gemessene Benchmark-Variante;
- externe Draft-Modell/MTP-Datei bleibt unter der Single-Model-Policy blockiert;
- exakte Reproduzierbarkeit und Tool-Korrektheit prüfen;
- nicht als Default aktivieren.

**Abnahme:** Aktivierung erfolgt nur, wenn Task-Erfolg und Stabilität gegenüber Baseline nicht sinken.

**Implementierung:** `approve_speculative_variant` ist opt-in und akzeptiert nur nicht-regressive, verifizierte Baseline-/Kandidatenraten; externe Draft-Modelle werden wegen der Single-Model-Policy abgelehnt.

### TASK-066 — Modell-/Backend-Autoauswahl `[x]` `[P2]`

- Profil je Workload speichern;
- Auswahl nach Qualität zuerst, danach Latenz/VRAM;
- kein „schnellstes“ Profil ohne Mindest-Verifier-Rate.

**Abnahme:** `fast`, `think`, `research` wählen reproduzierbare Profile; Wahl wird im Run-Audit festgehalten.

## 9. P1 — Evaluation, Trajectories und Fine-Tuning

### TASK-070 — Lynx Eval-Set erweitern `[x]` `[P1]`

Fälle für:

- no-tool, single-/multi-tool;
- invalid arguments und Retry;
- RLM-Artefakte;
- RAH-Delegation/Merge/Reject;
- Research-Claims/Widersprüche;
- MCP/Playwright/CLI/Kernels in sicheren Fixtures;
- Prompt Injection und Policy-Adversarial-Fälle.

**Abnahme:** versioniertes JSON/YAML-Set mit erwarteten Tools, Postconditions und Verifier-Ergebnis.

### TASK-071 — Evaluator-Metriken `[x]` `[P1]`

Ergänzen:

```text
valid_decision_rate
argument_validity
retry_success_rate
verification_pass_rate
citation_precision
research_coverage
rah_child_success_rate
merge_rejection_rate
llm_tokens / tool_calls / wall_time / VRAM
```

**Abnahme:** Metriken sind pro Fall und aggregiert reproduzierbar; fehlende Daten werden nicht als Erfolg gezählt.

### TASK-072 — Ablationsmatrix `[x]` `[P1]`

Vergleichen:

1. bare LLM;
2. normales Harness;
3. Harness + Tool Router;
4. Harness + Structured State;
5. Harness + Verifier;
6. RLM;
7. RAH;
8. RAH + Verifier.

**Abnahme:** Evaluation zeigt Qualitätsgewinn und Kosten jeder Schicht separat.

### TASK-073 — Verified Trajectory Dataset `[x]` `[P1]`

- State/Observation nicht leer oder pauschal rekonstruieren;
- nur erfolgreich verifizierte Runs exportieren;
- Secret-Redaction und Split nach Taskfamilie/Modell;
- Holdout gegen Leakage schützen.

**Abnahme:** jede Trainingszeile verweist auf Run, Verifier und relevante Action; unverified Run exportiert 0 Beispiele.

### TASK-074 — 27B-Teacher-Vergleich `[~]` `[P2]`

- Qwen3.8-27B nur als Offline-/Hybrid-Teacher evaluieren;
- keine Annahme, dass UD-Q4_K_M komfortabel auf 16 GB VRAM passt;
- Teacher-Traces gegen deterministische Verifier prüfen;
- Distillationseffekt gegen 4B/9B messen.

**Abnahme:** Teacher verbessert validierte Action-/Task-Daten, ohne Tool-Safety zu verschlechtern.

**Implementierung:** `TeacherStudentEvaluator` verifiziert identische Holdout-Cases, verwirft Reasoning und kennzeichnet die zulässige Kontrolle explizit als `same_model_control`. Der geforderte 27B-Teacher wird wegen der verbindlichen Single-Model-Policy nicht geladen oder ausgeführt; die Deferral-Evidenz steht in `build/task074-teacher-comparison-not-run.json`.

### TASK-075 — LoRA/QLoRA-Ablation `[~]` `[P3]`

Erst nach TASK-070 bis TASK-074:

- Tool-Auswahl;
- Argumente;
- Observation-Klassifikation;
- State-Machine-Action;
- kompakte strukturierte Ausgabe.

**Abnahme:** Holdout-Verbesserung bei gleicher oder besserer Verifier-/Safety-Rate; kein Training auf rohe unvalidierte Webdaten.

**Implementierung:** `LoRAExperimentSpec`, `evaluate_lora_ablation` und der verifizierte Trajectory-Dataset-Gate sind implementiert; nur der kanonische Qwen-Base-Checkpoint wird akzeptiert. `llama-finetune` weist quantisierte Checkpoints vor Dataset-/Optimizer-Allokation ab; der reproduzierte Q6_K-Probe-Lauf erzeugte bewusst kein Modell und keine Trainingsmetrik. Die Evidenz steht in `build/task075-training-probe.json` und enthält Commit, Build-/Binary-Hashes, exakten Probe-Aufruf sowie ROCm-/Vulkan-Exitcodes; kein Training wird ohne validierten Lauf behauptet.

## 10. P2 — A2A und interner Agent-Bus

### TASK-080 — Interner `AgentTask`-Bus `[x]` `[P2]`

- Async Queue für lokale logische Agents;
- Task-Status, Result, Artifact und Cancellation;
- ein gemeinsames Inference-Backend;
- kein A2A-Protokolloverhead im lokalen Standardpfad.

**Abnahme:** Researcher, Critic und Verifier können lokal über dieselben Task-/Artifact-Modelle kommunizieren.

### TASK-081 — A2A Mapping `[~]` `[P2]`

- `AgentDescriptor` auf Agent Card abbilden;
- Task/Message/Part/Artifact auf interne Modelle mappen;
- Artifact-only Handoff;
- Remote-Result als untrusted behandeln.

**Abnahme:** lokaler Task kann ohne Vollkontext in A2A-kompatibles Request/Result gemappt werden.

### TASK-082 — A2A Auth-/Transport-Härtung `[x]` `[P2]`

- HTTPS und Host-Allowlist;
- AuthN/AuthZ und Request-/Output-Limits;
- Remote-Timeout/Cancellation und Polling/Streaming-Lifecycle;
- keine Secrets oder System-Prompts im Payload.

**Abnahme:** unallowlisted, HTTP- oder übergroße Remote-Anfragen werden vor Netzwerkzugriff blockiert.

### TASK-083 — A2A Interoperability Test `[x]` `[P2]`

- gegen eine reale kompatible Test-Implementierung;
- Agent Discovery/Card;
- Task Submit, Status, Artifact, Failure;
- Replay mit Stub statt realem Remote-System.

**Abnahme:** Testnachweis liegt als nichtgeheimes Fixture/Run-Artefakt vor.

### TASK-084 — ACP-Adapter `[x]` `[P3]`

Nur bei konkretem Legacy-Bedarf: ACP-Nachrichten auf `AgentTask` mappen. Kein ACP-Core und kein zusätzlicher interner Transport ohne Interoperabilitätsfall.

## 11. P0/P2 — Security, Betrieb und Governance

### TASK-090 — Threat-/Injection-Regressionen `[x]` `[P0]`

- Web/MCP/Browser/Artifact Prompt Injection;
- untrusted Observation darf keine Action/Policy autorisieren;
- Child-Tool-Eskalation;
- Schema-/Payload-DoS;
- Replay-/Fork-Isolation.

**Abnahme:** jede Regression reproduzierbar als Test; gefährliche Action bleibt denied.

### TASK-091 — Secret Redaction und Trajectory Hygiene `[x]` `[P0]`

- Redaction für Tokens, Cookies, Headers, URLs mit Credentials, Env-Werte;
- Redaction vor Trajectory, Artifact und A2A;
- Tests für verschachtelte Daten und Fehlertexte;
- Rotation/Retention dokumentieren.

**Abnahme:** Test-Secrets erscheinen weder in JSONL noch in Artifact-/Flight-Recorder-Dateien.

### TASK-092 — Resource Governance `[x]` `[P1]`

- CPU/RAM/VRAM/Prozess-/FD-/Output-Limits;
- globale Concurrency;
- Kernel-/Browser-/MCP-Cleanup;
- Run-Abbruch bei Disk-/Artifact-Limit.

**Abnahme:** Limitüberschreitung erzeugt kontrollierten Status und keine hängenden Prozesse.

### TASK-093 — API- und Dauerbetrieb-Härtung `[x]` `[P2]`

- loopback default beibehalten;
- AuthN/AuthZ, TLS-Reverse-Proxy-Dokumentation, Rate-Limits;
- Log-/Artifact-Retention und Rotation;
- Health-/Readiness-/Metrics-Endpunkte ohne Secrets.

**Abnahme:** unauthentifizierte Runs werden verweigert; Dauerbetrieb ist mit systemd-Vorlagen reproduzierbar.

### TASK-094 — Audit-Schema versionieren `[x]` `[P1]`

- Event-Schema-Version;
- Run-/Task-/Child-/Branch-IDs;
- Model-/Backend-/Policy-/Budget-Snapshot;
- Migration oder explizite Inkompatibilität.

**Abnahme:** ein Run kann nachträglich mit seiner vollständigen Ausführungsumgebung identifiziert werden.

## 12. P2/P3 — Architektur und Dokumentation

### TASK-100 — A2A-/MCP-/RLM-/RAH-Architekturdiagramme `[x]` `[P2]`

- Northbound User/API;
- Southbound MCP/CLI/Browser/Kernel;
- internal Agent Bus;
- external A2A;
- Artifact/OKF/Trajectory-Layer;
- Security Boundaries.

### TASK-101 — Runbooks `[x]` `[P2]`

- Modell laden und Hash prüfen;
- llama.cpp ROCm/Vulkan bauen;
- Profiler ausführen;
- Live-Integration aktivieren;
- Replay/Fork/Resume;
- Incident bei Tool-/Child-/Kernel-Abbruch.

### TASK-102 — Architektur-Fitness-Tests `[x]` `[P3]`

Automatisch prüfen:

- Core bleibt dependency-free/async;
- Agent-Loop ruft keine URLs/Subprocesses direkt auf;
- externe Tools laufen nur über Gateway;
- Remote-Binding bleibt default deaktiviert;
- Child-Policies können nicht eskalieren.

### TASK-103 — Rust-Grenzen evaluieren `[x]` `[P3]`

Keine Implementierung vor Messung. Erst wenn Scheduler/Tool-Gateway nachweisbar zum Bottleneck wird, IPC-Vertrag definieren und gezielt auslagern.

## 13. Empfohlene Reihenfolge

### Milestone M1 — Sicherer Controller

`TASK-001` → `002` → `003` → `004` → `005` → `006`  
`TASK-010` → `011` → `012` → `013` → `014`  
`TASK-090` → `091`

### Milestone M2 — RAH v1

`TASK-020` → `021` → `022` → `023` → `024` → `025` → `026`

### Milestone M3 — RLM/Research

`TASK-030` → `031` → `032` → `033` → `034` → `035`  
`TASK-040` → `041` → `042` → `043` → `044` → `045`

### Milestone M4 — Modelle und Evals

`TASK-060` → `061` → `062` → `063` → `064`  
`TASK-070` → `071` → `072` → `073`

### Milestone M5 — Skills/Tools/A2A

`TASK-050` → `051` → `052`/`053`/`054`/`055` → `056` → `057`  
`TASK-080` → `081` → `082` → `083`

### Milestone M6 — Teacher/Fine-Tuning/Optimierung

`TASK-074` → `075` → `065` → `066` → `084` → `102`/`103`

## 14. Release-Gate für Lynx v0.2

Der Release ist erst erreicht, wenn:

- RAH mindestens zwei Ebenen bounded und replaybar ausführt;
- Root-/Child-Budget global eingehalten wird;
- Child-Result und Merge explizit verifiziert werden;
- RLM große Artefakte ohne unbounded Prompt verarbeitet;
- Research Claims und Citations nicht pauschal als verified gelten;
- `Qwen3.5-4B-Q6_K` mindestens ein reproduziertes Modell-/Backend-Profil hat (9B-Profile sind durch die verbindliche Single-Model-Policy ausgeschlossen);
- Eval-Metriken Harness-vs.-RAH-vs.-Verifier sichtbar vergleichen;
- Injection-, Secret-, Timeout- und Policy-Regressionen grün sind;
- keine automatische Knowledge-/Skill-/Prompt-Promotion aktiv ist;
- externe A2A/MCP/Browser-Aktionen weiterhin explizit opt-in und confirmation-gated sind.
