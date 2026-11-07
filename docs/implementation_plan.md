# Customer Success Agent Implementation Plan

## 1. Objective

Build a reproducible customer-support automation system that normalises tickets from four channels, classifies intent and urgency, retrieves identifiable knowledge-base passages, deterministically decides whether to respond or escalate, generates grounded cited answers, blocks unsafe responses, persists every decision, and processes an arbitrary evaluation file unattended while producing a metrics report.

The primary delivery gate is one documented command that accepts an input path and output directory and processes every ticket without intervention. The implementation should optimise for passing this gate reliably before adding optional sophistication.

This plan covers engineering work. The discovery problem statement, evaluation interpretation, and personal reflection must remain the student's own work, as required by the project instructions.

## 2. Current State Assessment

### Present

- Required top-level folders exist: `src`, `prompts`, `tests`, `evaluation`, `docs`, `data`, `storage`, and `.github/workflows`.
- The eight required source module names exist.
- A GitHub Actions workflow exists and invokes dependency installation and pytest.
- Environment variable names are present in a local `.env`.
- The 29-article documentation corpus is present and a Chroma database has been created locally.
- SQLite storage exists locally.

### Incomplete or blocking

- `src/ingest.py`, `classify.py`, `retrieve.py`, `route.py`, `generate.py`, `guardrails.py`, `logging_store.py`, `api.py`, and `evaluation/harness.py` are empty.
- `start.py` is a commented setup experiment, not an application entry point.
- `README.md`, `docs/architecture.md`, and `prompts/README.md` are empty.
- `tests` has no test suite.
- `.env.example` is missing.
- `.gitignore` ignores `.env` only; it must also ignore runtime storage, caches, results, Python cache files, and local environments.
- Generated Chroma and SQLite data currently sit inside the project and must not be treated as committed source artifacts.
- Only `documentation.json` has been copied into the working project. Development tickets and ground-truth responses remain in the source pack.
- The supplied pack does not contain the advertised `validation_tickets.json`. This must be obtained from the course source or raised with the provider. Until then, create a fixed development/validation split from the 500 development tickets and clearly label those results as development results.
- Dependency versions are not fully pinned, include duplicate `requests`, and mix old LangChain pins with open-ended packages. This jeopardises clean-checkout reproducibility.
- The working project is not currently inside a Git repository, so commit history and CI cannot yet satisfy submission requirements.

## 3. Evidence and Dataset Facts to Design Around

- 500 development tickets across email (212), chat (155), documentation comments (78), and forum (55).
- 22 intent classes and three urgency classes.
- 311 tickets are labelled for automatic response and 189 for escalation.
- 357 tickets are answerable from documentation.
- 87 tickets must never receive an automatic response.
- The always-escalate intent groups in the development labels are `compliance_request`, `feature_request`, `security_incident`, and `unclear_request`.
- The corpus contains 29 articles across ten documentation categories.
- 200 ground-truth responses provide required points and prohibited claims.
- The hidden assessment file will use the published ticket schema, but its path and ticket count must not be assumed.

## 4. Proposed Architecture

Use a deterministic local pipeline for ingestion, classification, retrieval, routing, validation, and logging. Use the external LLM only to draft grounded answers. This reduces rate-limit exposure, improves repeatability, and permits graceful degradation during provider outages.

```text
Ticket JSON
  -> ingest and schema validation
  -> local intent and urgency classifier
  -> Chroma retrieval with relevance threshold
  -> deterministic routing policy
       -> escalation package, or
       -> structured LLM answer with retrieved context
          -> guardrails and citation validation
             -> sent answer or blocked escalation
  -> terminal outcome, decision records, and metrics
```

Recommended supporting modules:

- `src/config.py`: typed environment configuration and path resolution.
- `src/models.py`: Pydantic request, intermediate-state, citation, decision, response, and evaluation models.
- `src/pipeline.py`: LangGraph state graph and the single `process_ticket` entry point.
- `src/llm_client.py`: OpenRouter client, retries, timeouts, structured output parsing, and response cache.
- `src/metrics.py`: Prometheus instruments and reusable metric calculations.
- `src/exceptions.py`: expected failure types mapped to safe fallbacks.
- `src/bootstrap.py`: idempotent corpus indexing, schema creation, and optional classifier training.
- `artifacts/`: versioned local classifier/vectoriser artifacts if training is not performed during setup.

`storage/` remains runtime-only and is recreated by bootstrap. Application paths must be based on the project root, never the terminal's current directory.

## 5. Core Data Contracts

Define these contracts before component implementation:

1. `NormalizedTicket`
   - ticket ID, channel, subject, body, combined text, received timestamp, customer metadata, and optional labels/history.
   - Handles absent optional fields, empty bodies, unknown extra fields, and Unicode.

2. `ClassificationResult`
   - intent, urgency, calibrated confidence, and ranked alternatives with probabilities.

3. `RetrievedPassage`
   - stable passage ID, document ID, title, section/chunk index, exact text, similarity score, and corpus metadata.

4. `RouteDecision`
   - action, threshold, policy checks, confidence, reason, and escalation summary.

5. `GeneratedAnswer`
   - answer text, claim-level citations, model metadata, prompt version, and generation status.

6. `GuardrailReport`
   - result per guardrail, detected evidence, blocking guardrail, and final action.

7. `ProcessingResult`
   - exactly one terminal outcome per ticket: `auto_respond`, `escalate`, or `block`, plus latency and error/fallback information.

## 6. Implementation Phases

### Phase 0 - Reproducible Foundation

Deliverables:

- Initialise Git at the project root and establish small, meaningful commits.
- Copy the permitted development data and ground-truth data into the working area or document their required external paths.
- Obtain the missing validation file or record the issue; do not invent it.
- Create `.env.example` containing placeholder values only.
- Expand `.gitignore` for `.env`, `.venv`, `storage/`, evaluation run output, caches, and Python artifacts.
- Resolve and fully pin a mutually compatible Python 3.11 dependency set.
- Split setup from runtime: bootstrap builds indexes/database; API and harness consume them.
- Add a bootstrap smoke check and verify setup from a fresh environment.

Definition of done:

- A new checkout can install dependencies and run bootstrap using only documented commands.
- No secret or machine-specific absolute path appears in source or Git history.

### Phase 1 - Models, Configuration, and Ingestion

Implement `config.py`, `models.py`, and `ingest.py` first.

Ingestion requirements:

- Accept `email`, `chat`, `docs_comment`, and `forum` through one parser.
- Preserve original body and channel while producing a common combined-text field.
- Validate required identifiers and timestamps without rejecting recoverable records.
- Convert malformed records into a logged safe escalation rather than crashing the batch.
- Keep labels/history optional so the same code works for labelled and unlabelled production inputs.

Tests:

- One fixture for each channel.
- Missing subject, empty body, missing optional metadata, unusual Unicode, unknown channel, bad timestamp, and malformed object.

Definition of done: four channel fixtures produce the same internal type and no fixture crashes the process.

### Phase 2 - Retrieval and Corpus Bootstrap

Implement an idempotent corpus indexer and `retrieve.py`.

Design:

- Parse article headings and numbered resolution steps before recursive splitting so a chunk does not break a resolution sequence unnecessarily.
- Compare at least two chunking configurations and record retrieval hit rate on development queries.
- Store stable passage IDs such as `DOC-AUTH-001#p03` plus the exact passage text.
- Return ranked passages and scores.
- Apply a measured relevance threshold and return an empty result when nothing is relevant.
- Detect corpus changes with a content hash and rebuild only when required.

Tests:

- Known questions retrieve expected document IDs.
- Feature/unclear requests can return no result.
- Every citation identifier resolves to an exact stored passage.
- A missing/corrupt index triggers a documented rebuild or safe error.

Definition of done: retrieval is traceable, thresholded, reproducible, and independently testable.

### Phase 3 - Classification and Confidence Calibration

Implement `classify.py` using a local supervised baseline:

- TF-IDF word and character features with logistic regression or linear SVM probability calibration.
- Separate intent and urgency models.
- Stratified development split with a fixed seed.
- Probability calibration using held-out data.
- Persist vectoriser, label encoders, model, training schema version, and metrics together.
- Return the chosen class, confidence, and top alternatives.
- Return a defined `unknown`/low-confidence result if inference fails.

This local classifier is preferable to an LLM classifier for the first working version because it is fast, deterministic, cheap, and supports measurable calibration. An LLM classifier can be evaluated later only if it improves held-out performance without breaking latency or reliability.

Definition of done:

- All tickets receive intent, urgency, confidence in `[0, 1]`, and alternatives.
- Per-class precision/recall and calibration tables can be generated.

### Phase 4 - Deterministic Routing

Implement `route.py` as a pure function.

Priority order:

1. Kill switch enabled -> escalate.
2. Malformed/empty input -> escalate.
3. Always-escalate predicted intent -> escalate.
4. Security, compliance, private-data, or prompt-injection concern -> escalate/block as policy specifies.
5. Classification confidence below measured threshold -> escalate.
6. No retrieval hit or retrieval score below threshold -> escalate.
7. Otherwise -> generate a draft.

Determine the confidence and retrieval thresholds from the development split, documenting the false-auto-response versus false-escalation trade-off. Do not hardcode `0.80` merely because it appears in an example.

Definition of done: identical inputs and state produce identical decisions and a human-readable reason.

### Phase 5 - Grounded Generation and Versioned Prompts

Implement `generate.py`, `llm_client.py`, and prompt files under `prompts/build/`.

Requirements:

- Place system instructions and retrieved passages outside the untrusted ticket delimiters.
- Require structured JSON output containing answer text and claim-level passage IDs.
- Permit citations only from the retrieved passage list.
- Instruct the model to state uncertainty and never invent missing facts.
- Prohibit refunds, timelines, account actions, or other commitments.
- Set deterministic generation parameters where supported.
- Version every prompt and record requirement IDs in `prompts/README.md`.
- Cache by model, prompt version, normalised ticket, and retrieved-passage hash.
- Use bounded exponential backoff for transient failures and rate limits.

Definition of done: a successful answer parses reliably and every cited passage was actually retrieved.

### Phase 6 - Guardrails and Safe Fallbacks

Implement `guardrails.py` so checks run on every candidate answer and can block it.

Minimum checks:

- PII/secrets: emails, phone numbers, API keys/tokens, account numbers, addresses where detectable.
- Citation integrity: unknown passage IDs, missing citations on factual claims, or citations not in the retrieval set.
- Grounding: claim/support checks against exact retrieved text, with conservative escalation on uncertainty.
- Instruction integrity: common prompt-injection patterns and attempted role/system override.
- Tone/scope: prohibited commitments, unsafe security guidance, or unsupported commercial promises.
- Confidence floor and kill-switch enforcement.

When a guardrail fires, do not silently redact and send. Block the response, log the evidence, and create an escalation package.

Definition of done: dedicated fixtures reliably trigger at least one block for each guardrail category, and a normal cited answer passes.

### Phase 7 - Decision Store, Observability, and Auditability

Implement `logging_store.py` and `metrics.py`.

Use two related SQLite tables:

- `ticket_runs`: one row per processed ticket with one terminal outcome. This is used to reconcile processed tickets with outcomes.
- `decisions`: multiple stage records per ticket for classification, routing, generation, and validation, following the governance minimum schema.

Store timestamps, ticket ID, stage, input summary/hash, model/version, prediction, confidence, alternatives, sources/scores, thresholds, action, reason, guardrail results, prompt version, requirement IDs, latency, and error/fallback information. Do not store unnecessary raw private data.

Expose Prometheus counters/histograms for ticket outcomes by channel, latency, guardrail activations, confidence distribution, provider failures, retrieval misses, and batch progress.

Definition of done:

- Every input has exactly one terminal `ticket_runs` record.
- Expected stage decisions are queryable and missing-stage audits are automated.
- Metrics endpoints do not start unexpectedly during imports or tests.

### Phase 8 - LangGraph Pipeline and FastAPI

Implement `pipeline.py` and `api.py`.

LangGraph nodes:

- ingest -> classify -> retrieve -> route -> generate (conditional) -> validate -> finalise/log.

Every node must return typed state and convert expected failure into a safe state transition. Unexpected ticket-level failures must be captured as logged escalations so the batch continues.

FastAPI endpoints:

- `GET /health`: liveness plus dependency readiness.
- `GET /ready`: corpus/model/database readiness.
- `POST /tickets/process`: process one ticket.
- `GET /decisions/{ticket_id}`: audit view with sensitive fields omitted.
- `GET /metrics`: Prometheus output.
- Optional authenticated `POST /admin/kill-switch` only if it can be secured; otherwise implement the switch as an environment/config file checked per ticket.

Definition of done: API examples cover automatic response, escalation, guardrail block, and provider outage.

### Phase 9 - Unattended Evaluation Harness

Implement `evaluation/harness.py` early, not after polishing the API.

Required CLI:

```powershell
python -m evaluation.harness --input <tickets.json> --output <result-directory>
```

Harness behaviour:

- Validate CLI paths and create the output directory.
- Load any number of records using the published schema.
- Process sequentially initially to respect free-tier limits; add bounded concurrency only after reliability is proven.
- Continue after per-ticket failures and produce a safe logged escalation.
- Persist incremental JSONL/checkpoints so an interrupted diagnostic run can be inspected.
- Produce `results.jsonl`, `metrics.json`, `classification_report.json`, `fairness_report.json`, `errors.jsonl`, and a run manifest containing date, code/config versions, thresholds, model, and ticket count.
- Exit non-zero only for run-level failures such as unreadable input or inability to initialise; ticket failures are represented in outputs.

Automatically calculate:

- Volume: processed, auto-answered, escalated, blocked.
- Business: first-contact resolution proxy, mean/median response time, escalation rate, repeat-contact baseline where labels exist.
- Technical: per-class intent precision/recall, urgency metrics, retrieval hit rate, citation accuracy proxy, median/p95 latency, provider availability.
- Governance: log reconciliation, activations by guardrail, PII detections, calibration bands, and group variation by tier, region, and language fluency.

Human-review metrics such as hallucination rate and satisfaction proxy must be clearly separated from automated metrics; the harness should emit a sampling file for review rather than fabricate those scores.

Definition of done: one command processes the full locally available evaluation set twice without manual intervention or dropped tickets.

### Phase 10 - Tests, Failure Injection, and CI

Create a layered pytest suite:

- Unit tests for each pure component.
- Contract tests for all data models.
- Retrieval tests against a small deterministic corpus fixture.
- Pipeline integration tests with a fake LLM client.
- API tests using FastAPI's test client.
- Harness test over a small mixed-channel fixture.
- Failure injection for empty retrieval, timeout, outage, HTTP 429, invalid model JSON, malformed ticket, database lock, and guardrail block.
- Reconciliation test: inputs equal terminal outcomes equal `ticket_runs` rows.
- Determinism test: same fixture twice produces the same classification/routing outcome.
- Secret scanning and import/startup smoke tests.

CI must use the same documented test command and must not need a live API key. Live-provider tests should be explicitly marked and excluded from the default suite.

Definition of done:

```powershell
python -m pytest tests -v
```

passes from a clean checkout in CI.

### Phase 11 - Documentation, Traceability, and Release Rehearsal

Complete:

- `README.md`: exact Python version, setup, configuration, bootstrap, API, harness, tests, output descriptions, troubleshooting, and no hidden manual steps.
- `docs/architecture.md`: components, data flow, thresholds, fallbacks, kill switch, and trade-offs.
- `prompts/README.md`: prompt ID, version, purpose, requirement IDs, inputs/outputs, change history, and evaluation notes.
- A requirement-to-code-to-test traceability table.
- AI-generated code attribution and AI-use declaration inputs for the report.
- Risk register, fairness audit, incident procedure, and kill-switch procedure in the required student workbooks/framework.

Release rehearsal:

1. Clone into a fresh directory.
2. Follow the README literally.
3. Bootstrap without pre-existing storage.
4. Submit one ticket from each channel.
5. Demonstrate automatic response, escalation, and guardrail block.
6. Disconnect the provider and confirm graceful escalation.
7. Run the full evaluation command unattended.
8. Reconcile results, terminal records, and decision logs.
9. Run the documented test command.
10. Scan the checkout and Git history for secrets.

## 7. Acceptance Criteria Traceability

| Criterion | Primary implementation | Proof |
|---|---|---|
| A1 clean checkout | pinned dependencies, bootstrap, README | fresh-clone rehearsal and CI |
| A2 four channels | `models.py`, `ingest.py` | four channel fixtures |
| A3 classification | `classify.py` | per-class report, confidence and alternatives assertions |
| A4 retrieval | `retrieve.py`, corpus index | expected-doc and passage-resolution tests |
| A5 deterministic routing | `route.py` | repeat-input determinism test |
| A6 grounded citations | `generate.py`, citation guardrail | citation resolution tests |
| A7 blocking guardrail | `guardrails.py` | engineered blocking fixtures |
| A8 persistent decisions | `logging_store.py` | reconciliation and schema tests |
| A9 unattended full run | `evaluation/harness.py` | full-file run with no intervention |
| A10 metrics report | `evaluation/harness.py`, `metrics.py` | output schema and calculation tests |
| A11 failure handling | pipeline fallbacks, retry/cache layer | injected failure suite |
| A12 one test command | pytest and CI | fresh-checkout command passes |

## 8. Recommended Build Order and Milestones

### Milestone 1 - End-to-end skeleton

Phases 0-4 plus basic logging. Use a stub generator and force escalation when generation is unavailable. Prove all four channels and deterministic routing.

### Milestone 2 - Gate-capable vertical slice

Phases 5-9 with one real guardrail, provider retry/fallback, and automatic metrics. Run all 500 development tickets or a fixed held-out split unattended.

### Milestone 3 - Reliability and quality

Phase 10, remaining guardrails, calibration, retrieval threshold tuning, and failure injection. Obtain and run the official validation set when available.

### Milestone 4 - Submission readiness

Phase 11, fairness analysis, governance artifacts, clean-clone rehearsal, prompt/requirement traceability, report inputs, and demonstration preparation.

## 9. Scope Control

Protect these items if time is short:

1. Unattended harness and no dropped tickets.
2. Safe deterministic routing and escalation.
3. Traceable retrieval and real citations.
4. Blocking guardrails.
5. Complete persistent logs and automatic metrics.
6. Clean-checkout setup and tests.

Defer UI work, sophisticated multi-agent behaviour, PostgreSQL, high-concurrency processing, and elaborate dashboards until all twelve acceptance criteria pass. FastAPI's generated documentation is sufficient as the first interface.

## 10. Immediate Next Actions

1. Place the project under Git and correct `.gitignore` before further generated files are created.
2. Add `.env.example` and resolve a fully pinned compatible dependency set.
3. Copy/reference development tickets and ground truth; request the missing validation set.
4. Implement typed contracts and four-channel ingestion with tests.
5. Refactor the setup experiment into an idempotent bootstrap module.
6. Build retrieval with stable passage IDs and an empty-result threshold.
7. Train and calibrate the local classifiers.
8. Implement deterministic routing and persistent decisions.
9. Create the harness immediately after the first vertical slice and exercise it continuously.
