# Prompt Register

Prompts are stored as versioned text files so changes can be reviewed and
evaluated independently from application code.

## GEN RESPONSE 001

| Field | Value |
| --- | --- |
| Current version | `v1` |
| File | `response_generator_v1.txt` |
| Purpose | Produce a concise, documentation-grounded customer response with machine-verifiable passage citations. |
| Requirement IDs | `A6`, `A7`, `FR-GEN-01`, `FR-SAFE-01` |
| Inputs | Predicted intent, predicted urgency, untrusted ticket fields, and retrieved passages with IDs. |
| Output | One JSON object containing `answer` and a complete `claims` array. |
| Failure behavior | Invalid JSON, missing claims, or invalid citations raise `GenerationError` and fail closed to escalation. |

### Change history

| Version | Change | Evaluation note |
| --- | --- | --- |
| `v1` | Initial grounded-response contract with prompt-injection boundaries, structured JSON, and passage-level citations. | Local structural and guardrail tests pass. Free-model evaluation shows intermittent citation-contract failures; these fail safely and are documented in `docs/architecture.md`. |

The user prompt is assembled at runtime in `src/generate.py` because it contains
ticket-specific and retrieval-specific values. The stable system contract is
kept here as the reviewable prompt artifact.
