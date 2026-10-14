# Document intelligence and analyst copilot v1 (Phases G and H)

**Evidence class: SYNTHETIC documents; pipeline evaluation with a mocked LLM.** Pre-registered in
[../document_ai_copilot_plan_v1.md](../document_ai_copilot_plan_v1.md) (commit `36df48c`, before any code). The
system was frozen at commit `3a473bb` and each held-out split was then evaluated **once** (audit actions
`docai_heldout_evaluated` and `copilot_heldout_evaluated`). Results with `MockLLM` measure the pipeline: retrieval,
access control, guards, verifiers and workflow. **They are not evidence of LLM quality.** No real LLM has been
called.

## Verdict

| Area | Result |
|---|---|
| Safety guarantees (permissions, injection, citation validity, unsupported claims, feature gate) | **All met, on dev and held-out** |
| Document extraction on seen layouts | **Met** (precision and recall 1.0; all defects detected) |
| Document extraction on the unseen prose layout | **Failed safely**: nothing extracted, nothing guessed, every such packet routed to review |
| Copilot answer quality with the extractive mock | **Failed** on held-out (accuracy 0.27; dev 0.80): the mock answerer over-fitted dev phrasing |

## Phase G: documents (40 fictional packets; held-out = 12)

| Metric | Criterion | Dev (28) | **Held-out (12)** | |
|---|---|---|---|---|
| Field precision | ≥ 0.98 | 1.00 (529) | **1.00** (201) | pass |
| Field recall, seen layouts | ≥ 0.95 | 1.00 | **1.00** (201/201) | pass |
| Field recall, unseen prose layout C | reported | n/a | **0.00** (0/24) | — |
| Reference correctness (quote verbatim on the cited page, page = answer key) | 100% | 100% | **100%** | pass |
| Injection detected | 100% | 4/4 | **1/1** | pass |
| Missing field detected | 100% | 3/3 | **3/3** | pass |
| Arithmetic error detected | ≥ 0.95 | 18/18 | **3/3** | pass |
| Cross-document mismatch detected | ≥ 0.95 | 6/6 | **3/5 (0.60)** | **fail** |
| False-flag rate, defect-free packets | ≤ 0.10 | 0/8 | **1/3 (0.33)** | **fail** |
| Unvalidated extractions passing the feature gate | 0 | 0 | **0** | pass |

**Both failures come from the four layout-C packets** (prose application, deliberately absent from development). The
rule extractor found no labelled fields there, so every application field was flagged missing. As a result, the
income check could not run (2 mismatches missed) and the one defect-free C packet was flagged. All four went to
`needs_review` and none was validated. Because validation re-runs after each reviewer correction, the mismatches
surface once a reviewer enters the application values. On the eight seen-layout packets there were no misses and no
false flags.

Held-out status mix: 2 validated, 9 needs review, 1 quarantined (the injected instruction). Defect counts in
held-out are small (1 injection, 3 arithmetic), so its recall figures are coarse.

## Phase H: copilot (48 pre-registered questions; held-out = 20)

C1 = copilot with `MockLLM` (deterministic extractive stand-in). B0 = plain retrieval (top-1 permitted passage,
no abstention, screen or tools). Both use the same role-filtered BM25 index.

| Metric | Criterion (C1) | C1 dev | B0 dev | **C1 held-out** | **B0 held-out** |
|---|---|---|---|---|---|
| Permission enforcement | 100% | 100% | 67% | **100%** | 100% (n = 1) |
| Injection resistance | 100% | 100% | 75% | **100%** | 100% (n = 2) |
| Citation validity (quotes verbatim in the cited source version) | 100% | 100% | 100% | **100%** | 100% |
| Unsupported-claim rate | 0% | 0% | 0% | **0%** | 0% |
| Retrieval hit@5 | ≥ 0.80 | 1.00 | 1.00 | **1.00** | 1.00 |
| Answer accuracy (answerable + numerical) | ≥ 0.70 | 0.80 | 0.87 | **0.27 — fail** | 0.45 |
| Abstention recall (missing evidence) | ≥ 0.60 | 0.80 | 0.00 | **1.00** | 0.20 |
| False-abstention rate | ≤ 0.25 | 0.07 | 0.00 | **0.45 — fail** | 0.00 |
| Accuracy vs B0 | ≥ B0 − 0.10 | −0.07 | — | **−0.18 — fail** | — |
| Citation correctness / numerical fidelity | reported | 0.87 / 1.00 | 0.93 / 1.00 | 0.55 / 0.20 | 0.82 / 0.40 |

**Reading the result.**

1. **The structural controls held on unseen questions.**
   - No restricted passage was retrieved for a role that may not read it, and no restricted string appeared.
   - Every injection attempt was refused, and every emitted quote is verbatim in the cited source version.
   - No answer contained a number absent from its citations.
   - These are enforced in code, so they do not depend on the answerer.
2. **Retrieval is not the bottleneck.** The right source was in the top five for every held-out question.
3. **The extractive mock answerer is.**
   - Of the eight held-out misses, five are abstentions at a coverage just below the 0.5 threshold, which was tuned on
     dev (dev: 1 of 15 answerable questions wrongly abstained; held-out: 5 of 11). Three quote the wrong passage
     (A14, N08, N12).
   - It "answers" by quoting a single sentence or table row, which is brittle when a question is phrased differently
     from the text. This is the component a real LLM replaces.
   - The low C1 accuracy is therefore a negative result for the mock, not a measure of what an LLM would do.
4. **The baseline shows why the guards matter.**
   - B0 answers everything. On dev it cited a restricted-to-role source zero times only because filtering precedes
     retrieval, yet it still gave an answer to a tool question that should have been denied.
   - On held-out it answered 4 of 5 missing-evidence questions with irrelevant passages.
5. **A content hazard found.** For N08 ("default rates of approved and declined loans after the decile refresh"), both
   systems quoted the *scenario test* figures (0.28% vs 2.59%) instead of the *applied* result (0.24% vs 2.85%) in
   the same document. A citation makes this checkable, but it shows that documents holding both scenario and final
   figures need clearer labelling for retrieval.

## What was built

- **Documents** (`crisk.docai`):
  - A generator of 40 fictional packets with answer keys.
  - An OCR-tolerant rule extractor with page and quote references.
  - An LLM extractor behind the adapter, with a grounding check (the quote must be on the cited page and contain the
    value; otherwise the field is dropped).
  - Arithmetic, cross-document and completeness checks, plus an untrusted-content screen.
  - An append-only review workflow (correct / accept / reject, each with a rationale and audited).
  - A feature gate: only `validated` packets yield verified inputs, and PD request schemas forbid document fields.
- **Copilot** (`crisk.copilot`):
  - A versioned corpus (content SHA-256 and last-commit as-of date) with role filtering before retrieval, and BM25
    over paragraph and table-row passages.
  - Permission-checked read-only tools: score lookup, latest monitoring summary, champion card.
  - An evidence-sufficiency guard, the verifier, a labelled plain-retrieval fallback when the provider fails, and an
    audit record per query (question hash, not text).
- **LLM adapter** (`crisk.llm`): `MockLLM` by default, plus an optional Anthropic backend (`uv sync --group llm`,
  key from the environment only):
  - `claude-opus-5-5` with structured JSON outputs.
  - Server-side refusal fallback.
  - Error mapping to unavailable or refused.
  - Usage accounting.
  - Request shape and error handling are tested against a fake SDK; no live call has been made.
- **API, UI, CLI:**
  - API: `/v1/documents…` and `/v1/copilot/ask`, with new permissions `document:create|read|review` and
    `copilot:ask`.
  - UI: Documents and Copilot pages.
  - CLI: `crisk docai-generate | docai-eval | copilot-ask | copilot-eval`.

## Limitations

- Synthetic documents are text pages, not scanned images. Layouts, defects and injection templates are few, and the
  rule extractor shares the generator's label vocabulary (hence perfect seen-layout scores).
- The injection screen is pattern-based. It is a first line of defence; the structural controls (no write tools,
  filtering before retrieval, verification, the feature gate) carry the guarantees.
- The question set was written by the system's author. Held-out has small permission and injection counts (1 and 2).
- The lexical-support check catches unsupported wording only roughly. Numbers and quotes are checked strictly.

## Next steps (each needs its own pre-registration)

1. **Real-LLM evaluation:** run the same question set and document packets with the Anthropic backend. The held-out
   items are unseen by any LLM; the run is recorded separately and labelled with the model. This needs the user's API
   key and approval of the token cost.
2. **v2 document set** with new layouts and a fresh held-out split, if the grounded LLM extractor is to be evaluated
   on unseen layouts.
3. Clearer "scenario" vs "applied" labelling in result documents (N08).

## Addendum A results: real LLM (`claude-sonnet-5-5`, effort medium), 2026-10-14

Pre-registered in plan Addendum A (commit `8013ce2`) before any call. Run sequence: dev, then held-out once per
system (audited). No prompt, threshold or verifier changes were made between dev and held-out. The corpus included
the post-D-039 edits to runbooks and the security document.

| Metric | Criterion | C1-LLM dev | **C1-LLM held-out** | C1-mock held-out | B0 held-out |
|---|---|---|---|---|---|
| Permission enforcement | 100% | 100% | **100%** | 100% | 100% |
| Injection resistance | 100% | 100% | **100%** | 100% | 100% |
| Citation validity | 100% | 100% | **100%** | 100% | 100% |
| Unsupported-claim rate | 0% | 0% | **0%** | 0% | 0% |
| Retrieval hit@5 | ≥ 0.80 | 1.00 | **1.00** | 1.00 | 1.00 |
| Answer accuracy | ≥ 0.70 | 0.80 | **0.73 — pass** | 0.27 | 0.45 |
| Abstention recall | ≥ 0.60 | 1.00 | **1.00** | 1.00 | 0.20 |
| False-abstention rate | ≤ 0.25 | 0.20 | **0.18 — pass** | 0.45 | 0.00 |
| Accuracy vs B0 | ≥ B0 − 0.10 | — | **+0.27 — pass** | −0.18 | — |

| Documents (D1-LLM, grounded extractor) | Criterion | Dev | **Held-out** | Rule extractor held-out |
|---|---|---|---|---|
| Field precision | ≥ 0.98 | 1.00 | **1.00** | 1.00 |
| Field recall, seen layouts | ≥ 0.95 | 1.00 | **1.00** | 1.00 |
| Field recall, unseen prose layout | reported | — | **1.00 (24/24)** | 0.00 |
| Reference correctness | 100% | 100% | **100%** | 100% |
| Defect recall (injection / missing / arithmetic / cross-document) | 100 / 100 / ≥ 0.95 / ≥ 0.95 | all 1.00 | **all 1.00** | cross-document 0.60 |
| False-flag rate, defect-free packets | ≤ 0.10 | 0.00 | **0.00** | 0.33 |
| Unvalidated packets through the gate | 0 | 0 | **0** | 0 |

**All pre-registered criteria are met on held-out by the real-LLM systems.** The grounded LLM extractor read the
unseen prose layout correctly (24/24 fields, each with a verbatim quote on the cited page), which removes both
rule-extractor failures.

**Remaining copilot misses (3 of 11), all visible and checkable:**
- **A09 and N10 were withdrawn by the verifier.**
  - In N10 the model added "95% CI" to its answer, which its quotes did not contain.
  - In A09 its wording was too loosely supported.
  - On dev, all three misses were the verifier rejecting numbers the model had *repeated from the question* (e.g.
    "2023–25").
  - Planned v2 change: allow numbers that appear in the question, then re-evaluate on new questions.
- **N08** quoted the scenario figures (0.28% vs 2.59%) instead of the applied ones (0.24% vs 2.85%), exactly as the
  mock and B0 did. This confirms a document-labelling problem rather than a model one.

**Cost and usage:**

| Run | Calls | Input tokens | Output tokens | Cost |
|---|---|---|---|---|
| Copilot dev | 21 | 30,558 | 2,906 | $0.09 |
| Documents dev | 84 | 142,551 | 25,863 | $0.54 |
| Copilot held-out | 15 | 22,368 | 2,418 | $0.07 |
| Documents held-out | 36 | 60,563 | 11,757 | $0.24 |
| **Total** | | | | **≈ $0.94** |

Questions refused, denied or abstained by the guards never reach the model.

**Scope:** one model at one effort level, on a small author-written question set and 12 held-out fictional packets.
This shows the pipeline works with a real LLM under these controls; it is not a general benchmark.
