"""Analyst copilot: cited answers from approved documents and read-only tools.

Plan: docs/document_ai_copilot_plan_v1.md.

Pipeline for each question:
1. Injection screen on the question (refuse if instruction-like text is found).
2. Deterministic tool routing; each tool is checked against the caller's API permissions (deny explicitly).
3. BM25 retrieval over passages the caller's role may read (access filtering happens before retrieval).
4. Evidence-sufficiency guard: specific terms in the question (codes, versions, years, numbers) must appear in the
   evidence, otherwise abstain.
5. Answerer (LLM adapter). Provider failure falls back to plain retrieval, labelled as such (RB-06).
6. Verifier: every citation must point to supplied evidence with a verbatim quote, every number in the answer must
   appear in a cited quote, and the answer must be lexically supported by its quotes. Failing answers are withdrawn
   and become abstentions.
Every query is written to the audit chain (question hash, outcome, cited passages, provider).

Read-only by construction: tools only read the audit store, the champion pointer and the latest monitoring report.
There is no SQL, write or policy/model-change capability.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass

from crisk.copilot.corpus import Corpus, Passage, Retriever
from crisk.llm.adapter import AnswerDraft, Citation, Evidence, LLMError
from crisk.llm.text import content_tokens, display, norm, numbers, specific_terms, term_set
from crisk.security.untrusted import clean_text, screen_text
from crisk.utils import sha256_text, utc_now

MAX_QUESTION = 1000
TOP_K = 5
MIN_SUPPORT = 0.5
LABEL = ("Answers come only from approved, versioned project documents and read-only tools, with citations. "
         "Not advice; check the cited sources.")
_SCORE_ID = re.compile(r"\b[0-9a-f]{32}\b")
_MONITORING = re.compile(r"\b(latest|current|recent)\b.{0,20}\b(monitoring|alerts?)\b|\bmonitoring (status|summary)\b",
                         re.IGNORECASE)
_CHAMPION = re.compile(r"\b(current|serving|live)\b.{0,10}\b(champion|model)\b|\bmodel card\b", re.IGNORECASE)


@dataclass
class Caller:
    name: str
    role: str
    can: callable  # permission -> bool


# ---- read-only tools ---------------------------------------------------------------------------------------------
class Tools:
    def __init__(self, cfg=None, store=None):
        self.cfg, self.store = cfg, store

    def route(self, question: str) -> list[tuple[str, str, str | None]]:
        """(tool, permission, argument) for each tool the question needs."""
        out = []
        m = _SCORE_ID.search(question.lower())
        if m:
            out.append(("score_lookup", "score:read", m.group(0)))
        if _MONITORING.search(question):
            out.append(("monitoring_summary", "monitoring:read", None))
        if _CHAMPION.search(question):
            out.append(("champion_card", "model:read", None))
        return out

    def run(self, tool: str, arg: str | None) -> Evidence | None:
        return getattr(self, tool)(arg)

    def score_lookup(self, score_id: str) -> Evidence | None:
        rec = self.store.get_score(score_id) if self.store else None
        if rec is None:
            return None
        resp = json.loads(rec["response"])
        reasons = "; ".join(f"{r.get('code')} {r.get('text') or r.get('feature')}"
                            for r in resp.get("reason_codes", []))
        pd_txt = f"{rec['pd']:.4f}" if rec["pd"] is not None else "not produced"
        text = (f"Score {score_id} (status {rec['status']}) has PD {pd_txt} and risk decile "
                f"{resp.get('risk_decile')}; approved use {resp.get('approved_use')}.\n"
                f"Model {rec['model_id']}, scored at {rec['scored_at']}.\nReason codes: {reasons or 'none'}.")
        return Evidence(f"tool:score:{score_id}", text, "tool.score_lookup", "Scoring audit record", rec["scored_at"])

    def monitoring_summary(self, _: str | None = None) -> Evidence | None:
        path = self.cfg.reports_dir / "monitoring" / "latest.json" if self.cfg else None
        if path is None or not path.exists():
            return None
        r = json.loads(path.read_text(encoding="utf-8"))
        lines = [f"Monitoring report generated {r['generated_at']} for model {r['model_id']}."]
        for a in r.get("alerts", []):
            lines.append(f"Alert {a['alert_id']}: severity {a['severity']}, {a['periods']} periods "
                         f"({a['first_period']} to {a['last_period']}); action {a['action']}.")
        if len(lines) == 1:
            lines.append("No alerts.")
        return Evidence("tool:monitoring:latest", "\n".join(lines), "tool.monitoring_summary",
                        "Latest monitoring report",
                        r["generated_at"][:10])

    def champion_card(self, _: str | None = None) -> Evidence | None:
        from crisk.models.registry import champion

        ch = champion(self.cfg) if self.cfg else None
        if not ch:
            return None
        text = (f"Current champion model {ch['model_id']}.\nApproved use: {ch.get('approved_use')}.\n"
                f"Promoted at {ch.get('promoted_at')} by reviewer {ch.get('reviewer')}.")
        return Evidence("tool:champion", text, "tool.champion_card", "Champion pointer",
                        (ch.get("promoted_at") or "")[:10])


# ---- verifier ----------------------------------------------------------------------------------------------------
def verify(draft: AnswerDraft, evidence: dict[str, Evidence], question: str) -> tuple[bool, str | None]:
    if draft.abstain:
        return True, None
    if not draft.answer.strip():
        return False, "empty_answer"
    if not draft.citations:
        return False, "no_citations"
    for c in draft.citations:
        ev = evidence.get(c.evidence_id)
        if ev is None:
            return False, f"unknown_evidence:{c.evidence_id}"
        if len(norm(c.quote)) < 8 or norm(c.quote) not in norm(ev.text):
            return False, f"quote_not_verbatim:{c.evidence_id}"
    quoted = " ".join(c.quote for c in draft.citations)
    missing = numbers(draft.answer) - numbers(quoted)
    if missing:
        return False, f"unsupported_number:{sorted(missing)[0]}"
    ans = set(content_tokens(draft.answer))
    if ans:
        support = len(ans & set(content_tokens(quoted + " " + question))) / len(ans)
        if support < MIN_SUPPORT:
            return False, f"low_support:{support:.2f}"
    return True, None


# ---- copilot -----------------------------------------------------------------------------------------------------
class Copilot:
    def __init__(self, corpus: Corpus, llm, tools: Tools | None = None, store=None):
        self.corpus, self.llm, self.tools, self.store = corpus, llm, tools or Tools(), store
        self._retrievers: dict[str, Retriever] = {}

    def retriever(self, role: str) -> Retriever:
        if role not in self._retrievers:
            self._retrievers[role] = Retriever(self.corpus.for_role(role))
        return self._retrievers[role]

    def _evidence(self, p: Passage) -> Evidence:
        return Evidence(p.pid, p.text, p.source_id, p.section, p.as_of)

    def _citation(self, c: Citation, ev: Evidence) -> dict:
        src = self.corpus.sources.get(ev.source_id)
        p = next((x for x in self.corpus.passages if x.pid == c.evidence_id), None)
        return {"evidence_id": c.evidence_id, "source_id": ev.source_id, "path": src["path"] if src else None,
                "version": src["version"][:12] if src else None, "as_of": ev.as_of,
                "lines": f"{p.line_start}-{p.line_end}" if p else None, "quote": display(c.quote)}

    def ask(self, question: str, caller: Caller) -> dict:
        qid = uuid.uuid4().hex
        q = clean_text(question).strip()[:MAX_QUESTION]
        base = {"query_id": qid, "question": q, "role": caller.role, "answered_at": utc_now(), "label": LABEL,
                "llm": self.llm.name, "answer": "", "citations": [], "as_of": None, "retrieved": [], "tools": []}
        hits = screen_text(q)
        if hits:
            return self._finish(base, "refuse", f"prompt_injection_suspected:{hits[0].pattern}", caller)
        evidence: list[Evidence] = []
        for tool, perm, arg in self.tools.route(q):
            if not caller.can(perm):
                return self._finish(base, "deny", f"permission_denied: role '{caller.role}' lacks '{perm}' "
                                                  f"needed for {tool}", caller)
            ev = self.tools.run(tool, arg)
            base["tools"].append({"tool": tool, "result": "ok" if ev else "no_data"})
            if ev:
                evidence.append(ev)
        ranked = self.retriever(caller.role).search(q, TOP_K)
        base["retrieved"] = [{"evidence_id": p.pid, "score": s} for p, s in ranked]
        evidence += [self._evidence(p) for p, _ in ranked]
        if not evidence:
            return self._finish(base, "abstain", "no_evidence_available_to_role", caller)
        ev_text = norm(" ".join(e.text + " " + e.title for e in evidence))
        ev_terms = term_set(ev_text)
        missing = sorted(t for t in specific_terms(q) if t not in ev_terms and t.replace("-", "") not in ev_terms)
        missing += sorted(n for n in numbers(q) if n not in numbers(ev_text))
        if missing:
            return self._finish(base, "abstain", f"evidence_missing_for:{','.join(missing)}", caller)
        by_id = {e.evidence_id: e for e in evidence}
        try:
            draft = self.llm.answer(q, evidence)
        except LLMError as e:  # provider failure: plain retrieval, labelled (RB-06)
            top = evidence[0]
            base.update(answer=display(top.text), as_of=top.as_of, fallback=f"plain_retrieval ({type(e).__name__})",
                        citations=[self._citation(Citation(top.evidence_id, ln), top) for ln in _lines(top.text)])
            return self._finish(base, "answer", "llm_unavailable_fallback", caller)
        ok, why = verify(draft, by_id, q)
        if not ok:
            if self.store is not None:
                self.store.append_event(actor=f"{caller.name} ({caller.role})", action="copilot_answer_withdrawn",
                                        entity_type="copilot_query", entity_id=qid, details={"reason": why})
            return self._finish(base, "abstain", f"answer_withdrawn:{why}", caller)
        if draft.abstain:
            return self._finish(base, "abstain", draft.reason or "insufficient_evidence", caller)
        cites = [self._citation(c, by_id[c.evidence_id]) for c in draft.citations]
        dates = [c["as_of"] for c in cites if c["as_of"] and c["as_of"][:2] == "20"]
        base.update(answer=draft.answer, citations=cites, as_of=max(dates) if dates else None)
        return self._finish(base, "answer", None, caller)

    def _finish(self, base: dict, outcome: str, reason: str | None, caller: Caller) -> dict:
        base.update(outcome=outcome, reason=reason)
        if outcome in ("abstain", "refuse", "deny"):
            base["answer"] = {"abstain": "I can't answer this from the sources available to you.",
                              "refuse": "This question looks like an instruction to the assistant rather than a "
                                        "question, so it was not processed. Rephrase it as a question.",
                              "deny": "Your role does not have access to the data this question needs."}[outcome]
            base["citations"] = []
        base["usage"] = self.llm.usage.as_dict()
        if self.store is not None:
            self.store.append_event(actor=f"{caller.name} ({caller.role})", action="copilot_query",
                                    entity_type="copilot_query", entity_id=base["query_id"],
                                    details={"question_sha256": sha256_text(base["question"]), "outcome": outcome,
                                             "reason": reason, "llm": self.llm.name,
                                             "cited": [c["evidence_id"] for c in base["citations"]]})
        return base


def _lines(text: str) -> list[str]:
    """Each non-empty line is quoted separately (a table-row passage holds two non-adjacent source lines)."""
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


def plain_retrieval(corpus: Corpus, role: str, question: str, retriever: Retriever | None = None) -> dict:
    """Baseline B0: top-1 permitted passage as the answer; no abstention, screen or tools."""
    r = retriever or Retriever(corpus.for_role(role))
    ranked = r.search(question, TOP_K)
    out = {"outcome": "answer", "reason": None, "answer": "", "citations": [], "role": role,
           "retrieved": [{"evidence_id": p.pid, "score": s} for p, s in ranked], "llm": "none (plain retrieval)"}
    if ranked:
        p = ranked[0][0]
        src = corpus.sources[p.source_id]
        out["answer"] = display(p.text)
        out["citations"] = [{"evidence_id": p.pid, "source_id": p.source_id, "path": p.path,
                             "version": src["version"][:12], "as_of": p.as_of, "lines": f"{p.line_start}-{p.line_end}",
                             "quote": display(ln)} for ln in _lines(p.text)]
    return out
