"""Versioned credit policy engine (Phase E), separate from the PD model.

A policy is a YAML list of rules ``{id, when, outcome, reason}``. ``when`` is a small boolean expression over
application fields plus ``risk_decile``, ``score_status`` and ``pd``. It is parsed with a whitelisted AST evaluator
(comparisons, and/or/not, constants, names); arbitrary code cannot run. A comparison involving a missing (None)
value is False, so missing data never satisfies a threshold. Use ``is None`` rules to route missing data explicitly.

Decision precedence: decline > refer > approve. Every triggered rule is returned with its reason, so the decision is
fully explained by the policy (rule reasons), with the model's reason codes attached as context.
Governance guard: a policy that references ``pd`` is refused unless the serving model is approved for 'pd_level'.
"""

from __future__ import annotations

import ast
import operator
from dataclasses import dataclass
from pathlib import Path

from crisk.config import REPO_ROOT, load_yaml
from crisk.utils import sha256_file

OUTCOME_RANK = {"approve": 0, "refer": 1, "decline": 2}
_CMP = {ast.Lt: operator.lt, ast.LtE: operator.le, ast.Gt: operator.gt, ast.GtE: operator.ge,
        ast.Eq: operator.eq, ast.NotEq: operator.ne}


class PolicyError(ValueError):
    pass


def _compile(expr: str) -> ast.Expression:
    tree = ast.parse(expr, mode="eval")
    allowed = (ast.Expression, ast.BoolOp, ast.And, ast.Or, ast.UnaryOp, ast.Not, ast.Compare, ast.Name, ast.Load,
               ast.Constant, *_CMP, ast.Is, ast.IsNot)
    for node in ast.walk(tree):
        if not isinstance(node, allowed):
            raise PolicyError(f"Disallowed syntax in rule {expr!r}: {type(node).__name__}")
    return tree


def _eval(node: ast.AST, ctx: dict):
    if isinstance(node, ast.Expression):
        return _eval(node.body, ctx)
    if isinstance(node, ast.BoolOp):
        vals = (_eval(v, ctx) for v in node.values)
        return all(vals) if isinstance(node.op, ast.And) else any(vals)
    if isinstance(node, ast.UnaryOp):
        return not _eval(node.operand, ctx)
    if isinstance(node, ast.Name):
        if node.id not in ctx:
            raise PolicyError(f"Unknown field {node.id!r}")
        return ctx[node.id]
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Compare):
        left = _eval(node.left, ctx)
        for op, comp in zip(node.ops, node.comparators, strict=True):
            right = _eval(comp, ctx)
            if isinstance(op, ast.Is):
                ok = left is right
            elif isinstance(op, ast.IsNot):
                ok = left is not right
            elif left is None or right is None:
                ok = False  # missing data never satisfies a threshold
            else:
                ok = _CMP[type(op)](left, right)
            if not ok:
                return False
            left = right
        return True
    raise PolicyError(f"Unsupported node {type(node).__name__}")


def _names(tree: ast.AST) -> set[str]:
    return {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}


@dataclass
class Rule:
    id: str
    when: str
    outcome: str
    reason: str
    tree: ast.Expression


@dataclass
class Policy:
    policy_id: str
    version: str
    effective_from: str
    rules: list[Rule]
    sha256: str
    source: str

    @classmethod
    def load(cls, path: str | Path) -> Policy:
        p = Path(path)
        p = p if p.is_absolute() else REPO_ROOT / p
        raw = load_yaml(p)
        rules, seen = [], set()
        for r in raw["rules"]:
            if r["outcome"] not in OUTCOME_RANK:
                raise PolicyError(f"Rule {r['id']}: outcome must be one of {sorted(OUTCOME_RANK)}")
            if r["id"] in seen:
                raise PolicyError(f"Duplicate rule id {r['id']}")
            seen.add(r["id"])
            rules.append(Rule(r["id"], r["when"], r["outcome"], r["reason"], _compile(r["when"])))
        return cls(raw["policy_id"], str(raw["version"]), str(raw["effective_from"]), rules, sha256_file(p),
                   str(p.relative_to(REPO_ROOT)) if p.is_relative_to(REPO_ROOT) else str(p))

    @property
    def uses_pd(self) -> bool:
        return any("pd" in _names(r.tree) for r in self.rules)

    def evaluate(self, application: dict, score: dict, approved_use: str) -> dict:
        """Deterministic decision for one application and its stored score response."""
        if self.uses_pd and approved_use != "pd_level":
            raise PolicyError(f"Policy {self.policy_id} {self.version} uses PD thresholds, but the model is approved "
                              f"for '{approved_use}' only (PD rules need 'pd_level')")
        ctx = {**application, "risk_decile": score.get("risk_decile"), "score_status": score.get("status"),
               "pd": score.get("pd")}
        triggered = [{"rule_id": r.id, "outcome": r.outcome, "reason": r.reason}
                     for r in self.rules if _eval(r.tree, ctx)]
        decision = max((t["outcome"] for t in triggered), key=OUTCOME_RANK.get, default="approve")
        return {"decision": decision, "policy_id": self.policy_id, "policy_version": self.version,
                "policy_sha256": self.sha256,
                "decisive_rules": [t for t in triggered if t["outcome"] == decision],
                "triggered_rules": triggered,
                "model_reason_codes": score.get("reason_codes", []), "risk_decile": score.get("risk_decile"),
                "score_id": score.get("score_id"), "model_id": (score.get("model") or {}).get("model_id")}
