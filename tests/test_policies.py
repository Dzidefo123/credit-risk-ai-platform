"""Policy engine: deterministic boundaries, precedence, safe expressions, governance guard, simulation."""

import numpy as np
import pandas as pd
import pytest

from crisk.policies.engine import Policy, PolicyError, _compile
from crisk.policies.simulate import economics, simulate

POLICY = "configs/policies/demo_policy_v1.yaml"
BASE = dict(credit_score=720, orig_dti=36, orig_ltv=80, occupancy_status="P", orig_upb=300000.0)
SCORED = {"status": "scored", "risk_decile": 5, "pd": 0.002, "reason_codes": [], "score_id": "s1"}
USE = "demonstration_rank_ordering"


def decide(app=None, score=None):
    return Policy.load(POLICY).evaluate({**BASE, **(app or {})}, {**SCORED, **(score or {})}, USE)


def test_approve_by_default_and_versioned():
    d = decide()
    assert d["decision"] == "approve" and d["triggered_rules"] == []
    assert d["policy_version"] == "1.0.0" and len(d["policy_sha256"]) == 64


@pytest.mark.parametrize("field,ok,bad,rule", [
    ("credit_score", 620, 619, "E01"), ("orig_dti", 50, 51, "E02"), ("orig_ltv", 97, 98, "E03")])
def test_eligibility_boundaries(field, ok, bad, rule):
    assert decide({field: ok})["decision"] == "approve"
    d = decide({field: bad})
    assert d["decision"] == "decline" and [r["rule_id"] for r in d["decisive_rules"]] == [rule]


def test_compound_rule_and_precedence():
    assert decide({"occupancy_status": "I", "orig_ltv": 86})["decision"] == "decline"     # E04
    assert decide({"occupancy_status": "P", "orig_ltv": 86})["decision"] == "approve"
    assert decide(score={"risk_decile": 9})["decision"] == "refer"                       # R02
    d = decide({"orig_dti": 46}, {"risk_decile": 10})                                   # R01 + R02 -> decline wins
    assert d["decision"] == "decline" and {r["rule_id"] for r in d["triggered_rules"]} == {"R01", "R02"}
    assert decide({"orig_upb": 800001.0})["decision"] == "refer"                         # X01


def test_missing_data_never_satisfies_thresholds_but_is_routed():
    d = decide({"credit_score": None})
    assert d["decision"] == "refer" and [r["rule_id"] for r in d["triggered_rules"]] == ["D02"]
    d = decide(score={"status": "unavailable", "risk_decile": None, "pd": None})
    assert d["decision"] == "refer" and "D01" in {r["rule_id"] for r in d["triggered_rules"]}


def test_rule_expressions_cannot_run_code():
    for bad in ("__import__('os').system('x')", "credit_score.__class__", "[1][0] > 0", "f(1)"):
        with pytest.raises(PolicyError):
            _compile(bad)


def test_pd_rules_refused_without_pd_level_approval(tmp_path):
    p = tmp_path / "pd_policy.yaml"
    p.write_text("policy_id: x\nversion: '1'\neffective_from: '2026-01-01'\nrules:\n"
                 "  - {id: P1, when: 'pd > 0.05', outcome: decline, reason: high pd}\n")
    pol = Policy.load(p)
    with pytest.raises(PolicyError, match="pd_level"):
        pol.evaluate(BASE, SCORED, USE)
    assert pol.evaluate(BASE, {**SCORED, "pd": 0.06}, "pd_level")["decision"] == "decline"


def test_simulation_rates_and_economics_are_consistent():
    rng = np.random.default_rng(0)
    n = 2000
    frame = pd.DataFrame({"credit_score": rng.integers(600, 820, n), "orig_dti": rng.integers(10, 55, n),
                          "orig_ltv": rng.integers(50, 99, n), "occupancy_status": "P",
                          "orig_upb": 300000.0, "risk_decile": rng.integers(1, 11, n), "pd": 0.01,
                          "label": (rng.random(n) < 0.02).astype(int)})
    a = {"lgd_base": 0.25, "net_margin_base": 0.01, "lgd_sensitivity": [0.25], "net_margin_sensitivity": [0.01],
         "operating_cost_per_loan_usd": 1500, "manual_review_cost_per_loan_usd": 250}
    res = simulate(frame, [Policy.load(POLICY)], USE, a)["1.0.0"]
    assert res["approve_rate"] + res["refer_rate"] + res["decline_rate"] == pytest.approx(1.0)
    e = res["economics_base_referred_declined"]
    funded = e["funded_loans"]
    assert e["net_margin_usd"] == pytest.approx(funded * 300000 * 0.01)
    assert e["contribution_usd"] == pytest.approx(e["net_margin_usd"] - e["credit_losses_usd"] - funded * 1500)
    assert economics(frame, np.zeros(n, bool), 0.25, 0.01, a)["funded_loans"] == 0
