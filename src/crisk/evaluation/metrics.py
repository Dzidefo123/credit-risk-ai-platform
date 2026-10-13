"""Discrimination and calibration metrics for binary PD models.

Calibration intercept ("calibration-in-the-large") is estimated with the logit of the
prediction as a fixed offset; calibration slope is the coefficient on logit(p) in an
unpenalised logistic recalibration model. Ideal values: intercept 0, slope 1.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score, roc_curve

EPS = 1e-12


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype="float64"), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype="float64")))


def fit_logistic_1d(x: np.ndarray, y: np.ndarray, offset: np.ndarray | None = None,
                    fit_slope: bool = True, iters: int = 100) -> tuple[float, float]:
    """Unpenalised logistic regression y ~ a + b*x (+ offset) by damped Newton-Raphson.

    Step-halving guarantees the log-likelihood never decreases. Plain Newton overshoots and diverges
    when the offset predictions are far from the observed rate (found in the v3 backtest: a window
    with ~3% defaults against ~0.2% predicted PDs drove the intercept to -1.6e8).
    """
    x = np.asarray(x, dtype="float64")
    y = np.asarray(y, dtype="float64")
    off = np.zeros_like(x) if offset is None else np.asarray(offset, dtype="float64")
    X = np.column_stack([np.ones_like(x), x]) if fit_slope else np.ones((len(x), 1))
    beta = np.zeros(X.shape[1])

    def loglik(b: np.ndarray) -> float:
        z = X @ b + off
        return float(np.sum(y * z - np.logaddexp(0.0, z)))

    ll = loglik(beta)
    for _ in range(iters):
        p = sigmoid(X @ beta + off)
        w = np.clip(p * (1 - p), 1e-12, None)
        grad = X.T @ (y - p)
        hess = X.T @ (X * w[:, None]) + 1e-10 * np.eye(X.shape[1])
        step = np.linalg.solve(hess, grad)
        t = 1.0
        while t > 1e-10:
            cand = beta + t * step
            ll_cand = loglik(cand)
            if ll_cand >= ll - 1e-12:
                break
            t *= 0.5
        beta, ll = cand, ll_cand
        if np.max(np.abs(t * step)) < 1e-10:
            break
    return (float(beta[0]), float(beta[1])) if fit_slope else (float(beta[0]), 1.0)


def calibration_intercept_slope(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    lp = logit(p)
    intercept, _ = fit_logistic_1d(lp, y, offset=lp, fit_slope=False)
    _, slope = fit_logistic_1d(lp, y)
    return intercept, slope


def reliability_table(y: np.ndarray, p: np.ndarray, bins: int = 10) -> list[dict]:
    """Quantile bins of predicted PD with observed default rate."""
    y = np.asarray(y)
    p = np.asarray(p)
    if len(p) == 0:
        return []
    edges = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, max(len(edges) - 2, 0))
    rows = []
    for b in range(max(len(edges) - 1, 1)):
        m = idx == b
        if m.sum() == 0:
            continue
        rows.append({"bin": b, "n": int(m.sum()), "mean_pd": float(p[m].mean()), "observed_rate": float(y[m].mean()),
                     "defaults": int(y[m].sum())})
    return rows


def bootstrap_ci(y: np.ndarray, p: np.ndarray, n_boot: int = 200, seed: int = 0, alpha: float = 0.05) -> dict:
    """Loan-level bootstrap CIs for ROC-AUC, Brier and observed/expected ratio.

    Loan-level resampling ignores common shocks within cohorts, so these intervals understate
    out-of-time uncertainty; report them with that caveat.
    """
    y = np.asarray(y).astype(int)
    p = np.clip(np.asarray(p, dtype="float64"), EPS, 1 - EPS)
    rng = np.random.default_rng(seed)
    n = len(y)
    stats = {"roc_auc": [], "brier": [], "observed_expected_ratio": []}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb, pb = y[idx], p[idx]
        if 0 < yb.sum() < n:
            stats["roc_auc"].append(roc_auc_score(yb, pb))
        stats["brier"].append(float(np.mean((pb - yb) ** 2)))
        stats["observed_expected_ratio"].append(float(yb.mean() / pb.mean()))
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {k: {"low": float(np.percentile(v, lo)), "high": float(np.percentile(v, hi)), "n_boot": len(v)}
            for k, v in stats.items() if v}


def binary_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    y = np.asarray(y).astype(int)
    p = np.clip(np.asarray(p, dtype="float64"), EPS, 1 - EPS)
    n, events = len(y), int(y.sum())
    out: dict = {"n": n, "events": events, "prevalence": events / n if n else None,
                 "mean_pd": float(p.mean()) if n else None,
                 "observed_expected_ratio": float((events / n) / p.mean()) if n else None}
    if n == 0:
        return out
    out["brier"] = float(brier_score_loss(y, p))
    out["log_loss"] = float(log_loss(y, p, labels=[0, 1]))
    if 0 < events < n:
        auc = float(roc_auc_score(y, p))
        fpr, tpr, _ = roc_curve(y, p)
        out.update(roc_auc=auc, gini=2 * auc - 1, ks=float(np.max(tpr - fpr)),
                   pr_auc=float(average_precision_score(y, p)))
        ci, cs = calibration_intercept_slope(y, p)
        out.update(calibration_intercept=ci, calibration_slope=cs)
    else:
        out.update(roc_auc=None, gini=None, ks=None, pr_auc=None, calibration_intercept=None,
                   calibration_slope=None, note="single class: discrimination undefined")
    out["reliability"] = reliability_table(y, p)
    return out
