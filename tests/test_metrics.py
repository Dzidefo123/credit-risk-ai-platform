import numpy as np
import pytest

from crisk.evaluation.metrics import binary_metrics, calibration_intercept_slope, sigmoid


def test_perfectly_calibrated_predictions_have_zero_intercept_unit_slope():
    rng = np.random.default_rng(0)
    p = sigmoid(rng.normal(-3, 1, 200_000))
    y = rng.random(len(p)) < p
    a, b = calibration_intercept_slope(y, p)
    assert a == pytest.approx(0, abs=0.03)
    assert b == pytest.approx(1, abs=0.03)


def test_overconfident_predictions_have_slope_below_one():
    rng = np.random.default_rng(1)
    z = rng.normal(-3, 1, 100_000)
    y = rng.random(len(z)) < sigmoid(z)
    _, b = calibration_intercept_slope(y, sigmoid(2 * z + 3))
    assert b == pytest.approx(0.5, abs=0.05)


def test_metric_values_on_known_case():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.4, 0.35, 0.8])
    m = binary_metrics(y, p)
    assert m["roc_auc"] == pytest.approx(0.75)
    assert m["gini"] == pytest.approx(0.5)
    assert m["ks"] == pytest.approx(0.5)
    assert m["brier"] == pytest.approx(np.mean((p - y) ** 2))


def test_offset_fit_converges_when_far_from_truth():
    """Regression: plain Newton diverged (intercept -1.6e8) when predictions were ~15x too low."""
    from crisk.evaluation.metrics import fit_logistic_1d, logit

    rng = np.random.default_rng(3)
    p0 = np.full(50_000, 0.002)
    y = (rng.random(len(p0)) < 0.03).astype(float)
    a, _ = fit_logistic_1d(logit(p0), y, offset=logit(p0), fit_slope=False)
    assert a == pytest.approx(logit(np.array([y.mean()]))[0] - logit(np.array([0.002]))[0], abs=1e-6)
    a2, b2 = fit_logistic_1d(logit(p0) + rng.normal(0, 0.5, len(p0)), y)
    assert np.isfinite(a2) and np.isfinite(b2)


def test_single_class_is_reported_not_crashed():
    m = binary_metrics(np.zeros(10), np.full(10, 0.1))
    assert m["roc_auc"] is None and "single class" in m["note"]
