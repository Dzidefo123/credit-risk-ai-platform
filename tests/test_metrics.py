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


def test_single_class_is_reported_not_crashed():
    m = binary_metrics(np.zeros(10), np.full(10, 0.1))
    assert m["roc_auc"] is None and "single class" in m["note"]
