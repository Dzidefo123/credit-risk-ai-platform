"""Model bundle: fitted preprocessing + estimator + calibration + explanation metadata."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from crisk.evaluation.metrics import fit_logistic_1d, logit, sigmoid
from crisk.features.origination import FeatureSpec


def make_preprocessor(spec: FeatureSpec, min_frequency: float) -> ColumnTransformer:
    """One transformer per input column so every output column maps back to one feature."""
    steps = []
    for c in spec.numeric_columns:
        steps.append((f"num_{c}", Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
            ("scale", StandardScaler()),
        ]), [c]))
    for c in spec.categorical:
        steps.append((f"cat_{c}", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=min_frequency,
                                                sparse_output=False), [c]))
    return ColumnTransformer(steps, remainder="drop", verbose_feature_names_out=True)


def _source_feature(transformed_name: str) -> str:
    prefix = transformed_name.split("__", 1)[0]
    return prefix.split("_", 1)[1]


@dataclass
class ModelBundle:
    model_family: str
    feature_spec: FeatureSpec
    pipeline: Pipeline
    platt_a: float = 0.0
    platt_b: float = 1.0
    calibrated: bool = False
    transformed_names: list[str] = field(default_factory=list)
    transformed_means: np.ndarray | None = None
    train_categories: dict[str, list[str]] = field(default_factory=dict)

    # ---- fitting -----------------------------------------------------------------------------------
    @classmethod
    def fit_logistic(cls, X: pd.DataFrame, y: np.ndarray, spec: FeatureSpec, C: float, max_iter: int,
                     min_frequency: float, seed: int, class_weight=None) -> ModelBundle:
        pipe = Pipeline([
            ("pre", make_preprocessor(spec, min_frequency)),
            ("clf", LogisticRegression(C=C, max_iter=max_iter, class_weight=class_weight, random_state=seed)),
        ])
        pipe.fit(X[spec.model_columns], y)
        pre = pipe.named_steps["pre"]
        Xt = pre.transform(X[spec.model_columns])
        cats = {c: sorted(str(v) for v in X[c].dropna().unique()) for c in spec.categorical}
        return cls(model_family="logistic_regression", feature_spec=spec, pipeline=pipe,
                   transformed_names=list(pre.get_feature_names_out()), transformed_means=Xt.mean(axis=0),
                   train_categories=cats)

    def fit_platt(self, X_cal: pd.DataFrame, y_cal: np.ndarray) -> None:
        a, b = fit_logistic_1d(logit(self.predict_raw(X_cal)), y_cal)
        self.platt_a, self.platt_b, self.calibrated = a, b, True

    # ---- prediction --------------------------------------------------------------------------------
    def predict_raw(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(X[self.feature_spec.model_columns])[:, 1]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        raw = self.predict_raw(X)
        return sigmoid(self.platt_a + self.platt_b * logit(raw)) if self.calibrated else raw

    def contributions(self, X: pd.DataFrame) -> pd.DataFrame:
        """Per-feature additive contributions to the uncalibrated logit, relative to the training mean.

        Exact for logistic regression: logit(x) - logit(mean) = sum_j coef_j * (x_t,j - mean_t,j). One-hot
        and missing-indicator columns are summed back to their source feature. These are model-internal
        attributions, not causal effects.
        """
        pre = self.pipeline.named_steps["pre"]
        coef = self.pipeline.named_steps["clf"].coef_.ravel()
        Xt = pre.transform(X[self.feature_spec.model_columns])
        contrib = (Xt - self.transformed_means) * coef
        df = pd.DataFrame(contrib, columns=self.transformed_names, index=X.index)
        groups = [_source_feature(n) for n in self.transformed_names]
        return df.T.groupby(groups, sort=False).sum().T

    def unseen_categories(self, X: pd.DataFrame) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for c in self.feature_spec.categorical:
            seen = set(self.train_categories.get(c, []))
            new = sorted({str(v) for v in X[c].dropna().unique()} - seen)
            if new:
                out[c] = new
        return out

    def coefficients(self) -> dict[str, float]:
        coef = self.pipeline.named_steps["clf"].coef_.ravel()
        return dict(zip(self.transformed_names, map(float, coef), strict=True))
