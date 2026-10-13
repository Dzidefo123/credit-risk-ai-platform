"""Model bundle: fitted preprocessing + estimator + calibration + explanation metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from crisk.evaluation.metrics import fit_logistic_1d, logit, sigmoid
from crisk.features.origination import FeatureSpec

# Pooled level for rare or unseen categories in the lightgbm frame.
OTHER_CATEGORY = "__OTHER__"


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
    """Fitted model + calibration + explanation metadata.

    ``pipeline`` holds the estimator: an sklearn Pipeline (logistic) or an LGBMClassifier (lightgbm).
    New fields have plain ``None`` defaults so bundles pickled by earlier versions still load.
    """

    model_family: str
    feature_spec: FeatureSpec
    pipeline: Any
    platt_a: float = 0.0
    platt_b: float = 1.0
    calibrated: bool = False
    transformed_names: list[str] = field(default_factory=list)
    transformed_means: np.ndarray | None = None
    train_categories: dict[str, list[str]] = field(default_factory=dict)
    macro: Any = None  # crisk.macro.pmms.MarketRateTable snapshot used for features (if any)
    lgb_categories: dict[str, list[str]] | None = None  # fixed category levels for lightgbm

    # ---- fitting -----------------------------------------------------------------------------------
    @classmethod
    def fit_logistic(cls, X: pd.DataFrame, y: np.ndarray, spec: FeatureSpec, C: float, max_iter: int,
                     min_frequency: float, seed: int, class_weight=None, macro=None) -> ModelBundle:
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
                   train_categories=cats, macro=macro)

    @classmethod
    def fit_lightgbm(cls, X: pd.DataFrame, y: np.ndarray, spec: FeatureSpec, params: dict,
                     monotone: dict[str, int], min_frequency: float, seed: int, early_stopping_mask: np.ndarray,
                     early_stopping_rounds: int, macro=None) -> tuple[ModelBundle, dict]:
        """Gradient-boosted trees. Rounds chosen by early stopping on a later-in-time slice of the
        training window (``early_stopping_mask``), then refitted on the whole training split."""
        import lightgbm as lgb

        levels = {}
        for c in spec.categorical:
            freq = X[c].value_counts(normalize=True)
            levels[c] = sorted(str(v) for v in freq[freq >= min_frequency].index) + [OTHER_CATEGORY]
        bundle = cls(model_family="lightgbm", feature_spec=spec, pipeline=None, macro=macro, lgb_categories=levels,
                     train_categories={c: sorted(str(v) for v in X[c].dropna().unique()) for c in spec.categorical})
        Xl = bundle._lgb_frame(X)
        constraints = [int(monotone.get(c, 0)) for c in spec.model_columns]
        base = dict(params, random_state=seed, monotone_constraints=constraints, verbose=-1)
        es = np.asarray(early_stopping_mask, dtype=bool)
        probe = lgb.LGBMClassifier(**base)
        probe.fit(Xl[~es], y[~es], eval_set=[(Xl[es], y[es])], eval_metric="binary_logloss",
                  callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)])
        best = int(probe.best_iteration_ or base.get("n_estimators", 100))
        final = lgb.LGBMClassifier(**{**base, "n_estimators": best})
        final.fit(Xl, y)
        bundle.pipeline = final
        info = {"best_iteration": best, "early_stopping_rows": int(es.sum()), "monotone_constraints":
                {c: v for c, v in zip(spec.model_columns, constraints, strict=True) if v}}
        return bundle, info

    def _lgb_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X[self.feature_spec.model_columns].copy()
        for c, levels in (self.lgb_categories or {}).items():
            s = out[c].astype(str)
            s = s.where(s.isin(levels), OTHER_CATEGORY)
            out[c] = pd.Categorical(s, categories=levels)
        return out

    def fit_platt(self, X_cal: pd.DataFrame, y_cal: np.ndarray) -> None:
        a, b = fit_logistic_1d(logit(self.predict_raw(X_cal)), y_cal)
        self.platt_a, self.platt_b, self.calibrated = a, b, True

    # ---- prediction --------------------------------------------------------------------------------
    def predict_raw(self, X: pd.DataFrame) -> np.ndarray:
        if self.model_family == "lightgbm":
            return self.pipeline.predict_proba(self._lgb_frame(X))[:, 1]
        return self.pipeline.predict_proba(X[self.feature_spec.model_columns])[:, 1]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        raw = self.predict_raw(X)
        return sigmoid(self.platt_a + self.platt_b * logit(raw)) if self.calibrated else raw

    def contributions(self, X: pd.DataFrame) -> pd.DataFrame:
        """Per-feature additive contributions to the uncalibrated logit.

        Logistic: exact, relative to the training mean (coef_j * (x_t,j - mean_t,j)), one-hot and
        missing-indicator columns summed back to their source feature. LightGBM: TreeSHAP values in
        log-odds (LightGBM ``pred_contrib``), relative to the model's expected value. Both are model-internal
        attributions, not causal effects.
        """
        if self.model_family == "lightgbm":
            contrib = self.pipeline.predict(self._lgb_frame(X), pred_contrib=True)
            return pd.DataFrame(contrib[:, :-1], columns=self.feature_spec.model_columns, index=X.index)
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
        """Logistic coefficients, or LightGBM total-gain importances (not signed effects)."""
        if self.model_family == "lightgbm":
            gain = self.pipeline.booster_.feature_importance(importance_type="gain")
            return dict(zip(self.feature_spec.model_columns, map(float, gain), strict=True))
        coef = self.pipeline.named_steps["clf"].coef_.ravel()
        return dict(zip(self.transformed_names, map(float, coef), strict=True))
