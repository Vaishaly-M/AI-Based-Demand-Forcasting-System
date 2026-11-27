"""Pre-test model comparison, validation-only selection, and winner-only test."""

from dataclasses import dataclass
from hashlib import sha256
from importlib.metadata import version
import platform
import math

import pandas as pd

from src.demand_planning.evaluation.metrics import forecast_metrics
from src.demand_planning.evaluation.splits import chronological_split
from src.demand_planning.features.calendar_lags import LAGS, ROLLING_WINDOWS, build_features
from src.demand_planning.forecasting.models import MODEL_NAMES, RANDOM_SEED, FittedModel, fit_model, predict_recursive

EXPERIMENT_VERSION = "phase4-v1"
HORIZONS = (7, 14, 30)


@dataclass
class EvaluationResult:
    """Audit tables and the selected model fitted only through validation end."""

    selected_model: FittedModel
    validation_metrics: pd.DataFrame
    cv_metrics: pd.DataFrame
    test_metrics: pd.DataFrame
    predictions: pd.DataFrame
    metadata: dict


def series_fingerprint(history: pd.Series) -> str:
    """Hash values and dates so cached results cannot silently reuse changed data."""
    return sha256(pd.util.hash_pandas_object(history, index=True).values.tobytes()).hexdigest()


def select_model(validation_metrics: pd.DataFrame) -> str:
    """Choose 30-day WAPE, then MAE/RMSE/fixed model order; all-zero uses MAE."""
    candidates = validation_metrics.loc[validation_metrics.horizon.eq(30)].copy()
    if len(candidates) != len(MODEL_NAMES) or set(candidates.model) != set(MODEL_NAMES):
        raise ValueError("Selection requires exactly one 30-day validation score for every model.")
    candidates["model_order"] = candidates.model.map({name: index for index, name in enumerate(MODEL_NAMES)})
    use_wape = candidates.wape_pct.notna().all()
    if not use_wape and candidates.wape_pct.notna().any():
        raise ValueError("All models must be scored against the same validation targets.")
    columns = (["wape_pct"] if use_wape else []) + ["mae", "rmse", "model_order"]
    return str(candidates.sort_values(columns).iloc[0].model)


def _evaluate_origin(history: pd.Series, actual: pd.Series, names: tuple[str, ...], stage: str,
                     fold: int) -> tuple[list[dict], list[pd.DataFrame], dict[str, FittedModel]]:
    """Train each candidate at one origin and score common forecast prefixes."""
    scores, predictions, models = [], [], {}
    for name in names:
        fitted = fit_model(name, history)
        forecast = predict_recursive(fitted, history, 30)
        if forecast.date.tolist() != actual.index.tolist():
            raise ValueError("Holdout dates must exactly match the 30-day forecast dates.")
        forecast["model"] = name
        forecast["actual"] = actual.to_numpy()
        forecast["stage"] = stage
        forecast["fold"] = fold
        predictions.append(forecast)
        for horizon in HORIZONS:
            scores.append({"model": name, "stage": stage, "fold": fold, "horizon": horizon,
                           "origin": history.index[-1],
                           **forecast_metrics(actual.iloc[:horizon], forecast.forecast.iloc[:horizon])})
        models[name] = fitted
    return scores, predictions, models


def run_evaluation(history: pd.Series, source: str = "Provided series", provenance: str = "Unverified",
                   store: int | None = None, family: str | None = None) -> EvaluationResult:
    """Run fixed settings with up to three diagnostic expanding-window folds.

    CV origins are inside the training period, 30 days apart, with >=56 prior
    days. Dedicated validation selects the winner; final test is used only after
    selection. No full-history fit, tuning, early stopping, or test leaderboard.
    """
    split = chronological_split(history)
    origins = sorted(len(split.train) - 30 * offset for offset in (1, 2, 3)
                     if len(split.train) - 30 * offset >= 56)
    cv_scores, all_predictions = [], []
    for fold, origin in enumerate(origins, 1):
        scores, predictions, _ = _evaluate_origin(split.train.iloc[:origin], split.train.iloc[origin:origin + 30],
                                                  MODEL_NAMES, "rolling_cv", fold)
        cv_scores.extend(scores)
        all_predictions.extend(predictions)
    scores, predictions, _ = _evaluate_origin(split.train, split.validation, MODEL_NAMES, "validation", 0)
    validation_metrics = pd.DataFrame(scores)
    all_predictions.extend(predictions)
    winner = select_model(validation_metrics)
    pretest = pd.concat([split.train, split.validation])
    test_scores, predictions, models = _evaluate_origin(pretest, split.test, (winner,), "test", 0)
    all_predictions.extend(predictions)
    selected = models[winner]
    versions = {package: version(package) for package in
                ("pandas", "numpy", "scikit-learn", "xgboost", "joblib")}
    metadata = {
        "experiment_version": EXPERIMENT_VERSION, "source": source, "provenance": provenance,
        "store": store, "family": family, "data_fingerprint": series_fingerprint(history),
        "python": platform.python_version(), "dependencies": versions, "random_seed": RANDOM_SEED,
        "split_boundaries": split.summary().assign(start=lambda d: d.start.astype(str), end=lambda d: d.end.astype(str)).to_dict("records"),
        "cv_origins": [str(split.train.index[origin - 1].date()) for origin in origins],
        "cv_policy": "Up to 3 non-overlapping 30-day holdouts within training; diagnostic only",
        "selection_policy": "30-day validation WAPE; all-zero fallback MAE; ties MAE, RMSE, fixed model order",
        "model_order": list(MODEL_NAMES), "selected_model": winner,
        "feature_columns": list(build_features(split.train).columns),
        "selected_model_feature_columns": list(selected.feature_columns),
        "lags": list(LAGS), "rolling_windows": list(ROLLING_WINDOWS),
        "rolling_std_ddof": 0, "warmup_days": 28, "horizons": list(HORIZONS),
        "trained_through": str(selected.trained_through.date()), "training_rows": selected.training_rows,
        "artifact_scope": "Selected model refit on train plus validation, excluding test; not a full-history future model",
        "preprocessing": "Linear Regression: StandardScaler fitted within each training window; trees: no scaling; no imputation",
        "prediction_policy": "Recursive; negative predictions clipped before feedback and scoring; no future actuals",
        "parameters": {name: _parameters(name) for name in MODEL_NAMES},
    }
    return EvaluationResult(selected, validation_metrics, pd.DataFrame(cv_scores), pd.DataFrame(test_scores),
                            pd.concat(all_predictions, ignore_index=True), metadata)


def _parameters(name: str) -> dict:
    """Record configured estimator parameters without serializing live objects."""
    from src.demand_planning.forecasting.models import make_estimator

    if name == "Seasonal Naive":
        return {"period": 7}
    estimator = make_estimator(name)
    parameters = estimator.get_params(deep=True)
    return {key: str(value) if isinstance(value, float) and not math.isfinite(value) else value
            for key, value in parameters.items()
            if value is None or isinstance(value, (str, bool, int, float))}
