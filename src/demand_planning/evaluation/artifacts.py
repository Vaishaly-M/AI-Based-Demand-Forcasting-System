"""Write trusted local artifacts and transparent audit tables; never load uploads."""

import json
from pathlib import Path

import joblib

from src.demand_planning.evaluation.experiment import EvaluationResult


def save_evaluation(result: EvaluationResult, directory: Path) -> Path:
    """Create a new run directory; refuse to overwrite any existing run.

    Joblib uses pickle: only trusted local artifacts should ever be loaded.
    Bundle includes the final 28 pre-test observations for recursive prediction.
    """
    metadata_json = json.dumps(result.metadata, indent=2, allow_nan=False)
    directory.mkdir(parents=True, exist_ok=False)
    pretest_rows = result.predictions.loc[result.predictions.stage.eq("validation")]
    history_tail = pretest_rows.loc[pretest_rows.model.eq("Seasonal Naive"), ["date", "actual"]].tail(28)
    bundle = {"model": result.selected_model, "metadata": result.metadata,
              "history_tail": history_tail.set_index("date").actual}
    joblib.dump(bundle, directory / "selected_model.joblib")
    (directory / "metadata.json").write_text(metadata_json, encoding="utf-8")
    result.validation_metrics.to_csv(directory / "validation_metrics.csv", index=False)
    result.cv_metrics.to_csv(directory / "rolling_cv_metrics.csv", index=False)
    result.test_metrics.to_csv(directory / "selected_test_metrics.csv", index=False)
    result.predictions.to_csv(directory / "predictions.csv", index=False)
    return directory
