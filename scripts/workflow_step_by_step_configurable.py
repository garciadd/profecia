#!/usr/bin/env python3
"""
Configurable end-to-end workflow driven by config/train.toml and config/data.toml.

Pipeline:
1. Resolve TOML configuration.
2. Preprocess raw variables into processed .npy arrays when needed.
3. Load processed arrays as xarray DataArrays.
4. Build train/test masks with the new split semantics:
   - spatial_pixel
   - random_observation
5. Export tabular train/test arrays with full row traceability.
6. Optionally run RandomizedSearchCV.
7. Train the final model with optional MLflow logging.
8. Evaluate on test and save metrics/figures.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from pprint import pformat

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import io
from src.data.eda import load_processed_dataset
from src.evaluation.regression import (
    build_pixel_timeseries_dataframe,
    build_prediction_dataframe,
    compute_global_regression_metrics,
    compute_pixel_regression_metrics,
    rank_best_and_worst_pixels,
    summarize_pixel_metrics,
)
from src.project_config import resolve_train_config
from src.training.dataset import export_train_test_data
from src.training.hyperparam_search import run_hyperparameter_search
from src.training.split import make_train_test_split
from src.training.train import (
    load_test_arrays,
    load_train_arrays,
    save_trained_pipeline,
    train_tabular_model,
)

LOGGER = logging.getLogger(__name__)

CONFIG_PATH = Path(os.getenv("PROFECIA_TRAIN_CONFIG", "config/train.toml"))
STOP_ON_ERROR = os.getenv("PROFECIA_STOP_ON_ERROR", "true").lower().strip() not in {"0", "false", "no"}
LOG_SHAP = os.getenv("PROFECIA_LOG_SHAP", "false").lower().strip() in {"1", "true", "yes"}


def configure_logging(cfg: dict) -> Path:
    log_dir = cfg["model_dir"] / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"workflow_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.log"

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)
    root.addHandler(stream_handler)
    root.addHandler(file_handler)
    return log_path


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(io._to_jsonable(payload), f, ensure_ascii=False, indent=2)


def build_mlflow_config(cfg: dict) -> dict | None:
    mlflow_cfg = cfg.get("mlflow", {})
    mlflow_enabled = bool(mlflow_cfg.get("enabled", False))
    if not mlflow_enabled:
        return None

    experiment_name = (
        mlflow_cfg.get("experiment_name")
        or os.getenv("PROFECIA_MLFLOW_EXPERIMENT")
        or f"profecia/{cfg['processed_run_name']}/{cfg['split_mode']}"
    )
    return {
        "tracking_uri": mlflow_cfg.get("tracking_uri") or os.getenv("MLFLOW_TRACKING_URI"),
        "user": mlflow_cfg.get("tracking_username") or os.getenv("MLFLOW_TRACKING_USERNAME"),
        "password": mlflow_cfg.get("tracking_password") or os.getenv("MLFLOW_TRACKING_PASSWORD"),
        "experiment_name": experiment_name,
        "run_name": cfg["model_run_name"],
    }


def preprocess_variables(cfg: dict) -> Path:
    data_cfg = cfg["data"]
    output_dir = data_cfg["output_dir"]
    expected_run_config = io.build_processed_run_config(
        variable_names=data_cfg["variable_names"],
        temporal_resolution=data_cfg["temporal_resolution"],
        mask_names=data_cfg["mask_names"],
        start_year=data_cfg["start_year"],
        end_year_inclusive=data_cfg["end_year_inclusive"],
        dtype=data_cfg["dtype"],
        roi=data_cfg["roi"],
    )
    status = io.processed_run_status(
        output_dir=output_dir,
        variable_names=data_cfg["variable_names"],
        expected_config=expected_run_config,
    )

    LOGGER.info("Processed run: %s", output_dir)
    LOGGER.info("Processed status: %s", pformat(status))
    if status["complete"] and status["config_matches"]:
        LOGGER.info("Processed data already exists and matches config; skipping preprocessing.")
        return output_dir

    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict = {}
    for variable in data_cfg["variable_names"]:
        LOGGER.info("Preprocessing variable %s", variable)
        results.update(
            io.load_and_save_variable(
                raw_dir=data_cfg["raw_dir"],
                output_dir=output_dir,
                variable=variable,
                mask_dir=data_cfg["mask_dir"],
                mask_names=data_cfg["mask_names"],
                roi=data_cfg["roi"],
                start_year=data_cfg["start_year"],
                end_year_inclusive=data_cfg["end_year_inclusive"],
                dtype=data_cfg["dtype"],
                temporal_resolution=data_cfg["temporal_resolution"],
                save_output=True,
            )
        )

    io.save_processed_metadata(
        output_dir=output_dir,
        variable_results=results,
        temporal_resolution=data_cfg["temporal_resolution"],
        roi=data_cfg["roi"],
        start_year=data_cfg["start_year"],
        end_year_inclusive=data_cfg["end_year_inclusive"],
        dtype=data_cfg["dtype"],
    )
    io.save_metadata_json(output_dir, expected_run_config, filename="run_config.json")
    LOGGER.info("Preprocessed data saved in %s", output_dir)
    return output_dir


def build_split_and_export(cfg: dict, processed_dir: Path) -> tuple[dict, dict]:
    LOGGER.info("Loading processed dataset from %s", processed_dir)
    data_dict, processed_metadata = load_processed_dataset(
        input_dir=processed_dir,
        variable_names=cfg["variable_names"],
        reference_variable=cfg["target_name"],
    )
    target = data_dict[cfg["target_name"]]
    predictors: dict[str, object] = {}
    for name in cfg["predictor_names"]:
        if name in data_dict:
            predictors[name] = data_dict[name]
            continue

        expanded_names = sorted(key for key in data_dict if key.startswith(f"{name}_"))
        if not expanded_names:
            raise KeyError(name)

        LOGGER.info(
            "Predictor %s expanded to derived variables: %s",
            name,
            ", ".join(expanded_names),
        )
        for expanded_name in expanded_names:
            predictors[expanded_name] = data_dict[expanded_name]

    LOGGER.info("Building split mode=%s", cfg["split_mode"])
    split_result = make_train_test_split(
        split_mode=cfg["split_mode"],
        target=target,
        predictors=predictors,
        train_fraction=cfg["train_fraction"],
        test_fraction=cfg["test_fraction"],
        seed=cfg["seed"],
        min_valid_fraction=cfg["min_valid_fraction"],
    )

    extra_masks = {
        key: value
        for key, value in split_result.items()
        if key.endswith("_mask") and key not in {"train_mask", "test_mask"}
    }
    LOGGER.info("Exporting tabular arrays to %s", cfg["model_data_dir"])
    dataset_metadata = export_train_test_data(
        target=target,
        predictors=predictors,
        train_mask=split_result["train_mask"],
        test_mask=split_result["test_mask"],
        output_dir=cfg["model_data_dir"],
        split_metadata=split_result["metadata"],
        extra_masks=extra_masks,
    )

    bundle = {
        "data_dict": data_dict,
        "processed_metadata": processed_metadata,
        "target": target,
        "predictors": predictors,
        "split_result": split_result,
        "dataset_metadata": dataset_metadata,
    }
    return bundle, dataset_metadata


def train_final_model(cfg: dict, dataset_metadata: dict) -> dict:
    X_train, y_train, trace_train, _ = load_train_arrays(
        cfg["model_data_dir"],
        mmap_mode=cfg["mmap_mode"],
        return_trace=True,
    )

    model_params = dict(cfg["model_params"])
    hyperparameter_search = {"enabled": False}

    if cfg["cv_config"]["enabled"]:
        LOGGER.info("Running RandomizedSearchCV")
        cv_result = run_hyperparameter_search(
            X_train=X_train,
            y_train=y_train,
            split_mode=cfg["split_mode"],
            model_name=cfg["model_name"],
            scaler_name=cfg["scaler_name"],
            random_state=cfg["random_state"],
            base_model_params=model_params,
            search_space=cfg["cv_config"]["search_space"],
            pixel_id_train=trace_train.get("pixel_id_train"),
            cv_config=cfg["cv_config"],
            output_dir=cfg["model_artifacts_dir"] / "cv",
        )
        model_params.update(cv_result["best_params"])
        hyperparameter_search = cv_result["metadata"]
        LOGGER.info("Best CV score: %s", cv_result["best_score"])
        LOGGER.info("Best params: %s", pformat(cv_result["best_params"]))

    train_result = train_tabular_model(
        X_train=X_train,
        y_train=y_train,
        model_name=cfg["model_name"],
        scaler_name=cfg["scaler_name"],
        random_state=cfg["random_state"],
        mlflow_config=build_mlflow_config(cfg),
        log_shap=LOG_SHAP,
        **model_params,
    )
    train_result["train_info"]["hyperparameter_search"] = hyperparameter_search

    saved_paths = save_trained_pipeline(
        output_dir=cfg["model_artifacts_dir"],
        model=train_result["model"],
        scaler=train_result["scaler"],
        train_info=train_result["train_info"],
        dataset_metadata=dataset_metadata,
    )
    train_result["saved_paths"] = saved_paths
    LOGGER.info("Model artifacts saved: %s", pformat(saved_paths))
    return train_result


def save_evaluation_outputs(
    cfg: dict,
    prediction_df: pd.DataFrame,
    pixel_metrics_df: pd.DataFrame,
    global_metrics: dict,
    pixel_metrics_summary: dict,
) -> dict[str, Path]:
    out_dir = cfg["model_figures_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir = cfg["model_dir"] / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    summary_path = report_dir / "summary_metrics.csv"
    pixel_metrics_path = report_dir / "pixel_metrics.csv"
    prediction_sample_path = report_dir / "prediction_sample.csv"

    pd.DataFrame([
        {
            "model_run_name": cfg["model_run_name"],
            **global_metrics,
            **{f"pixel_{k}": v for k, v in pixel_metrics_summary.items()},
        }
    ]).to_csv(summary_path, index=False)
    pixel_metrics_df.to_csv(pixel_metrics_path, index=False)
    prediction_df.head(20_000).to_csv(prediction_sample_path, index=False)

    global_hexbin_path = out_dir / "hexbin_global.png"
    residual_hist_path = out_dir / "residual_hist.png"
    best_worst_pixels_path = out_dir / "best_worst_pixels_hexbin.png"

    fig, ax = plt.subplots(figsize=(7, 7))
    hb = ax.hexbin(prediction_df["y_true"], prediction_df["y_pred"], gridsize=60, mincnt=1)
    xy_min = min(prediction_df["y_true"].min(), prediction_df["y_pred"].min())
    xy_max = max(prediction_df["y_true"].max(), prediction_df["y_pred"].max())
    ax.plot([xy_min, xy_max], [xy_min, xy_max], "--", color="black", linewidth=1)
    ax.set_xlabel("LAI real")
    ax.set_ylabel("LAI predicted")
    ax.set_title(f"Test global hexbin | {cfg['model_run_name']}")
    fig.colorbar(hb, ax=ax, label="Samples")
    fig.tight_layout()
    fig.savefig(global_hexbin_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(prediction_df["residual"], bins=60, color="#CC6677", alpha=0.85)
    ax.axvline(0.0, linestyle="--", color="black", linewidth=1)
    ax.set_title(f"Residuals | {cfg['model_run_name']}")
    ax.set_xlabel("Residual = true - predicted")
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    fig.savefig(residual_hist_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    ranking = rank_best_and_worst_pixels(pixel_metrics_df, metric="r2", top_k=4)
    selected_pixels = pd.concat(
        [ranking["best"].assign(group="best"), ranking["worst"].assign(group="worst")],
        ignore_index=True,
    )
    if len(selected_pixels) > 0:
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        for ax, (_, row) in zip(axes.ravel(), selected_pixels.iterrows()):
            ts_df = build_pixel_timeseries_dataframe(prediction_df, pixel_id=int(row["pixel_id"]))
            if len(ts_df) >= 20:
                hb = ax.hexbin(ts_df["y_true"], ts_df["y_pred"], gridsize=20, mincnt=1)
                fig.colorbar(hb, ax=ax, shrink=0.75)
            else:
                ax.scatter(ts_df["y_true"], ts_df["y_pred"], s=20)
            xy_min = min(ts_df["y_true"].min(), ts_df["y_pred"].min())
            xy_max = max(ts_df["y_true"].max(), ts_df["y_pred"].max())
            ax.plot([xy_min, xy_max], [xy_min, xy_max], "--", color="black")
            ax.set_title(f"{row['group']} | pixel {int(row['pixel_id'])} | R2={row['r2']:.3f}")
            ax.set_xlabel("True")
            ax.set_ylabel("Predicted")
        fig.tight_layout()
        fig.savefig(best_worst_pixels_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return {
        "summary": summary_path,
        "pixel_metrics": pixel_metrics_path,
        "prediction_sample": prediction_sample_path,
        "global_hexbin": global_hexbin_path,
        "residual_hist": residual_hist_path,
        "best_worst_pixels": best_worst_pixels_path,
    }


def log_evaluation_to_mlflow(mlflow_run_id: str | None, global_metrics: dict, pixel_metrics_summary: dict, paths: dict[str, Path]) -> None:
    if not mlflow_run_id:
        LOGGER.info("No MLflow run id; evaluation is saved only on disk.")
        return

    try:
        import mlflow
    except Exception:
        LOGGER.exception("MLflow import failed while logging evaluation artifacts.")
        return

    with mlflow.start_run(run_id=mlflow_run_id):
        for name, value in global_metrics.items():
            if isinstance(value, (int, float)) and np.isfinite(value):
                mlflow.log_metric(f"test_{name}", float(value))
        for name, value in pixel_metrics_summary.items():
            if isinstance(value, (int, float)) and np.isfinite(value):
                mlflow.log_metric(f"pixel_{name}", float(value))
        for path in paths.values():
            if path.exists():
                mlflow.log_artifact(str(path), artifact_path="evaluation")


def evaluate_model(cfg: dict, bundle: dict, train_result: dict) -> dict:
    X_test, y_test, trace_test, dataset_metadata = load_test_arrays(
        cfg["model_data_dir"],
        mmap_mode=cfg["mmap_mode"],
    )
    if train_result["scaler"] is not None:
        X_test_used = train_result["scaler"].transform(X_test).astype(np.float32, copy=False)
    else:
        X_test_used = X_test

    y_pred = np.asarray(train_result["model"].predict(X_test_used)).reshape(-1)
    target = bundle["target"]
    prediction_df = build_prediction_dataframe(
        y_true=y_test,
        y_pred=y_pred,
        pixel_id=trace_test["pixel_id_test"],
        lat_idx=trace_test["lat_idx_test"],
        lon_idx=trace_test["lon_idx_test"],
        time_idx=trace_test["time_idx_test"],
        latitude_values=target.latitude.values,
        longitude_values=target.longitude.values,
        time_values=pd.to_datetime(dataset_metadata["time_values"]),
    )
    prediction_df["residual"] = prediction_df["y_true"] - prediction_df["y_pred"]

    global_metrics = compute_global_regression_metrics(y_true=y_test, y_pred=y_pred)
    global_metrics["bias"] = float(np.mean(y_pred - np.asarray(y_test).reshape(-1)))
    if len(y_pred) > 1:
        global_metrics["pearson_r"] = float(np.corrcoef(np.asarray(y_test).reshape(-1), y_pred)[0, 1])
    else:
        global_metrics["pearson_r"] = np.nan

    pixel_metrics_df = compute_pixel_regression_metrics(
        y_true=y_test,
        y_pred=y_pred,
        pixel_id=trace_test["pixel_id_test"],
        lat_idx=trace_test["lat_idx_test"],
        lon_idx=trace_test["lon_idx_test"],
        latitude=target.latitude.values[np.asarray(trace_test["lat_idx_test"]).astype(int)],
        longitude=target.longitude.values[np.asarray(trace_test["lon_idx_test"]).astype(int)],
    )
    pixel_metrics_summary = summarize_pixel_metrics(pixel_metrics_df)
    paths = save_evaluation_outputs(cfg, prediction_df, pixel_metrics_df, global_metrics, pixel_metrics_summary)
    log_evaluation_to_mlflow(
        mlflow_run_id=train_result.get("mlflow_run_id"),
        global_metrics=global_metrics,
        pixel_metrics_summary=pixel_metrics_summary,
        paths=paths,
    )

    LOGGER.info("Global metrics: %s", pformat(global_metrics))
    return {
        "global_metrics": global_metrics,
        "pixel_metrics_summary": pixel_metrics_summary,
        "evaluation_paths": {k: str(v) for k, v in paths.items()},
    }


def run() -> dict:
    cfg = resolve_train_config(CONFIG_PATH)
    log_path = configure_logging(cfg)
    LOGGER.info("Log file: %s", log_path)
    LOGGER.info("Resolved config: %s", pformat(cfg))

    processed_dir = preprocess_variables(cfg)
    bundle, dataset_metadata = build_split_and_export(cfg, processed_dir)
    train_result = train_final_model(cfg, dataset_metadata)
    evaluation = evaluate_model(cfg, bundle, train_result)

    summary = {
        "executed_at_utc": datetime.now(UTC).isoformat(),
        "config_path": str(CONFIG_PATH),
        "processed_dir": str(processed_dir),
        "model_data_dir": str(cfg["model_data_dir"]),
        "model_artifacts_dir": str(cfg["model_artifacts_dir"]),
        "model_figures_dir": str(cfg["model_figures_dir"]),
        "split_mode": cfg["split_mode"],
        "dataset_metadata": dataset_metadata,
        "train_info": train_result["train_info"],
        "saved_paths": train_result["saved_paths"],
        "evaluation": evaluation,
    }
    summary_path = cfg["model_dir"] / "workflow_step_by_step_summary.json"
    save_json(summary_path, summary)
    LOGGER.info("Workflow summary saved in %s", summary_path)
    return summary


def main() -> int:
    try:
        run()
    except Exception:
        LOGGER.exception("Workflow failed.")
        if STOP_ON_ERROR:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
