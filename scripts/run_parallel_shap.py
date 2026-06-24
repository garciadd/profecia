#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.io import MASK_MAP, load_mask
from src.evaluation.regression import build_prediction_dataframe
from src.explainability.shap import run_shap_analysis
from src.project_config import resolve_train_config
from src.training.train import load_test_arrays, load_train_arrays, load_trained_pipeline

LOGGER = logging.getLogger("parallel_shap")
DEFAULT_MACRO_LABELS = {
    1: "Tropical",
    2: "Dry",
    3: "Temperate",
    4: "Continental",
    5: "Polar",
}
DEFAULT_LANDCOVER_LABELS = {
    10: "Tree cover",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    70: "Snow and ice",
    90: "Herbaceous wetland",
    100: "Moss and lichen",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SHAP analysis from the trained pipeline using multiple CPU cores.")
    parser.add_argument("--config", default="config/train.toml", help="Path to train.toml")
    parser.add_argument("--data-dir", help="Override model data directory")
    parser.add_argument("--artifacts-dir", help="Override model artifacts directory")
    parser.add_argument("--masks-dir", help="Override masks directory")
    parser.add_argument("--output-dir", help="Directory where SHAP artifacts will be written")
    parser.add_argument("--n-jobs", type=int, default=16, help="Number of worker processes for SHAP")
    parser.add_argument("--chunk-size", type=int, default=0, help="Rows per SHAP chunk; 0 means auto")
    parser.add_argument("--threads-per-worker", type=int, default=1, help="BLAS/OpenMP threads per worker")
    parser.add_argument("--mp-start-method", choices=["spawn", "fork", "forkserver"], default="spawn", help="Multiprocessing start method for SHAP workers")
    parser.add_argument("--parallel-backend", choices=["auto", "threads", "processes"], default="auto", help="Parallel backend for SHAP chunk execution")
    parser.add_argument("--max-samples-per-group", type=int, help="Override max_samples_per_group")
    parser.add_argument("--max-samples-total", type=int, help="Override max_samples_total")
    parser.add_argument("--sample-fraction", type=float, help="Override sample_fraction")
    parser.add_argument("--min-group-samples", type=int, help="Override min_group_samples")
    parser.add_argument("--group-column", default="landcover_label", help="Metadata column used for stratified SHAP sampling")
    parser.add_argument("--local-error-column", default="abs_error", help="Column used to select the local force plot row")
    parser.add_argument("--include-train", action="store_true", help="Include training rows together with test rows")
    parser.add_argument("--target-year", type=int, help="Filter observations to a specific year before SHAP sampling")
    parser.add_argument("--random-state", type=int, help="Override SHAP random state")
    parser.add_argument(
        "--presample-random-state",
        type=int,
        help="Random state used for early row subsampling before predictions/SHAP preparation",
    )
    parser.add_argument(
        "--max-test-rows",
        type=int,
        help="Limit test rows early, before scaling/predictions, to reduce RAM usage",
    )
    parser.add_argument(
        "--max-train-rows",
        type=int,
        help="Limit train rows early, before scaling/predictions, to reduce RAM usage",
    )
    parser.add_argument("--log-file", help="Override log path; defaults to <output-dir>/shap_run.log")
    parser.add_argument("--model-n-jobs", type=int, help="Temporarily override model.n_jobs before predictions/SHAP")
    return parser.parse_args()


def configure_logging(log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)


def load_label_maps(mask_dir: Path) -> tuple[dict[int, str], dict[int, str]]:
    mask_metadata_path = mask_dir / "mask_metadata.json"
    try:
        if not mask_metadata_path.exists():
            LOGGER.info("No %s found; using default climate/landcover labels", mask_metadata_path)
            return DEFAULT_MACRO_LABELS, DEFAULT_LANDCOVER_LABELS

        with open(mask_metadata_path, "r", encoding="utf-8") as f:
            mask_metadata = json.load(f)
    except PermissionError:
        LOGGER.warning(
            "Permission denied while reading %s; continuing with default climate/landcover labels",
            mask_metadata_path,
        )
        return DEFAULT_MACRO_LABELS, DEFAULT_LANDCOVER_LABELS

    macro_labels = {int(k): v for k, v in mask_metadata["masks"]["climate"]["labels"].items()}
    landcover_labels = {int(k): v for k, v in mask_metadata["masks"]["landcover"]["labels"].items()}
    return macro_labels, landcover_labels


def load_available_masks(mask_dir: Path, latitude_values: np.ndarray, longitude_values: np.ndarray) -> dict[str, Any]:
    available_masks: dict[str, Any] = {}
    for mask_name in ["climate", "landcover"]:
        mask_path = mask_dir / MASK_MAP[mask_name]
        try:
            if not mask_path.exists():
                LOGGER.info("Mask %s not found; skipping", mask_path)
                continue
            available_masks[mask_name] = load_mask(
                mask_dir=mask_dir,
                mask_name=mask_name,
                latitude=latitude_values,
                longitude=longitude_values,
            )
        except PermissionError:
            LOGGER.warning("Permission denied while reading mask %s; skipping", mask_path)
            continue
    return available_masks


def add_mask_columns(
    prediction_df: pd.DataFrame,
    available_masks: dict[str, Any],
    macro_labels: dict[int, str],
    landcover_labels: dict[int, str],
) -> pd.DataFrame:
    df = prediction_df.copy()
    lat_idx = df["lat_idx"].to_numpy(dtype=int)
    lon_idx = df["lon_idx"].to_numpy(dtype=int)

    if "climate" in available_masks:
        climate_arr = available_masks["climate"].values
        climate_codes = climate_arr[lat_idx, lon_idx].astype(int)
        df["climate_code"] = climate_codes
        df["climate_label"] = pd.Series(climate_codes).map(macro_labels).fillna("Unknown")

    if "landcover" in available_masks:
        landcover_arr = available_masks["landcover"].values
        landcover_codes = landcover_arr[lat_idx, lon_idx].astype(int)
        df["landcover_code"] = landcover_codes
        df["landcover_label"] = pd.Series(landcover_codes).map(landcover_labels).fillna("Unknown")

    return df


def build_prediction_frame(
    model,
    X_used: np.ndarray,
    y_true: np.ndarray,
    trace: dict[str, np.ndarray],
    split_label: str,
    latitude_values: np.ndarray,
    longitude_values: np.ndarray,
    time_values: pd.DatetimeIndex,
) -> pd.DataFrame:
    predict_t0 = time.perf_counter()
    y_pred = np.asarray(model.predict(X_used)).reshape(-1)
    LOGGER.info("Predicted %s rows for split=%s in %.2fs", len(y_pred), split_label, time.perf_counter() - predict_t0)

    df = build_prediction_dataframe(
        y_true=y_true,
        y_pred=y_pred,
        pixel_id=trace[f"pixel_id_{split_label}"],
        lat_idx=trace[f"lat_idx_{split_label}"],
        lon_idx=trace[f"lon_idx_{split_label}"],
        time_idx=trace[f"time_idx_{split_label}"],
        latitude_values=latitude_values,
        longitude_values=longitude_values,
        time_values=time_values,
    )
    df["residual"] = df["y_pred"] - df["y_true"]
    df["abs_error"] = df["residual"].abs()
    df["sq_error"] = df["residual"] ** 2
    df["split"] = split_label
    return df


def maybe_scale_features(scaler, X: np.ndarray) -> np.ndarray:
    if scaler is None:
        return X
    scaled = scaler.transform(X)
    return scaled.astype(np.float32, copy=False)


def subsample_rows(
    X: np.ndarray,
    y: np.ndarray,
    trace: dict[str, np.ndarray],
    split_label: str,
    max_rows: int | None,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    if max_rows is None or max_rows <= 0 or len(X) <= int(max_rows):
        return X, y, trace

    rng = np.random.default_rng(random_state)
    selected_idx = np.sort(rng.choice(len(X), size=int(max_rows), replace=False))
    trace_subset = {
        key: values[selected_idx]
        for key, values in trace.items()
        if key.endswith(f"_{split_label}")
    }
    LOGGER.info(
        "Early-subsampled split=%s from %s rows to %s rows before predictions",
        split_label,
        int(len(X)),
        int(len(selected_idx)),
    )
    return X[selected_idx], y[selected_idx], trace_subset


def resolve_shap_random_state(args: argparse.Namespace, cfg: dict[str, Any], train_info: dict[str, Any], dataset_metadata: dict[str, Any]) -> int:
    if args.random_state is not None:
        return int(args.random_state)
    return int(
        train_info.get("random_state")
        or cfg.get("random_state")
        or dataset_metadata.get("split_metadata", {}).get("seed", 42)
    )


def main() -> int:
    args = parse_args()
    cfg = resolve_train_config(args.config)

    data_dir = Path(args.data_dir) if args.data_dir else Path(cfg["model_data_dir"])
    artifacts_dir = Path(args.artifacts_dir) if args.artifacts_dir else Path(cfg["model_artifacts_dir"])
    masks_dir = Path(args.masks_dir) if args.masks_dir else Path(cfg["mask_dir"])
    output_dir = Path(args.output_dir) if args.output_dir else artifacts_dir / "shap_parallel"
    log_file = Path(args.log_file) if args.log_file else output_dir / "shap_run.log"
    configure_logging(log_file)

    overall_t0 = time.perf_counter()
    LOGGER.info("Starting parallel SHAP run")
    LOGGER.info("config=%s", Path(args.config).resolve())
    LOGGER.info("data_dir=%s", data_dir)
    LOGGER.info("artifacts_dir=%s", artifacts_dir)
    LOGGER.info("masks_dir=%s", masks_dir)
    LOGGER.info("output_dir=%s", output_dir)

    train_info_path = artifacts_dir / "train_info.json"
    if not train_info_path.exists():
        raise FileNotFoundError(f"Missing train_info.json: {train_info_path}")
    with open(train_info_path, "r", encoding="utf-8") as f:
        train_info = json.load(f)

    load_t0 = time.perf_counter()
    X_test, y_test, test_trace, dataset_metadata = load_test_arrays(
        input_dir=data_dir,
        mmap_mode=cfg["mmap_mode"],
    )
    model, scaler = load_trained_pipeline(
        model_path=train_info["model_path"],
        scaler_path=train_info["scaler_path"],
    )
    LOGGER.info("Loaded test arrays and trained pipeline in %.2fs", time.perf_counter() - load_t0)

    presample_random_state = int(
        args.presample_random_state
        if args.presample_random_state is not None
        else args.random_state
        if args.random_state is not None
        else cfg.get("random_state", 42)
    )

    X_test, y_test, test_trace = subsample_rows(
        X=X_test,
        y=y_test,
        trace=test_trace,
        split_label="test",
        max_rows=args.max_test_rows,
        random_state=presample_random_state,
    )

    if args.model_n_jobs is not None and hasattr(model, "set_params"):
        model.set_params(n_jobs=int(args.model_n_jobs))
        LOGGER.info("Overrode model.n_jobs to %s", int(args.model_n_jobs))
    elif args.n_jobs > 1 and hasattr(model, "set_params") and "n_jobs" in model.get_params():
        model.set_params(n_jobs=1)
        LOGGER.info("Set model.n_jobs=1 to avoid nested parallelism while SHAP uses multiple processes")

    feature_names = dataset_metadata.get("feature_names") or train_info.get("feature_names")
    if feature_names is None or len(feature_names) != X_test.shape[1]:
        feature_names = [f"f{i}" for i in range(X_test.shape[1])]

    latitude_size = int(dataset_metadata["latitude_size"])
    longitude_size = int(dataset_metadata["longitude_size"])
    latitude_values = np.linspace(-89.75, 89.75, latitude_size, dtype=np.float32)
    longitude_values = np.linspace(-179.75, 179.75, longitude_size, dtype=np.float32)
    time_values = pd.to_datetime(dataset_metadata["time_values"])

    macro_labels, landcover_labels = load_label_maps(masks_dir)
    available_masks = load_available_masks(masks_dir, latitude_values, longitude_values)

    prepare_t0 = time.perf_counter()
    X_test_used = maybe_scale_features(scaler, X_test)
    prediction_test_df = build_prediction_frame(
        model=model,
        X_used=X_test_used,
        y_true=y_test,
        trace=test_trace,
        split_label="test",
        latitude_values=latitude_values,
        longitude_values=longitude_values,
        time_values=time_values,
    )
    prediction_test_df = add_mask_columns(prediction_test_df, available_masks, macro_labels, landcover_labels)

    blocks: list[dict[str, Any]] = [
        {
            "split": "test",
            "X_used": np.asarray(X_test_used, dtype=np.float32),
            "prediction_df": prediction_test_df,
        }
    ]

    if args.include_train:
        train_load_t0 = time.perf_counter()
        X_train, y_train, train_trace, train_metadata = load_train_arrays(
            input_dir=data_dir,
            mmap_mode=cfg["mmap_mode"],
            return_trace=True,
        )
        LOGGER.info("Loaded train arrays in %.2fs", time.perf_counter() - train_load_t0)
        train_feature_names = train_metadata.get("feature_names") or feature_names
        if list(train_feature_names) != list(feature_names):
            raise ValueError("Train and test feature names do not match; cannot merge SHAP inputs")

        X_train, y_train, train_trace = subsample_rows(
            X=X_train,
            y=y_train,
            trace=train_trace,
            split_label="train",
            max_rows=args.max_train_rows,
            random_state=presample_random_state + 1,
        )

        X_train_used = maybe_scale_features(scaler, X_train)
        prediction_train_df = build_prediction_frame(
            model=model,
            X_used=X_train_used,
            y_true=y_train,
            trace=train_trace,
            split_label="train",
            latitude_values=latitude_values,
            longitude_values=longitude_values,
            time_values=time_values,
        )
        prediction_train_df = add_mask_columns(prediction_train_df, available_masks, macro_labels, landcover_labels)
        blocks.insert(
            0,
            {
                "split": "train",
                "X_used": np.asarray(X_train_used, dtype=np.float32),
                "prediction_df": prediction_train_df,
            },
        )

    prepared_blocks: list[dict[str, Any]] = []
    for block in blocks:
        block_df = block["prediction_df"].copy()
        row_positions = np.arange(len(block_df), dtype=int)
        if args.target_year is not None:
            year_mask = pd.to_datetime(block_df["time"], errors="coerce").dt.year == int(args.target_year)
            block_df = block_df.loc[year_mask].copy()
            row_positions = row_positions[year_mask.to_numpy()]

        if block_df.empty:
            LOGGER.info("Split=%s has no rows after filtering; skipping", block["split"])
            continue

        prepared_blocks.append(
            {
                "split": block["split"],
                "X_used": np.asarray(block["X_used"][row_positions], dtype=np.float32),
                "prediction_df": block_df.reset_index(drop=True),
            }
        )

    if not prepared_blocks:
        raise ValueError("No rows available for SHAP after applying the selected filters")

    X_shap_input = np.concatenate([block["X_used"] for block in prepared_blocks], axis=0)
    shap_prediction_df = pd.concat([block["prediction_df"] for block in prepared_blocks], ignore_index=True)
    LOGGER.info(
        "Prepared SHAP input with %s rows across splits=%s in %.2fs",
        int(X_shap_input.shape[0]),
        sorted(shap_prediction_df["split"].dropna().unique().tolist()),
        time.perf_counter() - prepare_t0,
    )

    shap_cfg = cfg["explainability"]["shap"]
    random_state = resolve_shap_random_state(args, cfg, train_info, dataset_metadata)

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "shap_run_args.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": str(Path(args.config).resolve()),
                "data_dir": str(data_dir),
                "artifacts_dir": str(artifacts_dir),
                "masks_dir": str(masks_dir),
                "output_dir": str(output_dir),
                "n_jobs": int(args.n_jobs),
                "chunk_size": None if args.chunk_size in (None, 0) else int(args.chunk_size),
                "threads_per_worker": int(args.threads_per_worker),
                "mp_start_method": str(args.mp_start_method),
                "parallel_backend": str(args.parallel_backend),
                "include_train": bool(args.include_train),
                "target_year": None if args.target_year is None else int(args.target_year),
                "presample_random_state": int(presample_random_state),
                "max_test_rows": None if args.max_test_rows is None else int(args.max_test_rows),
                "max_train_rows": None if args.max_train_rows is None else int(args.max_train_rows),
                "group_column": args.group_column,
                "local_error_column": args.local_error_column,
                "random_state": int(random_state),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    shap_t0 = time.perf_counter()
    result = run_shap_analysis(
        model=model,
        X=X_shap_input,
        output_dir=output_dir,
        feature_names=feature_names,
        sample_metadata_df=shap_prediction_df,
        random_state=random_state,
        group_column=args.group_column,
        local_error_column=args.local_error_column,
        max_samples_per_group=(
            int(args.max_samples_per_group)
            if args.max_samples_per_group is not None
            else int(shap_cfg["max_samples_per_group"])
        ),
        max_samples_total=(
            int(args.max_samples_total)
            if args.max_samples_total is not None
            else shap_cfg["max_samples_total"]
        ),
        sample_fraction=(
            float(args.sample_fraction)
            if args.sample_fraction is not None
            else float(shap_cfg["sample_fraction"])
        ),
        min_group_samples=(
            int(args.min_group_samples)
            if args.min_group_samples is not None
            else int(shap_cfg["min_group_samples"])
        ),
        n_jobs=int(args.n_jobs),
        chunk_size=None if args.chunk_size in (None, 0) else int(args.chunk_size),
        threads_per_worker=int(args.threads_per_worker),
        mp_start_method=str(args.mp_start_method),
        parallel_backend=str(args.parallel_backend),
    )
    LOGGER.info("SHAP analysis completed in %.2fs", time.perf_counter() - shap_t0)

    summary_path = output_dir / "shap_summary.json"
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            summary_payload = json.load(f)
    else:
        summary_payload = result["summary"]

    summary_payload["script_runtime_seconds"] = float(time.perf_counter() - overall_t0)
    summary_payload["splits_included"] = sorted(shap_prediction_df["split"].dropna().unique().tolist())
    summary_payload["target_year"] = None if args.target_year is None else int(args.target_year)
    summary_payload["include_train"] = bool(args.include_train)
    summary_payload["log_file"] = str(log_file)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, ensure_ascii=False, indent=2)

    LOGGER.info("Run finished in %.2fs", time.perf_counter() - overall_t0)
    LOGGER.info("Artifacts written to %s", output_dir)
    LOGGER.info("Log written to %s", log_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
