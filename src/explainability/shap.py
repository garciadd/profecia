from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


LOGGER = logging.getLogger(__name__)
SUPPORTED_TREE_MODEL_NAMES = {
    "RandomForestRegressor",
    "HistGradientBoostingRegressor",
}


def _normalize_feature_names(feature_names: list[str] | None, n_features: int) -> list[str]:
    if feature_names is None or len(feature_names) != n_features:
        return [f"f{i}" for i in range(n_features)]
    return [str(name) for name in feature_names]


def _ensure_supported_model(model) -> str:
    model_name = model.__class__.__name__
    if model_name not in SUPPORTED_TREE_MODEL_NAMES:
        raise TypeError(
            "SHAP explainability is currently configured for tree-based regressors. "
            f"Supported models: {sorted(SUPPORTED_TREE_MODEL_NAMES)}. "
            f"Loaded model: {model_name}."
        )
    return model_name


def _coerce_shap_values(raw_shap_values, n_rows: int) -> np.ndarray:
    if isinstance(raw_shap_values, list):
        raw_shap_values = raw_shap_values[0]

    shap_values = np.asarray(raw_shap_values)
    if shap_values.ndim == 3:
        if shap_values.shape[0] == n_rows:
            shap_values = shap_values[:, :, 0]
        else:
            shap_values = shap_values[0]

    if shap_values.ndim != 2:
        raise ValueError(f"Unexpected SHAP values shape: {shap_values.shape}")

    return shap_values


def _select_shap_indices(
    n_rows: int,
    sample_metadata_df: pd.DataFrame | None,
    group_column: str | None,
    max_samples_per_group: int,
    max_samples_total: int | None,
    sample_fraction: float,
    random_state: int,
) -> np.ndarray:
    rng = np.random.default_rng(random_state)

    if sample_metadata_df is not None and group_column and group_column in sample_metadata_df.columns:
        valid_df = sample_metadata_df[
            sample_metadata_df[group_column].notna() & (sample_metadata_df[group_column].astype(str) != "NoData")
        ]
        if not valid_df.empty:
            selected_indices: list[int] = []
            group_counts = valid_df[group_column].value_counts()
            LOGGER.info("Selecting SHAP sample stratified by %s", group_column)
            for group_value, n_total in group_counts.items():
                class_indices = valid_df[valid_df[group_column] == group_value].index.to_numpy(dtype=int)
                if n_total >= max_samples_per_group * 2:
                    n_select = max_samples_per_group
                else:
                    n_select = max(1, int(n_total * sample_fraction))

                sampled_idx = rng.choice(class_indices, size=n_select, replace=False)
                selected_indices.extend(sampled_idx.tolist())
                LOGGER.info(
                    "SHAP group=%s total=%s selected=%s",
                    group_value,
                    int(n_total),
                    int(n_select),
                )

            shap_idx = np.sort(np.asarray(selected_indices, dtype=int))
            if max_samples_total is not None and shap_idx.size > max_samples_total:
                shap_idx = np.sort(rng.choice(shap_idx, size=max_samples_total, replace=False))
            return shap_idx

    if max_samples_total is None:
        max_samples_total = min(n_rows, max_samples_per_group)

    if max_samples_total >= n_rows:
        return np.arange(n_rows, dtype=int)

    return np.sort(rng.choice(n_rows, size=max_samples_total, replace=False))


def _save_summary_plot(shap_values: np.ndarray, X_shap_df: pd.DataFrame, output_path: Path, plot_type: str | None = None) -> None:
    import matplotlib.pyplot as plt
    import shap

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_shap_df, plot_type=plot_type, show=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def _save_force_plot(
    expected_value: float,
    shap_row: np.ndarray,
    feature_row: pd.Series,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt
    import shap

    plt.figure(figsize=(14, 3.5))
    shap.force_plot(expected_value, shap_row, feature_row, matplotlib=True, show=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def run_shap_analysis(
    model,
    X: np.ndarray,
    output_dir: str | Path,
    feature_names: list[str] | None = None,
    sample_metadata_df: pd.DataFrame | None = None,
    random_state: int = 42,
    group_column: str | None = "landcover_label",
    local_error_column: str = "abs_error",
    max_samples_per_group: int = 1000,
    max_samples_total: int | None = None,
    sample_fraction: float = 0.5,
    min_group_samples: int = 30,
) -> dict[str, Any]:
    import shap

    if X.ndim != 2:
        raise ValueError(f"X must be a 2D array. Received: {X.shape}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_name = _ensure_supported_model(model)
    feature_names = _normalize_feature_names(feature_names, X.shape[1])
    shap_idx = _select_shap_indices(
        n_rows=X.shape[0],
        sample_metadata_df=sample_metadata_df,
        group_column=group_column,
        max_samples_per_group=max_samples_per_group,
        max_samples_total=max_samples_total,
        sample_fraction=sample_fraction,
        random_state=random_state,
    )

    if shap_idx.size == 0:
        raise ValueError("No rows were selected for SHAP analysis.")

    X_shap = np.asarray(X[shap_idx], dtype=np.float32)
    X_shap_df = pd.DataFrame(X_shap, columns=feature_names)

    if sample_metadata_df is not None:
        shap_meta_df = sample_metadata_df.iloc[shap_idx].reset_index(drop=True).copy()
    else:
        shap_meta_df = pd.DataFrame(index=np.arange(len(shap_idx)))

    explainer = shap.TreeExplainer(model)
    shap_values = _coerce_shap_values(explainer.shap_values(X_shap), n_rows=X_shap.shape[0])
    expected_value = float(np.ravel(explainer.expected_value)[0])

    shap_importance_df = (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": np.abs(shap_values).mean(axis=0)})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    shap_values_df = pd.DataFrame(shap_values, columns=[f"shap_{name}" for name in feature_names])
    shap_full_df = pd.concat([shap_meta_df, X_shap_df, shap_values_df], axis=1)

    np.save(output_dir / "shap_values.npy", shap_values)
    np.save(output_dir / "shap_sample_indices.npy", shap_idx)
    X_shap_df.to_csv(output_dir / "X_shap_sample.csv", index=False)
    shap_meta_df.to_csv(output_dir / "shap_sample_metadata.csv", index=False)
    shap_importance_df.to_csv(output_dir / "shap_feature_importance.csv", index=False)
    shap_values_df.to_csv(output_dir / "shap_values.csv", index=False)
    shap_full_df.to_csv(output_dir / "shap_full_dataset.csv", index=False)

    local_rank = 0
    if local_error_column in shap_meta_df.columns:
        local_rank = int(pd.to_numeric(shap_meta_df[local_error_column], errors="coerce").fillna(-np.inf).idxmax())

    local_feature_row = X_shap_df.iloc[[local_rank]].reset_index(drop=True)
    local_shap_df = (
        pd.DataFrame(
            {
                "feature": feature_names,
                "feature_value": X_shap_df.iloc[local_rank].to_numpy(),
                "shap_value": shap_values[local_rank],
                "abs_shap_value": np.abs(shap_values[local_rank]),
            }
        )
        .sort_values("abs_shap_value", ascending=False)
        .reset_index(drop=True)
    )
    local_meta_row = shap_meta_df.iloc[[local_rank]].reset_index(drop=True)
    local_meta_row.to_csv(output_dir / "shap_local_observation_metadata.csv", index=False)
    local_feature_row.to_csv(output_dir / "shap_local_observation_features.csv", index=False)
    local_shap_df.to_csv(output_dir / "shap_local_observation_values.csv", index=False)

    landcover_summaries: list[dict[str, Any]] = []
    if group_column and group_column in shap_meta_df.columns:
        for group_value, group in shap_meta_df.groupby(group_column, sort=True):
            if len(group) < min_group_samples:
                LOGGER.info(
                    "Skipping SHAP summary for %s=%s because only %s samples are available",
                    group_column,
                    group_value,
                    int(len(group)),
                )
                continue

            group_positions = group.index.to_numpy(dtype=int)
            group_slug = str(group_value).lower().replace(" ", "_").replace("/", "-")
            group_importance_df = (
                pd.DataFrame(
                    {
                        "feature": feature_names,
                        "mean_abs_shap": np.abs(shap_values[group_positions]).mean(axis=0),
                        group_column: group_value,
                        "n_samples": int(len(group_positions)),
                    }
                )
                .sort_values("mean_abs_shap", ascending=False)
                .reset_index(drop=True)
            )
            csv_name = f"shap_feature_importance_{group_column}_{group_slug}.csv"
            png_name = f"shap_feature_importance_{group_column}_{group_slug}.png"
            group_importance_df.to_csv(output_dir / csv_name, index=False)
            _save_summary_plot(
                shap_values[group_positions],
                X_shap_df.iloc[group_positions],
                output_dir / png_name,
                plot_type="bar",
            )
            landcover_summaries.append(
                {
                    group_column: group_value,
                    "n_samples": int(len(group_positions)),
                    "file": csv_name,
                    "plot_file": png_name,
                }
            )

    pd.DataFrame(landcover_summaries).to_csv(output_dir / "shap_landcover_summary.csv", index=False)

    summary_payload = {
        "model_name": model_name,
        "n_shap_samples": int(len(shap_idx)),
        "n_features": int(len(feature_names)),
        "random_state": int(random_state),
        "expected_value": expected_value,
        "feature_names": feature_names,
        "group_column": group_column,
        "local_error_column": local_error_column,
    }
    with open(output_dir / "shap_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, ensure_ascii=False, indent=2)

    _save_summary_plot(shap_values, X_shap_df, output_dir / "shap_summary_bar.png", plot_type="bar")
    _save_summary_plot(shap_values, X_shap_df, output_dir / "shap_summary_beeswarm.png")
    _save_force_plot(
        expected_value=expected_value,
        shap_row=shap_values[local_rank],
        feature_row=X_shap_df.iloc[local_rank],
        output_path=output_dir / "shap_local_force.png",
    )

    LOGGER.info("Saved SHAP artifacts to %s", output_dir)
    return {
        "output_dir": output_dir,
        "summary": summary_payload,
        "shap_indices": shap_idx,
        "shap_importance_df": shap_importance_df,
        "shap_meta_df": shap_meta_df,
        "local_shap_df": local_shap_df,
    }


def run_configured_shap_analysis(
    train_cfg: dict[str, Any],
    model,
    X: np.ndarray,
    train_info: dict[str, Any] | None = None,
    dataset_metadata: dict[str, Any] | None = None,
    prediction_df: pd.DataFrame | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any] | None:
    explainability_cfg = train_cfg.get("explainability", {})
    if not explainability_cfg.get("enabled", False):
        LOGGER.info("Explainability disabled in config; skipping SHAP analysis")
        return None

    if not explainability_cfg.get("run_after_training", False):
        LOGGER.info("Explainability auto-run disabled in config; skipping SHAP analysis")
        return None

    if explainability_cfg.get("method", "shap") != "shap":
        raise ValueError(
            f"Unsupported explainability method: {explainability_cfg.get('method')}. Only 'shap' is available."
        )

    shap_cfg = explainability_cfg.get("shap", {})
    train_info = train_info or {}
    dataset_metadata = dataset_metadata or {}
    feature_names = dataset_metadata.get("feature_names") or train_info.get("feature_names")
    random_state = int(
        train_info.get("random_state")
        or train_cfg.get("random_state")
        or dataset_metadata.get("split_metadata", {}).get("seed", 42)
    )

    if output_dir is None:
        output_dir = Path(train_cfg["model_artifacts_dir"]) / explainability_cfg.get("output_subdir", "shap")

    return run_shap_analysis(
        model=model,
        X=X,
        output_dir=output_dir,
        feature_names=feature_names,
        sample_metadata_df=prediction_df,
        random_state=random_state,
        group_column=explainability_cfg.get("group_column", "landcover_label"),
        local_error_column=explainability_cfg.get("local_error_column", "abs_error"),
        max_samples_per_group=int(shap_cfg.get("max_samples_per_group", 1000)),
        max_samples_total=shap_cfg.get("max_samples_total"),
        sample_fraction=float(shap_cfg.get("sample_fraction", 0.5)),
        min_group_samples=int(shap_cfg.get("min_group_samples", 30)),
    )
