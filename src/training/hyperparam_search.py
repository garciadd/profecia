from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline

from src.models.tabular_models import build_model
from src.training.train import build_scaler


def _build_cv(split_mode: str, n_splits: int, seed: int, groups: np.ndarray | None):
    split_mode = split_mode.lower().strip()

    if n_splits < 2:
        raise ValueError("cv.n_splits debe ser >= 2.")

    if split_mode == "spatial_pixel":
        if groups is None:
            raise ValueError("La CV spatial_pixel necesita pixel_id_train como groups.")

        n_groups = int(np.unique(groups).size)
        if n_groups < n_splits:
            raise ValueError(
                f"No hay suficientes pixeles de train para GroupKFold: {n_groups} grupos para {n_splits} folds."
            )
        return GroupKFold(n_splits=n_splits)

    if split_mode == "random_observation":
        return KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    raise ValueError(
        f"split_mode no soportado para CV: '{split_mode}'. Usa 'spatial_pixel' o 'random_observation'."
    )


def _build_pipeline(
    model_name: str,
    scaler_name: str | None,
    random_state: int,
    base_model_params: dict[str, Any] | None,
) -> Pipeline:
    steps = []
    scaler = build_scaler(scaler_name)
    if scaler is not None:
        steps.append(("scaler", scaler))

    model = build_model(
        model_name=model_name,
        random_state=random_state,
        **(base_model_params or {}),
    )
    steps.append(("model", model))
    return Pipeline(steps)


def _prefix_search_space(search_space: dict[str, Any]) -> dict[str, Any]:
    return {
        key if key.startswith("model__") else f"model__{key}": value
        for key, value in search_space.items()
    }


def _strip_model_prefix(params: dict[str, Any]) -> dict[str, Any]:
    clean = {}
    for key, value in params.items():
        clean[key.removeprefix("model__")] = value
    return clean


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def run_hyperparameter_search(
    X_train: np.ndarray,
    y_train: np.ndarray,
    split_mode: str,
    model_name: str,
    scaler_name: str | None,
    random_state: int,
    base_model_params: dict[str, Any] | None,
    search_space: dict[str, Any],
    pixel_id_train: np.ndarray | None = None,
    cv_config: dict[str, Any] | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """
    Ejecuta RandomizedSearchCV sobre X_train/y_train.

    La estrategia de folds depende de split_mode:
    - spatial_pixel: GroupKFold por pixel_id_train.
    - random_observation: KFold aleatorio sobre observaciones.
    """
    if X_train.ndim != 2:
        raise ValueError(f"X_train debe ser 2D. Recibido: {X_train.shape}")
    y_train = np.asarray(y_train).reshape(-1)
    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError("X_train e y_train no tienen el mismo numero de filas.")
    if not search_space:
        raise ValueError("cv.search_space no puede estar vacio si cv.enabled=true.")

    cv_config = cv_config or {}
    n_splits = int(cv_config.get("n_splits", 5))
    n_iter = int(cv_config.get("n_iter", 20))
    scoring = cv_config.get("scoring", "neg_root_mean_squared_error")
    n_jobs = int(cv_config.get("n_jobs", 1))
    verbose = int(cv_config.get("verbose", 1))

    groups = np.asarray(pixel_id_train) if pixel_id_train is not None else None
    if groups is not None and groups.shape[0] != X_train.shape[0]:
        raise ValueError("pixel_id_train no tiene el mismo numero de filas que X_train.")

    cv = _build_cv(
        split_mode=split_mode,
        n_splits=n_splits,
        seed=random_state,
        groups=groups,
    )
    pipeline = _build_pipeline(
        model_name=model_name,
        scaler_name=scaler_name,
        random_state=random_state,
        base_model_params=base_model_params,
    )

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=_prefix_search_space(search_space),
        n_iter=n_iter,
        scoring=scoring,
        n_jobs=n_jobs,
        cv=cv,
        refit=True,
        random_state=random_state,
        verbose=verbose,
        return_train_score=True,
    )

    fit_kwargs = {}
    if split_mode.lower().strip() == "spatial_pixel":
        fit_kwargs["groups"] = groups

    search.fit(X_train, y_train, **fit_kwargs)

    best_params = _strip_model_prefix(search.best_params_)
    cv_results_df = pd.DataFrame(search.cv_results_)

    output_paths: dict[str, str] = {}
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        cv_results_path = output_dir / "cv_results.csv"
        best_params_path = output_dir / "best_params.json"
        summary_path = output_dir / "cv_summary.json"

        cv_results_df.to_csv(cv_results_path, index=False)
        output_paths["cv_results_path"] = str(cv_results_path)

        with open(best_params_path, "w", encoding="utf-8") as f:
            json.dump(_json_safe(best_params), f, ensure_ascii=False, indent=2)
        output_paths["best_params_path"] = str(best_params_path)

        summary = {
            "enabled": True,
            "split_mode": split_mode,
            "cv_class": cv.__class__.__name__,
            "n_splits": n_splits,
            "n_iter": n_iter,
            "scoring": scoring,
            "best_score": float(search.best_score_),
            "best_params": _json_safe(best_params),
            "base_model_params": _json_safe(base_model_params or {}),
            "search_space": _json_safe(search_space),
            **output_paths,
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        output_paths["summary_path"] = str(summary_path)

    return {
        "best_params": best_params,
        "best_score": float(search.best_score_),
        "best_estimator": search.best_estimator_,
        "cv_results": cv_results_df,
        "search": search,
        "metadata": {
            "enabled": True,
            "split_mode": split_mode,
            "cv_class": cv.__class__.__name__,
            "n_splits": n_splits,
            "n_iter": n_iter,
            "scoring": scoring,
            "best_score": float(search.best_score_),
            "best_params": _json_safe(best_params),
            "base_model_params": _json_safe(base_model_params or {}),
            "search_space": _json_safe(search_space),
            **output_paths,
        },
    }
