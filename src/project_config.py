from __future__ import annotations

from pathlib import Path
import tomllib


def _load_toml(path: str | Path) -> dict:
    path = Path(path)
    with open(path, "rb") as f:
        return tomllib.load(f)


def _normalize_mask_names(mask_names: list[str]) -> list[str]:
    return [m.lower().strip() for m in mask_names]


def _normalize_variable_names(variable_names: list[str]) -> list[str]:
    return [str(v).upper().strip() for v in variable_names]


def _resolve_mlflow_config(raw: dict) -> dict:
    mlflow_raw = raw.get("mlflow", {})
    return {
        "enabled": bool(mlflow_raw.get("enabled", False)),
        "tracking_uri": mlflow_raw.get("tracking_uri"),
        "tracking_username": mlflow_raw.get("tracking_username"),
        "tracking_password": mlflow_raw.get("tracking_password"),
        "experiment_name": mlflow_raw.get("experiment_name"),
    }


def _resolve_explainability_config(raw: dict) -> dict:
    explainability_raw = raw.get("explainability", {})
    shap_raw = explainability_raw.get("shap", {})
    method = str(explainability_raw.get("method", "shap")).lower().strip()

    return {
        "enabled": bool(explainability_raw.get("enabled", False)),
        "run_after_training": bool(explainability_raw.get("run_after_training", False)),
        "method": method,
        "output_subdir": str(explainability_raw.get("output_subdir", method)).strip(),
        "group_column": explainability_raw.get("group_column", "landcover_label"),
        "local_error_column": explainability_raw.get("local_error_column", "abs_error"),
        "shap": {
            "max_samples_per_group": int(shap_raw.get("max_samples_per_group", 1000)),
            "max_samples_total": (
                None if shap_raw.get("max_samples_total") is None else int(shap_raw["max_samples_total"])
            ),
            "sample_fraction": float(shap_raw.get("sample_fraction", 0.5)),
            "min_group_samples": int(shap_raw.get("min_group_samples", 30)),
        },
    }


def _resolve_project_path(project_raw: dict, key: str, default: Path) -> Path:
    value = project_raw.get(key)
    if value is None:
        return default
    return Path(value)


def _fraction_tag(value: float) -> str:
    pct = value * 100
    if float(pct).is_integer():
        return str(int(pct))
    return str(pct).replace(".", "p")


def _normalize_data_value_type(data_value_type: str) -> str:
    data_value_type = data_value_type.lower().strip()
    if data_value_type not in {"real", "anomaly"}:
        raise ValueError("data.data_value_type debe ser 'real' o 'anomaly'.")
    return data_value_type


def build_processed_run_name(
    mask_names: list[str],
    temporal_resolution: str,
    data_value_type: str = "real",
    detrend_theil_sen: bool = False,
) -> str:
    tokens = _normalize_mask_names(mask_names)
    if not tokens:
        tokens = ["nomask"]
    tokens.append(temporal_resolution.lower().strip())
    data_value_type = _normalize_data_value_type(data_value_type)
    if data_value_type != "real":
        tokens.append(data_value_type)
    if detrend_theil_sen:
        tokens.append("theilsen")
    return "_".join(tokens)


def build_split_name(split_mode: str, train_fraction: float, test_fraction: float) -> str:
    return f"{split_mode}_tr{_fraction_tag(train_fraction)}_te{_fraction_tag(test_fraction)}"


def build_model_run_name(model_name: str, processed_run_name: str) -> str:
    model_name = model_name.lower().strip()
    return f"{model_name}_{processed_run_name}"


def resolve_data_config(config_path: str | Path = "config/data.toml") -> dict:
    config_path = Path(config_path)
    raw = _load_toml(config_path)

    project_raw = raw["project"]
    main_dir = Path(project_raw["main_dir"])
    raw_dir = _resolve_project_path(project_raw, "raw_dir", main_dir / "raw")
    processed_base_dir = _resolve_project_path(project_raw, "processed_base_dir", main_dir / "processed")
    mask_dir = _resolve_project_path(project_raw, "mask_dir", main_dir / "masks")
    temporal_resolution = raw["data"]["temporal_resolution"].lower().strip()
    data_value_type = _normalize_data_value_type(raw["data"].get("data_value_type", "real"))
    detrend_theil_sen = bool(raw["data"].get("detrend_theil_sen", False))
    mask_names = _normalize_mask_names(raw["data"].get("mask_names", []))
    variable_names = _normalize_variable_names(raw["data"]["variable_names"])
    target_name = str(raw["data"]["target_name"]).upper().strip()

    if not variable_names:
        raise ValueError("data.variable_names no puede estar vacío.")
    if target_name not in variable_names:
        raise ValueError("data.target_name debe estar incluido en data.variable_names.")

    predictor_names = [v for v in variable_names if v != target_name]

    cfg = {
        "config_path": str(config_path),
        "main_dir": main_dir,
        "raw_dir": raw_dir,
        "processed_base_dir": processed_base_dir,
        "mask_dir": mask_dir,
        "variable_names": variable_names,
        "target_name": target_name,
        "predictor_names": predictor_names,
        "temporal_resolution": temporal_resolution,
        "data_value_type": data_value_type,
        "detrend_theil_sen": detrend_theil_sen,
        "mask_names": mask_names,
        "start_year": int(raw["data"]["start_year"]),
        "end_year_inclusive": int(raw["data"]["end_year_inclusive"]),
        "dtype": raw["data"].get("dtype", "float32"),
        "roi": raw["data"].get("roi"),
        "mlflow": _resolve_mlflow_config(raw),
        "run_name": build_processed_run_name(
            mask_names,
            temporal_resolution,
            data_value_type=data_value_type,
            detrend_theil_sen=detrend_theil_sen,
        ),
    }
    cfg["output_dir"] = cfg["processed_base_dir"] / cfg["run_name"]
    return cfg


def resolve_train_config(config_path: str | Path = "config/train.toml") -> dict:
    config_path = Path(config_path)
    raw = _load_toml(config_path)

    data_cfg_path = raw["project"].get("data_config", "config/data.toml")
    data_cfg_path = (config_path.parent / data_cfg_path).resolve() if not Path(data_cfg_path).is_absolute() else Path(data_cfg_path)
    data_cfg = resolve_data_config(data_cfg_path)
    project_raw = raw.get("project", {})

    split_mode = raw["split"]["mode"].lower().strip()
    train_fraction = float(raw["split"]["train_fraction"])
    test_fraction = float(raw["split"]["test_fraction"])
    model_name = raw["model"]["name"].lower().strip()
    model_run_name = build_model_run_name(model_name, data_cfg["run_name"])
    cv_raw = raw.get("cv", {})
    cv_search_space = dict(cv_raw.get("search_space", {}).get(model_name, {}))

    cfg = {
        "config_path": str(config_path),
        "data_config_path": str(data_cfg_path),
        "data": data_cfg,
        "main_dir": data_cfg["main_dir"],
        "raw_dir": data_cfg["raw_dir"],
        "processed_run_name": data_cfg["run_name"],
        "processed_dir": data_cfg["output_dir"],
        "mask_dir": data_cfg["mask_dir"],
        "variable_names": list(data_cfg["variable_names"]),
        "target_name": data_cfg["target_name"],
        "predictor_names": list(data_cfg["predictor_names"]),
        "mlflow": dict(data_cfg["mlflow"]),
        "explainability": _resolve_explainability_config(raw),
        "split_mode": split_mode,
        "train_fraction": train_fraction,
        "test_fraction": test_fraction,
        "min_valid_fraction": float(raw["split"].get("min_valid_fraction", 0.0)),
        "seed": int(raw["split"].get("seed", 42)),
        "split_name": build_split_name(split_mode, train_fraction, test_fraction),
        "model_name": model_name,
        "model_run_name": model_run_name,
        "scaler_name": raw["model"].get("scaler"),
        "random_state": int(raw["model"].get("random_state", 42)),
        "model_params": dict(raw["model"].get("params", {})),
        "cv_config": {
            "enabled": bool(cv_raw.get("enabled", False)),
            "n_splits": int(cv_raw.get("n_splits", 5)),
            "n_iter": int(cv_raw.get("n_iter", 20)),
            "scoring": cv_raw.get("scoring", "neg_root_mean_squared_error"),
            "n_jobs": int(cv_raw.get("n_jobs", 1)),
            "verbose": int(cv_raw.get("verbose", 1)),
            "search_space": cv_search_space,
        },
        "mmap_mode": raw.get("runtime", {}).get("mmap_mode", "r"),
    }
    model_root_dir = _resolve_project_path(project_raw, "model_base_dir", cfg["main_dir"] / "models")
    cfg["model_base_root_dir"] = model_root_dir
    cfg["model_base_dir"] = model_root_dir / cfg["model_run_name"]
    cfg["model_dir"] = cfg["model_base_dir"] / cfg["split_mode"]
    cfg["model_data_dir"] = cfg["model_dir"] / "data"
    cfg["model_artifacts_dir"] = cfg["model_dir"] / "artifacts"
    cfg["model_figures_dir"] = cfg["model_dir"] / "figures"
    return cfg
