from __future__ import annotations

from pathlib import Path
import os

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

from src.reproducibility.config import ReproducibilityConfig


_PATH_KEYS = {
    "main_dir", "raw_dir", "processed_base_dir", "processed_dir", "mask_dir",
    "output_dir", "model_base_root_dir", "model_base_dir", "model_dir",
    "model_data_dir", "model_artifacts_dir", "model_figures_dir",
}


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

    reproducibility = ReproducibilityConfig.from_mapping(
        raw.get("reproducibility"),
        base_dir=Path.cwd(),
    )
    if reproducibility.input_catalogue is None:
        reproducibility = ReproducibilityConfig(
            create_rocrate=reproducibility.create_rocrate,
            reproducible_run=reproducibility.reproducible_run,
            output_dir=reproducibility.output_dir,
            upstream_rocrate=reproducibility.upstream_rocrate,
            input_catalogue=data_cfg_path,
            model_artifact=reproducibility.model_artifact,
            croissant=reproducibility.croissant,
            code=reproducibility.code,
            licenses=reproducibility.licenses,
            environment=reproducibility.environment,
            upstream=reproducibility.upstream,
            inputs=reproducibility.inputs,
            verification=reproducibility.verification,
        )

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
        "reproducibility": reproducibility,
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
    reproduction_work_dir = os.getenv("PROFECIA_WORK_DIR")
    if reproduction_work_dir:
        work = Path(reproduction_work_dir).resolve()
        cfg["raw_dir"] = work / "inputs"
        cfg["mask_dir"] = work / "derived-inputs" / "masks"
        cfg["processed_dir"] = work / "processed"
        cfg["data"]["raw_dir"] = cfg["raw_dir"]
        cfg["data"]["mask_dir"] = cfg["mask_dir"]
        cfg["data"]["output_dir"] = cfg["processed_dir"]
        cfg["model_base_root_dir"] = work / "outputs"
        cfg["model_base_dir"] = cfg["model_base_root_dir"] / cfg["model_run_name"]
        cfg["model_dir"] = cfg["model_base_dir"] / cfg["split_mode"]
        cfg["model_data_dir"] = cfg["model_dir"] / "data"
        cfg["model_artifacts_dir"] = cfg["model_dir"] / "artifacts"
        cfg["model_figures_dir"] = cfg["model_dir"] / "figures"
    return cfg


def resolve_effective_train_config(config_path: str | Path, work_dir: str | Path) -> dict:
    """Load the redacted, fully resolved configuration packaged for reproduction."""
    import json

    config_path = Path(config_path).resolve()
    crate_dir = config_path.parent.parent
    work = Path(work_dir).resolve()
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    if raw.get("effective_config_format") != "profecia-reproducibility-1":
        raise ValueError(f"Unsupported effective configuration format: {config_path}")

    def restore(value):
        if isinstance(value, dict):
            return {key: restore(item) for key, item in value.items()}
        if isinstance(value, list):
            return [restore(item) for item in value]
        if isinstance(value, str):
            return value.replace("${WORK_DIR}", str(work))
        return value

    cfg = restore(raw)
    cfg.pop("effective_config_format", None)
    cfg.pop("work_dir_variable", None)
    for key in _PATH_KEYS:
        if cfg.get(key):
            cfg[key] = Path(cfg[key])
    if isinstance(cfg.get("data"), dict):
        for key in _PATH_KEYS:
            if cfg["data"].get(key):
                cfg["data"][key] = Path(cfg["data"][key])
    cfg["config_path"] = str(crate_dir / "config" / "train.toml")
    cfg["data_config_path"] = str(crate_dir / "config" / "data.toml")
    cfg["reproducibility"] = ReproducibilityConfig.from_mapping(
        cfg.get("reproducibility"), base_dir=crate_dir
    )
    return cfg
