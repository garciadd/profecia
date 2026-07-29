from __future__ import annotations

from pathlib import Path
import tomllib


def _load_toml(path: str | Path) -> dict:
    path = Path(path)
    with open(path, "rb") as f:
        return tomllib.load(f)

def _resolve_path(value: str | Path, base_dir: Path) -> Path:
    """Resuelve una ruta absoluta o relativa respecto a base_dir."""

    path = Path(value).expanduser()

    if path.is_absolute():
        return path.resolve()

    return (base_dir / path).resolve()

def _normalize_mask_names(mask_names: list[str]) -> list[str]:
    return [m.lower().strip() for m in mask_names]


def _normalize_variable_names(variable_names: list[str]) -> list[str]:
    return [str(v).upper().strip() for v in variable_names]

def _normalize_categorical_filters(
    categorical_filters: dict | None,
) -> dict[str, list[int]]:
    """Normaliza los identificadores de clases categóricas."""

    normalized: dict[str, list[int]] = {}

    for name, values in (categorical_filters or {}).items():
        filter_name = str(name).lower().strip()

        selected_ids: list[int] = []
        for value in values or []:
            class_id = int(value)

            if class_id not in selected_ids:
                selected_ids.append(class_id)

        normalized[filter_name] = selected_ids

    return normalized

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
    data_value_type = str(data_value_type).lower().strip()

    allowed = {"real", "anomaly", "trend"}

    if data_value_type not in allowed:
        raise ValueError("data.data_value_type debe ser 'real', 'anomaly' o 'trend'.")

    return data_value_type

def build_processed_run_name(
    mask_names: list[str],
    start_year: int,
    end_year_inclusive: int,
    temporal_resolution: str,
    data_value_type: str = "real",
) -> str:
    mask_tokens = _normalize_mask_names(mask_names)

    if not mask_tokens:
        mask_tokens = ["nomask"]

    if end_year_inclusive < start_year:
        raise ValueError("end_year_inclusive debe ser mayor o igual que start_year.")

    temporal_resolution = temporal_resolution.lower().strip()
    if temporal_resolution not in {"monthly", "annual"}:
        raise ValueError(
            "temporal_resolution debe ser 'monthly' o 'annual'."
        )

    data_value_type = _normalize_data_value_type(data_value_type)

    tokens = [
        *mask_tokens,
        temporal_resolution,
        data_value_type,
        f"{start_year}_{end_year_inclusive}",
    ]

    return "_".join(tokens)

def build_split_name(split_mode: str, train_fraction: float, test_fraction: float) -> str:
    return f"{split_mode}_tr{_fraction_tag(train_fraction)}_te{_fraction_tag(test_fraction)}"


def build_model_run_name(model_name: str, processed_run_name: str) -> str:
    model_name = model_name.lower().strip()
    return f"{model_name}_{processed_run_name}"


def resolve_data_config(
    config_path: str | Path = "config/data.toml",
    paths_config_path: str | Path | None = None,
) -> dict:
    """Lee data.toml y paths.toml y devuelve una configuración unificada.

    data.toml contiene únicamente la selección activa del experimento.

    paths.toml contiene las rutas, el catálogo de variables, las máscaras
    disponibles y la configuración global del proyecto.
    """

    config_path = Path(config_path).expanduser().resolve()

    if paths_config_path is None:
        paths_config_path = config_path.with_name("paths.toml")
    else:
        paths_config_path = Path(
            paths_config_path
        ).expanduser().resolve()

    data_raw = _load_toml(config_path)
    paths_raw = _load_toml(paths_config_path)

    if "data" not in data_raw:
        raise ValueError(
            f"No existe la sección [data] en {config_path}"
        )

    if "paths" not in paths_raw:
        raise ValueError(
            f"No existe la sección [paths] en "
            f"{paths_config_path}"
        )

    data_section = data_raw["data"]
    paths_section = paths_raw["paths"]

    # ------------------------------------------------------------------
    # Rutas generales
    # ------------------------------------------------------------------

    project_dir_value = paths_section.get("project_dir", paths_config_path.parent.parent)
    project_dir = _resolve_path(project_dir_value, paths_config_path.parent)
    required_paths = {"data_dir", "processed_base_dir", "mask_dir"}
    missing_paths = sorted(required_paths - set(paths_section))

    if missing_paths:
        raise ValueError(
            "Faltan rutas obligatorias en [paths]: "
            + ", ".join(missing_paths)
        )

    data_dir = _resolve_path(paths_section["data_dir"], project_dir)
    processed_base_dir = _resolve_path(paths_section["processed_base_dir"], project_dir)
    mask_dir = _resolve_path(paths_section["mask_dir"], project_dir)

    # ------------------------------------------------------------------
    # Catálogo de variables
    # ------------------------------------------------------------------

    files_section = paths_raw.get("files", {}) or {}

    available_variables: set[str] = set()
    variable_file_map: dict[str, str] = {}
    variable_groups: dict[str, list[str]] = {}

    for group_name, group_files in files_section.items():
        if group_name == "masks":
            continue

        if not isinstance(group_files, dict):
            continue

        group_variables: list[str] = []

        for variable_name, relative_path in group_files.items():
            normalized_name = str(
                variable_name
            ).upper().strip()

            available_variables.add(normalized_name)
            group_variables.append(normalized_name)
            variable_file_map[normalized_name] = str(
                relative_path
            )

        variable_groups[str(group_name)] = group_variables

    variable_names = _normalize_variable_names(
        data_section.get("variable_names", [])
    )

    target_name = str(
        data_section.get("target_name", "LAI")
    ).upper().strip()

    if not variable_names:
        raise ValueError(
            "data.variable_names no puede estar vacío."
        )

    if target_name not in variable_names:
        raise ValueError(
            "data.target_name debe estar incluido en "
            "data.variable_names."
        )

    unknown_variables = sorted(
        set(variable_names) - available_variables
    )

    if unknown_variables:
        raise ValueError(
            "Las siguientes variables seleccionadas no existen "
            "en paths.toml: "
            + ", ".join(unknown_variables)
        )

    predictor_names = [
        variable
        for variable in variable_names
        if variable != target_name
    ]

    # ------------------------------------------------------------------
    # Máscaras binarias
    # ------------------------------------------------------------------

    masks_files_section = (files_section.get("masks", {}) or {})
    binary_mask_file_map = dict(masks_files_section.get("binary", {}) or {})
    categorical_mask_file_map = dict(masks_files_section.get("categorical", {}) or {})
    mask_names = _normalize_mask_names(data_section.get("mask_names", []))
    unknown_binary_masks = sorted(set(mask_names) - set(binary_mask_file_map))

    if unknown_binary_masks:
        raise ValueError(
            "Las siguientes máscaras binarias no existen "
            "en paths.toml: "
            + ", ".join(unknown_binary_masks)
        )

    # ------------------------------------------------------------------
    # Filtros categóricos
    # ------------------------------------------------------------------

    categorical_filters = _normalize_categorical_filters(
        data_section.get("categorical_filters", {})
    )

    categorical_metadata = (
        (paths_raw.get("masks", {}) or {})
        .get("categorical", {})
        or {}
    )

    unknown_categorical_masks = sorted(
        set(categorical_filters)
        - set(categorical_mask_file_map)
    )

    if unknown_categorical_masks:
        raise ValueError(
            "Los siguientes filtros categóricos no existen "
            "en paths.toml: "
            + ", ".join(unknown_categorical_masks)
        )

    for mask_name, selected_classes in (
        categorical_filters.items()
    ):
        mask_metadata = (
            categorical_metadata.get(mask_name, {}) or {}
        )

        raw_classes = mask_metadata.get("classes", {}) or {}

        valid_class_ids = {
            int(class_id)
            for class_id in raw_classes
        }

        invalid_classes = sorted(
            set(selected_classes) - valid_class_ids
        )

        if invalid_classes:
            raise ValueError(
                f"El filtro categórico {mask_name!r} contiene "
                "clases no definidas en paths.toml: "
                + ", ".join(
                    str(class_id)
                    for class_id in invalid_classes
                )
            )

    # ------------------------------------------------------------------
    # Configuración del experimento
    # ------------------------------------------------------------------

    temporal_resolution = str(
        data_section.get(
            "temporal_resolution",
            "annual",
        )
    ).lower().strip()

    if temporal_resolution not in {"monthly", "annual"}:
        raise ValueError(
            "data.temporal_resolution debe ser "
            "'monthly' o 'annual'."
        )

    data_value_type = _normalize_data_value_type(
        data_section.get(
            "data_value_type",
            "real",
        )
    )

    start_year = int(data_section.get("start_year", 1982))
    end_year_inclusive = int(data_section.get("end_year_inclusive", 2022))
    if start_year > end_year_inclusive:
        raise ValueError("data.start_year debe ser menor o igual que data.end_year_inclusive.")

    roi = data_section.get("roi")

    run_name = build_processed_run_name(
        mask_names=mask_names,
        start_year=start_year,
        end_year_inclusive=end_year_inclusive,
        temporal_resolution=temporal_resolution,
        data_value_type=data_value_type,
    )

    binary_mask_metadata = ((paths_raw.get("masks", {}) or {}).get("binary", {}) or {})

    cfg = {
    # Archivos de configuración
    "config_path": str(config_path),
    "paths_config_path": str(paths_config_path),

    # Rutas
    "project_dir": project_dir,
    "main_dir": project_dir,
    "data_dir": data_dir,
    "raw_dir": data_dir,
    "processed_base_dir": processed_base_dir,
    "mask_dir": mask_dir,

    # Catálogo de variables
    "available_variables": sorted(available_variables),
    "variable_file_map": variable_file_map,
    "variable_groups": variable_groups,

    # Catálogo de máscaras
    "binary_mask_file_map": binary_mask_file_map,
    "binary_mask_metadata": binary_mask_metadata,
    "categorical_mask_file_map": categorical_mask_file_map,
    "categorical_mask_metadata": categorical_metadata,

    # Selección activa
    "variable_names": variable_names,
    "target_name": target_name,
    "predictor_names": predictor_names,
    "temporal_resolution": temporal_resolution,
    "data_value_type": data_value_type,
    "mask_names": mask_names,
    "categorical_filters": categorical_filters,
    "start_year": start_year,
    "end_year_inclusive": end_year_inclusive,
    "dtype": str(data_section.get("dtype", "float32")),
    "roi": roi,

    # Configuración global
    "mlflow": _resolve_mlflow_config(paths_raw),

    # Salida
    "run_name": run_name,
}

    cfg["output_dir"] = processed_base_dir / run_name

    return cfg


def resolve_train_config(config_path: str | Path = "config/train.toml") -> dict:
    config_path = Path(config_path).expanduser().resolve()
    raw = _load_toml(config_path)

    data_cfg_path = raw["project"].get("data_config", "config/data.toml")
    data_cfg_path = (config_path.parent / data_cfg_path).resolve() if not Path(data_cfg_path).is_absolute() else Path(data_cfg_path)
    paths_cfg_value = raw.get("project", {}).get("paths_config")
    paths_cfg_path = None
    if paths_cfg_value is not None:
        paths_cfg_path = Path(paths_cfg_value).expanduser()
        if not paths_cfg_path.is_absolute():
            paths_cfg_path = (config_path.parent / paths_cfg_path).resolve()
    data_cfg = resolve_data_config(data_cfg_path, paths_cfg_path)
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
    model_root_dir = _resolve_project_path(
        project_raw,
        "model_base_dir",
        cfg["main_dir"] / "models",
    ).expanduser()
    if not model_root_dir.is_absolute():
        model_root_dir = (config_path.parent / model_root_dir).resolve()
    cfg["model_base_root_dir"] = model_root_dir
    cfg["model_base_dir"] = model_root_dir / cfg["model_run_name"]
    cfg["model_dir"] = cfg["model_base_dir"] / cfg["split_mode"]
    cfg["model_data_dir"] = cfg["model_dir"] / "data"
    cfg["model_artifacts_dir"] = cfg["model_dir"] / "artifacts"
    cfg["model_figures_dir"] = cfg["model_dir"] / "figures"
    return cfg
