"""Lectura de config/paths.toml y construcción del catálogo dinámico de PROFECIA."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

try:  # Python >= 3.11
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore

GROUP_LABELS: dict[str, str] = {
    "climatic": "Climáticas",
    "human": "Humanas",
    "soil": "Suelo / topografía / biodiversidad",
    "landcover": "Cobertura del suelo",
}


def load_paths_config(path: str | Path) -> dict[str, Any]:
    """Carga config/paths.toml."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No existe el catálogo de rutas/variables: {path}")
    with path.open("rb") as f:
        return tomllib.load(f)


def get_project_paths(paths_config: dict[str, Any]) -> dict[str, Path]:
    """Devuelve rutas raíz del catálogo como Path.

    Las rutas pueden ser absolutas o relativas. La resolución final frente al proyecto
    la decide el llamador si lo necesita.
    """
    raw = paths_config.get("paths", {}) or {}
    return {key: Path(value).expanduser() for key, value in raw.items() if isinstance(value, str)}


def get_target_file_map(paths_config: dict[str, Any]) -> dict[str, str]:
    return dict(((paths_config.get("files", {}) or {}).get("target", {}) or {}))


def get_target_names(paths_config: dict[str, Any]) -> list[str]:
    targets = list(get_target_file_map(paths_config).keys())
    if "LAI" in targets:
        return ["LAI"]
    return targets or ["LAI"]


def get_predictor_file_groups(paths_config: dict[str, Any]) -> "OrderedDict[str, dict[str, str]]":
    """Devuelve los grupos de predictores disponibles con sus ficheros.

    Excluye [files.target] y [files.masks].
    """
    files = paths_config.get("files", {}) or {}
    out: "OrderedDict[str, dict[str, str]]" = OrderedDict()
    for section, values in files.items():
        if section in {"target", "masks"}:
            continue
        if not isinstance(values, dict):
            continue
        label = GROUP_LABELS.get(section, section)
        out[label] = dict(values)
    return out


def get_predictor_groups(paths_config: dict[str, Any]) -> "OrderedDict[str, list[str]]":
    groups = get_predictor_file_groups(paths_config)
    return OrderedDict((label, list(file_map.keys())) for label, file_map in groups.items())


def flatten_predictor_groups(variable_groups: dict[str, list[str]]) -> list[str]:
    out: list[str] = []
    for variables in variable_groups.values():
        for var in variables:
            if var not in out:
                out.append(var)
    return out


def get_all_available_variables(paths_config: dict[str, Any], include_target: bool = True) -> list[str]:
    out: list[str] = []
    if include_target:
        out.extend(get_target_names(paths_config))
    for variables in get_predictor_groups(paths_config).values():
        for var in variables:
            if var not in out:
                out.append(var)
    return out


def get_binary_mask_file_map(paths_config: dict[str, Any]) -> dict[str, str]:
    files = paths_config.get("files", {}) or {}
    masks = files.get("masks", {}) or {}
    return dict(masks.get("binary", {}) or {})


def get_categorical_mask_file_map(paths_config: dict[str, Any]) -> dict[str, str]:
    files = paths_config.get("files", {}) or {}
    masks = files.get("masks", {}) or {}
    return dict(masks.get("categorical", {}) or {})


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "si", "sí"}


def get_binary_mask_catalog(paths_config: dict[str, Any], only_enabled: bool = True) -> "OrderedDict[str, dict[str, Any]]":
    """Catálogo de máscaras binarias con filename + metadatos."""
    file_map = get_binary_mask_file_map(paths_config)
    meta_root = ((paths_config.get("masks", {}) or {}).get("binary", {}) or {})
    out: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
    for name, filename in file_map.items():
        meta = dict(meta_root.get(name, {}) or {})
        enabled = _as_bool(meta.get("enabled"), True)
        if only_enabled and not enabled:
            continue
        mode = meta.get("mode", "exclude")
        if mode not in {"include", "exclude"}:
            mode = "exclude"
        out[name] = {
            "name": name,
            "filename": filename,
            "label": meta.get("label", name),
            "description": meta.get("description", ""),
            "mode": mode,
            "default": _as_bool(meta.get("default"), False),
            "enabled": enabled,
        }
    return out


def _parse_class_map(raw: dict[str, Any]) -> "OrderedDict[int, str]":
    classes: "OrderedDict[int, str]" = OrderedDict()
    for key, label in raw.items():
        try:
            class_id = int(key)
        except Exception:
            continue
        classes[class_id] = str(label)
    return OrderedDict(sorted(classes.items(), key=lambda item: item[0]))


def _parse_default_selected(raw: Any) -> list[int]:
    if not raw:
        return []
    out: list[int] = []
    for value in raw:
        try:
            out.append(int(value))
        except Exception:
            continue
    return out


def get_categorical_mask_catalog(paths_config: dict[str, Any], only_enabled: bool = True) -> "OrderedDict[str, dict[str, Any]]":
    """Catálogo de máscaras categóricas con filename, clases y metadatos."""
    file_map = get_categorical_mask_file_map(paths_config)
    meta_root = ((paths_config.get("masks", {}) or {}).get("categorical", {}) or {})
    out: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
    for name, filename in file_map.items():
        meta = dict(meta_root.get(name, {}) or {})
        enabled = _as_bool(meta.get("enabled"), True)
        if only_enabled and not enabled:
            continue
        classes = _parse_class_map(meta.get("classes", {}) or {})
        out[name] = {
            "name": name,
            "filename": filename,
            "label": meta.get("label", name),
            "description": meta.get("description", ""),
            "classes": classes,
            "default_selected": _parse_default_selected(meta.get("default_selected", [])),
            "enabled": enabled,
        }
    return out


def default_binary_mask_names(binary_catalog: dict[str, dict[str, Any]]) -> list[str]:
    return [name for name, meta in binary_catalog.items() if bool(meta.get("default", False))]


def default_categorical_filters(categorical_catalog: dict[str, dict[str, Any]]) -> dict[str, list[int]]:
    return {
        name: list(meta.get("default_selected", []) or [])
        for name, meta in categorical_catalog.items()
    }
