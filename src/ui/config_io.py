"""Lectura y escritura de data.toml generado desde la interfaz."""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:  # Python >= 3.11
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore

try:
    import tomli_w
except ModuleNotFoundError:  # pragma: no cover
    tomli_w = None  # type: ignore

from .state import ProfeciaUIState


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo de configuración: {path}")
    with path.open("rb") as f:
        return tomllib.load(f)


def _format_toml_value(value: Any) -> str:
    if isinstance(value, str):
        return '"' + value.replace('"', '\\"') + '"'
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return str(value)
    if isinstance(value, list):
        return "[" + ", ".join(_format_toml_value(v) for v in value) + "]"
    if isinstance(value, dict):
        items = ", ".join(f"{k} = {_format_toml_value(v)}" for k, v in value.items())
        return "{ " + items + " }"
    raise TypeError(f"Tipo no soportado en fallback TOML: {type(value)!r}")


def _dump_toml_fallback(config: dict[str, Any]) -> str:
    lines: list[str] = []
    for section, values in config.items():
        if not isinstance(values, dict):
            continue
        simple_items: dict[str, Any] = {}
        nested_items: dict[str, dict[str, Any]] = {}
        for key, value in values.items():
            if isinstance(value, dict):
                nested_items[key] = value
            else:
                simple_items[key] = value

        lines.append(f"[{section}]")
        for key, value in simple_items.items():
            lines.append(f"{key} = {_format_toml_value(value)}")
        lines.append("")

        for nested_key, nested_values in nested_items.items():
            lines.append(f"[{section}.{nested_key}]")
            for key, value in nested_values.items():
                lines.append(f"{key} = {_format_toml_value(value)}")
            lines.append("")
    return "\n".join(lines)


def save_config(
    config: dict[str, Any],
    path: str | Path,
    backup: bool = True,
) -> None:
    """Guarda un data.toml válido generado desde la interfaz."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if tomli_w is not None:
        text = tomli_w.dumps(config)
    else:
        text = _dump_toml_fallback(config)

    # Verifica el contenido antes de modificar el archivo existente.
    try:
        tomllib.loads(text)
    except Exception as exc:
        raise ValueError(
            "La configuración generada por la interfaz no es TOML válido."
        ) from exc

    if backup and path.exists():
        backup_path = path.with_suffix(path.suffix + ".bak")
        backup_path.write_text(
            path.read_text(encoding="utf-8"),
            encoding="utf-8",
        )

    path.write_text(text, encoding="utf-8")


def update_config_from_state(config: dict[str, Any], state: ProfeciaUIState) -> dict[str, Any]:
    """Genera un data.toml limpio desde el estado actual.

    Las rutas, nombres de ficheros y catálogo de variables/máscaras viven en paths.toml.
    Aquí se guarda solo la configuración activa del experimento.
    """
    previous_data = config.get("data", {}) or {}
    state.dtype = state.dtype or str(previous_data.get("dtype", "float32"))
    return {"data": state.as_data_update()}
