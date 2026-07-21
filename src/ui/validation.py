"""Validaciones de configuración para la interfaz PROFECIA."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .state import ProfeciaUIState


@dataclass
class ValidationResult:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def as_html(self) -> str:
        if not self.errors and not self.warnings:
            return '<div class="profecia-ok">Configuración válida.</div>'
        parts: list[str] = []
        if self.errors:
            parts.append("<b>Errores</b><ul>" + "".join(f"<li>{e}</li>" for e in self.errors) + "</ul>")
        if self.warnings:
            parts.append("<b>Advertencias</b><ul>" + "".join(f"<li>{w}</li>" for w in self.warnings) + "</ul>")
        css_class = "profecia-error" if self.errors else "profecia-warning"
        return f'<div class="{css_class}">' + "".join(parts) + "</div>"


def validate_state(
    state: ProfeciaUIState,
    *,
    lat: np.ndarray | None = None,
    lon: np.ndarray | None = None,
    valid_mask: np.ndarray | None = None,
    available_variables: list[str] | None = None,
    binary_mask_catalog: dict[str, dict[str, Any]] | None = None,
    categorical_mask_catalog: dict[str, dict[str, Any]] | None = None,
    skipped_filters: list[str] | None = None,
) -> ValidationResult:
    result = ValidationResult()

    if state.temporal_resolution not in {"annual", "monthly"}:
        result.errors.append("Resolución temporal no válida. Usa 'annual' o 'monthly'.")

    if state.data_value_type not in {"real", "anomaly", "trend"}:
        result.errors.append("Tipo de dato no válido. Usa 'real', 'anomaly' o 'trend'.")

    if state.start_year > state.end_year_inclusive:
        result.errors.append("El año de inicio debe ser menor o igual que el año final.")

    if state.lat_min >= state.lat_max:
        result.errors.append("Lat min debe ser menor que Lat max.")

    if state.lon_min == state.lon_max:
        result.errors.append("Lon min y Lon max no pueden ser iguales.")

    if lat is not None:
        lat_min_available = float(np.nanmin(lat))
        lat_max_available = float(np.nanmax(lat))
        if state.lat_min < lat_min_available or state.lat_max > lat_max_available:
            result.warnings.append(
                f"El ROI sale del rango latitudinal disponible ({lat_min_available:.2f}, {lat_max_available:.2f})."
            )

    if lon is not None:
        lon_min_available = float(np.nanmin(lon))
        lon_max_available = float(np.nanmax(lon))
        if state.lon_min < lon_min_available or state.lon_max > lon_max_available:
            result.warnings.append(
                f"El ROI sale del rango longitudinal disponible ({lon_min_available:.2f}, {lon_max_available:.2f})."
            )

    if state.target_name != "LAI":
        result.errors.append("Por ahora la única variable objetivo permitida es LAI.")

    if "LAI" not in state.variable_names:
        result.errors.append("LAI debe estar incluido en variable_names como target.")

    if available_variables is not None:
        available = set(available_variables)
        unknown_variables = [v for v in state.variable_names if v not in available]
        if unknown_variables:
            result.errors.append(
                "Hay variables seleccionadas que no existen en paths.toml: " + ", ".join(unknown_variables)
            )

    if binary_mask_catalog is not None:
        unknown_masks = [m for m in state.mask_names if m not in binary_mask_catalog]
        if unknown_masks:
            result.errors.append(
                "Hay máscaras binarias seleccionadas que no existen en paths.toml: " + ", ".join(unknown_masks)
            )

    if categorical_mask_catalog is not None:
        for name, selected_ids in state.categorical_filters.items():
            if name not in categorical_mask_catalog:
                if selected_ids:
                    result.errors.append(f"El filtro categórico {name!r} no existe en paths.toml.")
                continue
            valid_ids = set(categorical_mask_catalog[name].get("classes", {}).keys())
            invalid_ids = [int(v) for v in selected_ids if int(v) not in valid_ids]
            if invalid_ids:
                result.errors.append(
                    f"El filtro categórico {name!r} contiene clases no definidas: {invalid_ids}."
                )

    if skipped_filters:
        result.warnings.append(
            "Hay filtros seleccionados sin archivo de máscara cargado, por lo que no se han aplicado: "
            + ", ".join(skipped_filters)
        )

    if state.data_value_type == "trend" and (state.end_year_inclusive - state.start_year + 1) < 10:
        result.warnings.append("La tendencia con menos de 10 años puede ser inestable.")

    if valid_mask is not None and np.count_nonzero(valid_mask) == 0:
        result.errors.append("No quedan celdas válidas tras aplicar ROI y filtros espaciales.")

    predictors = [v for v in state.variable_names if v != state.target_name]
    if len(predictors) == 0:
        result.warnings.append("No hay predictores seleccionados aparte del target.")

    return result
