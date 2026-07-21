"""Estado de configuración de la interfaz PROFECIA."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

DEFAULT_ROI = {
    "lat_min": -60.0,
    "lat_max": 80.0,
    "lon_min": -180.0,
    "lon_max": 180.0,
}


def _as_int_list(values: Any) -> list[int]:
    if not values:
        return []
    out: list[int] = []
    for value in values:
        try:
            out.append(int(value))
        except Exception:
            continue
    return out


@dataclass
class ProfeciaUIState:
    """Configuración activa controlada por la interfaz.

    La UI expone únicamente el experimento: resolución, tipo de dato, ROI,
    máscaras/filtros espaciales, variables y target. Las rutas viven en paths.toml.
    """

    temporal_resolution: str = "annual"
    data_value_type: str = "real"
    start_year: int = 1982
    end_year_inclusive: int = 2022
    lat_min: float = DEFAULT_ROI["lat_min"]
    lat_max: float = DEFAULT_ROI["lat_max"]
    lon_min: float = DEFAULT_ROI["lon_min"]
    lon_max: float = DEFAULT_ROI["lon_max"]
    mask_names: list[str] = field(default_factory=lambda: ["land"])
    categorical_filters: dict[str, list[int]] = field(default_factory=dict)
    variable_names: list[str] = field(default_factory=lambda: ["LAI"])
    target_name: str = "LAI"
    dtype: str = "float32"

    @property
    def roi_dict(self) -> dict[str, float]:
        return {
            "lat_min": float(self.lat_min),
            "lat_max": float(self.lat_max),
            "lon_min": float(self.lon_min),
            "lon_max": float(self.lon_max),
        }

    @property
    def roi_bounds_leaflet(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """Bounds en formato ipyleaflet: ((south, west), (north, east))."""
        return ((float(self.lat_min), float(self.lon_min)), (float(self.lat_max), float(self.lon_max)))

    @property
    def predictor_names(self) -> list[str]:
        return [var for var in self.variable_names if var != self.target_name]

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        *,
        available_variables: list[str] | None = None,
        target_names: list[str] | None = None,
        default_mask_names: list[str] | None = None,
        default_categorical_filters: dict[str, list[int]] | None = None,
    ) -> "ProfeciaUIState":
        data = config.get("data", {}) or {}
        roi = data.get("roi") or DEFAULT_ROI

        target_names = target_names or ["LAI"]
        target_name = data.get("target_name") or "LAI"
        if "LAI" in target_names:
            target_name = "LAI"
        elif target_name not in target_names:
            target_name = target_names[0]

        raw_variables = list(data.get("variable_names") or [target_name])
        if target_name not in raw_variables:
            raw_variables.insert(0, target_name)

        if available_variables is not None:
            allowed = set(available_variables)
            variable_names = [v for v in raw_variables if v in allowed]
            if target_name not in variable_names:
                variable_names.insert(0, target_name)
        else:
            variable_names = raw_variables

        mask_names = list(data.get("mask_names") or default_mask_names or ["land"])

        categorical_defaults = default_categorical_filters or {}
        categorical_filters = {
            str(name): list(values)
            for name, values in categorical_defaults.items()
        }
        raw_categorical = data.get("categorical_filters", {}) or {}
        for name, values in raw_categorical.items():
            categorical_filters[str(name)] = _as_int_list(values)

        return cls(
            temporal_resolution=data.get("temporal_resolution", "annual"),
            data_value_type=data.get("data_value_type", "real"),
            start_year=int(data.get("start_year", 1982)),
            end_year_inclusive=int(data.get("end_year_inclusive", 2022)),
            lat_min=float(roi.get("lat_min", DEFAULT_ROI["lat_min"])),
            lat_max=float(roi.get("lat_max", DEFAULT_ROI["lat_max"])),
            lon_min=float(roi.get("lon_min", DEFAULT_ROI["lon_min"])),
            lon_max=float(roi.get("lon_max", DEFAULT_ROI["lon_max"])),
            mask_names=mask_names,
            categorical_filters=categorical_filters,
            variable_names=variable_names,
            target_name=target_name,
            dtype=str(data.get("dtype", "float32")),
        )

    def as_data_update(self) -> dict[str, Any]:
        """Campos que se vuelcan en la sección [data] del TOML."""
        variable_names = list(self.variable_names)
        if self.target_name not in variable_names:
            variable_names.insert(0, self.target_name)
        # Target fijo en primera posición para evitar ambigüedad en el pipeline.
        variable_names = [self.target_name] + [v for v in variable_names if v != self.target_name]

        return {
            "temporal_resolution": self.temporal_resolution,
            "data_value_type": self.data_value_type,
            "roi": self.roi_dict,
            "mask_names": list(self.mask_names),
            "start_year": int(self.start_year),
            "end_year_inclusive": int(self.end_year_inclusive),
            "dtype": self.dtype or "float32",
            "variable_names": variable_names,
            "target_name": self.target_name,
            "categorical_filters": {
                str(name): [int(v) for v in values]
                for name, values in self.categorical_filters.items()
            },
        }
