from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import theilslopes
import xarray as xr


@dataclass
class PreprocessResult:
    data: xr.DataArray
    products: dict[str, xr.DataArray | dict[str, xr.DataArray]]
    metadata: dict[str, Any]


def normalize_data_value_type(data_value_type: str) -> str:
    value = str(data_value_type).lower().strip()
    if value not in {"real", "anomaly", "trend"}:
        raise ValueError(
            "data_value_type debe ser 'real', 'anomaly' o 'trend'."
        )
    return value


def validate_preprocess_options(
    data_value_type: str = "real",
    detrend_theil_sen: bool = False,
) -> dict[str, Any]:
    """Valida las opciones de preprocesado.

    ``detrend_theil_sen`` se conserva únicamente por compatibilidad con
    llamadas antiguas. La configuración nueva usa ``data_value_type='trend'``.
    """
    if detrend_theil_sen:
        raise ValueError(
            "detrend_theil_sen ya no forma parte de la configuración. "
            "Usa data_value_type='trend'."
        )
    return {"data_value_type": normalize_data_value_type(data_value_type)}


def calculate_climatology(
    da: xr.DataArray,
    temporal_resolution: str,
) -> xr.DataArray:
    resolution = str(temporal_resolution).lower().strip()

    if resolution == "monthly":
        climatology = da.groupby("time.month").mean(dim="time", skipna=True)
        climatology.attrs["climatology_type"] = "monthly_mean"
    elif resolution == "annual":
        climatology = da.mean(dim="time", skipna=True)
        climatology.attrs["climatology_type"] = "historical_mean"
    else:
        raise ValueError(
            "temporal_resolution debe ser 'monthly' o 'annual'."
        )

    climatology = climatology.rename("climatology")
    climatology.attrs = {
        **dict(da.attrs),
        **dict(climatology.attrs),
        "preprocess_product": "climatology",
    }
    return climatology


def calculate_anomaly(
    da: xr.DataArray,
    climatology: xr.DataArray,
    temporal_resolution: str,
) -> xr.DataArray:
    resolution = str(temporal_resolution).lower().strip()

    if resolution == "monthly":
        anomaly = da.groupby("time.month") - climatology
        reference = "monthly_climatology"
    elif resolution == "annual":
        anomaly = da - climatology
        reference = "historical_mean"
    else:
        raise ValueError(
            "temporal_resolution debe ser 'monthly' o 'annual'."
        )

    anomaly = anomaly.rename("anomaly")
    anomaly.attrs = {
        **dict(da.attrs),
        "data_value_type": "anomaly",
        "preprocess_product": "anomaly",
        "anomaly_reference": reference,
    }
    return anomaly


def time_in_years_centered(da: xr.DataArray) -> xr.DataArray:
    time = pd.to_datetime(da["time"].values)
    years = np.asarray(
        (time - time[0]) / pd.Timedelta(days=365.2425),
        dtype=np.float64,
    )
    years -= np.nanmean(years)

    out = xr.DataArray(
        years,
        coords={"time": da["time"]},
        dims=("time",),
        name="trend_time_years_centered",
    )
    out.attrs["preprocess_product"] = "trend_time_years_centered"
    out.attrs["time_unit"] = "years_centered"
    return out


def _theil_sen_slope_1d(y: np.ndarray, x: np.ndarray) -> np.float64:
    valid = np.isfinite(y) & np.isfinite(x)
    if int(valid.sum()) < 2:
        return np.float64(np.nan)
    return np.float64(theilslopes(y[valid], x[valid]).slope)


def calculate_theil_sen_trend_components(
    da: xr.DataArray,
) -> dict[str, xr.DataArray]:
    time_years = time_in_years_centered(da)
    slope = xr.apply_ufunc(
        _theil_sen_slope_1d,
        da,
        time_years,
        input_core_dims=[["time"], ["time"]],
        output_core_dims=[[]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.float64],
    )
    slope = slope.rename("trend_slope")
    slope.attrs = {
        **dict(da.attrs),
        "preprocess_product": "trend_slope",
        "trend_method": "theil_sen",
        "trend_unit": f"{da.attrs.get('units', '')}/year".strip("/"),
    }
    return {
        "slope": slope,
        "time_years_centered": time_years,
    }


def reconstruct_trend(
    components: dict[str, xr.DataArray],
    dims: tuple[str, ...],
) -> xr.DataArray:
    trend = components["slope"] * components["time_years_centered"]
    trend = trend.transpose(*dims).rename("trend")
    trend.attrs = {
        **dict(components["slope"].attrs),
        "data_value_type": "trend",
        "preprocess_product": "trend",
        "trend_method": "theil_sen",
        "trend_storage": "slope_plus_centered_time",
    }
    return trend


def apply_preprocessing(
    da: xr.DataArray,
    temporal_resolution: str,
    data_value_type: str = "real",
    detrend_theil_sen: bool = False,
) -> PreprocessResult:
    options = validate_preprocess_options(
        data_value_type=data_value_type,
        detrend_theil_sen=detrend_theil_sen,
    )
    value_type = options["data_value_type"]
    products: dict[str, xr.DataArray | dict[str, xr.DataArray]] = {}

    if value_type == "real":
        out = da.copy()
        steps: list[str] = []
        anomaly_reference = None
        trend_method = None

    elif value_type == "anomaly":
        climatology = calculate_climatology(da, temporal_resolution)
        out = calculate_anomaly(da, climatology, temporal_resolution)
        products["climatology"] = climatology
        steps = ["climatology", "anomaly"]
        anomaly_reference = out.attrs["anomaly_reference"]
        trend_method = None

    else:
        components = calculate_theil_sen_trend_components(da)
        out = reconstruct_trend(components, tuple(da.dims))
        products["trend"] = components
        steps = ["theil_sen_trend"]
        anomaly_reference = None
        trend_method = "theil_sen"

    out = out.astype(da.dtype)
    out.attrs = {
        **dict(out.attrs),
        "data_value_type": value_type,
    }

    metadata = {
        "data_value_type": value_type,
        "preprocessing_applied": bool(steps),
        "preprocessing_steps": steps,
        "anomaly_applied": value_type == "anomaly",
        "anomaly_reference": anomaly_reference,
        "trend_applied": value_type == "trend",
        "trend_method": trend_method,
        "trend_time_unit": "years_centered" if value_type == "trend" else None,
        "trend_storage": (
            "slope_plus_centered_time" if value_type == "trend" else None
        ),
        "reconstruction_order": _reconstruction_order(value_type),
    }
    return PreprocessResult(data=out, products=products, metadata=metadata)


def _reconstruction_order(data_value_type: str) -> list[str]:
    if data_value_type == "anomaly":
        return ["add_climatology"]
    return []


def _load_metadata(metadata_path: str | Path) -> dict[str, Any]:
    with open(metadata_path, "r", encoding="utf-8") as file:
        return json.load(file)


def _time_coord(spec: dict[str, Any]) -> pd.DatetimeIndex:
    frequency = spec["frequency"]
    if frequency == "MS":
        return pd.date_range(spec["time_min"], spec["time_max"], freq="MS")
    if frequency == "YS":
        return pd.date_range(spec["time_min"], spec["time_max"], freq="YS")
    raise ValueError(
        f"Frecuencia temporal no soportada para reconstrucción: {frequency}"
    )


def _spatial_coord(start: float, size: int, resolution: float) -> np.ndarray:
    return start + np.arange(size, dtype=np.float64) * resolution


def _coords_for_dims(
    variable_meta: dict[str, Any],
    dims: tuple[str, ...],
) -> dict[str, Any]:
    grid = variable_meta["grid"]
    temporal = variable_meta["temporal_grid"]
    coords: dict[str, Any] = {}

    for dim in dims:
        if dim == "time":
            coords["time"] = _time_coord(temporal)
        elif dim == "latitude":
            coords["latitude"] = _spatial_coord(
                grid["lat_min"],
                grid["latitude_size"],
                grid["latitude_resolution_deg"],
            )
        elif dim == "longitude":
            coords["longitude"] = _spatial_coord(
                grid["lon_min"],
                grid["longitude_size"],
                grid["longitude_resolution_deg"],
            )
        elif dim == "month":
            coords["month"] = np.arange(1, 13, dtype=np.int64)

    return coords


def _load_product(
    variable_meta: dict[str, Any],
    product_meta: dict[str, Any],
    name: str | None = None,
) -> xr.DataArray:
    array = np.load(product_meta["path"])
    dims = tuple(product_meta["dims"])
    coords = _coords_for_dims(variable_meta, dims)
    return xr.DataArray(array, coords=coords, dims=dims, name=name)


def reconstruct_original_signal(
    metadata_path: str | Path,
    variable_name: str,
    processed_data: xr.DataArray | np.ndarray | None = None,
) -> xr.DataArray:
    """Reconstruye la señal real desde un producto ``real`` o ``anomaly``.

    Una serie de tendencia no contiene la variabilidad residual ni la media
    original, por lo que no permite reconstruir por sí sola la señal real.
    """
    metadata = _load_metadata(metadata_path)
    variable_name = variable_name.upper()
    variable_meta = metadata["variables"][variable_name]
    processing = variable_meta["processing"]
    value_type = processing.get("data_value_type", "real")

    if value_type == "trend":
        raise ValueError(
            "No se puede reconstruir la señal original únicamente desde "
            "un producto de tendencia."
        )

    final_dims = tuple(
        variable_meta.get("final_dims", ("time", "latitude", "longitude"))
    )
    if processed_data is None:
        processed_data = np.load(variable_meta["array_path"])

    if isinstance(processed_data, xr.DataArray):
        reconstructed = processed_data.copy()
    else:
        reconstructed = xr.DataArray(
            processed_data,
            coords=_coords_for_dims(variable_meta, final_dims),
            dims=final_dims,
            name=variable_name,
        )

    if value_type == "anomaly":
        climatology_meta = processing["preprocess_products"]["climatology"]
        climatology = _load_product(
            variable_meta,
            climatology_meta,
            name="climatology",
        )
        if "month" in climatology.dims:
            reconstructed = reconstructed.groupby("time.month") + climatology
        else:
            reconstructed = reconstructed + climatology

    reconstructed = reconstructed.rename(f"{variable_name}_reconstructed")
    reconstructed.attrs["reconstructed_from"] = variable_name
    reconstructed.attrs["metadata_path"] = str(metadata_path)
    return reconstructed
