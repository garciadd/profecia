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
    data_value_type = data_value_type.lower().strip()
    if data_value_type not in {"real", "anomaly"}:
        raise ValueError("data_value_type debe ser 'real' o 'anomaly'.")
    return data_value_type

def validate_preprocess_options(
    data_value_type: str = "real",
    detrend_theil_sen: bool = False,
) -> dict[str, Any]:
    data_value_type = normalize_data_value_type(data_value_type)
    return {
        "data_value_type": data_value_type,
        "detrend_theil_sen": bool(detrend_theil_sen),
    }

def calculate_climatology(da: xr.DataArray, temporal_resolution: str) -> xr.DataArray:
    temporal_resolution = temporal_resolution.lower().strip()
    if temporal_resolution == "monthly":
        climatology = da.groupby("time.month").mean(dim="time", skipna=True)
        climatology = climatology.rename("climatology")
        climatology.attrs = dict(da.attrs)
        climatology.attrs["preprocess_product"] = "climatology"
        climatology.attrs["climatology_type"] = "monthly_mean"
        return climatology

    if temporal_resolution == "annual":
        climatology = da.mean(dim="time", skipna=True).rename("climatology")
        climatology.attrs = dict(da.attrs)
        climatology.attrs["preprocess_product"] = "climatology"
        climatology.attrs["climatology_type"] = "historical_mean"
        return climatology

    raise ValueError("temporal_resolution debe ser 'monthly' o 'annual'.")

def calculate_anomaly(
    da: xr.DataArray,
    climatology: xr.DataArray,
    temporal_resolution: str,
) -> xr.DataArray:
    temporal_resolution = temporal_resolution.lower().strip()
    if temporal_resolution == "monthly":
        anomaly = da.groupby("time.month") - climatology
        reference = "monthly_climatology"
    elif temporal_resolution == "annual":
        anomaly = da - climatology
        reference = "historical_mean"
    else:
        raise ValueError("temporal_resolution debe ser 'monthly' o 'annual'.")

    anomaly = anomaly.rename("anomaly")
    anomaly.attrs = dict(da.attrs)
    anomaly.attrs["data_value_type"] = "anomaly"
    anomaly.attrs["preprocess_product"] = "anomaly"
    anomaly.attrs["anomaly_reference"] = reference
    return anomaly

def time_in_years_centered(da: xr.DataArray) -> xr.DataArray:
    time = pd.to_datetime(da["time"].values)
    years = (time - time[0]) / pd.Timedelta(days=365.2425)
    years = np.asarray(years, dtype=np.float64)
    years = years - np.nanmean(years)
    out = xr.DataArray(years, coords={"time": da["time"]}, dims=("time",), name="trend_time_years_centered")
    out.attrs["preprocess_product"] = "trend_time_years_centered"
    out.attrs["time_unit"] = "years_centered"
    return out

def _theil_sen_slope_1d(y: np.ndarray, x: np.ndarray) -> np.float64:
    valid = np.isfinite(y) & np.isfinite(x)
    if int(valid.sum()) < 2:
        return np.float64(0.0)
    slope = theilslopes(y[valid], x[valid]).slope
    return np.float64(slope)

def calculate_theil_sen_trend_components(da: xr.DataArray) -> dict[str, xr.DataArray]:
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
    slope.attrs = dict(da.attrs)
    slope.attrs["preprocess_product"] = "trend_slope"
    slope.attrs["trend_method"] = "theil_sen"
    slope.attrs["trend_unit"] = f"{da.attrs.get('units', '')}/year".strip("/")
    return {"slope": slope, "time_years_centered": time_years}

def reconstruct_trend(components: dict[str, xr.DataArray], dims: tuple[str, ...]) -> xr.DataArray:
    trend = components["slope"] * components["time_years_centered"]
    trend = trend.transpose(*dims)
    trend = trend.rename("trend")
    trend.attrs = dict(components["slope"].attrs)
    trend.attrs["preprocess_product"] = "trend"
    trend.attrs["trend_storage"] = "slope_plus_centered_time"
    return trend

def remove_theil_sen_trend(da: xr.DataArray) -> tuple[xr.DataArray, dict[str, xr.DataArray]]:
    components = calculate_theil_sen_trend_components(da)
    trend = reconstruct_trend(components, tuple(da.dims))
    out = da - trend
    out.attrs = dict(da.attrs)
    out.attrs["detrended"] = "true"
    out.attrs["detrend_method"] = "theil_sen"
    out.attrs["detrend_time_unit"] = "years_centered"
    out = out.astype(da.dtype)
    return out, components

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
    products: dict[str, xr.DataArray | dict[str, xr.DataArray]] = {}
    steps: list[str] = []
    out = da.copy()
    out.attrs = dict(da.attrs)
    out.attrs["data_value_type"] = "real"
    out.attrs["detrended"] = "false"

    anomaly_reference = None
    trend_target = "real"
    if options["data_value_type"] == "anomaly":
        climatology = calculate_climatology(out, temporal_resolution)
        anomaly = calculate_anomaly(out, climatology, temporal_resolution)
        products["climatology"] = climatology
        products["anomaly"] = anomaly
        out = anomaly
        anomaly_reference = anomaly.attrs["anomaly_reference"]
        trend_target = "anomaly"
        steps.extend(["climatology", "anomaly"])

    if options["detrend_theil_sen"]:
        out, trend_components = remove_theil_sen_trend(out)
        products["trend"] = trend_components
        steps.append("theil_sen_trend_removed")
        if trend_target == "anomaly":
            out = out.rename("detrended_anomaly")
            out.attrs["preprocess_product"] = "detrended_anomaly"
            products["detrended_anomaly"] = out
        else:
            out = out.rename("detrended")
            out.attrs["preprocess_product"] = "detrended"
            products["detrended"] = out

    metadata = {
        **options,
        "preprocessing_applied": bool(steps),
        "preprocessing_steps": steps,
        "anomaly_applied": options["data_value_type"] == "anomaly",
        "anomaly_reference": anomaly_reference,
        "detrend_method": "theil_sen" if options["detrend_theil_sen"] else None,
        "detrend_time_unit": "years_centered" if options["detrend_theil_sen"] else None,
        "trend_storage": "slope_plus_centered_time" if options["detrend_theil_sen"] else None,
        "trend_target": trend_target if options["detrend_theil_sen"] else None,
        "reconstruction_order": _reconstruction_order(options),
    }
    return PreprocessResult(data=out, products=products, metadata=metadata)

def _reconstruction_order(options: dict[str, Any]) -> list[str]:
    order = []
    if options["detrend_theil_sen"]:
        order.append("add_trend_from_slope_and_centered_time")
    if options["data_value_type"] == "anomaly":
        order.append("add_climatology")
    return order

def _load_metadata(metadata_path: str | Path) -> dict[str, Any]:
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)

def _time_coord(spec: dict[str, Any]) -> pd.DatetimeIndex:
    start = spec["time_min"]
    end = spec["time_max"]
    frequency = spec["frequency"]
    if frequency == "MS":
        return pd.date_range(start=start, end=end, freq="MS")
    if frequency == "YS":
        return pd.date_range(start=start, end=end, freq="YS")
    raise ValueError(f"Frecuencia temporal no soportada para reconstrucción: {frequency}")

def _spatial_coord(start: float, size: int, resolution: float) -> np.ndarray:
    return start + np.arange(size, dtype=np.float64) * resolution

def _coords_for_dims(variable_meta: dict[str, Any], dims: tuple[str, ...]) -> dict[str, Any]:
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
                grid["spatial_resolution_deg"],
            )
        elif dim == "longitude":
            coords["longitude"] = _spatial_coord(
                grid["lon_min"],
                grid["longitude_size"],
                grid["spatial_resolution_deg"],
            )
        elif dim == "month":
            coords["month"] = np.arange(1, 13, dtype=np.int64)
    return coords

def _load_product(
    variable_meta: dict[str, Any],
    product_meta: dict[str, Any],
    path_key: str = "path",
    name: str | None = None,
) -> xr.DataArray:
    path = product_meta[path_key]
    arr = np.load(path)
    dims = tuple(product_meta["dims"])
    coords = _coords_for_dims(variable_meta, dims)
    return xr.DataArray(arr, coords=coords, dims=dims, name=name)

def reconstruct_original_signal(
    metadata_path: str | Path,
    variable_name: str,
    processed_data: xr.DataArray | np.ndarray | None = None,
) -> xr.DataArray:
    """
    Reconstruye la señal original de una variable procesada usando metadata.json
    y los productos auxiliares guardados como .npy.
    """
    metadata = _load_metadata(metadata_path)
    variable_name = variable_name.upper()
    variable_meta = metadata["variables"][variable_name]
    processing = variable_meta["processing"]

    final_dims = tuple(variable_meta.get("final_dims", ("time", "latitude", "longitude")))
    if processed_data is None:
        processed_data = np.load(variable_meta["array_path"])
    if isinstance(processed_data, xr.DataArray):
        reconstructed = processed_data.copy()
    else:
        coords = _coords_for_dims(variable_meta, final_dims)
        reconstructed = xr.DataArray(processed_data, coords=coords, dims=final_dims, name=variable_name)

    products = processing.get("preprocess_products", {})

    if processing.get("detrend_theil_sen"):
        trend_meta = products["trend"]
        slope = _load_product(variable_meta, trend_meta["slope"], path_key="path", name="trend_slope")
        time_years = _load_product(
            variable_meta,
            trend_meta["time_years_centered"],
            path_key="path",
            name="trend_time_years_centered",
        )
        trend = reconstruct_trend({"slope": slope, "time_years_centered": time_years}, tuple(reconstructed.dims))
        reconstructed = reconstructed + trend

    if processing.get("anomaly_applied"):
        climatology = _load_product(variable_meta, products["climatology"], name="climatology")
        if "month" in climatology.dims:
            reconstructed = reconstructed.groupby("time.month") + climatology
        else:
            reconstructed = reconstructed + climatology

    reconstructed = reconstructed.rename(f"{variable_name}_reconstructed")
    reconstructed.attrs["reconstructed_from"] = variable_name
    reconstructed.attrs["metadata_path"] = str(metadata_path)
    return reconstructed
