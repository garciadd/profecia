
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any

import gc
import json

import numpy as np
import pandas as pd
import xarray as xr


FILE_MAP = {
    "LAI": "lai_1982_2022_monthly_0.5deg.nc",
    "SM1": "swvl1_1982_2022_monthly_0.5deg.nc",
    "SM2": "subswc_1982_2022_monthly_0.5deg.nc",
    "TP": "tp_1982_2022_monthly_0.5deg.nc",
    "T2M": "t2m_1982_2022_monthly_0.5deg.nc",
    "SSRD": "ssrd_1982_2022_monthly_0.5deg.nc",
    "VPD": "vpd_1982_2022_monthly_0.5deg.nc",
}

MASK_MAP = {
    "land": "land_mask_0p5deg.npy",
    "ebf": "ebf_mask_0p5deg.npy",
    "bs": "bs_mask_0p5deg.npy",
    "climate": "climate_mask_0p5_5classes.npy",
    "landcover": "landcover_mask_0p5_7classes.npy",
}

STANDARD_DIM_NAMES = {
    "lat": "latitude",
    "latitude": "latitude",
    "y": "latitude",
    "lon": "longitude",
    "longitude": "longitude",
    "x": "longitude",
    "time": "time",
}

ANNUAL_AGGREGATION_RULES = {
    "LAI": "mean",
    "SM1": "mean",
    "SM2": "mean",
    "TP": "sum",
    "T2M": "mean",
    "SSRD": "sum",
    "VPD": "mean",
}

CLIMATE_VALID_CODES = {1, 2, 3, 4, 5}
LANDCOVER_VALID_CODES = {10, 20, 30, 40, 70, 90, 100}


@dataclass(frozen=True)
class ROI:
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float


def _get_path(base_dir: str | Path, name: str, file_map: dict[str, str]) -> Path:
    name = name.lower()
    if name not in file_map:
        raise ValueError(f"'{name}' no soportado. Disponibles: {list(file_map)}")
    path = Path(base_dir) / file_map[name]
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo: {path}")
    return path


def _to_jsonable(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


def save_metadata_json(output_dir: str | Path, metadata: dict, filename: str = "metadata.json") -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(metadata), f, ensure_ascii=False, indent=2)
    return path


def save_npy(output_dir: str | Path, name: str, data: xr.DataArray | np.ndarray) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    arr = data.values if isinstance(data, xr.DataArray) else data
    path = output_dir / f"{name}.npy"
    np.save(path, arr)
    return path


def _standardize_dataset(ds: xr.Dataset) -> xr.Dataset:
    rename_dict = {}
    for name in list(ds.dims) + list(ds.coords):
        if name in STANDARD_DIM_NAMES and STANDARD_DIM_NAMES[name] != name:
            rename_dict[name] = STANDARD_DIM_NAMES[name]
    if rename_dict:
        ds = ds.rename(rename_dict)
    for coord in ("time", "latitude", "longitude"):
        if coord in ds.coords:
            ds = ds.sortby(coord)
    return ds


def _get_single_data_var(ds: xr.Dataset) -> str:
    data_vars = list(ds.data_vars)
    if len(data_vars) != 1:
        raise ValueError(f"Se esperaba un netCDF univariable. Encontradas: {data_vars}")
    return data_vars[0]


def _validate_coords(da: xr.DataArray) -> None:
    expected = {"time", "latitude", "longitude"}
    if not expected.issubset(set(da.dims)):
        raise ValueError(f"Dimensiones inesperadas: {da.dims}")

    lat = da["latitude"].values
    lon = da["longitude"].values
    if float(lat.min()) < -90 or float(lat.max()) > 90:
        raise ValueError("Latitudes fuera de rango.")
    if float(lon.min()) < -180 or float(lon.max()) > 180:
        raise ValueError("Longitudes fuera de rango esperado [-180, 180].")
    if np.any(np.diff(lat) <= 0):
        raise ValueError("Latitude debe estar en orden ascendente.")
    if np.any(np.diff(lon) <= 0):
        raise ValueError("Longitude debe estar en orden ascendente.")


def _select_time(da: xr.DataArray, start_year: int | None, end_year_inclusive: int | None) -> xr.DataArray:
    if start_year is None and end_year_inclusive is None:
        return da
    if start_year is None or end_year_inclusive is None:
        raise ValueError("Debes pasar start_year y end_year_inclusive juntos.")
    if end_year_inclusive < start_year:
        raise ValueError("end_year_inclusive debe ser >= start_year.")
    return da.sel(time=slice(f"{start_year}-01-01", f"{end_year_inclusive}-12-31"))


def _select_roi(da: xr.DataArray, roi: ROI | None) -> xr.DataArray:
    if roi is None:
        return da
    if not (-90 <= roi.lat_min < roi.lat_max <= 90):
        raise ValueError("ROI lat inválida.")
    if not (-180 <= roi.lon_min < roi.lon_max <= 180):
        raise ValueError("ROI lon inválida.")
    out = da.sel(latitude=slice(roi.lat_min, roi.lat_max), longitude=slice(roi.lon_min, roi.lon_max))
    if out.sizes["latitude"] == 0 or out.sizes["longitude"] == 0:
        raise ValueError("Selección ROI vacía.")
    return out


def load_netcdf(
    base_dir: str | Path,
    variable: str,
    roi: ROI | None = None,
    start_year: int | None = None,
    end_year_inclusive: int | None = None,
    dtype: str = "float32",
) -> tuple[xr.DataArray, dict]:
    variable = variable.upper()
    path = _get_path(base_dir, variable.lower(), {k.lower(): v for k, v in FILE_MAP.items()})

    ds = xr.open_dataset(path)
    ds = _standardize_dataset(ds)
    var_name = _get_single_data_var(ds)

    da = ds[var_name].transpose("time", "latitude", "longitude")
    da = _select_time(da, start_year, end_year_inclusive)
    da = _select_roi(da, roi)
    da = da.astype(dtype)
    _validate_coords(da)

    meta = {
        "variable_requested": variable,
        "variable_in_file": var_name,
        "filename": path.name,
        "path": str(path),
        "shape_loaded": tuple(int(x) for x in da.shape),
        "dtype": str(da.dtype),
        "units": da.attrs.get("units", ""),
        "time_min": str(pd.to_datetime(da["time"].values[0])) if da.sizes["time"] else None,
        "time_max": str(pd.to_datetime(da["time"].values[-1])) if da.sizes["time"] else None,
        "lat_min": float(da["latitude"].min()),
        "lat_max": float(da["latitude"].max()),
        "lon_min": float(da["longitude"].min()),
        "lon_max": float(da["longitude"].max()),
    }

    ds.close()
    return da, meta


def _validate_full_years(da: xr.DataArray) -> None:
    counts = da["time"].dt.year.to_series().value_counts().sort_index()
    incomplete = counts[counts != 12]
    if len(incomplete) > 0:
        raise ValueError(f"Años incompletos: {incomplete.to_dict()}")


def aggregate_time(
    da: xr.DataArray,
    variable_name: str,
    temporal_resolution: str = "monthly",
    annual_rule: str | None = None,
    require_full_years: bool = True,
) -> xr.DataArray:
    temporal_resolution = temporal_resolution.lower()
    variable_name = variable_name.upper()

    if temporal_resolution == "monthly":
        out = da.copy()
        out.attrs = dict(da.attrs)
        out.attrs["temporal_resolution"] = "monthly"
        return out

    if temporal_resolution != "annual":
        raise ValueError("temporal_resolution debe ser 'monthly' o 'annual'.")

    if require_full_years:
        _validate_full_years(da)

    rule = annual_rule or ANNUAL_AGGREGATION_RULES[variable_name]
    if rule not in {"mean", "sum"}:
        raise ValueError("annual_rule debe ser 'mean' o 'sum'.")

    grouped = da.groupby("time.year")
    out = grouped.mean(dim="time", skipna=True) if rule == "mean" else grouped.sum(dim="time", skipna=True)

    years = out["year"].values
    out = out.rename({"year": "time"})
    out = out.assign_coords(time=pd.to_datetime([f"{int(y)}-01-01" for y in years]))
    out.attrs = dict(da.attrs)
    out.attrs["temporal_resolution"] = "annual"
    out.attrs["annual_aggregation_rule"] = rule
    return out


def load_mask(
    mask_dir: str | Path,
    mask_name: str,
    latitude: np.ndarray,
    longitude: np.ndarray,
) -> xr.DataArray:
    """
    Carga una máscara desde NPY usando MASK_MAP.
    Se asume que las máscaras ya están guardadas en convención del pipeline:
    latitude ascendente (-90 -> 90), longitude ascendente (-180 -> 180).
    """
    path = _get_path(mask_dir, mask_name, MASK_MAP)
    arr = np.load(path)

    expected_shape = (len(latitude), len(longitude))
    if arr.shape != expected_shape:
        raise ValueError(
            f"Shape de la máscara {arr.shape} no coincide con el grid esperado {expected_shape} para {mask_name}"
        )

    return xr.DataArray(
        arr,
        coords={"latitude": latitude, "longitude": longitude},
        dims=("latitude", "longitude"),
        name=mask_name,
    )


def _validate_mask_alignment(da: xr.DataArray, mask: xr.DataArray, name: str) -> None:
    if tuple(mask.dims) != ("latitude", "longitude"):
        raise ValueError(f"{name} debe tener dims ('latitude', 'longitude').")
    if da.sizes["latitude"] != mask.sizes["latitude"] or da.sizes["longitude"] != mask.sizes["longitude"]:
        raise ValueError(f"{name} no coincide en shape espacial.")
    if not np.array_equal(da["latitude"].values, mask["latitude"].values):
        raise ValueError(f"{name} no comparte latitude.")
    if not np.array_equal(da["longitude"].values, mask["longitude"].values):
        raise ValueError(f"{name} no comparte longitude.")


def build_combined_filter_mask(
    da: xr.DataArray,
    masks: dict[str, xr.DataArray] | None = None,
) -> tuple[xr.DataArray | None, dict[str, Any]]:
    if not masks:
        return None, {"mask_names": [], "combined_fraction_kept": None}

    allowed = {"land", "bs", "ebf", "climate", "landcover"}
    unknown = set(masks) - allowed
    if unknown:
        raise ValueError(f"Máscaras no soportadas: {sorted(unknown)}")

    parts = []
    for name, mask in masks.items():
        _validate_mask_alignment(da, mask, name)

        if name == "land":
            part = mask.astype(bool)
        elif name in {"bs", "ebf"}:
            part = ~mask.astype(bool)
        elif name == "climate":
            part = xr.apply_ufunc(np.isin, mask, np.array(sorted(CLIMATE_VALID_CODES))).astype(bool)
        elif name == "landcover":
            part = xr.apply_ufunc(np.isin, mask, np.array(sorted(LANDCOVER_VALID_CODES))).astype(bool)
        parts.append(part)

    combined = parts[0]
    for part in parts[1:]:
        combined = combined & part

    combined = combined.rename("combined_filter_mask")
    info = {
        "mask_names": list(masks.keys()),
        "combined_pixels_kept": int(combined.sum()),
        "combined_total_pixels": int(combined.size),
        "combined_fraction_kept": float(combined.mean()),
    }
    return combined, info


def apply_filter_mask(da: xr.DataArray, filter_mask: xr.DataArray | None) -> xr.DataArray:
    if filter_mask is None:
        return da
    _validate_mask_alignment(da, filter_mask, "filter_mask")
    return da.where(filter_mask)


def load_and_save_variable(
    raw_dir: str | Path,
    output_dir: str | Path,
    variable: str,
    mask_dir: str | Path | None = None,
    mask_names: list[str] | None = None,
    roi: ROI | None = None,
    start_year: int | None = None,
    end_year_inclusive: int | None = None,
    dtype: str = "float32",
    temporal_resolution: str = "monthly",
    annual_rule: str | None = None,
    require_full_years: bool = True,
    save_output: bool = True,
) -> dict:
    da_raw, meta_load = load_netcdf(
        base_dir=raw_dir,
        variable=variable,
        roi=roi,
        start_year=start_year,
        end_year_inclusive=end_year_inclusive,
        dtype=dtype,
    )

    da_agg = aggregate_time(
        da=da_raw,
        variable_name=variable,
        temporal_resolution=temporal_resolution,
        annual_rule=annual_rule,
        require_full_years=require_full_years,
    )

    masks = {}
    if mask_names:
        if mask_dir is None:
            raise ValueError("Si usas mask_names, debes pasar mask_dir.")
        for name in mask_names:
            masks[name] = load_mask(mask_dir, name, da_agg["latitude"].values, da_agg["longitude"].values)

    combined_mask, mask_info = build_combined_filter_mask(da_agg, masks if masks else None)
    da_final = apply_filter_mask(da_agg, combined_mask)

    output_path = save_npy(output_dir, variable.upper(), da_final) if save_output else None

    result = {
        "logical_name": variable.upper(),
        "array_path": str(output_path) if output_path is not None else None,
        "load_metadata": meta_load,
        "processing": {
            "temporal_resolution": temporal_resolution,
            "annual_rule": annual_rule,
            "require_full_years": require_full_years,
            "mask_dir": str(mask_dir) if mask_dir is not None else None,
            **mask_info,
        },
        "final_shape": tuple(int(x) for x in da_final.shape),
        "final_time_min": str(pd.to_datetime(da_final["time"].values[0])) if da_final.sizes["time"] else None,
        "final_time_max": str(pd.to_datetime(da_final["time"].values[-1])) if da_final.sizes["time"] else None,
        "final_units": da_final.attrs.get("units", ""),
        "final_dtype": str(da_final.dtype),
    }

    del da_raw, da_agg, da_final, combined_mask, masks
    gc.collect()
    return result


def build_processed_metadata(
    variable_results: dict[str, dict],
    temporal_resolution: str,
    roi: ROI | None = None,
    start_year: int | None = None,
    end_year_inclusive: int | None = None,
    dtype: str = "float32",
) -> dict:
    if temporal_resolution.lower() not in {"monthly", "annual"}:
        raise ValueError("temporal_resolution debe ser 'monthly' o 'annual'.")
    if not variable_results:
        raise ValueError("variable_results no puede estar vacío.")

    return {
        "dataset_config": {
            "temporal_resolution": temporal_resolution.lower(),
            "roi": roi,
            "start_year": start_year,
            "end_year_inclusive": end_year_inclusive,
            "dtype": dtype,
        },
        "variables": {k.upper(): v for k, v in variable_results.items()},
    }


def save_processed_metadata(
    output_dir: str | Path,
    variable_results: dict[str, dict],
    temporal_resolution: str,
    roi: ROI | None = None,
    start_year: int | None = None,
    end_year_inclusive: int | None = None,
    dtype: str = "float32",
    filename: str = "metadata.json",
) -> Path:
    metadata = build_processed_metadata(
        variable_results=variable_results,
        temporal_resolution=temporal_resolution,
        roi=roi,
        start_year=start_year,
        end_year_inclusive=end_year_inclusive,
        dtype=dtype,
    )
    return save_metadata_json(output_dir, metadata, filename)
