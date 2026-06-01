
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
import re
from typing import Any

import gc
import json

import numpy as np
import pandas as pd
import xarray as xr

from src.data import preprocess


FILE_MAP = {
    "LAI": "lai_1982_2022_monthly_0.5deg.nc",
    "SM1": "swvl1_1982_2022_monthly_0.5deg.nc",
    "SM2": "subswc_1982_2022_monthly_0.5deg.nc",
    "SM1-2": "swvl1-2_1982_2022_monthly_0.5deg.nc",
    "TP": "tp_1982_2022_monthly_0.5deg.nc",
    "T2M": "t2m_1982_2022_monthly_0.5deg.nc",
    "SSRD": "ssrd_1982_2022_monthly_0.5deg.nc",
    "VPD": "vpd_1982_2022_monthly_0.5deg.nc",
    "D2M": "d2m_1982_2022_monthly_0.5deg.nc",
    "PEV": "pev_1982_2022_monthly_0.5deg.nc",
    "WIND": "wind_1982_2022_monthly_0.5deg.nc",
    "SPEI": "spei06_univariable_1982_2022_monthly_0.5deg.nc",
    "CO2": "../annual/human/co2_1982_2022_annual_0.5deg.nc",
    "HFP": "../annual/human/hfp_1982_2022_annual_0.5deg.nc",
    "NDEP": "../annual/human/ndep_1982_2022_annual_0.5deg.nc",
    "TLU": "../annual/human/tlu_1982_2022_annual_0.5deg.nc",
    "ELEVATION": "soil/elevation_1982_2022_monthly_0.5deg.nc",
    "PH": "soil/ph_1982_2022_monthly_0.5deg.nc",
    "RICHNESS": "soil/richness_1982_2022_monthly_0.5deg.nc",
    "BULK": "soil/bulk_1982_2022_monthly_0.5deg.nc",
    "CEC": "soil/cec_1982_2022_monthly_0.5deg.nc",
    "CLAY": "soil/clay_1982_2022_monthly_0.5deg.nc",
    "SAND": "soil/sand_1982_2022_monthly_0.5deg.nc",
    "SILT": "soil/silt_1982_2022_monthly_0.5deg.nc",
    "SOC": "soil/soc_1982_2022_monthly_0.5deg.nc",
    "TOTAL_N": "soil/total_n_1982_2022_monthly_0.5deg.nc",
    "LC_STATIC": "landcover_static_1982_2022_monthly_0.5deg.nc",
    "LC_3CLASS": "../annual/landcover_3classes_1982_2022_annual_0.5deg.nc",
    "LC_7CLASS": "../annual/landcover_7classes_1982_2022_annual_0.5deg.nc",
}

MASK_MAP = {
    "land": "land_mask_0p5deg.npy",
    "ebf": "ebf_mask_0p5deg.npy",
    "bs": "bs_mask_0p5deg.npy",
    "snow_ice": "snow_ice_mask_0p5deg.npy",
    "climate": "climate_mask_0p5_5classes.npy",
    "landcover": "landcover_mask_0p5_7classes.npy",
    # Per-class landcover masks (binary)
    "landcover_cropland": "landcover_cropland_0p5deg.npy",
    "landcover_forest": "landcover_forest_0p5deg.npy",
    "landcover_grassland": "landcover_grassland_0p5deg.npy",
    "landcover_shrubland": "landcover_shrubland_0p5deg.npy",
    "landcover_tundra": "landcover_tundra_0p5deg.npy",
    "landcover_barren": "landcover_barren_0p5deg.npy",
    "landcover_snow_ice": "landcover_snow_ice_0p5deg.npy",
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
    "SM1-2": "mean",
    "TP": "sum",
    "T2M": "mean",
    "SSRD": "sum",
    "VPD": "mean",
    "D2M": "mean",
    "PEV": "mean",
    "WIND": "mean",
    "SPEI": "mean",
    "CO2": "mean",
    "HFP": "mean",
    "NDEP": "mean",
    "TLU": "mean",
    "ELEVATION": "mean",
    "PH": "mean",
    "RICHNESS": "mean",
    "BULK": "mean",
    "CEC": "mean",
    "CLAY": "mean",
    "SAND": "mean",
    "SILT": "mean",
    "SOC": "mean",
    "TOTAL_N": "mean",
    "LC_STATIC": "mean",
    "LC_3CLASS": "mean",
    "LC_7CLASS": "mean",
}

CLIMATE_VALID_CODES = {1, 2, 3, 4, 5}
LANDCOVER_VALID_CODES = {10, 20, 30, 40, 70, 90, 100}
LAGGED_VARIABLE_PATTERN = re.compile(r"^(?P<base>[A-Z0-9_]+)_LAG_(?P<lag>\d+)$")


@dataclass(frozen=True)
class ROI:
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float


def _parse_variable_request(variable: str) -> dict[str, Any]:
    variable = variable.upper().strip()
    match = LAGGED_VARIABLE_PATTERN.fullmatch(variable)
    if not match:
        return {
            "requested_name": variable,
            "base_name": variable,
            "lag_steps": 0,
            "is_lagged": False,
        }

    lag_steps = int(match.group("lag"))
    if lag_steps <= 0:
        raise ValueError("El sufijo _LAG_N requiere N >= 1.")

    return {
        "requested_name": variable,
        "base_name": match.group("base"),
        "lag_steps": lag_steps,
        "is_lagged": True,
    }


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


def save_masks_png(output_dir: str | Path, masks: dict[str, xr.DataArray], dpi: int = 150) -> dict[str, str]:
    """
    Guarda cada máscara del diccionario `masks` como un PNG en
    `output_dir / "masks"` usando matplotlib. Devuelve un diccionario
    {mask_name: path_str} con las rutas guardadas.
    """
    from matplotlib import pyplot as plt

    output_dir = Path(output_dir) / "masks"
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = {}
    for name, da in masks.items():
        fig, ax = plt.subplots(figsize=(10, 4))
        try:
            da.plot(ax=ax)
            ax.set_title(name)
            path = output_dir / f"{name}.png"
            fig.savefig(path, bbox_inches="tight", dpi=dpi)
            saved[name] = str(path)
        finally:
            plt.close(fig)

    return saved


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


def _apply_lagged_shift(
    da: xr.DataArray,
    lag_steps: int,
    temporal_resolution: str,
) -> xr.DataArray:
    if lag_steps < 0:
        raise ValueError("lag_steps debe ser >= 0.")
    if lag_steps == 0:
        return da

    shifted = da.shift(time=lag_steps)
    shifted.attrs = dict(da.attrs)

    temporal_resolution = temporal_resolution.lower().strip()
    if temporal_resolution == "monthly":
        lag_unit = "months"
    elif temporal_resolution == "annual":
        lag_unit = "years"
    else:
        raise ValueError("temporal_resolution debe ser 'monthly' o 'annual'.")

    shifted.attrs["lag_steps"] = lag_steps
    shifted.attrs["lag_temporal_unit"] = lag_unit
    shifted.attrs["lag_note"] = f"Shifted by {lag_steps} {lag_unit}; first {lag_steps} steps are NaN."
    return shifted


def load_netcdf(
    base_dir: str | Path,
    variable: str,
    roi: ROI | None = None,
    start_year: int | None = None,
    end_year_inclusive: int | None = None,
    dtype: str = "float32",
) -> tuple[xr.DataArray, dict]:
    variable_info = _parse_variable_request(variable)
    variable = variable_info["requested_name"]
    base_variable = variable_info["base_name"]
    path = _get_path(base_dir, base_variable.lower(), {k.lower(): v for k, v in FILE_MAP.items()})

    ds = xr.open_dataset(path)
    ds = _standardize_dataset(ds)
    var_name = _get_single_data_var(ds)

    if "class" in ds.dims:
        da = ds[var_name].transpose("time", "class", "latitude", "longitude")
    else:
        da = ds[var_name].transpose("time", "latitude", "longitude")
    da = _select_time(da, start_year, end_year_inclusive)
    da = _select_roi(da, roi)
    da = da.astype(dtype)
    _validate_coords(da)

    meta = {
        "variable_requested": variable,
        "variable_base": base_variable,
        "variable_in_file": var_name,
        "filename": path.name,
        "path": str(path),
        "is_lagged": variable_info["is_lagged"],
        "lag_steps": variable_info["lag_steps"],
        "lag_applied": False,
        "lag_apply_stage": None,
        "lag_temporal_unit": None,
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


def _has_single_value_per_year(da: xr.DataArray) -> bool:
    counts = da["time"].dt.year.to_series().value_counts().sort_index()
    return len(counts) > 0 and bool((counts == 1).all())


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

    if _has_single_value_per_year(da):
        out = da.copy()
        out.attrs = dict(da.attrs)
        out.attrs["temporal_resolution"] = "annual"
        out.attrs["annual_aggregation_rule"] = "identity"
        return out

    if require_full_years:
        _validate_full_years(da)

    for clave in ANNUAL_AGGREGATION_RULES:
        if clave in variable_name:
            variable_name = clave 

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

    allowed = MASK_MAP.keys()
    unknown = set(masks) - allowed
    if unknown:
        raise ValueError(f"Máscaras no soportadas: {sorted(unknown)}")

    parts = []
    for name, mask in masks.items():
        _validate_mask_alignment(da, mask, name)

        if name == "land":
            part = mask.astype(bool)
        elif name in {"bs", "ebf", "landcover_cropland", "landcover_forest", 
                    "landcover_grassland", "landcover_shrubland", "landcover_tundra", "landcover_barren", "landcover_snow_ice"}:
            part = ~mask.astype(bool)
        elif name == "climate":
            part = xr.apply_ufunc(np.isin, mask, np.array(sorted(CLIMATE_VALID_CODES))).astype(bool)
        elif name == "landcover":
            part = xr.apply_ufunc(np.isin, mask, np.array(sorted(LANDCOVER_VALID_CODES))).astype(bool)
        else:
            part = mask.astype(bool)
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


def _dataarray_summary(da: xr.DataArray) -> dict[str, Any]:
    return {
        "shape": tuple(int(x) for x in da.shape),
        "dims": tuple(str(x) for x in da.dims),
        "dtype": str(da.dtype),
    }


def _coord_resolution(values: np.ndarray) -> float | None:
    if len(values) < 2:
        return None
    diffs = np.diff(values.astype(float))
    if not np.allclose(diffs, diffs[0]):
        return None
    return float(diffs[0])


def _time_frequency(temporal_resolution: str) -> str:
    temporal_resolution = temporal_resolution.lower().strip()
    if temporal_resolution == "monthly":
        return "MS"
    if temporal_resolution == "annual":
        return "YS"
    raise ValueError("temporal_resolution debe ser 'monthly' o 'annual'.")


def _grid_metadata(da: xr.DataArray) -> dict[str, Any]:
    latitude = da["latitude"].values
    longitude = da["longitude"].values
    lat_resolution = _coord_resolution(latitude)
    lon_resolution = _coord_resolution(longitude)
    spatial_resolution = lat_resolution if lat_resolution is not None and np.isclose(lat_resolution, lon_resolution) else None
    return {
        "lat_min": float(latitude[0]),
        "lat_max": float(latitude[-1]),
        "lon_min": float(longitude[0]),
        "lon_max": float(longitude[-1]),
        "latitude_size": int(len(latitude)),
        "longitude_size": int(len(longitude)),
        "latitude_order": "ascending" if np.all(np.diff(latitude) > 0) else "unknown",
        "longitude_order": "ascending" if np.all(np.diff(longitude) > 0) else "unknown",
        "latitude_resolution_deg": lat_resolution,
        "longitude_resolution_deg": lon_resolution,
        "spatial_resolution_deg": spatial_resolution,
    }


def _temporal_metadata(da: xr.DataArray, temporal_resolution: str) -> dict[str, Any]:
    return {
        "time_min": str(pd.to_datetime(da["time"].values[0])) if da.sizes["time"] else None,
        "time_max": str(pd.to_datetime(da["time"].values[-1])) if da.sizes["time"] else None,
        "time_size": int(da.sizes["time"]),
        "temporal_resolution": temporal_resolution.lower().strip(),
        "frequency": _time_frequency(temporal_resolution),
        "time_order": "ascending",
    }


def _product_metadata(path: Path | None, da: xr.DataArray) -> dict[str, Any]:
    meta = {
        "path": str(path) if path is not None else None,
        **_dataarray_summary(da),
    }
    if "month" in da.dims:
        meta["month_values"] = [int(x) for x in da["month"].values]
    return meta


def save_preprocess_products(
    output_dir: str | Path,
    variable_name: str,
    products: dict[str, xr.DataArray | dict[str, xr.DataArray]],
    save_output: bool = True,
) -> dict[str, Any]:
    product_dir = Path(output_dir) / "preprocess" / variable_name
    saved: dict[str, Any] = {}

    for product_name, product in products.items():
        if isinstance(product, xr.DataArray):
            path = save_npy(product_dir, product_name, product) if save_output else None
            saved[product_name] = _product_metadata(path, product)
            continue

        if product_name == "trend":
            slope = product["slope"]
            time_years = product["time_years_centered"]
            slope_path = save_npy(product_dir, "trend_slope", slope) if save_output else None
            time_path = save_npy(product_dir, "trend_time_years_centered", time_years) if save_output else None
            saved["trend"] = {
                "storage": "slope_plus_centered_time",
                "description": "Reconstruir tendencia eliminada como trend_slope * trend_time_years_centered.",
                "slope": _product_metadata(slope_path, slope),
                "time_years_centered": _product_metadata(time_path, time_years),
            }
            continue

        saved[product_name] = {"path": None, "storage": "unsupported"}

    return saved


def _process_and_save_single_dataarray(
    da_raw,
    output_dir: str | Path,
    variable_name: str,
    meta_load: dict,
    mask_dir: str | Path | None = None,
    masks: dict | None = None,
    temporal_resolution: str = "monthly",
    annual_rule: str | None = None,
    require_full_years: bool = True,
    data_value_type: str = "real",
    detrend_theil_sen: bool = False,
    save_output: bool = True,
) -> dict:
    """Procesa una única DataArray 3D y devuelve metadata de salida + processing info."""
    variable_info = {
        "requested_name": str(meta_load.get("variable_requested", variable_name)).upper(),
        "base_name": str(meta_load.get("variable_base", variable_name)).upper(),
        "lag_steps": int(meta_load.get("lag_steps", 0)),
        "is_lagged": bool(meta_load.get("is_lagged", False)),
    }
    lag_steps = variable_info["lag_steps"]
    temporal_resolution = temporal_resolution.lower().strip()

    if temporal_resolution == "monthly":
        da_for_aggregation = _apply_lagged_shift(
            da_raw,
            lag_steps=lag_steps,
            temporal_resolution=temporal_resolution,
        )
        lag_apply_stage = "pre_aggregation" if lag_steps > 0 else None
    elif temporal_resolution == "annual":
        da_for_aggregation = da_raw
        lag_apply_stage = "post_aggregation" if lag_steps > 0 else None
    else:
        raise ValueError("temporal_resolution debe ser 'monthly' o 'annual'.")

    da_agg = aggregate_time(
        da=da_for_aggregation,
        variable_name=variable_name,
        temporal_resolution=temporal_resolution,
        annual_rule=annual_rule,
        require_full_years=require_full_years,
    )

    if temporal_resolution == "annual" and lag_steps > 0:
        da_agg = _apply_lagged_shift(
            da_agg,
            lag_steps=lag_steps,
            temporal_resolution=temporal_resolution,
        )

    combined_mask, mask_info = build_combined_filter_mask(da_agg, masks if masks else None)
    da_masked = apply_filter_mask(da_agg, combined_mask)
    preprocess_result = preprocess.apply_preprocessing(
        da_masked,
        temporal_resolution=temporal_resolution,
        data_value_type=data_value_type,
        detrend_theil_sen=detrend_theil_sen,
    )
    da_final = preprocess_result.data
    preprocess_products = save_preprocess_products(
        output_dir=output_dir,
        variable_name=variable_name,
        products=preprocess_result.products,
        save_output=save_output,
    )

    load_meta_out = dict(meta_load)
    load_meta_out["lag_applied"] = bool(lag_steps > 0)
    load_meta_out["lag_apply_stage"] = lag_apply_stage
    if lag_steps > 0:
        load_meta_out["lag_temporal_unit"] = "months" if temporal_resolution == "monthly" else "years"

    output_path = save_npy(output_dir, variable_name, da_final) if save_output else None

    result = {
        variable_name: {
            "logical_name": variable_name,
            "array_path": str(output_path) if output_path is not None else None,
            "load_metadata": load_meta_out,
            "processing": {
                "temporal_resolution": temporal_resolution,
                "annual_rule": annual_rule,
                "require_full_years": require_full_years,
                **preprocess_result.metadata,
                "preprocess_products": preprocess_products,
                "mask_dir": str(mask_dir) if mask_dir is not None else None,
                "is_lagged": variable_info["is_lagged"],
                "lag_steps": lag_steps,
                "lag_apply_stage": lag_apply_stage,
                "lag_temporal_unit": load_meta_out["lag_temporal_unit"],
                **mask_info,
            },
            "grid": _grid_metadata(da_final),
            "temporal_grid": _temporal_metadata(da_final, temporal_resolution),
            "final_dims": tuple(str(x) for x in da_final.dims),
            "final_shape": tuple(int(x) for x in da_final.shape),
            "final_time_min": str(pd.to_datetime(da_final["time"].values[0])) if da_final.sizes["time"] else None,
            "final_time_max": str(pd.to_datetime(da_final["time"].values[-1])) if da_final.sizes["time"] else None,
            "final_units": da_final.attrs.get("units", ""),
            "final_dtype": str(da_final.dtype),
            "reconstruction": {
                "supported": bool(preprocess_result.metadata["preprocessing_applied"]),
                "metadata_source": "metadata.json",
                "order": preprocess_result.metadata["reconstruction_order"],
                "products_root": str(Path(output_dir) / "preprocess" / variable_name),
            },
        }
    }

    del da_agg, da_masked, da_final, combined_mask, preprocess_result
    gc.collect()

    return result

def _process_and_save_multiclass_dataarray(
    da_raw,
    output_dir: str | Path,
    variable: str,
    meta_load: dict,
    mask_dir: str | Path | None = None,
    masks: dict | None = None,
    temporal_resolution: str = "monthly",
    annual_rule: str | None = None,
    require_full_years: bool = True,
    data_value_type: str = "real",
    detrend_theil_sen: bool = False,
    save_output: bool = True,
) -> dict:
    """Procesa una DataArray con dimensión 'class' y guarda una salida por clase."""

    outputs = {}  # ← aquí guardaremos el resultado de cada clase por separado
    last_mask_info = {}  # ← guardamos mask_info de la última iteración para devolverlo

    classes = da_raw["class"].values  # ← extraemos los valores reales de las clases

    for i, c in enumerate(classes):
        da_class = da_raw.isel({"class": i})  
        # ← seleccionamos una sola clase y eliminamos la dimensión 'class'

        class_variable_name = f"{variable.upper()}_CLASS_{c}"
        # ← construimos un nombre único para el npy de esta clase

        class_result = _process_and_save_single_dataarray(
            da_raw=da_class,
            output_dir=output_dir,
            variable_name=class_variable_name,
            meta_load=meta_load,
            mask_dir=mask_dir,
            masks=masks,
            temporal_resolution=temporal_resolution,
            annual_rule=annual_rule,
            require_full_years=require_full_years,
            data_value_type=data_value_type,
            detrend_theil_sen=detrend_theil_sen,
            save_output=save_output,
        )

        outputs.update(class_result)

        del da_class
        gc.collect()

    return outputs


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
    data_value_type: str = "real",
    detrend_theil_sen: bool = False,
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

    masks = {}
    if mask_names:
        if mask_dir is None:
            raise ValueError("Si usas mask_names, debes pasar mask_dir.")
        for name in mask_names:
            masks[name] = load_mask(
                mask_dir,
                name,
                da_raw["latitude"].values,
                da_raw["longitude"].values,
            )
            # ← aquí usamos da_raw, porque todavía no sabemos si la variable tiene class o no

    if "class" in da_raw.dims:
        result = _process_and_save_multiclass_dataarray(
            da_raw=da_raw,
            output_dir=output_dir,
            variable=variable,
            meta_load=meta_load,
            mask_dir=mask_dir,
            masks=masks if masks else None,
            temporal_resolution=temporal_resolution,
            annual_rule=annual_rule,
            require_full_years=require_full_years,
            data_value_type=data_value_type,
            detrend_theil_sen=detrend_theil_sen,
            save_output=save_output,
        )
    else:
        result = _process_and_save_single_dataarray(
            da_raw=da_raw,
            output_dir=output_dir,
            variable_name=variable.upper(),
            meta_load=meta_load,
            mask_dir=mask_dir,
            masks=masks if masks else None,
            temporal_resolution=temporal_resolution,
            annual_rule=annual_rule,
            require_full_years=require_full_years,
            data_value_type=data_value_type,
            detrend_theil_sen=detrend_theil_sen,
            save_output=save_output,
        )

    del da_raw, masks
    gc.collect()
    return result


def build_processed_metadata(
    variable_results: dict[str, dict],
    temporal_resolution: str,
    roi: ROI | None = None,
    start_year: int | None = None,
    end_year_inclusive: int | None = None,
    dtype: str = "float32",
    data_value_type: str = "real",
    detrend_theil_sen: bool = False,
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
            **preprocess.validate_preprocess_options(
                data_value_type=data_value_type,
                detrend_theil_sen=detrend_theil_sen,
            ),
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
    data_value_type: str = "real",
    detrend_theil_sen: bool = False,
    filename: str = "metadata.json",
) -> Path:
    metadata = build_processed_metadata(
        variable_results=variable_results,
        temporal_resolution=temporal_resolution,
        roi=roi,
        start_year=start_year,
        end_year_inclusive=end_year_inclusive,
        dtype=dtype,
        data_value_type=data_value_type,
        detrend_theil_sen=detrend_theil_sen,
    )
    return save_metadata_json(output_dir, metadata, filename)

def build_processed_run_config(
    variable_names: list[str],
    temporal_resolution: str,
    mask_names: list[str] | None = None,
    start_year: int | None = None,
    end_year_inclusive: int | None = None,
    dtype: str = "float32",
    roi: ROI | None = None,
    data_value_type: str = "real",
    detrend_theil_sen: bool = False,
) -> dict:
    options = preprocess.validate_preprocess_options(
        data_value_type=data_value_type,
        detrend_theil_sen=detrend_theil_sen,
    )
    return {
        "variable_names": [str(v).upper() for v in variable_names],
        "temporal_resolution": temporal_resolution.lower().strip(),
        "mask_names": [str(m).lower().strip() for m in (mask_names or [])],
        "start_year": start_year,
        "end_year_inclusive": end_year_inclusive,
        "dtype": str(dtype),
        "roi": _to_jsonable(roi),
        **options,
    }


def load_processed_run_config(
    output_dir: str | Path,
    filename: str = "run_config.json",
) -> dict | None:
    path = Path(output_dir) / filename
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def processed_run_status(
    output_dir: str | Path,
    variable_names: list[str],
    expected_config: dict | None = None,
    metadata_filename: str = "metadata.json",
    run_config_filename: str = "run_config.json",
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    variable_outputs: dict[str, list[Path]] = {}
    for variable in variable_names:
        variable_name = str(variable).upper()
        exact_path = output_dir / f"{variable_name}.npy"
        if exact_path.exists():
            variable_outputs[variable_name] = [exact_path]
        else:
            variable_outputs[variable_name] = sorted(output_dir.glob(f"{variable_name}_*.npy"))

    metadata_path = output_dir / metadata_filename
    run_config_path = output_dir / run_config_filename

    files_present = all(bool(paths) for paths in variable_outputs.values())
    metadata_present = metadata_path.exists()
    run_config_present = run_config_path.exists()
    complete = output_dir.exists() and files_present and metadata_present and run_config_present

    saved_config = load_processed_run_config(output_dir, filename=run_config_filename)
    config_matches = expected_config is None or (saved_config == _to_jsonable(expected_config))

    return {
        "output_dir": str(output_dir),
        "exists": output_dir.exists(),
        "complete": bool(complete),
        "files_present": bool(files_present),
        "metadata_present": bool(metadata_present),
        "run_config_present": bool(run_config_present),
        "config_matches": bool(config_matches),
        "saved_config": saved_config,
        "missing_files": [
            str(output_dir / f"{variable_name}.npy")
            for variable_name, paths in variable_outputs.items()
            if not paths
        ],
        "variable_outputs": {
            variable_name: [str(path) for path in paths]
            for variable_name, paths in variable_outputs.items()
        },
        "metadata_path": str(metadata_path),
        "run_config_path": str(run_config_path),
    }
