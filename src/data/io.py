
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
    "SM1_2": "mean",
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

LAGGED_VARIABLE_PATTERN = re.compile(r"^(?P<base>[A-Z0-9_]+)_LAG_(?P<lag>\d+)$")


@dataclass(frozen=True)
class ROI:
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float


def _normalize_roi(roi: ROI | dict[str, float] | None) -> ROI | None:
    if roi is None or isinstance(roi, ROI):
        return roi
    if isinstance(roi, dict):
        return ROI(
            lat_min=float(roi["lat_min"]),
            lat_max=float(roi["lat_max"]),
            lon_min=float(roi["lon_min"]),
            lon_max=float(roi["lon_max"]),
        )
    raise TypeError("roi debe ser ROI, dict o None.")


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
    """Resuelve una ruta registrada en paths.toml."""

    base_dir = Path(base_dir).expanduser()

    normalized_name = str(name).upper().strip()
    normalized_map = {
        str(key).upper().strip(): str(value)
        for key, value in file_map.items()
    }

    if normalized_name not in normalized_map:
        available = ", ".join(sorted(normalized_map))

        raise ValueError(
            f"{normalized_name!r} no está registrado. "
            f"Disponibles: {available}"
        )

    path = (base_dir / normalized_map[normalized_name]).resolve()

    if not path.exists():
        raise FileNotFoundError(
            f"No existe el archivo para {normalized_name!r}: {path}"
        )

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
    variable_file_map: dict[str, str],
    roi: ROI | dict[str, float] | None = None,
    start_year: int | None = None,
    end_year_inclusive: int | None = None,
    dtype: str = "float32",
) -> tuple[xr.DataArray, dict]:
    
    variable_info = _parse_variable_request(variable)
    variable = variable_info["requested_name"]
    base_variable = variable_info["base_name"]
    path = _get_path(base_dir, base_variable, variable_file_map)

    ds = xr.open_dataset(path)
    ds = _standardize_dataset(ds)
    var_name = _get_single_data_var(ds)

    if "class" in ds.dims:
        da = ds[var_name].transpose("time", "class", "latitude", "longitude")
    else:
        da = ds[var_name].transpose("time", "latitude", "longitude")
    da = _select_time(da, start_year, end_year_inclusive)
    da = _select_roi(da, _normalize_roi(roi))
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


def _infer_temporal_resolution(da: xr.DataArray) -> str:
    """Detecta si la serie contiene un valor mensual o anual por año."""
    if "time" not in da.dims or da.sizes.get("time", 0) == 0:
        raise ValueError("La variable no contiene pasos temporales.")

    counts = da["time"].dt.year.to_series().value_counts().sort_index()

    if bool((counts == 12).all()):
        return "monthly"
    if bool((counts == 1).all()):
        return "annual"

    raise ValueError(
        "No se puede determinar una resolución temporal regular. "
        f"Pasos por año: {counts.to_dict()}"
    )


def _resolve_annual_rule_variable(variable_name: str) -> str:
    """Obtiene el nombre base usado por ANNUAL_AGGREGATION_RULES."""
    base_name = _parse_variable_request(variable_name)["base_name"]

    if base_name in ANNUAL_AGGREGATION_RULES:
        return base_name

    candidates = sorted(ANNUAL_AGGREGATION_RULES, key=len, reverse=True)
    for candidate in candidates:
        if base_name.startswith(f"{candidate}_"):
            return candidate

    raise ValueError(
        "No existe una regla de agregación anual para "
        f"{variable_name!r}."
    )


def _expand_annual_to_monthly(da: xr.DataArray) -> xr.DataArray:
    """Repite cada valor anual en los doce meses del mismo año."""
    parts: list[xr.DataArray] = []

    for index in range(da.sizes["time"]):
        annual_value = da.isel(time=index, drop=True)
        year = int(da["time"].dt.year.isel(time=index).item())
        monthly_time = pd.date_range(f"{year}-01-01", periods=12, freq="MS")
        parts.append(annual_value.expand_dims(time=monthly_time))

    out = xr.concat(parts, dim="time").transpose(*da.dims)
    out.attrs = dict(da.attrs)
    out.attrs["native_temporal_resolution"] = "annual"
    out.attrs["temporal_resolution"] = "monthly"
    out.attrs["temporal_conversion"] = "annual_to_monthly_repeat"
    out.attrs["annual_aggregation_rule"] = None
    return out


def aggregate_time(
    da: xr.DataArray,
    variable_name: str,
    temporal_resolution: str = "monthly",
    annual_rule: str | None = None,
    require_full_years: bool = True,
) -> xr.DataArray:
    """Armoniza la resolución temporal manteniendo la API existente."""
    requested_resolution = str(temporal_resolution).lower().strip()
    if requested_resolution not in {"monthly", "annual"}:
        raise ValueError("temporal_resolution debe ser 'monthly' o 'annual'.")

    native_resolution = _infer_temporal_resolution(da)

    if native_resolution == requested_resolution:
        out = da.copy()
        out.attrs = dict(da.attrs)
        out.attrs["native_temporal_resolution"] = native_resolution
        out.attrs["temporal_resolution"] = requested_resolution
        out.attrs["temporal_conversion"] = "identity"
        out.attrs["annual_aggregation_rule"] = (
            "identity" if requested_resolution == "annual" else None
        )
        return out

    if native_resolution == "annual" and requested_resolution == "monthly":
        return _expand_annual_to_monthly(da)

    if require_full_years:
        _validate_full_years(da)

    rule_variable = _resolve_annual_rule_variable(variable_name)
    rule = annual_rule or ANNUAL_AGGREGATION_RULES[rule_variable]
    if rule not in {"mean", "sum"}:
        raise ValueError("annual_rule debe ser 'mean' o 'sum'.")

    grouped = da.groupby("time.year")
    if rule == "mean":
        out = grouped.mean(dim="time", skipna=True)
    else:
        out = grouped.sum(dim="time", skipna=True)

    years = out["year"].values
    out = out.rename({"year": "time"})
    out = out.assign_coords(
        time=pd.to_datetime([f"{int(year)}-01-01" for year in years])
    )
    out.attrs = dict(da.attrs)
    out.attrs["native_temporal_resolution"] = "monthly"
    out.attrs["temporal_resolution"] = "annual"
    out.attrs["temporal_conversion"] = f"monthly_to_annual_{rule}"
    out.attrs["annual_aggregation_rule"] = rule
    return out


def load_mask(
    mask_dir: str | Path,
    mask_name: str,
    latitude: np.ndarray,
    longitude: np.ndarray,
    mask_file_map: dict[str, str] | None = None,
) -> xr.DataArray:
    """Carga una máscara global de 0.5 grados y la recorta al grid solicitado."""
    if mask_file_map is None:
        raise ValueError("Debes proporcionar mask_file_map para cargar la máscara.")

    path = _get_path(mask_dir, mask_name, mask_file_map)
    array = np.load(path)
    if array.ndim != 2:
        raise ValueError(
            f"La máscara {mask_name!r} debe ser bidimensional. "
            f"Shape encontrada: {array.shape}."
        )

    latitude = np.asarray(latitude, dtype=np.float64)
    longitude = np.asarray(longitude, dtype=np.float64)

    if array.shape == (len(latitude), len(longitude)):
        return xr.DataArray(
            array,
            coords={"latitude": latitude, "longitude": longitude},
            dims=("latitude", "longitude"),
            name=mask_name,
        )

    if array.shape != (360, 720):
        raise ValueError(
            f"La máscara global {mask_name!r} debe tener shape (360, 720). "
            f"Shape encontrada: {array.shape}."
        )

    global_mask = xr.DataArray(
        array,
        coords={
            "latitude": np.arange(-90.0, 90.0, 0.5),
            "longitude": np.arange(-180.0, 180.0, 0.5),
        },
        dims=("latitude", "longitude"),
        name=mask_name,
    )

    try:
        selected = global_mask.sel(
            latitude=xr.DataArray(latitude, dims="latitude"),
            longitude=xr.DataArray(longitude, dims="longitude"),
        )
    except KeyError as exc:
        raise ValueError(
            f"La máscara {mask_name!r} no comparte el grid de la variable."
        ) from exc

    return selected.assign_coords(latitude=latitude, longitude=longitude)


def _validate_mask_alignment(
    da: xr.DataArray,
    mask: xr.DataArray,
    name: str,
) -> None:
    if tuple(mask.dims) != ("latitude", "longitude"):
        raise ValueError(f"{name} debe tener dims ('latitude', 'longitude').")
    if (
        da.sizes["latitude"] != mask.sizes["latitude"]
        or da.sizes["longitude"] != mask.sizes["longitude"]
    ):
        raise ValueError(f"{name} no coincide en shape espacial.")
    if not np.array_equal(da["latitude"].values, mask["latitude"].values):
        raise ValueError(f"{name} no comparte latitude.")
    if not np.array_equal(da["longitude"].values, mask["longitude"].values):
        raise ValueError(f"{name} no comparte longitude.")


def build_combined_filter_mask(
    da: xr.DataArray,
    masks: dict[str, xr.DataArray] | None = None,
    binary_mask_metadata: dict[str, dict[str, Any]] | None = None,
    categorical_masks: dict[str, xr.DataArray] | None = None,
    categorical_filters: dict[str, list[int]] | None = None,
    categorical_mask_metadata: dict[str, dict[str, Any]] | None = None,
) -> tuple[xr.DataArray | None, dict[str, Any]]:
    """Combina máscaras binarias y selecciones categóricas."""
    binary_masks = masks or {}
    categorical_masks = categorical_masks or {}
    categorical_filters = categorical_filters or {}

    parts: list[xr.DataArray] = []
    binary_info: dict[str, dict[str, Any]] = {}
    categorical_info: dict[str, dict[str, Any]] = {}

    if binary_masks and binary_mask_metadata is None:
        raise ValueError("Falta binary_mask_metadata.")

    for name, mask in binary_masks.items():
        _validate_mask_alignment(da, mask, name)
        metadata = (binary_mask_metadata or {}).get(name)
        if metadata is None:
            raise ValueError(f"Máscara binaria no registrada: {name!r}.")
        if not bool(metadata.get("enabled", True)):
            raise ValueError(f"La máscara {name!r} está deshabilitada.")

        mode = str(metadata.get("mode", "")).lower().strip()
        if mode == "include":
            part = mask.astype(bool)
        elif mode == "exclude":
            part = ~mask.astype(bool)
        else:
            raise ValueError(
                f"Modo no válido para {name!r}: {mode!r}. "
                "Debe ser 'include' o 'exclude'."
            )

        parts.append(part)
        binary_info[name] = {
            "mode": mode,
            "pixels_kept": int(part.sum()),
            "total_pixels": int(part.size),
            "fraction_kept": float(part.mean()),
        }

    for name, selected_classes in categorical_filters.items():
        selected_classes = sorted({int(value) for value in selected_classes})
        if not selected_classes:
            continue
        if name not in categorical_masks:
            raise ValueError(f"No se ha cargado la máscara categórica {name!r}.")

        metadata = (categorical_mask_metadata or {}).get(name, {})
        if not bool(metadata.get("enabled", True)):
            raise ValueError(f"La máscara categórica {name!r} está deshabilitada.")

        mask = categorical_masks[name]
        _validate_mask_alignment(da, mask, name)
        part = xr.apply_ufunc(
            np.isin,
            mask,
            np.asarray(selected_classes, dtype=np.int64),
        ).astype(bool)
        parts.append(part)
        categorical_info[name] = {
            "selected_classes": selected_classes,
            "pixels_kept": int(part.sum()),
            "total_pixels": int(part.size),
            "fraction_kept": float(part.mean()),
        }

    if not parts:
        return None, {
            "mask_names": [],
            "categorical_filters": {},
            "combined_fraction_kept": None,
        }

    combined = parts[0]
    for part in parts[1:]:
        combined = combined & part
    combined = combined.rename("combined_filter_mask")

    info = {
        "mask_names": list(binary_masks),
        "binary_masks": binary_info,
        "categorical_filters": {
            name: values["selected_classes"]
            for name, values in categorical_info.items()
        },
        "categorical_masks": categorical_info,
        "combined_pixels_kept": int(combined.sum()),
        "combined_total_pixels": int(combined.size),
        "combined_fraction_kept": float(combined.mean()),
    }
    return combined, info


def apply_filter_mask(
    da: xr.DataArray,
    filter_mask: xr.DataArray | None,
) -> xr.DataArray:
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
    binary_mask_metadata: dict[str, dict[str, Any]] | None = None,
    categorical_masks: dict | None = None,
    categorical_filters: dict[str, list[int]] | None = None,
    categorical_mask_metadata: dict[str, dict[str, Any]] | None = None,
    temporal_resolution: str = "monthly",
    annual_rule: str | None = None,
    require_full_years: bool = True,
    data_value_type: str = "real",
    detrend_theil_sen: bool = False,
    save_output: bool = True,
) -> dict:
    """Procesa una única DataArray 3D y devuelve sus metadatos."""
    variable_info = {
        "requested_name": str(
            meta_load.get("variable_requested", variable_name)
        ).upper(),
        "base_name": str(meta_load.get("variable_base", variable_name)).upper(),
        "lag_steps": int(meta_load.get("lag_steps", 0)),
        "is_lagged": bool(meta_load.get("is_lagged", False)),
    }
    lag_steps = variable_info["lag_steps"]
    temporal_resolution = temporal_resolution.lower().strip()

    native_temporal_resolution = _infer_temporal_resolution(da_raw)

    if temporal_resolution == "annual" and native_temporal_resolution == "monthly":
        da_for_aggregation = _apply_lagged_shift(
            da_raw, lag_steps, "monthly"
        )
        lag_apply_stage = "pre_aggregation" if lag_steps > 0 else None
    else:
        da_for_aggregation = da_raw
        lag_apply_stage = "post_aggregation" if lag_steps > 0 else None

    da_agg = aggregate_time(
        da=da_for_aggregation,
        variable_name=variable_name,
        temporal_resolution=temporal_resolution,
        annual_rule=annual_rule,
        require_full_years=require_full_years,
    )

    if lag_steps > 0 and lag_apply_stage == "post_aggregation":
        da_agg = _apply_lagged_shift(
            da_agg, lag_steps, temporal_resolution
        )

    combined_mask, mask_info = build_combined_filter_mask(
        da=da_agg,
        masks=masks,
        binary_mask_metadata=binary_mask_metadata,
        categorical_masks=categorical_masks,
        categorical_filters=categorical_filters,
        categorical_mask_metadata=categorical_mask_metadata,
    )
    da_masked = apply_filter_mask(da_agg, combined_mask)
    preprocess_result = preprocess.apply_preprocessing(
        da_masked,
        temporal_resolution=temporal_resolution,
        data_value_type=data_value_type,
        detrend_theil_sen=detrend_theil_sen,
    )
    da_final = preprocess_result.data
    preprocess_products = save_preprocess_products(
        output_dir, variable_name, preprocess_result.products, save_output
    )

    load_meta_out = dict(meta_load)
    load_meta_out["lag_applied"] = lag_steps > 0
    load_meta_out["lag_apply_stage"] = lag_apply_stage
    load_meta_out["lag_temporal_unit"] = (
        "months" if temporal_resolution == "monthly" else "years"
    ) if lag_steps > 0 else None

    output_path = save_npy(output_dir, variable_name, da_final) if save_output else None
    reconstruction_supported = data_value_type in {"real", "anomaly"}

    result = {
        variable_name: {
            "logical_name": variable_name,
            "array_path": str(output_path) if output_path else None,
            "load_metadata": load_meta_out,
            "processing": {
                "temporal_resolution": temporal_resolution,
                "native_temporal_resolution": da_agg.attrs.get(
                    "native_temporal_resolution",
                    native_temporal_resolution,
                ),
                "requested_temporal_resolution": temporal_resolution,
                "temporal_conversion": da_agg.attrs.get(
                    "temporal_conversion",
                    "identity",
                ),
                "annual_rule": da_agg.attrs.get(
                    "annual_aggregation_rule"
                ),
                "native_time_size": int(da_raw.sizes["time"]),
                "final_time_size": int(da_final.sizes["time"]),
                "require_full_years": require_full_years,
                **preprocess_result.metadata,
                "preprocess_products": preprocess_products,
                "mask_dir": str(mask_dir) if mask_dir else None,
                "is_lagged": variable_info["is_lagged"],
                "lag_steps": lag_steps,
                "lag_apply_stage": lag_apply_stage,
                "lag_temporal_unit": load_meta_out["lag_temporal_unit"],
                **mask_info,
            },
            "grid": _grid_metadata(da_final),
            "temporal_grid": _temporal_metadata(da_final, temporal_resolution),
            "final_dims": tuple(str(value) for value in da_final.dims),
            "final_shape": tuple(int(value) for value in da_final.shape),
            "final_time_min": (
                str(pd.to_datetime(da_final["time"].values[0]))
                if da_final.sizes["time"] else None
            ),
            "final_time_max": (
                str(pd.to_datetime(da_final["time"].values[-1]))
                if da_final.sizes["time"] else None
            ),
            "final_units": da_final.attrs.get("units", ""),
            "final_dtype": str(da_final.dtype),
            "reconstruction": {
                "supported": reconstruction_supported,
                "metadata_source": "metadata.json",
                "order": preprocess_result.metadata["reconstruction_order"],
                "products_root": str(
                    Path(output_dir) / "preprocess" / variable_name
                ),
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
    binary_mask_metadata: dict[str, dict[str, Any]] | None = None,
    categorical_masks: dict | None = None,
    categorical_filters: dict[str, list[int]] | None = None,
    categorical_mask_metadata: dict[str, dict[str, Any]] | None = None,
    temporal_resolution: str = "monthly",
    annual_rule: str | None = None,
    require_full_years: bool = True,
    data_value_type: str = "real",
    detrend_theil_sen: bool = False,
    save_output: bool = True,
) -> dict:
    outputs = {}
    for index, class_value in enumerate(da_raw["class"].values):
        class_name = f"{variable.upper()}_CLASS_{class_value}"
        outputs.update(
            _process_and_save_single_dataarray(
                da_raw=da_raw.isel({"class": index}),
                output_dir=output_dir,
                variable_name=class_name,
                meta_load=meta_load,
                mask_dir=mask_dir,
                masks=masks,
                binary_mask_metadata=binary_mask_metadata,
                categorical_masks=categorical_masks,
                categorical_filters=categorical_filters,
                categorical_mask_metadata=categorical_mask_metadata,
                temporal_resolution=temporal_resolution,
                annual_rule=annual_rule,
                require_full_years=require_full_years,
                data_value_type=data_value_type,
                detrend_theil_sen=detrend_theil_sen,
                save_output=save_output,
            )
        )
    return outputs


def load_and_save_variable(
    raw_dir: str | Path,
    output_dir: str | Path,
    variable: str,
    variable_file_map: dict[str, str],
    mask_dir: str | Path | None = None,
    binary_mask_file_map: dict[str, str] | None = None,
    binary_mask_metadata: dict[str, dict[str, Any]] | None = None,
    mask_names: list[str] | None = None,
    roi: ROI | dict[str, float] | None = None,
    start_year: int | None = None,
    end_year_inclusive: int | None = None,
    dtype: str = "float32",
    temporal_resolution: str = "monthly",
    annual_rule: str | None = None,
    require_full_years: bool = True,
    data_value_type: str = "real",
    detrend_theil_sen: bool = False,
    save_output: bool = True,
    categorical_mask_file_map: dict[str, str] | None = None,
    categorical_mask_metadata: dict[str, dict[str, Any]] | None = None,
    categorical_filters: dict[str, list[int]] | None = None,
) -> dict:
    da_raw, meta_load = load_netcdf(
        base_dir=raw_dir,
        variable=variable,
        variable_file_map=variable_file_map,
        roi=roi,
        start_year=start_year,
        end_year_inclusive=end_year_inclusive,
        dtype=dtype,
    )

    binary_masks: dict[str, xr.DataArray] = {}
    for name in mask_names or []:
        if mask_dir is None or binary_mask_file_map is None:
            raise ValueError(
                "Para usar máscaras binarias debes pasar mask_dir y "
                "binary_mask_file_map."
            )
        binary_masks[name] = load_mask(
            mask_dir,
            name,
            da_raw["latitude"].values,
            da_raw["longitude"].values,
            binary_mask_file_map,
        )

    categorical_masks: dict[str, xr.DataArray] = {}
    for name, selected_classes in (categorical_filters or {}).items():
        if not selected_classes:
            continue
        if mask_dir is None or categorical_mask_file_map is None:
            raise ValueError(
                "Para usar filtros categóricos debes pasar mask_dir y "
                "categorical_mask_file_map."
            )
        categorical_masks[name] = load_mask(
            mask_dir,
            name,
            da_raw["latitude"].values,
            da_raw["longitude"].values,
            categorical_mask_file_map,
        )

    common = dict(
        output_dir=output_dir,
        meta_load=meta_load,
        mask_dir=mask_dir,
        masks=binary_masks or None,
        binary_mask_metadata=binary_mask_metadata,
        categorical_masks=categorical_masks or None,
        categorical_filters=categorical_filters,
        categorical_mask_metadata=categorical_mask_metadata,
        temporal_resolution=temporal_resolution,
        annual_rule=annual_rule,
        require_full_years=require_full_years,
        data_value_type=data_value_type,
        detrend_theil_sen=detrend_theil_sen,
        save_output=save_output,
    )

    if "class" in da_raw.dims:
        result = _process_and_save_multiclass_dataarray(
            da_raw=da_raw, variable=variable, **common
        )
    else:
        result = _process_and_save_single_dataarray(
            da_raw=da_raw, variable_name=variable.upper(), **common
        )

    del da_raw, binary_masks, categorical_masks
    gc.collect()
    return result


def build_processed_metadata(
    variable_results: dict[str, dict],
    temporal_resolution: str,
    roi: ROI | dict[str, float] | None = None,
    start_year: int | None = None,
    end_year_inclusive: int | None = None,
    dtype: str = "float32",
    data_value_type: str = "real",
    detrend_theil_sen: bool = False,
    mask_names: list[str] | None = None,
    categorical_filters: dict[str, list[int]] | None = None,
) -> dict:
    if temporal_resolution.lower() not in {"monthly", "annual"}:
        raise ValueError("temporal_resolution debe ser 'monthly' o 'annual'.")
    if not variable_results:
        raise ValueError("variable_results no puede estar vacío.")

    return {
        "dataset_config": {
            "temporal_resolution": temporal_resolution.lower(),
            "roi": _to_jsonable(roi),
            "start_year": start_year,
            "end_year_inclusive": end_year_inclusive,
            "dtype": dtype,
            "mask_names": [str(name).lower() for name in (mask_names or [])],
            "categorical_filters": {
                str(name).lower(): sorted({int(value) for value in values})
                for name, values in (categorical_filters or {}).items()
            },
            **preprocess.validate_preprocess_options(
                data_value_type=data_value_type,
                detrend_theil_sen=detrend_theil_sen,
            ),
        },
        "variables": {key.upper(): value for key, value in variable_results.items()},
    }


def save_processed_metadata(
    output_dir: str | Path,
    variable_results: dict[str, dict],
    temporal_resolution: str,
    roi: ROI | dict[str, float] | None = None,
    start_year: int | None = None,
    end_year_inclusive: int | None = None,
    dtype: str = "float32",
    data_value_type: str = "real",
    detrend_theil_sen: bool = False,
    filename: str = "metadata.json",
    mask_names: list[str] | None = None,
    categorical_filters: dict[str, list[int]] | None = None,
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
        mask_names=mask_names,
        categorical_filters=categorical_filters,
    )
    return save_metadata_json(output_dir, metadata, filename)


def build_processed_run_config(
    variable_names: list[str],
    temporal_resolution: str,
    mask_names: list[str] | None = None,
    start_year: int | None = None,
    end_year_inclusive: int | None = None,
    dtype: str = "float32",
    roi: ROI | dict[str, float] | None = None,
    data_value_type: str = "real",
    detrend_theil_sen: bool = False,
    categorical_filters: dict[str, list[int]] | None = None,
) -> dict:
    return {
        "variable_names": [str(value).upper() for value in variable_names],
        "temporal_resolution": temporal_resolution.lower().strip(),
        "mask_names": [str(value).lower().strip() for value in (mask_names or [])],
        "categorical_filters": {
            str(name).lower(): sorted({int(value) for value in values})
            for name, values in (categorical_filters or {}).items()
        },
        "start_year": start_year,
        "end_year_inclusive": end_year_inclusive,
        "dtype": str(dtype),
        "roi": _to_jsonable(roi),
        **preprocess.validate_preprocess_options(
            data_value_type=data_value_type,
            detrend_theil_sen=detrend_theil_sen,
        ),
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
