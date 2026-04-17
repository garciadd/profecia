import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import rasterio


# Configuración interna para controlar RAM

DEFAULT_RANDOM_SEED = 42
DEFAULT_CORR_MAX_POINTS = 200_000
DEFAULT_STATS_MAX_POINTS = 500_000
DEFAULT_CATEGORY_MAX_PIXELS = 2_000
DEFAULT_SCATTER_MAX_ATTEMPTS = 10


# Helpers internos

def _to_float32_view(arr: np.ndarray) -> np.ndarray:
    """
    Devuelve vista float32 si ya lo es; si no, convierte.
    """
    if arr.dtype == np.float32:
        return arr
    return arr.astype(np.float32, copy=False)


def _corr_from_vectors(a: np.ndarray, b: np.ndarray) -> float:
    """
    Correlación de Pearson entre dos vectores 1D ya filtrados.
    """
    if a.size < 2 or b.size < 2:
        return np.nan

    a = a.astype(np.float32, copy=False)
    b = b.astype(np.float32, copy=False)

    a_mean = a.mean()
    b_mean = b.mean()

    a_centered = a - a_mean
    b_centered = b - b_mean

    denom = np.sqrt((a_centered ** 2).sum() * (b_centered ** 2).sum())
    if denom == 0:
        return np.nan

    return float((a_centered * b_centered).sum() / denom)


def _sample_valid_values_1d(
    arr: np.ndarray,
    max_points: int = DEFAULT_STATS_MAX_POINTS,
    seed: int = DEFAULT_RANDOM_SEED,
    max_attempts: int = DEFAULT_SCATTER_MAX_ATTEMPTS,
) -> np.ndarray:
    """
    Muestra aleatoriamente valores válidos de un array sin construir
    el vector completo de índices válidos.

    Útil para cuantiles e histogramas.
    """
    flat = arr.ravel()
    n_total = flat.size
    if n_total == 0:
        return np.array([], dtype=np.float32)

    rng = np.random.default_rng(seed)
    collected = []

    target_size = min(max_points, n_total)
    chunk_size = min(max(target_size * 2, 10_000), n_total)

    for _ in range(max_attempts):
        idx = rng.integers(0, n_total, size=chunk_size, endpoint=False)
        values = _to_float32_view(flat[idx])
        values = values[np.isfinite(values)]

        if values.size > 0:
            collected.append(values)

        current_size = sum(x.size for x in collected)
        if current_size >= target_size:
            break

    if not collected:
        return np.array([], dtype=np.float32)

    out = np.concatenate(collected)
    if out.size > target_size:
        idx = rng.choice(out.size, size=target_size, replace=False)
        out = out[idx]

    return out


def _sample_valid_pairs_flat(
    da1: xr.DataArray,
    da2: xr.DataArray,
    max_points: int = DEFAULT_CORR_MAX_POINTS,
    seed: int = DEFAULT_RANDOM_SEED,
    max_attempts: int = DEFAULT_SCATTER_MAX_ATTEMPTS,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Muestra pares válidos (a,b) de dos DataArray sin construir
    una máscara booleana gigante global.
    """
    a = da1.values.ravel()
    b = da2.values.ravel()

    n_total = a.size
    if n_total == 0:
        return (
            np.array([], dtype=np.float32),
            np.array([], dtype=np.float32),
        )

    rng = np.random.default_rng(seed)
    collected_a = []
    collected_b = []

    target_size = min(max_points, n_total)
    chunk_size = min(max(target_size * 2, 20_000), n_total)

    for _ in range(max_attempts):
        idx = rng.integers(0, n_total, size=chunk_size, endpoint=False)
        a_chunk = _to_float32_view(a[idx])
        b_chunk = _to_float32_view(b[idx])

        valid = np.isfinite(a_chunk) & np.isfinite(b_chunk)
        if valid.any():
            collected_a.append(a_chunk[valid])
            collected_b.append(b_chunk[valid])

        current_size = sum(x.size for x in collected_a)
        if current_size >= target_size:
            break

    if not collected_a:
        return (
            np.array([], dtype=np.float32),
            np.array([], dtype=np.float32),
        )

    a_out = np.concatenate(collected_a)
    b_out = np.concatenate(collected_b)

    if a_out.size > target_size:
        idx = rng.choice(a_out.size, size=target_size, replace=False)
        a_out = a_out[idx]
        b_out = b_out[idx]

    return a_out, b_out


def _sample_category_pixel_indices(
    category_mask: xr.DataArray,
    category_values: list[int] | tuple[int, ...],
    max_pixels: int = DEFAULT_CATEGORY_MAX_PIXELS,
    seed: int = DEFAULT_RANDOM_SEED,
) -> np.ndarray:
    """
    Devuelve índices planos de píxeles pertenecientes a una o varias categorías.
    Trabaja sobre la máscara 2D, que sí es manejable en RAM.
    """
    mask_arr = np.asarray(category_mask.values)

    condition = np.zeros(mask_arr.shape, dtype=bool)
    for value in category_values:
        condition |= (mask_arr == value)

    valid_idx = np.flatnonzero(condition.ravel())
    if valid_idx.size == 0:
        return valid_idx

    rng = np.random.default_rng(seed)
    n = min(max_pixels, valid_idx.size)
    return rng.choice(valid_idx, size=n, replace=False)


def _category_pair_correlation(
    da1: xr.DataArray,
    da2: xr.DataArray,
    category_mask: xr.DataArray,
    category_values: list[int] | tuple[int, ...],
    seed: int = DEFAULT_RANDOM_SEED,
    max_pixels: int = DEFAULT_CATEGORY_MAX_PIXELS,
) -> float:
    """
    Correlación entre dos variables condicionada por una máscara categórica.
    Muestrea píxeles espaciales y usa sus series temporales completas.
    """
    pixel_idx = _sample_category_pixel_indices(
        category_mask=category_mask,
        category_values=category_values,
        max_pixels=max_pixels,
        seed=seed,
    )

    if pixel_idx.size == 0:
        return np.nan

    n_lon = da1.sizes["longitude"]

    lat_idx = pixel_idx // n_lon
    lon_idx = pixel_idx % n_lon

    a = _to_float32_view(
        da1.isel(
            latitude=xr.DataArray(lat_idx, dims="points"),
            longitude=xr.DataArray(lon_idx, dims="points"),
        ).values
    )
    b = _to_float32_view(
        da2.isel(
            latitude=xr.DataArray(lat_idx, dims="points"),
            longitude=xr.DataArray(lon_idx, dims="points"),
        ).values
    )

    a = a.ravel()
    b = b.ravel()

    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 2:
        return np.nan

    return _corr_from_vectors(a[valid], b[valid])


def _require_monthly_time(da: xr.DataArray, function_name: str) -> None:
    """
    Valida que un DataArray tenga resolución mensual.
    """
    times = pd.to_datetime(da["time"].values)
    if len(times) < 2:
        raise ValueError(f"{function_name} requiere al menos 2 timestamps.")

    month_steps = np.diff(times.values.astype("datetime64[M]")).astype(int)

    if not np.all(month_steps == 1):
        raise ValueError(
            f"{function_name} solo tiene sentido para series mensuales regulares. "
            "El DataArray recibido no parece mensual."
        )


def _category_pair_correlation_by_month(
    da1: xr.DataArray,
    da2: xr.DataArray,
    category_mask: xr.DataArray,
    category_values: list[int] | tuple[int, ...],
    month: int,
    seed: int = DEFAULT_RANDOM_SEED,
    max_pixels: int = DEFAULT_CATEGORY_MAX_PIXELS,
) -> float:
    """
    Igual que _category_pair_correlation, pero restringiendo a un mes del año.
    Solo tiene sentido en resolución mensual.
    """
    _require_monthly_time(da1, "_category_pair_correlation_by_month")
    _require_monthly_time(da2, "_category_pair_correlation_by_month")

    da1_m = da1.where(da1["time"].dt.month == month, drop=True)
    da2_m = da2.where(da2["time"].dt.month == month, drop=True)

    if da1_m.sizes["time"] == 0 or da2_m.sizes["time"] == 0:
        return np.nan

    return _category_pair_correlation(
        da1=da1_m,
        da2=da2_m,
        category_mask=category_mask,
        category_values=category_values,
        seed=seed + month,
        max_pixels=max_pixels,
    )


def _category_pair_correlation_with_lag(
    da1: xr.DataArray,
    da2: xr.DataArray,
    category_mask: xr.DataArray,
    category_values: list[int] | tuple[int, ...],
    lag_steps: int,
    seed: int = DEFAULT_RANDOM_SEED,
    max_pixels: int = DEFAULT_CATEGORY_MAX_PIXELS,
) -> float:
    """
    Corr(target(t), predictor(t-lag_steps)) condicionada por categoría.

    lag_steps se interpreta como:
    - meses si la serie es mensual
    - años si la serie es anual
    """
    if lag_steps < 0:
        raise ValueError("lag_steps debe ser >= 0")

    da2_lagged = da2.shift(time=lag_steps)

    return _category_pair_correlation(
        da1=da1,
        da2=da2_lagged,
        category_mask=category_mask,
        category_values=category_values,
        seed=seed + lag_steps,
        max_pixels=max_pixels,
    )


# Carga de datos

def load_metadata(metadata_path: str | Path) -> dict:
    """Carga el metadata.json del preprocesado."""
    metadata_path = Path(metadata_path)
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _get_variables_block(metadata: dict) -> dict:
    """
    Devuelve el bloque de variables del metadata.

    Soporta:
    - nuevo esquema: metadata["variables"]
    - fallback simple: metadata ya es un dict de variables
    """
    if "variables" in metadata and isinstance(metadata["variables"], dict):
        return metadata["variables"]

    return metadata


def infer_temporal_resolution_from_metadata(
    metadata: dict,
    reference_variable: str = "LAI",
) -> str:
    """
    Infere la resolución temporal a partir del metadata.
    """
    variables_block = _get_variables_block(metadata)

    if reference_variable not in variables_block:
        raise KeyError(
            f"La variable de referencia '{reference_variable}' no está en metadata. "
            f"Variables disponibles: {list(variables_block.keys())}"
        )

    ref = variables_block[reference_variable]
    return ref["processing"]["temporal_resolution"].lower()


def build_coordinates_from_metadata(
    metadata: dict,
    reference_variable: str = "LAI",
) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
    """
    Reconstruye coordenadas time, latitude y longitude a partir del metadata
    generado por io.py.

    Compatible con datasets mensuales o anuales.
    """
    variables_block = _get_variables_block(metadata)

    if reference_variable not in variables_block:
        raise KeyError(
            f"La variable de referencia '{reference_variable}' no está en metadata. "
            f"Variables disponibles: {list(variables_block.keys())}"
        )

    ref = variables_block[reference_variable]

    if "final_shape" not in ref:
        raise KeyError(
            f"metadata para '{reference_variable}' no contiene 'final_shape'."
        )

    n_time, n_lat, n_lon = ref["final_shape"]

    time_min = ref.get("final_time_min")
    time_max = ref.get("final_time_max")
    temporal_resolution = ref["processing"]["temporal_resolution"].lower()

    load_meta = ref.get("load_metadata", {})
    lat_min = load_meta.get("lat_min")
    lat_max = load_meta.get("lat_max")
    lon_min = load_meta.get("lon_min")
    lon_max = load_meta.get("lon_max")

    if any(v is None for v in [time_min, time_max, lat_min, lat_max, lon_min, lon_max]):
        raise ValueError(
            f"Metadata incompleto para '{reference_variable}'. "
            "Faltan límites temporales o espaciales."
        )

    if temporal_resolution == "monthly":
        time = pd.date_range(time_min, time_max, freq="MS")
    elif temporal_resolution == "annual":
        time = pd.date_range(time_min, time_max, freq="YS")
    else:
        raise ValueError(
            f"temporal_resolution desconocida: {temporal_resolution}"
        )

    latitude = np.linspace(lat_min, lat_max, n_lat, dtype=np.float32)
    longitude = np.linspace(lon_min, lon_max, n_lon, dtype=np.float32)

    if len(time) != n_time:
        raise ValueError(
            f"Número de fechas reconstruidas ({len(time)}) distinto de n_time ({n_time}). "
            f"temporal_resolution={temporal_resolution}, "
            f"time_min={time_min}, time_max={time_max}"
        )

    return time, latitude, longitude


def build_dataarray_from_npy(
    npy_path: str | Path,
    time: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    name: str,
) -> xr.DataArray:
    """
    Carga un .npy y lo reconstruye como DataArray.

    Se usa mmap_mode='r' para evitar copiar el array entero en RAM al cargarlo.
    """
    arr = np.load(npy_path, mmap_mode="r")

    expected_shape = (len(time), len(latitude), len(longitude))
    if arr.shape != expected_shape:
        raise ValueError(
            f"Shape inesperada para {name}: {arr.shape}. Esperada: {expected_shape}."
        )

    return xr.DataArray(
        arr,
        coords={
            "time": time,
            "latitude": latitude,
            "longitude": longitude,
        },
        dims=("time", "latitude", "longitude"),
        name=name,
    )


def load_processed_dataset(
    input_dir: str | Path,
    variable_names: list[str],
    reference_variable: str = "LAI",
) -> tuple[dict[str, xr.DataArray], dict]:
    """
    Carga metadata + variables .npy y devuelve un diccionario de DataArrays.
    """
    input_dir = Path(input_dir)
    metadata = load_metadata(input_dir / "metadata.json")

    variable_names = [v.upper() for v in variable_names]
    reference_variable = reference_variable.upper()

    time, latitude, longitude = build_coordinates_from_metadata(
        metadata,
        reference_variable=reference_variable,
    )

    data_dict: dict[str, xr.DataArray] = {}
    for name in variable_names:
        npy_path = input_dir / f"{name}.npy"
        if not npy_path.exists():
            raise FileNotFoundError(f"No existe el archivo: {npy_path}")

        data_dict[name] = build_dataarray_from_npy(
            npy_path=npy_path,
            time=time,
            latitude=latitude,
            longitude=longitude,
            name=name,
        )

    return data_dict, metadata


# EDA estructural y univariante

def dataset_overview(data_dict: dict[str, xr.DataArray]) -> pd.DataFrame:
    """
    Resumen estructural por variable: shape, dims, rango temporal/espacial y NaN.
    """
    rows = []

    for name, da in data_dict.items():
        n_total = da.size
        n_nan = int(da.isnull().sum().item())
        pct_nan = 100 * n_nan / n_total if n_total > 0 else np.nan

        rows.append(
            {
                "variable": name,
                "shape": tuple(da.shape),
                "dims": tuple(da.dims),
                "time_min": str(pd.to_datetime(da.time.min().item()).date()),
                "time_max": str(pd.to_datetime(da.time.max().item()).date()),
                "lat_min": float(da.latitude.min().item()),
                "lat_max": float(da.latitude.max().item()),
                "lon_min": float(da.longitude.min().item()),
                "lon_max": float(da.longitude.max().item()),
                "n_total": int(n_total),
                "n_nan": n_nan,
                "pct_nan": pct_nan,
            }
        )

    return pd.DataFrame(rows)


def univariate_stats(da: xr.DataArray, name: str | None = None) -> dict:
    """
    Estadísticos descriptivos básicos.

    min/max/mean/std son exactos.
    cuantiles se estiman por muestreo para no disparar RAM/CPU.
    """
    var_name = name or da.name or "variable"

    n_total = da.size
    n_nan = int(da.isnull().sum().item())
    pct_nan = 100 * n_nan / n_total if n_total > 0 else np.nan

    if n_nan == n_total:
        return {
            "variable": var_name,
            "min": np.nan,
            "max": np.nan,
            "mean": np.nan,
            "std": np.nan,
            "p01": np.nan,
            "p05": np.nan,
            "p25": np.nan,
            "p50": np.nan,
            "p75": np.nan,
            "p95": np.nan,
            "p99": np.nan,
            "n_nan": n_nan,
            "pct_nan": pct_nan,
        }

    sampled = _sample_valid_values_1d(
        arr=da.values,
        max_points=DEFAULT_STATS_MAX_POINTS,
        seed=DEFAULT_RANDOM_SEED,
    )

    if sampled.size == 0:
        q01 = q05 = q25 = q50 = q75 = q95 = q99 = np.nan
    else:
        q01, q05, q25, q50, q75, q95, q99 = np.quantile(
            sampled, [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
        )

    return {
        "variable": var_name,
        "min": float(da.min(skipna=True).item()),
        "max": float(da.max(skipna=True).item()),
        "mean": float(da.mean(skipna=True).item()),
        "std": float(da.std(skipna=True).item()),
        "p01": float(q01) if np.isfinite(q01) else np.nan,
        "p05": float(q05) if np.isfinite(q05) else np.nan,
        "p25": float(q25) if np.isfinite(q25) else np.nan,
        "p50": float(q50) if np.isfinite(q50) else np.nan,
        "p75": float(q75) if np.isfinite(q75) else np.nan,
        "p95": float(q95) if np.isfinite(q95) else np.nan,
        "p99": float(q99) if np.isfinite(q99) else np.nan,
        "n_nan": n_nan,
        "pct_nan": pct_nan,
    }


def univariate_stats_df(data_dict: dict[str, xr.DataArray]) -> pd.DataFrame:
    """DataFrame de estadísticos descriptivos por variable."""
    rows = [univariate_stats(da, name) for name, da in data_dict.items()]
    return pd.DataFrame(rows)


# EDA espacial y temporal

def temporal_mean_map(da: xr.DataArray) -> xr.DataArray:
    """Mapa de media temporal."""
    return da.mean(dim="time", skipna=True)


def temporal_std_map(da: xr.DataArray) -> xr.DataArray:
    """Mapa de desviación estándar temporal."""
    return da.std(dim="time", skipna=True)


def spatial_mean_timeseries(da: xr.DataArray) -> xr.DataArray:
    """Serie temporal de la media espacial."""
    return da.mean(dim=("latitude", "longitude"), skipna=True)


def monthly_climatology(da: xr.DataArray) -> xr.DataArray:
    """Climatología mensual de la media espacial. Solo mensual."""
    _require_monthly_time(da, "monthly_climatology")
    ts = spatial_mean_timeseries(da)
    return ts.groupby("time.month").mean(dim="time", skipna=True)


def monthly_anomalies(da: xr.DataArray) -> xr.DataArray:
    """Anomalías mensuales de la media espacial. Solo mensual."""
    _require_monthly_time(da, "monthly_anomalies")
    ts = spatial_mean_timeseries(da)
    clim = ts.groupby("time.month").mean(dim="time", skipna=True)
    return ts.groupby("time.month") - clim


# EDA multivariante

def _paired_valid_values(da1: xr.DataArray, da2: xr.DataArray) -> tuple[np.ndarray, np.ndarray]:
    """
    Extrae pares válidos aplanados para correlación.

    En esta versión se devuelve una muestra de pares válidos, no todos los puntos,
    para evitar reventar RAM.
    """
    return _sample_valid_pairs_flat(
        da1=da1,
        da2=da2,
        max_points=DEFAULT_CORR_MAX_POINTS,
        seed=DEFAULT_RANDOM_SEED,
    )


def global_flattened_correlation(da1: xr.DataArray, da2: xr.DataArray) -> float:
    """
    Correlación global aplanando tiempo y espacio.

    Versión aproximada por muestreo aleatorio de pares válidos.
    Mucho más ligera y suficiente para EDA.
    """
    a, b = _paired_valid_values(da1, da2)
    return _corr_from_vectors(a, b)


def target_predictor_correlation_table(
    target: xr.DataArray,
    predictors: dict[str, xr.DataArray],
) -> pd.DataFrame:
    """Tabla de correlación global target-predictors."""
    rows = []
    for name, da in predictors.items():
        rows.append(
            {
                "predictor": name,
                "correlation_with_target": global_flattened_correlation(target, da),
            }
        )

    return pd.DataFrame(rows).sort_values(
        "correlation_with_target", ascending=False
    ).reset_index(drop=True)


def global_correlation_matrix_sampled(
    data_dict: dict[str, xr.DataArray],
    max_points: int = DEFAULT_CORR_MAX_POINTS,
    seed: int = DEFAULT_RANDOM_SEED,
) -> pd.DataFrame:
    """
    Matriz de correlación global aproximada usando muestreo aleatorio.
    """
    var_names = list(data_dict.keys())
    n = len(var_names)
    corr = np.full((n, n), np.nan, dtype=np.float32)

    for i, name_i in enumerate(var_names):
        for j, name_j in enumerate(var_names):
            if j < i:
                corr[i, j] = corr[j, i]
                continue

            a, b = _sample_valid_pairs_flat(
                da1=data_dict[name_i],
                da2=data_dict[name_j],
                max_points=max_points,
                seed=seed + i * 100 + j,
            )
            value = _corr_from_vectors(a, b)

            corr[i, j] = value
            corr[j, i] = value

    return pd.DataFrame(corr, index=var_names, columns=var_names)


def global_correlation_matrix(
    data_dict: dict[str, xr.DataArray],
) -> pd.DataFrame:
    """
    Alias de compatibilidad.
    """
    return global_correlation_matrix_sampled(data_dict)


def pixelwise_correlation(da1: xr.DataArray, da2: xr.DataArray) -> xr.DataArray:
    """
    Correlación temporal por píxel.
    """
    return xr.corr(da1.astype(np.float32), da2.astype(np.float32), dim="time")


def sample_valid_pixels(
    da: xr.DataArray,
    n_pixels: int = 10,
    seed: int = DEFAULT_RANDOM_SEED,
) -> list[tuple[int, int]]:
    """
    Selecciona píxeles con al menos un dato válido en el tiempo.
    Devuelve índices (lat_idx, lon_idx).
    """
    valid_mask = da.notnull().any(dim="time").values
    valid_positions = np.argwhere(valid_mask)

    if len(valid_positions) == 0:
        return []

    rng = np.random.default_rng(seed)
    n_select = min(n_pixels, len(valid_positions))
    idx = rng.choice(len(valid_positions), size=n_select, replace=False)
    return [tuple(valid_positions[i]) for i in idx]


def pixel_series_correlation(
    da1: xr.DataArray,
    da2: xr.DataArray,
    pixel_idx: tuple[int, int],
) -> float:
    """Correlación temporal entre dos variables en un píxel dado."""
    i, j = pixel_idx
    s1 = _to_float32_view(da1.isel(latitude=i, longitude=j).values)
    s2 = _to_float32_view(da2.isel(latitude=i, longitude=j).values)

    mask = np.isfinite(s1) & np.isfinite(s2)
    if mask.sum() < 2:
        return np.nan

    return _corr_from_vectors(s1[mask], s2[mask])


def sample_pixel_correlation_table(
    target: xr.DataArray,
    predictors: dict[str, xr.DataArray],
    n_pixels: int = 10,
    seed: int = DEFAULT_RANDOM_SEED,
) -> pd.DataFrame:
    """
    Tabla simple de correlaciones temporales en píxeles aleatorios.
    """
    rows = []
    sampled_pixels = sample_valid_pixels(target, n_pixels=n_pixels, seed=seed)

    for i, j in sampled_pixels:
        row = {
            "lat_idx": int(i),
            "lon_idx": int(j),
            "latitude": float(target.latitude.values[i]),
            "longitude": float(target.longitude.values[j]),
        }

        for name, da in predictors.items():
            row[f"corr_{target.name}_{name}"] = pixel_series_correlation(target, da, (i, j))

        rows.append(row)

    return pd.DataFrame(rows)


def sample_scatter_dataframe(
    da_x: xr.DataArray,
    da_y: xr.DataArray,
    x_name: str,
    y_name: str,
    max_points: int = 5000,
    seed: int = DEFAULT_RANDOM_SEED,
) -> pd.DataFrame:
    """
    DataFrame muestreado para scatter/hexbin sin cargar todos los puntos válidos.
    """
    x, y = _sample_valid_pairs_flat(
        da1=da_x,
        da2=da_y,
        max_points=max_points,
        seed=seed,
    )

    if x.size == 0:
        return pd.DataFrame(columns=[x_name, y_name])

    return pd.DataFrame({x_name: x, y_name: y})


# EDA temporal target vs predictores

def monthly_target_predictor_correlation(
    target: xr.DataArray,
    predictor: xr.DataArray,
) -> dict[int, float]:
    """
    Correlación global target-predictor separada por mes calendario.
    Solo mensual.
    """
    _require_monthly_time(target, "monthly_target_predictor_correlation")
    _require_monthly_time(predictor, "monthly_target_predictor_correlation")

    results: dict[int, float] = {}

    for month in range(1, 13):
        target_m = target.where(target["time"].dt.month == month, drop=True)
        pred_m = predictor.where(predictor["time"].dt.month == month, drop=True)
        results[month] = global_flattened_correlation(target_m, pred_m)

    return results


def monthly_correlation_table(
    target: xr.DataArray,
    predictors: dict[str, xr.DataArray],
) -> pd.DataFrame:
    """Tabla mensual de correlación entre target y predictores. Solo mensual."""
    _require_monthly_time(target, "monthly_correlation_table")

    rows = []
    for name, da in predictors.items():
        row = {"predictor": name}
        row.update(monthly_target_predictor_correlation(target, da))
        rows.append(row)

    df = pd.DataFrame(rows)
    return df[["predictor", *list(range(1, 13))]]


def lagged_global_correlation(
    target: xr.DataArray,
    predictor: xr.DataArray,
    lag_steps: int,
) -> float:
    """
    Correlación global entre target(t) y predictor(t - lag_steps).

    lag_steps se interpreta como:
    - meses si la resolución es mensual
    - años si la resolución es anual
    """
    if lag_steps < 0:
        raise ValueError("lag_steps debe ser >= 0")

    predictor_lagged = predictor.shift(time=lag_steps)
    return global_flattened_correlation(target, predictor_lagged)


def lagged_correlation_table(
    target: xr.DataArray,
    predictors: dict[str, xr.DataArray],
    lags: list[int] | tuple[int, ...] = (0, 1, 2, 3),
) -> pd.DataFrame:
    """
    Tabla de correlaciones globales con distintos retardos temporales.
    Interpreta lag_k como corr(target(t), predictor(t-k)).
    """
    rows = []

    for name, da in predictors.items():
        row = {"predictor": name}
        for lag in lags:
            row[f"lag_{lag}"] = lagged_global_correlation(target, da, lag_steps=lag)
        rows.append(row)

    return pd.DataFrame(rows)


def best_lag_table(
    target: xr.DataArray,
    predictors: dict[str, xr.DataArray],
    lags: list[int] | tuple[int, ...] = (0, 1, 2, 3),
) -> pd.DataFrame:
    """
    Resume el mejor lag por predictor según la correlación absoluta.
    """
    lag_df = lagged_correlation_table(target, predictors, lags=lags)
    lag_cols = [c for c in lag_df.columns if c.startswith("lag_")]

    rows = []
    for _, row in lag_df.iterrows():
        vals = row[lag_cols].astype(float)
        best_col = vals.abs().idxmax()
        rows.append(
            {
                "predictor": row["predictor"],
                "best_lag": best_col,
                "best_correlation": float(vals[best_col]),
                "best_abs_correlation": float(np.abs(vals[best_col])),
            }
        )

    return pd.DataFrame(rows).sort_values(
        "best_abs_correlation", ascending=False
    ).reset_index(drop=True)


# Utilidades mínimas para plotting externo

def subplot_grid(n_items: int, ncols: int = 2) -> tuple[int, int]:
    """Devuelve (nrows, ncols) para una rejilla simple de subplots."""
    nrows = int(np.ceil(n_items / ncols))
    return nrows, ncols


# GeoTIFF / categorías

def load_tiff_as_dataarray(
    tiff_path: str | Path,
    latitude: np.ndarray,
    longitude: np.ndarray,
    name: str,
) -> xr.DataArray:
    """
    Carga un GeoTIFF 2D alineado con el grid del dataset principal
    y lo convierte en DataArray con coords latitude/longitude.
    """
    tiff_path = Path(tiff_path)

    with rasterio.open(tiff_path) as src:
        arr = src.read(1)

    expected_shape = (len(latitude), len(longitude))
    if arr.shape != expected_shape:
        raise ValueError(
            f"Shape del TIFF {arr.shape} no coincide con el grid esperado {expected_shape}"
        )

    arr = np.flipud(arr)

    return xr.DataArray(
        arr,
        coords={
            "latitude": latitude,
            "longitude": longitude,
        },
        dims=("latitude", "longitude"),
        name=name,
    )



def load_npy_mask_as_dataarray(
    npy_path: str | Path,
    latitude: np.ndarray,
    longitude: np.ndarray,
    name: str,
) -> xr.DataArray:
    """
    Carga una máscara 2D desde .npy ya guardada en convención del pipeline:
    latitude ascendente (-90 -> 90), longitude ascendente (-180 -> 180).
    """
    npy_path = Path(npy_path)
    arr = np.load(npy_path)

    expected_shape = (len(latitude), len(longitude))
    if arr.shape != expected_shape:
        raise ValueError(
            f"Shape del NPY {arr.shape} no coincide con el grid esperado {expected_shape}"
        )

    return xr.DataArray(
        arr,
        coords={
            "latitude": latitude,
            "longitude": longitude,
        },
        dims=("latitude", "longitude"),
        name=name,
    )

def apply_categorical_mask(
    da: xr.DataArray,
    category_mask: xr.DataArray,
    category_values: list[int] | tuple[int, ...],
) -> xr.DataArray:
    """
    Aplica una máscara categórica 2D a un DataArray 3D.
    """
    if set(category_mask.dims) != {"latitude", "longitude"}:
        raise ValueError("category_mask debe tener dims ('latitude', 'longitude')")

    condition = xr.zeros_like(category_mask, dtype=bool)
    for value in category_values:
        condition = condition | (category_mask == value)

    return da.where(condition)


def subset_data_dict_by_category(
    data_dict: dict[str, xr.DataArray],
    category_mask: xr.DataArray,
    category_values: list[int] | tuple[int, ...],
) -> dict[str, xr.DataArray]:
    """
    Devuelve un diccionario de variables enmascaradas para una o varias categorías.
    """
    return {
        name: apply_categorical_mask(da, category_mask, category_values)
        for name, da in data_dict.items()
    }


def valid_pixel_count(masked_da: xr.DataArray) -> int:
    """
    Número de píxeles espaciales con al menos un valor válido en el tiempo.
    """
    valid = masked_da.notnull().any(dim="time")
    return int(valid.sum().item())


def category_summary_table(
    data_dict: dict[str, xr.DataArray],
    category_mask: xr.DataArray,
    labels_map: dict[int, str],
    reference_variable: str = "LAI",
    ignore_codes: list[int] | tuple[int, ...] = (0,),
) -> pd.DataFrame:
    """
    Resumen rápido por categoría usando una variable de referencia,
    evitando crear subsets 3D completos.
    """
    da = data_dict[reference_variable]

    if set(category_mask.dims) != {"latitude", "longitude"}:
        raise ValueError("category_mask debe tener dims ('latitude', 'longitude')")

    n_time = da.sizes["time"]
    valid_count_per_pixel = da.notnull().sum(dim="time")

    rows = []

    for code, label in labels_map.items():
        if code in ignore_codes:
            continue

        cat_pixels = category_mask == code

        n_pixels = int(cat_pixels.sum().item())
        if n_pixels == 0:
            rows.append(
                {
                    "code": code,
                    "label": label,
                    "n_pixels": 0,
                    "n_valid_pixels": 0,
                    "n_total_values": 0,
                    "n_nan": 0,
                    "pct_nan": np.nan,
                }
            )
            continue

        n_valid_pixels = int(((valid_count_per_pixel > 0) & cat_pixels).sum().item())
        n_total_values = int(n_pixels * n_time)
        n_valid_values = int(valid_count_per_pixel.where(cat_pixels, 0).sum().item())

        n_nan = int(n_total_values - n_valid_values)
        pct_nan = 100 * n_nan / n_total_values if n_total_values > 0 else np.nan

        rows.append(
            {
                "code": code,
                "label": label,
                "n_pixels": n_pixels,
                "n_valid_pixels": n_valid_pixels,
                "n_total_values": n_total_values,
                "n_nan": n_nan,
                "pct_nan": pct_nan,
            }
        )

    return pd.DataFrame(rows).sort_values("n_pixels", ascending=False).reset_index(drop=True)


def category_target_predictor_correlation_table(
    data_dict: dict[str, xr.DataArray],
    category_mask: xr.DataArray,
    labels_map: dict[int, str],
    target_name: str,
    predictor_names: list[str],
    ignore_codes: list[int] | tuple[int, ...] = (0,),
) -> pd.DataFrame:
    """
    Tabla de correlaciones globales target-predictor condicionadas por categoría.
    """
    rows = []

    target = data_dict[target_name]

    for code, label in labels_map.items():
        if code in ignore_codes:
            continue

        for predictor_name in predictor_names:
            predictor = data_dict[predictor_name]

            corr_value = _category_pair_correlation(
                da1=target,
                da2=predictor,
                category_mask=category_mask,
                category_values=[code],
                seed=DEFAULT_RANDOM_SEED + code,
                max_pixels=DEFAULT_CATEGORY_MAX_PIXELS,
            )

            rows.append(
                {
                    "category_code": code,
                    "category_label": label,
                    "predictor": predictor_name,
                    "correlation_with_target": corr_value,
                }
            )

    return pd.DataFrame(rows)


def category_monthly_correlation_tables(
    data_dict: dict[str, xr.DataArray],
    category_mask: xr.DataArray,
    labels_map: dict[int, str],
    target_name: str,
    predictor_names: list[str],
    ignore_codes: list[int] | tuple[int, ...] = (0,),
) -> dict[str, pd.DataFrame]:
    """
    Devuelve una tabla mensual por categoría.
    Solo mensual.
    """
    target = data_dict[target_name]
    _require_monthly_time(target, "category_monthly_correlation_tables")

    out = {}

    for code, label in labels_map.items():
        if code in ignore_codes:
            continue

        rows = []
        for predictor_name in predictor_names:
            predictor = data_dict[predictor_name]
            row = {"predictor": predictor_name}

            for month in range(1, 13):
                row[month] = _category_pair_correlation_by_month(
                    da1=target,
                    da2=predictor,
                    category_mask=category_mask,
                    category_values=[code],
                    month=month,
                    seed=DEFAULT_RANDOM_SEED + code * 100 + month,
                    max_pixels=DEFAULT_CATEGORY_MAX_PIXELS,
                )

            rows.append(row)

        out[label] = pd.DataFrame(rows)[["predictor", *list(range(1, 13))]]

    return out


def category_lagged_correlation_tables(
    data_dict: dict[str, xr.DataArray],
    category_mask: xr.DataArray,
    labels_map: dict[int, str],
    target_name: str,
    predictor_names: list[str],
    lags: list[int] = [0, 1, 2, 3],
    ignore_codes: list[int] | tuple[int, ...] = (0,),
) -> dict[str, pd.DataFrame]:
    """
    Devuelve una tabla de correlaciones con lag por categoría.

    lag se interpreta como:
    - meses para series mensuales
    - años para series anuales
    """
    out = {}
    target = data_dict[target_name]

    for code, label in labels_map.items():
        if code in ignore_codes:
            continue

        rows = []
        for predictor_name in predictor_names:
            predictor = data_dict[predictor_name]
            row = {"predictor": predictor_name}

            for lag in lags:
                row[f"lag_{lag}"] = _category_pair_correlation_with_lag(
                    da1=target,
                    da2=predictor,
                    category_mask=category_mask,
                    category_values=[code],
                    lag_steps=lag,
                    seed=DEFAULT_RANDOM_SEED + code * 100 + lag,
                    max_pixels=DEFAULT_CATEGORY_MAX_PIXELS,
                )

            rows.append(row)

        out[label] = pd.DataFrame(rows)

    return out


# Predictor dominante: mapas globales y categóricos

def _pixelwise_correlation_block(
    target_block: np.ndarray,
    predictor_block: np.ndarray,
) -> np.ndarray:
    """
    Correlación temporal por píxel para un bloque espacial.
    """
    target_block = target_block.astype(np.float32, copy=False)
    predictor_block = predictor_block.astype(np.float32, copy=False)

    valid = np.isfinite(target_block) & np.isfinite(predictor_block)
    n_valid = valid.sum(axis=0).astype(np.int32)

    a = np.where(valid, target_block, 0.0)
    b = np.where(valid, predictor_block, 0.0)

    sum_a = a.sum(axis=0)
    sum_b = b.sum(axis=0)
    sum_ab = (a * b).sum(axis=0)
    sum_a2 = (a * a).sum(axis=0)
    sum_b2 = (b * b).sum(axis=0)

    out = np.full(sum_a.shape, np.nan, dtype=np.float32)

    enough = n_valid >= 2
    if not np.any(enough):
        return out

    n = n_valid[enough].astype(np.float32)

    num = sum_ab[enough] - (sum_a[enough] * sum_b[enough] / n)
    den_a = sum_a2[enough] - (sum_a[enough] ** 2 / n)
    den_b = sum_b2[enough] - (sum_b[enough] ** 2 / n)
    den = np.sqrt(den_a * den_b)

    valid_den = den > 0
    tmp = np.full(n.shape, np.nan, dtype=np.float32)
    tmp[valid_den] = num[valid_den] / den[valid_den]

    out[enough] = tmp
    return out


def dominant_predictor_map(
    target: xr.DataArray,
    predictors: dict[str, xr.DataArray],
    lat_block_size: int = 30,
    lon_block_size: int = 60,
) -> xr.DataArray:
    """
    Mapa global del predictor dominante por píxel.
    """
    predictor_names = list(predictors.keys())

    n_lat = target.sizes["latitude"]
    n_lon = target.sizes["longitude"]

    dominant_code = np.zeros((n_lat, n_lon), dtype=np.int16)
    best_abs_corr = np.full((n_lat, n_lon), -np.inf, dtype=np.float32)

    for pred_idx, pred_name in enumerate(predictor_names, start=1):
        predictor = predictors[pred_name]

        for lat_start in range(0, n_lat, lat_block_size):
            lat_end = min(lat_start + lat_block_size, n_lat)

            for lon_start in range(0, n_lon, lon_block_size):
                lon_end = min(lon_start + lon_block_size, n_lon)

                target_block = target.isel(
                    latitude=slice(lat_start, lat_end),
                    longitude=slice(lon_start, lon_end),
                ).values

                predictor_block = predictor.isel(
                    latitude=slice(lat_start, lat_end),
                    longitude=slice(lon_start, lon_end),
                ).values

                corr_block = _pixelwise_correlation_block(
                    target_block=target_block,
                    predictor_block=predictor_block,
                )

                abs_corr_block = np.abs(corr_block)
                current_best = best_abs_corr[lat_start:lat_end, lon_start:lon_end]

                better = np.isfinite(abs_corr_block) & (abs_corr_block > current_best)

                current_best[better] = abs_corr_block[better]
                best_abs_corr[lat_start:lat_end, lon_start:lon_end] = current_best

                dominant_sub = dominant_code[lat_start:lat_end, lon_start:lon_end]
                dominant_sub[better] = pred_idx
                dominant_code[lat_start:lat_end, lon_start:lon_end] = dominant_sub

    dominant_code[~np.isfinite(best_abs_corr)] = 0

    return xr.DataArray(
        dominant_code,
        coords={
            "latitude": target.latitude,
            "longitude": target.longitude,
        },
        dims=("latitude", "longitude"),
        name="dominant_predictor_map",
    )


def dominant_predictor_table_by_category(
    data_dict: dict[str, xr.DataArray],
    category_mask: xr.DataArray,
    labels_map: dict[int, str],
    target_name: str,
    predictor_names: list[str],
    ignore_codes: list[int] | tuple[int, ...] = (0,),
) -> pd.DataFrame:
    """
    Para cada categoría identifica el predictor con mayor correlación absoluta.
    """
    corr_df = category_target_predictor_correlation_table(
        data_dict=data_dict,
        category_mask=category_mask,
        labels_map=labels_map,
        target_name=target_name,
        predictor_names=predictor_names,
        ignore_codes=ignore_codes,
    ).copy()

    if corr_df.empty:
        return pd.DataFrame(
            columns=[
                "category_code",
                "category_label",
                "dominant_predictor",
                "correlation_with_target",
                "abs_correlation_with_target",
            ]
        )

    corr_df["abs_correlation_with_target"] = corr_df["correlation_with_target"].abs()

    rows = []
    for label in corr_df["category_label"].unique():
        sub = corr_df[corr_df["category_label"] == label].copy()

        sub_valid = sub[np.isfinite(sub["abs_correlation_with_target"])]
        if len(sub_valid) == 0:
            first_row = sub.iloc[0]
            rows.append(
                {
                    "category_code": int(first_row["category_code"]),
                    "category_label": first_row["category_label"],
                    "dominant_predictor": np.nan,
                    "correlation_with_target": np.nan,
                    "abs_correlation_with_target": np.nan,
                }
            )
            continue

        best_idx = sub_valid["abs_correlation_with_target"].idxmax()
        best_row = sub_valid.loc[best_idx]

        rows.append(
            {
                "category_code": int(best_row["category_code"]),
                "category_label": best_row["category_label"],
                "dominant_predictor": best_row["predictor"],
                "correlation_with_target": float(best_row["correlation_with_target"]),
                "abs_correlation_with_target": float(best_row["abs_correlation_with_target"]),
            }
        )

    return pd.DataFrame(rows).sort_values("category_code").reset_index(drop=True)


def categorical_dominant_predictor_map(
    category_mask: xr.DataArray,
    dominant_table: pd.DataFrame,
    predictor_names: list[str],
    nodata_code: int = 0,
) -> xr.DataArray:
    """
    Construye un mapa categórico 2D con el predictor dominante por clase.
    """
    predictor_to_code = {name: i + 1 for i, name in enumerate(predictor_names)}

    out = np.full(category_mask.shape, nodata_code, dtype=np.int16)
    mask_vals = category_mask.values

    for _, row in dominant_table.iterrows():
        class_code = int(row["category_code"])
        predictor_name = row["dominant_predictor"]

        if pd.isna(predictor_name):
            continue

        if predictor_name not in predictor_to_code:
            continue

        out[mask_vals == class_code] = predictor_to_code[predictor_name]

    return xr.DataArray(
        out,
        coords=category_mask.coords,
        dims=category_mask.dims,
        name="categorical_dominant_predictor_map",
    )


def predictor_code_dict(predictor_names: list[str]) -> dict[int, str]:
    """
    Diccionario código -> predictor.
    """
    return {i + 1: name for i, name in enumerate(predictor_names)}


def plot_dominant_predictor_map(
    dominant_map: xr.DataArray,
    predictor_names: list[str],
    title: str,
    figsize: tuple[int, int] = (12, 5),
    cmap_name: str = "tab10",
):
    """
    Representa un mapa categórico de predictor dominante.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm

    n_pred = len(predictor_names)

    base = plt.get_cmap(cmap_name, n_pred)
    colors = ["lightgray"] + [base(i) for i in range(n_pred)]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, n_pred + 1.5, 1), cmap.N)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.pcolormesh(
        dominant_map["longitude"],
        dominant_map["latitude"],
        dominant_map.values,
        cmap=cmap,
        norm=norm,
        shading="auto",
    )

    cbar = plt.colorbar(im, ax=ax, ticks=np.arange(0, n_pred + 1))
    cbar.ax.set_yticklabels(["NoData"] + predictor_names)

    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.tight_layout()
    plt.show()