import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

# VALIDACIÓN

def _infer_temporal_resolution(time_values: np.ndarray) -> str:
    """
    Infiere si la serie temporal parece monthly o annual.
    """
    time_values = pd.to_datetime(time_values)

    if len(time_values) < 2:
        return "unknown"

    month_steps = np.diff(time_values.values.astype("datetime64[M]")).astype(int)

    if np.all(month_steps == 1):
        return "monthly"

    if np.all(month_steps == 12):
        return "annual"

    return "unknown"


def _validate_inputs(
    target: xr.DataArray,
    predictors: dict[str, xr.DataArray],
    train_mask: xr.DataArray,
    test_mask: xr.DataArray,
) -> None:
    """
    Valida alineamiento y consistencia de inputs.
    """
    if tuple(target.dims) != ("time", "latitude", "longitude"):
        raise ValueError(
            f"target debe tener dims ('time', 'latitude', 'longitude'). Recibido: {target.dims}"
        )

    for name, da in predictors.items():
        if tuple(da.dims) != ("time", "latitude", "longitude"):
            raise ValueError(
                f"{name} debe tener dims ('time', 'latitude', 'longitude'). Recibido: {da.dims}"
            )

        if da.shape != target.shape:
            raise ValueError(f"{name} no está alineado con target. {da.shape} != {target.shape}")

        if not np.array_equal(da["time"].values, target["time"].values):
            raise ValueError(f"{name} no comparte las mismas coordenadas time que target.")

        if not np.array_equal(da["latitude"].values, target["latitude"].values):
            raise ValueError(f"{name} no comparte las mismas coordenadas latitude que target.")

        if not np.array_equal(da["longitude"].values, target["longitude"].values):
            raise ValueError(f"{name} no comparte las mismas coordenadas longitude que target.")

    if tuple(train_mask.dims) != ("latitude", "longitude"):
        raise ValueError(f"train_mask debe tener dims ('latitude', 'longitude'). Recibido: {train_mask.dims}")

    if tuple(test_mask.dims) != ("latitude", "longitude"):
        raise ValueError(f"test_mask debe tener dims ('latitude', 'longitude'). Recibido: {test_mask.dims}")

    if train_mask.shape != target.shape[1:]:
        raise ValueError("train_mask no tiene la shape espacial esperada.")

    if test_mask.shape != target.shape[1:]:
        raise ValueError("test_mask no tiene la shape espacial esperada.")

    if not np.array_equal(train_mask["latitude"].values, target["latitude"].values):
        raise ValueError("train_mask no comparte latitude con target.")

    if not np.array_equal(train_mask["longitude"].values, target["longitude"].values):
        raise ValueError("train_mask no comparte longitude con target.")

    if not np.array_equal(test_mask["latitude"].values, target["latitude"].values):
        raise ValueError("test_mask no comparte latitude con target.")

    if not np.array_equal(test_mask["longitude"].values, target["longitude"].values):
        raise ValueError("test_mask no comparte longitude con target.")

    overlap = train_mask.values & test_mask.values
    if overlap.any():
        raise ValueError("Train/Test se solapan.")


# CORE

def _process_block(
    target_block: np.ndarray,
    predictor_blocks: list[np.ndarray],
    mask_block: np.ndarray,
):
    """
    Convierte un bloque espacial a formato tabular.

    Returns
    -------
    tuple:
    - X: (n_samples_valid, n_features)
    - y: (n_samples_valid,)
    - lat_idx: índices locales de latitud de los píxeles del bloque
    - lon_idx: índices locales de longitud de los píxeles del bloque
    - time_idx: índices de tiempo locales (0..n_time-1) por muestra válida
    - pixel_pos_idx: índice del píxel dentro del conjunto de píxeles del bloque
    """
    lat_idx, lon_idx = np.where(mask_block)

    if len(lat_idx) == 0:
        return None

    y_2d = target_block[:, lat_idx, lon_idx]  # (time, n_pixels)

    X_list = []
    for p in predictor_blocks:
        X_list.append(p[:, lat_idx, lon_idx])  # (time, n_pixels)

    valid = np.isfinite(y_2d)
    for xb in X_list:
        valid &= np.isfinite(xb)

    if not valid.any():
        return None

    y = y_2d[valid]
    X = np.column_stack([xb[valid] for xb in X_list])

    time_idx, pixel_pos_idx = np.where(valid)

    return (
        X.astype(np.float32),
        y.astype(np.float32),
        lat_idx.astype(np.int32),
        lon_idx.astype(np.int32),
        time_idx.astype(np.int32),
        pixel_pos_idx.astype(np.int32),
    )


# EXPORT PRINCIPAL

def export_train_test_split(
    target: xr.DataArray,
    predictors: dict[str, xr.DataArray],
    train_mask: xr.DataArray,
    test_mask: xr.DataArray,
    output_dir: str | Path,
    prefix: str,
    lat_block_size: int = 10,
):
    """
    Exporta train/test a .npy, preservando información suficiente para
    reconstruir series temporales en evaluación.

    Guarda:
    - X_train.npy
    - y_train.npy
    - X_test.npy
    - y_test.npy
    - pixel_id_test.npy
    - lat_idx_test.npy
    - lon_idx_test.npy
    - time_idx_test.npy
    - dataset_metadata.json

    Notas
    -----
    - Cada fila de X/y corresponde a una observación (time, pixel).
    - En test se guarda trazabilidad completa por fila.
    """
    _validate_inputs(target, predictors, train_mask, test_mask)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_names = list(predictors.keys())

    # acumuladores
    X_train_list, y_train_list = [], []
    X_test_list, y_test_list = [], []

    pixel_id_list = []
    lat_idx_list = []
    lon_idx_list = []
    time_idx_list = []

    n_lat = target.sizes["latitude"]
    n_lon = target.sizes["longitude"]
    n_time = target.sizes["time"]

    time_values = pd.to_datetime(target["time"].values)
    temporal_resolution = _infer_temporal_resolution(time_values)

    for lat_start in range(0, n_lat, lat_block_size):
        lat_end = min(lat_start + lat_block_size, n_lat)

        target_block = target.isel(latitude=slice(lat_start, lat_end)).values
        predictor_blocks = [
            predictors[name].isel(latitude=slice(lat_start, lat_end)).values
            for name in feature_names
        ]

        train_block = train_mask.isel(latitude=slice(lat_start, lat_end)).values
        test_block = test_mask.isel(latitude=slice(lat_start, lat_end)).values

        # -------- TRAIN --------
        out = _process_block(target_block, predictor_blocks, train_block)

        if out is not None:
            X, y, *_ = out
            X_train_list.append(X)
            y_train_list.append(y)

        # -------- TEST --------
        out = _process_block(target_block, predictor_blocks, test_block)

        if out is not None:
            X, y, lat_idx_local, lon_idx_local, time_idx_local, pixel_pos_idx = out

            X_test_list.append(X)
            y_test_list.append(y)

            lat_global_per_pixel = lat_idx_local + lat_start
            lon_global_per_pixel = lon_idx_local

            pixel_id_per_pixel = lat_global_per_pixel * n_lon + lon_global_per_pixel

            pixel_id_per_sample = pixel_id_per_pixel[pixel_pos_idx]
            lat_idx_per_sample = lat_global_per_pixel[pixel_pos_idx]
            lon_idx_per_sample = lon_global_per_pixel[pixel_pos_idx]
            time_idx_per_sample = time_idx_local

            pixel_id_list.append(pixel_id_per_sample.astype(np.int32))
            lat_idx_list.append(lat_idx_per_sample.astype(np.int32))
            lon_idx_list.append(lon_idx_per_sample.astype(np.int32))
            time_idx_list.append(time_idx_per_sample.astype(np.int32))

    # CONCAT

    if len(X_train_list) == 0:
        X_train = np.empty((0, len(feature_names)), dtype=np.float32)
        y_train = np.empty((0,), dtype=np.float32)
    else:
        X_train = np.concatenate(X_train_list, axis=0)
        y_train = np.concatenate(y_train_list, axis=0)

    if len(X_test_list) == 0:
        X_test = np.empty((0, len(feature_names)), dtype=np.float32)
        y_test = np.empty((0,), dtype=np.float32)
        pixel_id = np.empty((0,), dtype=np.int32)
        lat_idx = np.empty((0,), dtype=np.int32)
        lon_idx = np.empty((0,), dtype=np.int32)
        time_idx = np.empty((0,), dtype=np.int32)
    else:
        X_test = np.concatenate(X_test_list, axis=0)
        y_test = np.concatenate(y_test_list, axis=0)
        pixel_id = np.concatenate(pixel_id_list, axis=0)
        lat_idx = np.concatenate(lat_idx_list, axis=0)
        lon_idx = np.concatenate(lon_idx_list, axis=0)
        time_idx = np.concatenate(time_idx_list, axis=0)

    # validación final
    if not (
        len(X_test) == len(y_test) == len(pixel_id) == len(lat_idx) == len(lon_idx) == len(time_idx)
    ):
        raise RuntimeError("Las longitudes de los arrays de test no coinciden.")

    if not (
        len(X_train) == len(y_train)
    ):
        raise RuntimeError("Las longitudes de X_train e y_train no coinciden.")

    if np.any((time_idx < 0) | (time_idx >= n_time)):
        raise RuntimeError("time_idx_test contiene índices fuera de rango.")

    # SAVE

    np.save(output_dir / "X_train.npy", X_train)
    np.save(output_dir / "y_train.npy", y_train)

    np.save(output_dir / "X_test.npy", X_test)
    np.save(output_dir / "y_test.npy", y_test)

    np.save(output_dir / "pixel_id_test.npy", pixel_id)
    np.save(output_dir / "lat_idx_test.npy", lat_idx)
    np.save(output_dir / "lon_idx_test.npy", lon_idx)
    np.save(output_dir / "time_idx_test.npy", time_idx)

    metadata = {
        "prefix": prefix,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "n_features": int(len(feature_names)),
        "feature_names": feature_names,
        "target": target.name,
        "target_dtype": str(target.dtype),
        "feature_dtypes": {name: str(predictors[name].dtype) for name in feature_names},
        "n_time": int(n_time),
        "time_start": str(pd.to_datetime(time_values[0])),
        "time_end": str(pd.to_datetime(time_values[-1])),
        "time_values": [str(pd.to_datetime(t)) for t in time_values],
        "temporal_resolution_inferred": temporal_resolution,
        "latitude_size": int(target.sizes["latitude"]),
        "longitude_size": int(target.sizes["longitude"]),
    }

    with open(output_dir / "dataset_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    return metadata