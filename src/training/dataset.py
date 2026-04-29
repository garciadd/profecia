from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr


def _infer_temporal_resolution(time_values: np.ndarray) -> str:
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
    required_dims = ("time", "latitude", "longitude")
    if tuple(target.dims) != required_dims:
        raise ValueError(f"target debe tener dims {required_dims}. Recibido: {target.dims}")

    for name, da in predictors.items():
        if tuple(da.dims) != required_dims:
            raise ValueError(f"{name} debe tener dims {required_dims}. Recibido: {da.dims}")
        if da.shape != target.shape:
            raise ValueError(f"{name} no está alineado con target. {da.shape} != {target.shape}")

    valid_dims = {("latitude", "longitude"), ("time", "latitude", "longitude")}
    if tuple(train_mask.dims) not in valid_dims or tuple(test_mask.dims) not in valid_dims:
        raise ValueError("train_mask y test_mask deben ser 2D o 3D.")
    if tuple(train_mask.dims) != tuple(test_mask.dims):
        raise ValueError("train_mask y test_mask deben tener las mismas dimensiones.")
    if np.any(train_mask.values & test_mask.values):
        raise ValueError("Train/Test se solapan.")


def _process_block(
    target_block: np.ndarray,
    predictor_blocks: list[np.ndarray],
    mask_block: np.ndarray,
):
    lat_idx, lon_idx = np.where(mask_block)
    if len(lat_idx) == 0:
        return None

    y_2d = target_block[:, lat_idx, lon_idx]
    X_list = [p[:, lat_idx, lon_idx] for p in predictor_blocks]

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


def _process_block_observation_level(
    target_block: np.ndarray,
    predictor_blocks: list[np.ndarray],
    train_mask_block: np.ndarray,
    test_mask_block: np.ndarray,
):
    valid = np.isfinite(target_block)
    for xb in predictor_blocks:
        valid &= np.isfinite(xb)

    train_valid = train_mask_block.astype(bool) & valid
    test_valid = test_mask_block.astype(bool) & valid

    def _extract(mask_3d: np.ndarray):
        if not mask_3d.any():
            return None

        time_idx, lat_idx, lon_idx = np.where(mask_3d)
        y = target_block[time_idx, lat_idx, lon_idx]
        X = np.column_stack([xb[time_idx, lat_idx, lon_idx] for xb in predictor_blocks])

        return (
            X.astype(np.float32),
            y.astype(np.float32),
            lat_idx.astype(np.int32),
            lon_idx.astype(np.int32),
            time_idx.astype(np.int32),
        )

    return _extract(train_valid), _extract(test_valid)


def export_train_test_data(
    target: xr.DataArray,
    predictors: dict[str, xr.DataArray],
    train_mask: xr.DataArray,
    test_mask: xr.DataArray,
    output_dir: str | Path,
    split_metadata: dict[str, Any] | None = None,
    lat_block_size: int = 10,
    extra_masks: dict[str, xr.DataArray] | None = None,
) -> dict[str, Any]:
    _validate_inputs(target, predictors, train_mask, test_mask)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_names = list(predictors.keys())
    X_train_list, y_train_list = [], []
    X_test_list, y_test_list = [], []
    pixel_id_train_list, lat_idx_train_list, lon_idx_train_list, time_idx_train_list = [], [], [], []
    pixel_id_test_list, lat_idx_test_list, lon_idx_test_list, time_idx_test_list = [], [], [], []

    n_lat = target.sizes["latitude"]
    n_lon = target.sizes["longitude"]
    n_time = target.sizes["time"]
    time_values = pd.to_datetime(target["time"].values)
    temporal_resolution = _infer_temporal_resolution(time_values)
    split_mode = "spatial_pixel" if tuple(train_mask.dims) == ("latitude", "longitude") else "random_observation"

    for lat_start in range(0, n_lat, lat_block_size):
        lat_end = min(lat_start + lat_block_size, n_lat)
        target_block = target.isel(latitude=slice(lat_start, lat_end)).values
        predictor_blocks = [predictors[name].isel(latitude=slice(lat_start, lat_end)).values for name in feature_names]

        if split_mode == "spatial_pixel":
            train_block = train_mask.isel(latitude=slice(lat_start, lat_end)).values
            test_block = test_mask.isel(latitude=slice(lat_start, lat_end)).values

            train_out = _process_block(target_block, predictor_blocks, train_block)
            if train_out is not None:
                X, y, lat_idx_local, lon_idx_local, time_idx_local, pixel_pos_idx = train_out
                X_train_list.append(X)
                y_train_list.append(y)

                lat_global_per_pixel = lat_idx_local + lat_start
                lon_global_per_pixel = lon_idx_local
                pixel_id_per_pixel = lat_global_per_pixel * n_lon + lon_global_per_pixel

                pixel_id_train_list.append(pixel_id_per_pixel[pixel_pos_idx].astype(np.int32))
                lat_idx_train_list.append(lat_global_per_pixel[pixel_pos_idx].astype(np.int32))
                lon_idx_train_list.append(lon_global_per_pixel[pixel_pos_idx].astype(np.int32))
                time_idx_train_list.append(time_idx_local.astype(np.int32))

            test_out = _process_block(target_block, predictor_blocks, test_block)
            if test_out is not None:
                X, y, lat_idx_local, lon_idx_local, time_idx_local, pixel_pos_idx = test_out
                X_test_list.append(X)
                y_test_list.append(y)

                lat_global_per_pixel = lat_idx_local + lat_start
                lon_global_per_pixel = lon_idx_local
                pixel_id_per_pixel = lat_global_per_pixel * n_lon + lon_global_per_pixel

                pixel_id_test_list.append(pixel_id_per_pixel[pixel_pos_idx].astype(np.int32))
                lat_idx_test_list.append(lat_global_per_pixel[pixel_pos_idx].astype(np.int32))
                lon_idx_test_list.append(lon_global_per_pixel[pixel_pos_idx].astype(np.int32))
                time_idx_test_list.append(time_idx_local.astype(np.int32))
        else:
            train_block = train_mask.isel(latitude=slice(lat_start, lat_end)).values
            test_block = test_mask.isel(latitude=slice(lat_start, lat_end)).values

            train_out, test_out = _process_block_observation_level(
                target_block=target_block,
                predictor_blocks=predictor_blocks,
                train_mask_block=train_block,
                test_mask_block=test_block,
            )

            if train_out is not None:
                X, y, lat_idx_local, lon_idx_local, time_idx_local = train_out
                X_train_list.append(X)
                y_train_list.append(y)

                lat_idx_global = lat_idx_local + lat_start
                lon_idx_global = lon_idx_local
                pixel_id = lat_idx_global * n_lon + lon_idx_global

                pixel_id_train_list.append(pixel_id.astype(np.int32))
                lat_idx_train_list.append(lat_idx_global.astype(np.int32))
                lon_idx_train_list.append(lon_idx_global.astype(np.int32))
                time_idx_train_list.append(time_idx_local.astype(np.int32))

            if test_out is not None:
                X, y, lat_idx_local, lon_idx_local, time_idx_local = test_out
                X_test_list.append(X)
                y_test_list.append(y)

                lat_idx_global = lat_idx_local + lat_start
                lon_idx_global = lon_idx_local
                pixel_id = lat_idx_global * n_lon + lon_idx_global

                pixel_id_test_list.append(pixel_id.astype(np.int32))
                lat_idx_test_list.append(lat_idx_global.astype(np.int32))
                lon_idx_test_list.append(lon_idx_global.astype(np.int32))
                time_idx_test_list.append(time_idx_local.astype(np.int32))

    n_features = len(feature_names)
    X_train = np.concatenate(X_train_list, axis=0) if X_train_list else np.empty((0, n_features), dtype=np.float32)
    y_train = np.concatenate(y_train_list, axis=0) if y_train_list else np.empty((0,), dtype=np.float32)
    X_test = np.concatenate(X_test_list, axis=0) if X_test_list else np.empty((0, n_features), dtype=np.float32)
    y_test = np.concatenate(y_test_list, axis=0) if y_test_list else np.empty((0,), dtype=np.float32)
    pixel_id_train = np.concatenate(pixel_id_train_list, axis=0) if pixel_id_train_list else np.empty((0,), dtype=np.int32)
    lat_idx_train = np.concatenate(lat_idx_train_list, axis=0) if lat_idx_train_list else np.empty((0,), dtype=np.int32)
    lon_idx_train = np.concatenate(lon_idx_train_list, axis=0) if lon_idx_train_list else np.empty((0,), dtype=np.int32)
    time_idx_train = np.concatenate(time_idx_train_list, axis=0) if time_idx_train_list else np.empty((0,), dtype=np.int32)
    pixel_id_test = np.concatenate(pixel_id_test_list, axis=0) if pixel_id_test_list else np.empty((0,), dtype=np.int32)
    lat_idx_test = np.concatenate(lat_idx_test_list, axis=0) if lat_idx_test_list else np.empty((0,), dtype=np.int32)
    lon_idx_test = np.concatenate(lon_idx_test_list, axis=0) if lon_idx_test_list else np.empty((0,), dtype=np.int32)
    time_idx_test = np.concatenate(time_idx_test_list, axis=0) if time_idx_test_list else np.empty((0,), dtype=np.int32)

    np.save(output_dir / "X_train.npy", X_train)
    np.save(output_dir / "y_train.npy", y_train)
    np.save(output_dir / "X_test.npy", X_test)
    np.save(output_dir / "y_test.npy", y_test)
    np.save(output_dir / "pixel_id_train.npy", pixel_id_train)
    np.save(output_dir / "lat_idx_train.npy", lat_idx_train)
    np.save(output_dir / "lon_idx_train.npy", lon_idx_train)
    np.save(output_dir / "time_idx_train.npy", time_idx_train)
    np.save(output_dir / "pixel_id_test.npy", pixel_id_test)
    np.save(output_dir / "lat_idx_test.npy", lat_idx_test)
    np.save(output_dir / "lon_idx_test.npy", lon_idx_test)
    np.save(output_dir / "time_idx_test.npy", time_idx_test)
    np.save(output_dir / "train_mask.npy", train_mask.values)
    np.save(output_dir / "test_mask.npy", test_mask.values)

    if extra_masks:
        for name, da in extra_masks.items():
            if da is not None:
                np.save(output_dir / f"{name}.npy", da.values)

    metadata = {
        "split_mode": split_mode,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "n_unique_train_pixels": int(np.unique(pixel_id_train).size) if len(pixel_id_train) > 0 else 0,
        "n_unique_test_pixels": int(np.unique(pixel_id_test).size) if len(pixel_id_test) > 0 else 0,
        "n_features": int(n_features),
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
        "split_metadata": split_metadata or {},
    }

    with open(output_dir / "dataset_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    if split_metadata is not None:
        with open(output_dir / "split_metadata.json", "w", encoding="utf-8") as f:
            json.dump(split_metadata, f, ensure_ascii=False, indent=2)

    return metadata
