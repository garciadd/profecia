from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr


DEFAULT_RANDOM_SEED = 42


def _validate_target_3d(target: xr.DataArray) -> None:
    required_dims = ("time", "latitude", "longitude")
    if tuple(target.dims) != required_dims:
        raise ValueError(f"target debe tener dims {required_dims}. Recibido: {target.dims}")


def _validate_predictors(target: xr.DataArray, predictors: dict[str, xr.DataArray] | None) -> None:
    if predictors is None:
        return

    for name, da in predictors.items():
        if tuple(da.dims) != ("time", "latitude", "longitude"):
            raise ValueError(
                f"{name} debe tener dims ('time', 'latitude', 'longitude'). Recibido: {da.dims}"
            )
        if da.shape != target.shape:
            raise ValueError(f"{name} no está alineado con target. {da.shape} != {target.shape}")
        if not np.array_equal(da["time"].values, target["time"].values):
            raise ValueError(f"{name} no comparte time con target.")
        if not np.array_equal(da["latitude"].values, target["latitude"].values):
            raise ValueError(f"{name} no comparte latitude con target.")
        if not np.array_equal(da["longitude"].values, target["longitude"].values):
            raise ValueError(f"{name} no comparte longitude con target.")


def build_valid_pixel_mask(
    target: xr.DataArray,
    predictors: dict[str, xr.DataArray] | None = None,
    min_valid_fraction: float = 0.0,
) -> xr.DataArray:
    _validate_target_3d(target)
    _validate_predictors(target, predictors)

    if not (0.0 <= min_valid_fraction <= 1.0):
        raise ValueError("min_valid_fraction debe estar entre 0 y 1.")

    valid = np.isfinite(target.values)
    if predictors is not None:
        for da in predictors.values():
            valid &= np.isfinite(da.values)

    if min_valid_fraction == 0.0:
        valid_pixel = valid.any(axis=0)
    else:
        valid_pixel = valid.mean(axis=0) >= min_valid_fraction

    return xr.DataArray(
        valid_pixel,
        coords={"latitude": target["latitude"].values, "longitude": target["longitude"].values},
        dims=("latitude", "longitude"),
        name="valid_pixel_mask",
    )


def build_valid_observation_mask(
    target: xr.DataArray,
    predictors: dict[str, xr.DataArray] | None = None,
    min_valid_fraction: float = 0.0,
) -> xr.DataArray:
    _validate_target_3d(target)
    _validate_predictors(target, predictors)

    valid = np.isfinite(target.values)
    if predictors is not None:
        for da in predictors.values():
            valid &= np.isfinite(da.values)

    valid_pixel_mask = build_valid_pixel_mask(
        target=target,
        predictors=predictors,
        min_valid_fraction=min_valid_fraction,
    )
    valid &= valid_pixel_mask.values[None, :, :]

    return xr.DataArray(
        valid,
        coords=target.coords,
        dims=target.dims,
        name="valid_observation_mask",
    )


def _sample_selected_and_test(
    valid_idx: np.ndarray,
    train_fraction: float,
    test_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not (0.0 < train_fraction <= 1.0):
        raise ValueError("train_fraction debe estar en el intervalo (0, 1].")
    if not (0.0 < test_fraction < 1.0):
        raise ValueError("test_fraction debe estar en el intervalo (0, 1).")
    if valid_idx.size == 0:
        raise ValueError("No hay elementos válidos para construir el split.")

    rng = np.random.default_rng(seed)

    n_total = int(valid_idx.size)
    n_selected = n_total if train_fraction >= 1.0 else max(1, int(round(train_fraction * n_total)))

    selected_idx = valid_idx
    if n_selected < n_total:
        selected_idx = rng.choice(valid_idx, size=n_selected, replace=False)

    n_test = max(1, int(round(test_fraction * n_selected)))
    n_test = min(n_test, n_selected - 1) if n_selected > 1 else 0

    test_idx = np.array([], dtype=selected_idx.dtype)
    if n_test > 0:
        test_idx = rng.choice(selected_idx, size=n_test, replace=False)

    return np.asarray(selected_idx), np.asarray(test_idx)


def make_spatial_pixel_split(
    target: xr.DataArray,
    predictors: dict[str, xr.DataArray] | None = None,
    train_fraction: float = 1.0,
    test_fraction: float = 0.1,
    seed: int = DEFAULT_RANDOM_SEED,
    min_valid_fraction: float = 0.0,
) -> dict[str, Any]:
    valid_pixel_mask = build_valid_pixel_mask(
        target=target,
        predictors=predictors,
        min_valid_fraction=min_valid_fraction,
    )
    valid_idx = np.flatnonzero(valid_pixel_mask.values.ravel())
    selected_idx, test_idx = _sample_selected_and_test(
        valid_idx=valid_idx,
        train_fraction=train_fraction,
        test_fraction=test_fraction,
        seed=seed,
    )

    selected_mask = np.zeros(valid_pixel_mask.shape, dtype=bool)
    selected_mask.ravel()[selected_idx] = True

    test_mask = np.zeros(valid_pixel_mask.shape, dtype=bool)
    if test_idx.size > 0:
        test_mask.ravel()[test_idx] = True

    train_mask = selected_mask & (~test_mask)

    metadata = {
        "split_mode": "spatial_pixel",
        "seed": int(seed),
        "train_fraction_requested": float(train_fraction),
        "test_fraction_requested": float(test_fraction),
        "min_valid_fraction": float(min_valid_fraction),
        "n_total_pixels": int(valid_pixel_mask.size),
        "n_valid_pixels": int(valid_pixel_mask.values.sum()),
        "n_selected_pixels": int(selected_mask.sum()),
        "n_train_pixels": int(train_mask.sum()),
        "n_test_pixels": int(test_mask.sum()),
    }

    return {
        "train_mask": xr.DataArray(train_mask, coords=valid_pixel_mask.coords, dims=valid_pixel_mask.dims, name="train_mask"),
        "test_mask": xr.DataArray(test_mask, coords=valid_pixel_mask.coords, dims=valid_pixel_mask.dims, name="test_mask"),
        "selected_pixel_mask": xr.DataArray(selected_mask, coords=valid_pixel_mask.coords, dims=valid_pixel_mask.dims, name="selected_pixel_mask"),
        "valid_pixel_mask": valid_pixel_mask,
        "metadata": metadata,
    }


def make_random_observation_split(
    target: xr.DataArray,
    predictors: dict[str, xr.DataArray],
    train_fraction: float = 1.0,
    test_fraction: float = 0.1,
    seed: int = DEFAULT_RANDOM_SEED,
    min_valid_fraction: float = 0.0,
) -> dict[str, Any]:
    valid_observation_mask = build_valid_observation_mask(
        target=target,
        predictors=predictors,
        min_valid_fraction=min_valid_fraction,
    )
    valid_idx = np.flatnonzero(valid_observation_mask.values.ravel())
    selected_idx, test_idx = _sample_selected_and_test(
        valid_idx=valid_idx,
        train_fraction=train_fraction,
        test_fraction=test_fraction,
        seed=seed,
    )

    selected_mask = np.zeros(valid_observation_mask.shape, dtype=bool)
    selected_mask.ravel()[selected_idx] = True

    test_mask = np.zeros(valid_observation_mask.shape, dtype=bool)
    if test_idx.size > 0:
        test_mask.ravel()[test_idx] = True

    train_mask = selected_mask & (~test_mask)

    metadata = {
        "split_mode": "random_observation",
        "seed": int(seed),
        "train_fraction_requested": float(train_fraction),
        "test_fraction_requested": float(test_fraction),
        "min_valid_fraction": float(min_valid_fraction),
        "n_total_observations": int(valid_observation_mask.size),
        "n_valid_observations": int(valid_observation_mask.values.sum()),
        "n_selected_observations": int(selected_mask.sum()),
        "n_train_observations": int(train_mask.sum()),
        "n_test_observations": int(test_mask.sum()),
        "n_unique_train_pixels": int(train_mask.any(axis=0).sum()),
        "n_unique_test_pixels": int(test_mask.any(axis=0).sum()),
    }

    return {
        "train_mask": xr.DataArray(train_mask, coords=valid_observation_mask.coords, dims=valid_observation_mask.dims, name="train_mask"),
        "test_mask": xr.DataArray(test_mask, coords=valid_observation_mask.coords, dims=valid_observation_mask.dims, name="test_mask"),
        "selected_observation_mask": xr.DataArray(selected_mask, coords=valid_observation_mask.coords, dims=valid_observation_mask.dims, name="selected_observation_mask"),
        "valid_observation_mask": valid_observation_mask,
        "valid_pixel_mask": build_valid_pixel_mask(target=target, predictors=predictors, min_valid_fraction=min_valid_fraction),
        "metadata": metadata,
    }


def make_train_test_split(
    split_mode: str,
    target: xr.DataArray,
    predictors: dict[str, xr.DataArray],
    train_fraction: float = 1.0,
    test_fraction: float = 0.1,
    seed: int = DEFAULT_RANDOM_SEED,
    min_valid_fraction: float = 0.0,
) -> dict[str, Any]:
    split_mode = split_mode.lower().strip()

    if split_mode == "spatial_pixel":
        return make_spatial_pixel_split(
            target=target,
            predictors=predictors,
            train_fraction=train_fraction,
            test_fraction=test_fraction,
            seed=seed,
            min_valid_fraction=min_valid_fraction,
        )

    if split_mode == "random_observation":
        return make_random_observation_split(
            target=target,
            predictors=predictors,
            train_fraction=train_fraction,
            test_fraction=test_fraction,
            seed=seed,
            min_valid_fraction=min_valid_fraction,
        )

    raise ValueError(
        f"split_mode no soportado: '{split_mode}'. Usa 'spatial_pixel' o 'random_observation'."
    )
