import json
from pathlib import Path
from typing import Iterable

import numpy as np
import xarray as xr


DEFAULT_RANDOM_SEED = 42
DEFAULT_IGNORE_CODES = (0,)


def _validate_lai_3d(lai: xr.DataArray) -> None:
    """
    Valida que LAI tenga dims (time, latitude, longitude).
    """
    required_dims = ("time", "latitude", "longitude")
    if tuple(lai.dims) != required_dims:
        raise ValueError(
            f"LAI debe tener dims {required_dims}. Recibido: {lai.dims}"
        )


def _validate_mask_2d(mask: xr.DataArray, name: str = "mask") -> None:
    """
    Valida que una máscara 2D tenga dims (latitude, longitude).
    """
    required_dims = ("latitude", "longitude")
    if tuple(mask.dims) != required_dims:
        raise ValueError(
            f"{name} debe tener dims {required_dims}. Recibido: {mask.dims}"
        )


def _validate_category_mask_2d(category_mask: xr.DataArray) -> None:
    """
    Valida que la máscara categórica tenga dims (latitude, longitude).
    """
    _validate_mask_2d(category_mask, name="category_mask")


def _validate_alignment(lai: xr.DataArray, category_mask: xr.DataArray) -> None:
    """
    Comprueba que LAI y la máscara categórica compartan exactamente
    la misma rejilla espacial.
    """
    _validate_lai_3d(lai)
    _validate_category_mask_2d(category_mask)

    if lai.sizes["latitude"] != category_mask.sizes["latitude"]:
        raise ValueError(
            "LAI y category_mask no tienen el mismo tamaño en latitude."
        )

    if lai.sizes["longitude"] != category_mask.sizes["longitude"]:
        raise ValueError(
            "LAI y category_mask no tienen el mismo tamaño en longitude."
        )

    if not np.array_equal(lai["latitude"].values, category_mask["latitude"].values):
        raise ValueError("LAI y category_mask no comparten las mismas coordenadas latitude.")

    if not np.array_equal(lai["longitude"].values, category_mask["longitude"].values):
        raise ValueError("LAI y category_mask no comparten las mismas coordenadas longitude.")


def build_valid_pixel_mask(
    lai: xr.DataArray,
    min_valid_fraction: float = 0.0,
) -> xr.DataArray:
    """
    Construye una máscara 2D de píxeles válidos a partir de LAI.

    Un píxel es válido si cumple:
    - min_valid_fraction = 0.0  -> al menos un valor no-NaN en el tiempo
    - min_valid_fraction > 0.0  -> fracción mínima de valores válidos
    """
    _validate_lai_3d(lai)

    if not (0.0 <= min_valid_fraction <= 1.0):
        raise ValueError("min_valid_fraction debe estar entre 0 y 1.")

    if min_valid_fraction == 0.0:
        valid_mask = lai.notnull().any(dim="time")
    else:
        valid_fraction = lai.notnull().mean(dim="time")
        valid_mask = valid_fraction >= min_valid_fraction

    valid_mask = valid_mask.rename("valid_pixel_mask")
    return valid_mask


def get_valid_category_codes(
    category_mask: xr.DataArray,
    valid_pixel_mask: xr.DataArray,
    ignore_codes: Iterable[int] = DEFAULT_IGNORE_CODES,
) -> list[int]:
    """
    Devuelve los códigos de categoría válidos presentes en píxeles elegibles.
    """
    _validate_category_mask_2d(category_mask)
    _validate_mask_2d(valid_pixel_mask, name="valid_pixel_mask")

    cat = np.asarray(category_mask.values)
    valid = np.asarray(valid_pixel_mask.values).astype(bool)

    ignore_codes = set(int(x) for x in ignore_codes)

    eligible_values = cat[valid]
    eligible_values = eligible_values[np.isfinite(eligible_values)]

    codes = []
    for code in np.unique(eligible_values):
        code_int = int(code)
        if code_int in ignore_codes:
            continue
        codes.append(code_int)

    return sorted(codes)


def build_eligible_pixel_mask(
    lai: xr.DataArray,
    category_mask: xr.DataArray,
    ignore_codes: Iterable[int] = DEFAULT_IGNORE_CODES,
    min_valid_fraction: float = 0.0,
) -> xr.DataArray:
    """
    Construye la máscara de píxeles elegibles para el split:

    - válidos según LAI
    - categoría finita
    - categoría no incluida en ignore_codes
    """
    _validate_alignment(lai, category_mask)

    valid_pixel_mask = build_valid_pixel_mask(
        lai=lai,
        min_valid_fraction=min_valid_fraction,
    )

    cat = np.asarray(category_mask.values)
    valid = np.asarray(valid_pixel_mask.values).astype(bool)

    ignore_codes = tuple(int(x) for x in ignore_codes)

    eligible = valid & np.isfinite(cat)
    for code in ignore_codes:
        eligible &= (cat != code)

    return xr.DataArray(
        eligible,
        coords=category_mask.coords,
        dims=category_mask.dims,
        name="eligible_pixel_mask",
    )


def make_category_stratified_pixel_subset(
    lai: xr.DataArray,
    category_mask: xr.DataArray,
    pixel_fraction: float = 1.0,
    seed: int = DEFAULT_RANDOM_SEED,
    ignore_codes: Iterable[int] = DEFAULT_IGNORE_CODES,
    min_valid_fraction: float = 0.0,
    category_labels: dict[int, str] | None = None,
) -> dict:
    """
    Selecciona un subconjunto espacial estratificado por categoría.

    Lógica:
    - parte de los píxeles elegibles
    - para cada categoría selecciona aleatoriamente pixel_fraction
    - el resto de píxeles elegibles se descartan para esta corrida

    Returns
    -------
    dict con:
    - selected_pixel_mask
    - eligible_pixel_mask
    - valid_pixel_mask
    - metadata
    """
    if not (0.0 < pixel_fraction <= 1.0):
        raise ValueError("pixel_fraction debe estar en el intervalo (0, 1].")

    _validate_alignment(lai, category_mask)

    valid_pixel_mask = build_valid_pixel_mask(
        lai=lai,
        min_valid_fraction=min_valid_fraction,
    )

    eligible_pixel_mask = build_eligible_pixel_mask(
        lai=lai,
        category_mask=category_mask,
        ignore_codes=ignore_codes,
        min_valid_fraction=min_valid_fraction,
    )

    cat = np.asarray(category_mask.values)
    eligible = np.asarray(eligible_pixel_mask.values).astype(bool)

    ignore_codes = tuple(int(x) for x in ignore_codes)
    rng = np.random.default_rng(seed)

    valid_codes = get_valid_category_codes(
        category_mask=category_mask,
        valid_pixel_mask=eligible_pixel_mask,
        ignore_codes=ignore_codes,
    )

    if len(valid_codes) == 0:
        raise ValueError("No se encontraron categorías válidas para seleccionar píxeles.")

    selected_mask = np.zeros(cat.shape, dtype=bool)
    summary = []

    for code in valid_codes:
        class_idx = np.flatnonzero(((cat == code) & eligible).ravel())
        n_total = len(class_idx)

        if n_total == 0:
            continue

        if pixel_fraction >= 1.0:
            n_selected = n_total
        else:
            n_selected = int(round(pixel_fraction * n_total))
            n_selected = max(1, n_selected)

        selected_idx = rng.choice(class_idx, size=n_selected, replace=False)
        selected_mask.ravel()[selected_idx] = True

        summary.append(
            {
                "category_code": int(code),
                "category_label": (
                    category_labels.get(int(code), f"class_{int(code)}")
                    if category_labels is not None
                    else f"class_{int(code)}"
                ),
                "n_eligible": int(n_total),
                "n_selected": int(n_selected),
                "n_dropped": int(n_total - n_selected),
                "pixel_fraction_real": float(n_selected / n_total),
            }
        )

    # validación
    if not np.all(selected_mask <= eligible):
        raise RuntimeError("selected_pixel_mask contiene píxeles no elegibles.")

    selected_mask_da = xr.DataArray(
        selected_mask,
        coords=category_mask.coords,
        dims=category_mask.dims,
        name="selected_pixel_mask",
    )

    metadata = {
        "pixel_fraction_requested": float(pixel_fraction),
        "seed": int(seed),
        "ignore_codes": list(ignore_codes),
        "min_valid_fraction": float(min_valid_fraction),
        "n_total_pixels": int(cat.size),
        "n_valid_pixels": int(valid_pixel_mask.values.sum()),
        "n_eligible_pixels": int(eligible.sum()),
        "n_selected_pixels": int(selected_mask.sum()),
        "category_codes_used": [int(x) for x in valid_codes],
        "category_labels": (
            {int(k): str(v) for k, v in category_labels.items()}
            if category_labels is not None
            else None
        ),
        "per_category": summary,
    }

    return {
        "selected_pixel_mask": selected_mask_da,
        "eligible_pixel_mask": eligible_pixel_mask,
        "valid_pixel_mask": valid_pixel_mask,
        "metadata": metadata,
    }


def make_stratified_spatial_split(
    lai: xr.DataArray,
    category_mask: xr.DataArray,
    test_fraction: float = 0.10,
    seed: int = DEFAULT_RANDOM_SEED,
    ignore_codes: Iterable[int] = DEFAULT_IGNORE_CODES,
    split_name: str = "spatial_stratified_split",
    category_labels: dict[int, str] | None = None,
    pixel_fraction: float = 1.0,
    subset_seed: int | None = None,
    min_valid_fraction: float = 0.0,
) -> dict:
    """
    Crea un split espacial estratificado por categoría con submuestreo previo opcional.

    Flujo:
    1) se construyen los píxeles elegibles
    2) se selecciona un % de píxeles por categoría (pixel_fraction)
    3) sobre esos píxeles seleccionados se aplica el split train/test estratificado

    Returns
    -------
    dict con:
    - train_mask
    - test_mask
    - selected_pixel_mask
    - eligible_pixel_mask
    - valid_pixel_mask
    - metadata
    """
    if not (0.0 < test_fraction < 1.0):
        raise ValueError("test_fraction debe estar entre 0 y 1.")

    if subset_seed is None:
        subset_seed = seed

    _validate_alignment(lai, category_mask)

    subset_result = make_category_stratified_pixel_subset(
        lai=lai,
        category_mask=category_mask,
        pixel_fraction=pixel_fraction,
        seed=subset_seed,
        ignore_codes=ignore_codes,
        min_valid_fraction=min_valid_fraction,
        category_labels=category_labels,
    )

    selected_pixel_mask = subset_result["selected_pixel_mask"]
    eligible_pixel_mask = subset_result["eligible_pixel_mask"]
    valid_pixel_mask = subset_result["valid_pixel_mask"]

    cat = np.asarray(category_mask.values)
    selected = np.asarray(selected_pixel_mask.values).astype(bool)

    ignore_codes = tuple(int(x) for x in ignore_codes)
    rng = np.random.default_rng(seed)

    valid_codes = get_valid_category_codes(
        category_mask=category_mask,
        valid_pixel_mask=selected_pixel_mask,
        ignore_codes=ignore_codes,
    )

    if len(valid_codes) == 0:
        raise ValueError("No se encontraron categorías válidas para construir el split.")

    test_mask = np.zeros(cat.shape, dtype=bool)
    summary = []

    for code in valid_codes:
        class_idx = np.flatnonzero(((cat == code) & selected).ravel())
        n_total = len(class_idx)

        if n_total == 0:
            continue

        n_test = int(round(test_fraction * n_total))
        n_test = max(1, n_test)

        if n_total > 1:
            n_test = min(n_test, n_total - 1)
        else:
            n_test = 0

        if n_test > 0:
            selected_idx = rng.choice(class_idx, size=n_test, replace=False)
            test_mask.ravel()[selected_idx] = True

        summary.append(
            {
                "category_code": int(code),
                "category_label": (
                    category_labels.get(int(code), f"class_{int(code)}")
                    if category_labels is not None
                    else f"class_{int(code)}"
                ),
                "n_selected": int(n_total),
                "n_test": int(n_test),
                "n_train": int(n_total - n_test),
                "test_fraction_real": float(n_test / n_total) if n_total > 0 else np.nan,
            }
        )

    train_mask = selected & (~test_mask)

    overlap = train_mask & test_mask
    if overlap.any():
        raise RuntimeError("Hay solapamiento entre train y test.")

    covered = train_mask | test_mask
    if not np.array_equal(covered, selected):
        raise RuntimeError("Train y test no cubren exactamente los píxeles seleccionados.")

    train_mask_da = xr.DataArray(
        train_mask,
        coords=category_mask.coords,
        dims=category_mask.dims,
        name="train_mask",
    )

    test_mask_da = xr.DataArray(
        test_mask,
        coords=category_mask.coords,
        dims=category_mask.dims,
        name="test_mask",
    )

    metadata = {
        "split_name": split_name,
        "seed": int(seed),
        "subset_seed": int(subset_seed),
        "ignore_codes": list(ignore_codes),
        "min_valid_fraction": float(min_valid_fraction),
        "pixel_fraction_requested": float(pixel_fraction),
        "test_fraction_requested": float(test_fraction),
        "n_total_pixels": int(cat.size),
        "n_valid_pixels": int(valid_pixel_mask.values.sum()),
        "n_eligible_pixels": int(eligible_pixel_mask.values.sum()),
        "n_selected_pixels": int(selected.sum()),
        "n_train_pixels": int(train_mask.sum()),
        "n_test_pixels": int(test_mask.sum()),
        "category_codes_used": [int(x) for x in valid_codes],
        "category_labels": (
            {int(k): str(v) for k, v in category_labels.items()}
            if category_labels is not None
            else None
        ),
        "pixel_subset_per_category": subset_result["metadata"]["per_category"],
        "split_per_category": summary,
    }

    return {
        "train_mask": train_mask_da,
        "test_mask": test_mask_da,
        "selected_pixel_mask": selected_pixel_mask,
        "eligible_pixel_mask": eligible_pixel_mask,
        "valid_pixel_mask": valid_pixel_mask,
        "metadata": metadata,
    }


def mask_to_pixel_ids(mask_2d: xr.DataArray) -> np.ndarray:
    """
    Convierte una máscara booleana 2D en pixel_ids planos.

    pixel_id = lat_idx * n_lon + lon_idx
    """
    _validate_mask_2d(mask_2d, name="mask_2d")

    arr = np.asarray(mask_2d.values).astype(bool)
    return np.flatnonzero(arr.ravel()).astype(np.int32)


def pixel_ids_to_mask(
    pixel_ids: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    name: str = "mask_from_pixel_ids",
) -> xr.DataArray:
    """
    Reconstruye una máscara 2D booleana a partir de pixel_ids.
    """
    n_lat = len(latitude)
    n_lon = len(longitude)

    out = np.zeros((n_lat, n_lon), dtype=bool)
    out.ravel()[pixel_ids] = True

    return xr.DataArray(
        out,
        coords={"latitude": latitude, "longitude": longitude},
        dims=("latitude", "longitude"),
        name=name,
    )


def save_spatial_split(
    output_dir: str | Path,
    train_mask: xr.DataArray,
    test_mask: xr.DataArray,
    metadata: dict,
    prefix: str,
    selected_pixel_mask: xr.DataArray | None = None,
    eligible_pixel_mask: xr.DataArray | None = None,
    valid_pixel_mask: xr.DataArray | None = None,
) -> None:
    """
    Guarda máscaras y metadata del split en disco.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / f"{prefix}_train_mask.npy", train_mask.values)
    np.save(output_dir / f"{prefix}_test_mask.npy", test_mask.values)

    if selected_pixel_mask is not None:
        np.save(output_dir / f"{prefix}_selected_pixel_mask.npy", selected_pixel_mask.values)

    if eligible_pixel_mask is not None:
        np.save(output_dir / f"{prefix}_eligible_pixel_mask.npy", eligible_pixel_mask.values)

    if valid_pixel_mask is not None:
        np.save(output_dir / f"{prefix}_valid_pixel_mask.npy", valid_pixel_mask.values)

    with open(output_dir / f"{prefix}_split_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)