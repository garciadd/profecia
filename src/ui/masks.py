"""Carga y combinación eficiente de filtros espaciales para PROFECIA."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .paths_io import get_project_paths


@dataclass
class SpatialMasks:
    binary: dict[str, np.ndarray] = field(default_factory=dict)
    categorical: dict[str, np.ndarray] = field(default_factory=dict)
    missing_binary: list[str] = field(default_factory=list)
    missing_categorical: list[str] = field(default_factory=list)

    @property
    def missing(self) -> list[str]:
        return self.missing_binary + self.missing_categorical


def load_grid(
    preview_dir: str | Path,
    shape: tuple[int, int] = (360, 720),
) -> tuple[np.ndarray, np.ndarray]:
    """Carga lat/lon o genera una malla global regular de 0.5 grados."""

    preview_dir = Path(preview_dir)

    lat_candidates = [
        preview_dir / "grid" / "lat.npy",
        preview_dir / "lat.npy",
    ]
    lon_candidates = [
        preview_dir / "grid" / "lon.npy",
        preview_dir / "lon.npy",
    ]

    lat = next(
        (np.load(path) for path in lat_candidates if path.exists()),
        None,
    )
    lon = next(
        (np.load(path) for path in lon_candidates if path.exists()),
        None,
    )

    if lat is None:
        lat = np.linspace(
            89.75,
            -89.75,
            shape[0],
            dtype=np.float32,
        )

    if lon is None:
        lon = np.linspace(
            -179.75,
            179.75,
            shape[1],
            dtype=np.float32,
        )

    return (
        np.asarray(lat).squeeze(),
        np.asarray(lon).squeeze(),
    )


def _resolve_candidates(
    *,
    name: str,
    filename: str,
    preview_dir: Path,
    paths_config: dict[str, Any],
) -> list[Path]:
    paths = get_project_paths(paths_config)
    data_dir = paths.get("data_dir")
    mask_dir = paths.get("mask_dir")

    candidates: list[Path] = [
        preview_dir / "masks" / filename,
        preview_dir / "masks" / f"{name}.npy",
        preview_dir / filename,
        preview_dir / f"{name}.npy",
    ]

    if mask_dir is not None:
        candidates.extend([mask_dir / filename, mask_dir / f"{name}.npy"])
    if data_dir is not None:
        candidates.extend([
            data_dir / "masks" / filename,
            data_dir / "masks" / f"{name}.npy",
            data_dir / filename,
            data_dir / f"{name}.npy",
        ])

    # Conserva orden, elimina duplicados.
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in candidates:
        p = Path(path).expanduser()
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique

def _load_binary_mask_file(path: Path) -> np.ndarray:
    arr = np.asarray(np.load(path)).squeeze()

    if arr.ndim != 2:
        raise ValueError(
            f"La máscara binaria {path} debe ser 2D. "
            f"Shape recibido: {arr.shape}"
        )

    if arr.dtype == bool:
        mask = arr
    else:
        mask = np.isfinite(arr) & (arr > 0)

    # Las máscaras están almacenadas de sur a norte.
    # La malla de la interfaz está definida de norte a sur.
    return np.flipud(mask)

def _load_categorical_mask_file(path: Path) -> np.ndarray:
    arr = np.asarray(np.load(path)).squeeze()

    if arr.ndim != 2:
        raise ValueError(
            f"La máscara categórica {path} debe ser 2D. "
            f"Shape recibido: {arr.shape}"
        )

    arr = arr.astype(np.int16, copy=False)

    # Sur-norte → norte-sur.
    return np.flipud(arr)

def load_spatial_masks(
    preview_dir: str | Path,
    paths_config: dict[str, Any],
    shape: tuple[int, int],
    binary_catalog: dict[str, dict[str, Any]],
    categorical_catalog: dict[str, dict[str, Any]],
) -> SpatialMasks:
    """Carga máscaras binarias y categóricas disponibles.

    Busca primero en previews/masks para que la interfaz sea rápida y portable. Si no
    encuentra ahí, intenta usar las rutas definidas en paths.toml.
    """
    preview_dir = Path(preview_dir)
    out = SpatialMasks()

    for name, meta in binary_catalog.items():
        filename = str(meta.get("filename", f"{name}.npy"))
        path = next((p for p in _resolve_candidates(name=name, filename=filename, preview_dir=preview_dir, paths_config=paths_config) if p.exists()), None)
        if path is None:
            out.missing_binary.append(name)
            continue
        mask = _load_binary_mask_file(path)
        if mask.shape != shape:
            raise ValueError(f"La máscara binaria {name!r} tiene shape {mask.shape}, pero se esperaba {shape}.")
        out.binary[name] = mask

    for name, meta in categorical_catalog.items():
        filename = str(meta.get("filename", f"{name}.npy"))
        path = next((p for p in _resolve_candidates(name=name, filename=filename, preview_dir=preview_dir, paths_config=paths_config) if p.exists()), None)
        if path is None:
            out.missing_categorical.append(name)
            continue
        mask = _load_categorical_mask_file(path)
        if mask.shape != shape:
            raise ValueError(f"La máscara categórica {name!r} tiene shape {mask.shape}, pero se esperaba {shape}.")
        out.categorical[name] = mask

    return out


def create_roi_mask(
    lat: np.ndarray,
    lon: np.ndarray,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
) -> np.ndarray:
    """Crea una máscara booleana 2D para el ROI.

    Soporta ROI normal y ROI cruzando antimeridiano si lon_min > lon_max.
    """
    lat = np.asarray(lat).squeeze()
    lon = np.asarray(lon).squeeze()

    if lat_min >= lat_max:
        raise ValueError("lat_min debe ser menor que lat_max.")
    if lon_min == lon_max:
        raise ValueError("lon_min y lon_max no pueden ser iguales.")

    lat_mask = (lat >= lat_min) & (lat <= lat_max)
    if lon_min < lon_max:
        lon_mask = (lon >= lon_min) & (lon <= lon_max)
    else:
        lon_mask = (lon >= lon_min) | (lon <= lon_max)

    return lat_mask[:, None] & lon_mask[None, :]


def combine_spatial_filters(
    *,
    selected_binary_masks: Iterable[str],
    categorical_filters: dict[str, list[int]],
    spatial_masks: SpatialMasks,
    shape: tuple[int, int],
    binary_catalog: dict[str, dict[str, Any]],
    categorical_catalog: dict[str, dict[str, Any]],
) -> tuple[np.ndarray, list[str]]:
    """Combina filtros espaciales.

    True = celda válida final.

    Máscaras binarias:
      - mode=include: conserva las celdas True de la máscara.
      - mode=exclude: elimina las celdas True de la máscara.

    Máscaras categóricas:
      - sin clases seleccionadas: no filtra.
      - con clases seleccionadas: conserva únicamente esas clases.
    """
    valid = np.ones(shape, dtype=bool)
    skipped: list[str] = []

    for name in selected_binary_masks:
        mask = spatial_masks.binary.get(name)
        if mask is None:
            skipped.append(name)
            continue
        mode = str(binary_catalog.get(name, {}).get("mode", "exclude"))
        if mode == "include":
            valid &= mask
        elif mode == "exclude":
            valid &= ~mask
        else:
            raise ValueError(f"Modo no reconocido para la máscara binaria {name!r}: {mode!r}")

    for name, class_ids in categorical_filters.items():
        selected_ids = [int(v) for v in class_ids if v is not None]
        if not selected_ids:
            continue
        mask = spatial_masks.categorical.get(name)
        if mask is None:
            skipped.append(name)
            continue
        valid &= np.isin(mask, selected_ids)

    return valid, skipped


def create_final_valid_mask(
    *,
    selected_binary_masks: Iterable[str],
    categorical_filters: dict[str, list[int]],
    spatial_masks: SpatialMasks,
    lat: np.ndarray,
    lon: np.ndarray,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    binary_catalog: dict[str, dict[str, Any]],
    categorical_catalog: dict[str, dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Devuelve máscara final, máscara ROI y filtros omitidos."""
    shape = (lat.size, lon.size)
    roi_mask = create_roi_mask(lat, lon, lat_min, lat_max, lon_min, lon_max)
    filter_valid, skipped = combine_spatial_filters(
        selected_binary_masks=selected_binary_masks,
        categorical_filters=categorical_filters,
        spatial_masks=spatial_masks,
        shape=shape,
        binary_catalog=binary_catalog,
        categorical_catalog=categorical_catalog,
    )
    return roi_mask & filter_valid, roi_mask, skipped


def valid_cell_stats(valid_mask: np.ndarray) -> dict[str, float | int]:
    total = int(valid_mask.size)
    valid = int(np.count_nonzero(valid_mask))
    pct = float(valid / total * 100.0) if total else 0.0
    return {"valid": valid, "total": total, "pct": pct}
