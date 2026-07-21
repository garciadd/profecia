"""Capas ligeras de previsualización para la interfaz PROFECIA."""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Literal

import matplotlib
import matplotlib.colors as mcolors
import numpy as np
from PIL import Image

DataValueType = Literal["real", "anomaly", "trend"]
TemporalResolution = Literal["annual", "monthly"]


def get_colormap(name: str):
    """Compatibilidad con matplotlib nuevo y antiguo."""
    if hasattr(matplotlib, "colormaps"):
        return matplotlib.colormaps.get_cmap(name)
    from matplotlib import cm  # pragma: no cover
    return cm.get_cmap(name)


def preview_layer_name(temporal_resolution: str, data_value_type: str) -> str:
    if data_value_type == "real":
        return f"lai_real_{temporal_resolution}_mean"
    if data_value_type == "anomaly":
        return f"lai_anomaly_{temporal_resolution}_std"
    if data_value_type == "trend":
        return f"lai_trend_{temporal_resolution}_slope"
    raise ValueError(f"Tipo de dato no reconocido: {data_value_type!r}")


def _candidate_layer_paths(preview_dir: Path, name: str) -> list[Path]:
    return [
        preview_dir / "layers" / f"{name}.npy",
        preview_dir / f"{name}.npy",
    ]


def _make_demo_layer(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Genera una capa sintética fija, ligera y espacialmente coherente.

    La capa solo sirve como fondo visual para representar el ROI y las máscaras.
    Sus valores están aproximadamente comprendidos entre 0 y 7.
    """

    lat2d, lon2d = np.meshgrid(lat, lon, indexing="ij")

    # Mayor valor cerca del ecuador y descenso progresivo hacia los polos.
    latitudinal = 5.2 * np.exp(-((lat2d / 38.0) ** 2))

    # Variación longitudinal suave para evitar bandas perfectamente uniformes.
    longitudinal = (
        0.7 * np.sin(np.deg2rad(lon2d * 1.3))
        + 0.4 * np.cos(np.deg2rad(lon2d * 2.1))
    )

    # Variación regional adicional.
    regional = (
        0.5
        * np.sin(np.deg2rad(lat2d * 2.0))
        * np.cos(np.deg2rad(lon2d * 0.8))
    )

    layer = 0.6 + latitudinal + longitudinal + regional

    return np.clip(layer, 0.0, 7.0).astype(np.float32)

def load_preview_layer(
    preview_dir: str | Path,
    temporal_resolution: str,
    data_value_type: str,
    lat: np.ndarray,
    lon: np.ndarray,
) -> tuple[np.ndarray, str]:
    """Genera la capa sintética fija utilizada por el mapa de la interfaz."""
    del preview_dir
    del temporal_resolution
    del data_value_type

    layer = _make_demo_layer(lat, lon)

    return (
        layer,
        "Capa sintética de referencia para visualizar el ROI y los filtros espaciales.",
    )

def _normalizer(values: np.ndarray, data_value_type: str) -> mcolors.Normalize:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return mcolors.Normalize(vmin=0.0, vmax=1.0)

    if data_value_type == "trend":
        p = float(np.nanpercentile(np.abs(finite), 98))
        p = p if p > 0 else 1.0
        return mcolors.TwoSlopeNorm(vmin=-p, vcenter=0.0, vmax=p)

    vmin = float(np.nanpercentile(finite, 2))
    vmax = float(np.nanpercentile(finite, 98))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin, vmax = float(np.nanmin(finite)), float(np.nanmax(finite))
    if vmin == vmax:
        vmax = vmin + 1.0
    return mcolors.Normalize(vmin=vmin, vmax=vmax)

WEB_MERCATOR_MAX_LAT = 85.05112878


def _rgba_to_web_mercator(
    rgba: np.ndarray,
    lat: np.ndarray,
) -> np.ndarray:
    """Reordena las filas de una imagen geográfica para Web Mercator.

    El array original tiene filas equiespaciadas en grados de latitud.
    Leaflet muestra el mapa en Web Mercator, donde las filas no están
    equiespaciadas en latitud.
    """
    lat = np.asarray(lat, dtype=np.float64).squeeze()

    if lat.ndim != 1:
        raise ValueError("lat debe ser un vector unidimensional.")

    if rgba.shape[0] != lat.size:
        raise ValueError(
            "El número de filas del PNG no coincide con lat: "
            f"{rgba.shape[0]} vs {lat.size}"
        )

    # Aseguramos que el array fuente está ordenado norte → sur.
    if lat[0] < lat[-1]:
        lat_source = lat[::-1]
        rgba_source = np.flipud(rgba)
    else:
        lat_source = lat
        rgba_source = rgba

    # Coordenada y uniforme en Web Mercator.
    max_lat_rad = np.deg2rad(WEB_MERCATOR_MAX_LAT)
    max_y = np.arcsinh(np.tan(max_lat_rad))

    mercator_y = np.linspace(
        max_y,
        -max_y,
        rgba.shape[0],
        dtype=np.float64,
    )

    # Transformación inversa de Web Mercator:
    # y -> latitud geográfica.
    target_lat = np.rad2deg(
        np.arctan(np.sinh(mercator_y))
    )

    # Índice de la fila fuente más próxima.
    source_indices = np.interp(
        target_lat,
        lat_source[::-1],
        np.arange(lat_source.size - 1, -1, -1),
    )

    source_indices = np.clip(
        np.rint(source_indices).astype(np.int64),
        0,
        lat_source.size - 1,
    )

    return rgba_source[source_indices, :]

def array_to_png_data_url(
    layer: np.ndarray,
    valid_mask: np.ndarray,
    lat: np.ndarray,
    data_value_type: str,
    invalid_rgba: tuple[int, int, int, int] = (120, 128, 140, 115),
    valid_alpha: int = 210,
) -> str:
    """Convierte la capa y las máscaras en un PNG para Leaflet."""

    del data_value_type

    if layer.shape != valid_mask.shape:
        raise ValueError(
            "layer y valid_mask deben tener el mismo shape: "
            f"{layer.shape} vs {valid_mask.shape}"
        )

    layer = np.asarray(layer, dtype=np.float32)
    valid_mask = np.asarray(valid_mask, dtype=bool)

    norm = mcolors.Normalize(vmin=0.0, vmax=7.0)
    cmap = get_colormap("YlGn")

    rgba_float = cmap(
        norm(np.nan_to_num(layer, nan=0.0))
    )
    rgba = (rgba_float * 255).astype(np.uint8)
    rgba[..., 3] = valid_alpha

    invalid = ~valid_mask | ~np.isfinite(layer)

    rgba[invalid, 0] = invalid_rgba[0]
    rgba[invalid, 1] = invalid_rgba[1]
    rgba[invalid, 2] = invalid_rgba[2]
    rgba[invalid, 3] = invalid_rgba[3]

    # Conversión de cuadrícula geográfica regular a Web Mercator.
    rgba = _rgba_to_web_mercator(rgba, lat)

    image = Image.fromarray(rgba, mode="RGBA")

    buffer = io.BytesIO()
    image.save(
        buffer,
        format="PNG",
        optimize=True,
    )

    encoded = base64.b64encode(
        buffer.getvalue()
    ).decode("ascii")

    return f"data:image/png;base64,{encoded}"