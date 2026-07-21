"""Vista cartográfica interactiva basada en ipyleaflet."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from ipywidgets import HTML

try:
    from ipyleaflet import ImageOverlay, LayersControl, Map, Rectangle, WidgetControl, basemaps
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError(
        "ipyleaflet no está instalado. Instala dependencias con: pip install -r requirements.txt"
    ) from exc


@dataclass
class MapBounds:
    south: float
    west: float
    north: float
    east: float

    @classmethod
    def from_lat_lon(
        cls,
        lat: np.ndarray,
        lon: np.ndarray,
    ) -> "MapBounds":
        """Devuelve los límites globales compatibles con Web Mercator."""

        del lat
        del lon

        return cls(
            south=-85.05112878,
            west=-180.0,
            north=85.05112878,
            east=180.0,
        )

    @property
    def leaflet(self) -> tuple[tuple[float, float], tuple[float, float]]:
        return ((self.south, self.west), (self.north, self.east))


class ProfeciaMapView:
    """Mapa con overlay raster y rectángulo ROI."""

    def __init__(
        self,
        lat: np.ndarray,
        lon: np.ndarray,
        initial_png_url: str | None = None,
        roi_bounds: tuple[tuple[float, float], tuple[float, float]] | None = None,
        height: str = "430px",
    ) -> None:
        self.lat = lat
        self.lon = lon
        self.map_bounds = MapBounds.from_lat_lon(lat, lon)
        self.roi_bounds = roi_bounds or self.map_bounds.leaflet

        self.map = Map(
            basemap=basemaps.CartoDB.PositronNoLabels,
            center=(10, 0),
            zoom=2,
            scroll_wheel_zoom=True,
            layout={"height": height, "width": "100%"},
        )

        self.overlay = ImageOverlay(
            url=initial_png_url or "",
            bounds=self.map_bounds.leaflet,
            name="Preview LAI / máscaras",
        )
        self.map.add_layer(self.overlay)

        self.roi_rectangle = Rectangle(
            bounds=self.roi_bounds,
            color="#F59E0B",
            fill=False,
            weight=3,
            name="ROI",
        )
        self.map.add_layer(self.roi_rectangle)

        self.map.add_control(LayersControl(position="topright"))
        self.map.add_control(WidgetControl(widget=self._legend(), position="bottomleft"))

    def _legend(self) -> HTML:
        return HTML(
            """
            <div class="profecia-map-legend">
              <span><i style="background:#cfe8ff"></i> océano / excluido</span>
              <span><i style="background:#78a85d"></i> válido</span>
              <span><i style="background:#f59e0b"></i> ROI</span>
              <span><i style="background:#808896"></i> enmascarado</span>
            </div>
            """
        )

    def update_overlay(self, png_data_url: str) -> None:
        self.overlay.url = png_data_url

    def update_roi(self, roi_bounds: tuple[tuple[float, float], tuple[float, float]]) -> None:
        self.roi_bounds = roi_bounds
        self.roi_rectangle.bounds = roi_bounds

    @property
    def widget(self) -> Any:
        return self.map
