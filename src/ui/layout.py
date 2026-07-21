"""Layout principal de la interfaz interactiva PROFECIA."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import ipywidgets as widgets
from IPython.display import display

from .config_io import load_config, save_config, update_config_from_state
from .masks import create_final_valid_mask, load_grid, load_spatial_masks, valid_cell_stats
from .paths_io import (
    default_binary_mask_names,
    default_categorical_filters,
    get_all_available_variables,
    get_binary_mask_catalog,
    get_categorical_mask_catalog,
    get_predictor_groups,
    get_target_names,
    load_paths_config,
)
from .preview import array_to_png_data_url, load_preview_layer
from .state import ProfeciaUIState
from .validation import validate_state
from .widgets import ProfeciaWidgets
from .map_view import ProfeciaMapView

CSS = """
<style>
.profecia-root, .profecia-root * {
    box-sizing: border-box;
}
.profecia-root {
    max-width: 100%;
    overflow-x: hidden !important;
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    color: #172033;
}
.profecia-header {
    display: flex;
    align-items: center;
    gap: 16px;
    margin: 6px 0 18px 0;
    max-width: 100%;
}
.profecia-logo-dot {
    width: 44px;
    height: 44px;
    border-radius: 999px;
    border: 3px solid #2563eb;
    color: #16a34a;
    font-size: 30px;
    font-weight: 800;
    line-height: 37px;
    text-align: center;
    background: #f8fbff;
}
.profecia-title {
    font-size: 42px;
    font-weight: 850;
    letter-spacing: -1px;
    color: #0f1e3a;
}
.profecia-separator {
    width: 1px;
    height: 34px;
    background: #cbd5e1;
}
.profecia-subtitle {
    font-size: 20px;
    color: #475569;
}
.profecia-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 16px 18px;
    margin-bottom: 14px;
    box-shadow: 0 3px 14px rgba(15, 23, 42, 0.07);
    max-width: 100%;
    overflow-x: hidden !important;
}
.profecia-card-title {
    font-size: 18px;
    font-weight: 780;
    color: #172033;
    margin-bottom: 2px;
}
.profecia-card-subtitle {
    font-size: 13px;
    color: #64748b;
    margin-bottom: 10px;
}
.profecia-small-label {
    font-size: 13px;
    color: #334155;
    font-weight: 720;
    margin: 4px 0 6px 0;
}
.profecia-roi-label {
    font-size: 12px;
    color: #64748b;
    font-weight: 700;
    margin-bottom: 3px;
}
.profecia-info {
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    border-radius: 10px;
    color: #2563eb;
    font-size: 13px;
    padding: 9px 12px;
    margin-top: 10px;
}
.profecia-summary {
    background: #f8fbff;
    border: 1px solid #bfdbfe;
    border-radius: 16px;
    padding: 12px 16px;
    margin-top: 6px;
    margin-bottom: 14px;
}
.profecia-summary-grid {
    display: grid;
    grid-template-columns: repeat(6, minmax(110px, 1fr));
    gap: 10px;
    align-items: center;
}
.profecia-summary-item {
    border-left: 1px solid #dbeafe;
    padding-left: 12px;
}
.profecia-summary-label {
    font-size: 12px;
    color: #64748b;
}
.profecia-summary-value {
    font-size: 15px;
    color: #0f172a;
    font-weight: 760;
}
.profecia-chip-row {
    display: flex;
    flex-wrap: wrap;
    gap: 7px;
    margin-top: 6px;
}
.profecia-chip, .profecia-target-chip {
    display: inline-block;
    background: #e8f1ff;
    color: #174ea6;
    border: 1px solid #bfdbfe;
    border-radius: 999px;
    padding: 4px 10px;
    font-size: 12px;
    font-weight: 700;
}
.profecia-target-chip {
    background: #fee2e2;
    color: #7f1d1d;
    border-color: #fecaca;
    font-size: 13px;
    padding: 6px 14px;
}
.profecia-chip-empty {
    background: #f1f5f9;
    color: #64748b;
    border-color: #cbd5e1;
}
.profecia-ok {
    background: #ecfdf5;
    border: 1px solid #bbf7d0;
    color: #166534;
    border-radius: 10px;
    padding: 10px 12px;
    font-size: 13px;
}
.profecia-warning {
    background: #fffbeb;
    border: 1px solid #fde68a;
    color: #92400e;
    border-radius: 10px;
    padding: 10px 12px;
    font-size: 13px;
}
.profecia-error {
    background: #fef2f2;
    border: 1px solid #fecaca;
    color: #991b1b;
    border-radius: 10px;
    padding: 10px 12px;
    font-size: 13px;
}
.profecia-map-legend {
    background: rgba(255,255,255,0.92);
    border: 1px solid #d7dde8;
    border-radius: 10px;
    padding: 8px 10px;
    box-shadow: 0 2px 8px rgba(15,23,42,0.15);
    font-size: 12px;
    color: #334155;
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
}
.profecia-map-legend i {
    width: 13px;
    height: 13px;
    display: inline-block;
    border-radius: 3px;
    margin-right: 5px;
    border: 1px solid rgba(15,23,42,0.18);
    vertical-align: -2px;
}
.profecia-root .widget-box,
.profecia-root .widget-hbox,
.profecia-root .widget-vbox,
.profecia-root .widget-gridbox,
.profecia-root .jupyter-widgets,
.profecia-root .lm-Widget,
.profecia-root .p-Widget {
    max-width: 100% !important;
    overflow-x: hidden !important;
}

/* SelectMultiple / Select de ipywidgets */
.profecia-root select,
.profecia-root select[multiple],
.profecia-root .widget-select select,
.profecia-root .widget-select-multiple select {
    width: 100% !important;
    max-width: 100% !important;
    overflow-x: hidden !important;
}

/* Output de Jupyter / VS Code */
.jp-OutputArea-output,
.jp-Cell-outputArea,
.output_subarea {
    max-width: 100% !important;
    overflow-x: hidden !important;
}
</style>
"""


def _card(title: str, subtitle: str, children: list[Any]) -> widgets.VBox:
    box = widgets.VBox(
        [
            widgets.HTML(f'<div class="profecia-card-title">{title}</div><div class="profecia-card-subtitle">{subtitle}</div>'),
            *children,
        ],
        layout=widgets.Layout(width="100%"),
    )
    box.add_class("profecia-card")
    return box


def _labels_for_categorical_filters(
    categorical_filters: dict[str, list[int]],
    categorical_catalog: dict[str, dict[str, Any]],
) -> str:
    parts: list[str] = []
    for name, values in categorical_filters.items():
        if not values:
            continue
        meta = categorical_catalog.get(name, {})
        label = meta.get("label", name)
        classes = meta.get("classes", {}) or {}
        selected_labels = [str(classes.get(int(v), v)) for v in values]
        parts.append(f"{label}: {', '.join(selected_labels)}")
    return " · ".join(parts) if parts else "sin filtros categóricos"


class ProfeciaConfiguratorUI:
    """Interfaz completa para configurar data.toml y previsualizar ROI/filtros."""

    def __init__(
        self,
        config_path: str | Path,
        paths_config_path: str | Path,
        preview_dir: str | Path,
        build_command: str | None = None,
    ) -> None:
        self.config_path = Path(config_path)
        self.paths_config_path = Path(paths_config_path)
        self.preview_dir = Path(preview_dir)
        self.build_command = build_command

        self.paths_config = load_paths_config(self.paths_config_path)
        if self.config_path.exists():
            self.config = load_config(self.config_path)
        else:
            self.config = {}

        self.variable_groups = get_predictor_groups(self.paths_config)
        self.target_names = get_target_names(self.paths_config)
        self.available_variables = get_all_available_variables(self.paths_config, include_target=True)
        self.binary_catalog = get_binary_mask_catalog(self.paths_config, only_enabled=True)
        self.categorical_catalog = get_categorical_mask_catalog(self.paths_config, only_enabled=True)

        self.state = ProfeciaUIState.from_config(
            self.config,
            available_variables=self.available_variables,
            target_names=self.target_names,
            default_mask_names=default_binary_mask_names(self.binary_catalog),
            default_categorical_filters=default_categorical_filters(self.categorical_catalog),
        )

        self.widgets = ProfeciaWidgets(
            self.state,
            binary_mask_catalog=self.binary_catalog,
            categorical_mask_catalog=self.categorical_catalog,
            variable_groups=self.variable_groups,
            target_names=self.target_names,
        )

        self.lat, self.lon = load_grid(self.preview_dir)
        self.shape = (self.lat.size, self.lon.size)
        self.spatial_masks = load_spatial_masks(
            self.preview_dir,
            self.paths_config,
            self.shape,
            self.binary_catalog,
            self.categorical_catalog,
        )

        initial_layer, _ = load_preview_layer(
            self.preview_dir,
            self.state.temporal_resolution,
            self.state.data_value_type,
            self.lat,
            self.lon,
        )
        initial_valid = initial_layer == initial_layer
        initial_png = array_to_png_data_url(initial_layer, initial_valid, self.lat, self.state.data_value_type)
        self.map_view = ProfeciaMapView(
            self.lat,
            self.lon,
            initial_png_url=initial_png,
            roi_bounds=self.state.roi_bounds_leaflet,
        )

        self.summary_html = widgets.HTML(value="")
        self.validation_html = widgets.HTML(value="")
        self.log_output = widgets.Output(layout=widgets.Layout(border="1px solid #e2e8f0", padding="8px"))
        self.mask_info = widgets.HTML(value="")
        self.preview_info = widgets.HTML(value="")

        self.widgets.register_on_change(self._on_widget_change)
        self.widgets.btn_update.on_click(lambda _: self.update_preview())
        self.widgets.btn_save.on_click(lambda _: self.save_current_config())

        self.root = self._build_layout()
        self.update_preview()

    def _on_widget_change(self) -> None:
        if self.widgets.auto_update.value:
            self.update_preview()

    def _build_header(self) -> widgets.HTML:
        return widgets.HTML(
            """
            <div class="profecia-header">
              <div class="profecia-logo-dot">⌁</div>
              <div class="profecia-title">PROFECIA</div>
              <div class="profecia-separator"></div>
              <div class="profecia-subtitle">Configurador interactivo del dataset</div>
            </div>
            """
        )

    def _build_layout(self) -> widgets.VBox:
        config_card = _card(
            "Configuración de datos",
            "Selecciona la representación temporal y el tipo de señal",
            [
                widgets.VBox(
                    [
                        widgets.HTML('<div class="profecia-small-label">Resolución temporal</div>'),
                        self.widgets.temporal_resolution,
                        widgets.HTML('<div class="profecia-small-label">Tipo de dato</div>'),
                        self.widgets.data_value_type,
                    ],
                    layout=widgets.Layout(gap="8px", width="100%"),
                )
            ],
        )

        domain_card = _card(
            "Dominio temporal y espacial",
            "Control fino del periodo de estudio y ROI",
            [
                widgets.HBox(
                    [
                        widgets.VBox(
                            [widgets.HTML('<div class="profecia-small-label">Periodo de análisis</div>'), self.widgets.period],
                            layout=widgets.Layout(width="46%", min_width="260px"),
                        ),
                        widgets.VBox(
                            [
                                widgets.HTML('<div class="profecia-small-label">ROI geográfico</div>'),
                                widgets.HBox(
                                    [
                                        self.widgets.roi_field("Lat min", self.widgets.lat_min),
                                        self.widgets.roi_field("Lat max", self.widgets.lat_max),
                                    ],
                                    layout=widgets.Layout(gap="12px", flex_flow="row wrap"),
                                ),
                                widgets.HBox(
                                    [
                                        self.widgets.roi_field("Lon min", self.widgets.lon_min),
                                        self.widgets.roi_field("Lon max", self.widgets.lon_max),
                                    ],
                                    layout=widgets.Layout(gap="12px", flex_flow="row wrap"),
                                ),
                            ],
                            layout=widgets.Layout(width="54%", min_width="290px"),
                        ),
                    ],
                    layout=widgets.Layout(gap="18px", flex_flow="row wrap"),
                )
            ],
        )

        mask_card = _card(
            "Filtros espaciales",
            "Máscaras binarias y filtros categóricos del dominio efectivo",
            [
                widgets.HTML('<div class="profecia-small-label">Máscaras binarias</div>'),
                self.widgets.binary_mask_grid(),
                widgets.HTML('<div class="profecia-small-label">Filtros categóricos</div>'),
                self.widgets.categorical_filter_boxes(),
                self.mask_info,
            ],
        )

        map_card = _card(
            "Mapa interactivo de dominio",
            "Previsualización de ROI, tierra/mar y filtros activos",
            [self.map_view.widget, self.preview_info],
        )

        variables_card = _card(
            "Variables y objetivo",
            "Explora y selecciona predictores del dataset. El target queda fijado a LAI.",
            [
                self.widgets.variable_search,
                widgets.HBox(
                    [
                        widgets.VBox([self.widgets.variables_accordion, self.widgets.selected_chips], layout=widgets.Layout(width="74%")),
                        widgets.VBox([self.widgets.target], layout=widgets.Layout(width="26%")),
                    ],
                    layout=widgets.Layout(gap="18px", flex_flow="row wrap"),
                ),
            ],
        )

        left_col = widgets.VBox([config_card, domain_card, mask_card], layout=widgets.Layout(width="100%"))
        right_col = widgets.VBox([map_card, variables_card], layout=widgets.Layout(width="100%"))
        main_grid = widgets.GridBox(
            [left_col, right_col],
            layout=widgets.Layout(
                width="100%",
                grid_template_columns="minmax(340px, 0.9fr) minmax(480px, 1.1fr)",
                grid_gap="18px",
                overflow="hidden",
            ),
        )

        summary_box = widgets.VBox([self.summary_html, self.validation_html], layout=widgets.Layout(width="100%"))
        summary_box.add_class("profecia-summary")

        action_card = _card(
            "Acciones",
            "Guardar el TOML o lanzar el pipeline real solo cuando la configuración esté validada",
            [self.widgets.action_bar(), self.log_output],
        )

        root = widgets.VBox([widgets.HTML(CSS), self._build_header(), main_grid, summary_box, action_card], layout=widgets.Layout(width="100%", overflow="hidden"))
        root.add_class("profecia-root")
        return root

    def read_state(self) -> ProfeciaUIState:
        self.state = self.widgets.read_state()
        return self.state

    def _compute_current_masks(self, state: ProfeciaUIState):
        return create_final_valid_mask(
            selected_binary_masks=state.mask_names,
            categorical_filters=state.categorical_filters,
            spatial_masks=self.spatial_masks,
            lat=self.lat,
            lon=self.lon,
            lat_min=state.lat_min,
            lat_max=state.lat_max,
            lon_min=state.lon_min,
            lon_max=state.lon_max,
            binary_catalog=self.binary_catalog,
            categorical_catalog=self.categorical_catalog,
        )

    def update_preview(self) -> None:
        state = self.read_state()

        try:
            valid_mask, roi_mask, skipped_filters = self._compute_current_masks(state)
            validation = validate_state(
                state,
                lat=self.lat,
                lon=self.lon,
                valid_mask=valid_mask,
                available_variables=self.available_variables,
                binary_mask_catalog=self.binary_catalog,
                categorical_mask_catalog=self.categorical_catalog,
                skipped_filters=skipped_filters,
            )
        except Exception as exc:
            self.validation_html.value = f'<div class="profecia-error"><b>Error al actualizar:</b> {exc}</div>'
            return

        try:
            layer, preview_msg = load_preview_layer(
                self.preview_dir,
                state.temporal_resolution,
                state.data_value_type,
                self.lat,
                self.lon,
            )
            png_url = array_to_png_data_url(layer, valid_mask, self.lat, state.data_value_type)
            self.map_view.update_overlay(png_url)
            self.map_view.update_roi(state.roi_bounds_leaflet)
        except Exception as exc:
            preview_msg = "No se pudo actualizar la capa de mapa."
            validation.errors.append(f"No se pudo actualizar la capa de mapa: {exc}")

        stats = valid_cell_stats(valid_mask)
        missing_text = ""
        if self.spatial_masks.missing:
            missing_text = " · máscaras no cargadas: " + ", ".join(self.spatial_masks.missing)
        skipped_text = ""
        if skipped_filters:
            skipped_text = " · filtros omitidos: " + ", ".join(skipped_filters)

        binary_text = ", ".join(state.mask_names) if state.mask_names else "sin máscaras binarias"
        categorical_text = _labels_for_categorical_filters(state.categorical_filters, self.categorical_catalog)
        self.mask_info.value = (
            f'<div class="profecia-info">Vista previa: binarias: <b>{binary_text}</b> · '
            f'categóricas: <b>{categorical_text}</b> · '
            f'celdas válidas: <b>{stats["valid"]:,}</b> / {stats["total"]:,} '
            f'({stats["pct"]:.2f} %){missing_text}{skipped_text}</div>'
        )
        self.preview_info.value = f'<div class="profecia-info">{preview_msg}</div>'
        self.summary_html.value = self._summary_html(state, stats)
        self.validation_html.value = validation.as_html()

    def _summary_html(self, state: ProfeciaUIState, stats: dict[str, float | int]) -> str:
        masks = ", ".join(state.mask_names) if state.mask_names else "ninguna"
        categorical = _labels_for_categorical_filters(state.categorical_filters, self.categorical_catalog)
        return f"""
        <div class="profecia-summary-grid">
            <div><b>Resumen de configuración</b></div>
            <div class="profecia-summary-item"><div class="profecia-summary-label">Resolución temporal</div><div class="profecia-summary-value">{state.temporal_resolution}</div></div>
            <div class="profecia-summary-item"><div class="profecia-summary-label">Tipo de dato</div><div class="profecia-summary-value">{state.data_value_type}</div></div>
            <div class="profecia-summary-item"><div class="profecia-summary-label">Periodo</div><div class="profecia-summary-value">{state.start_year} – {state.end_year_inclusive}</div></div>
            <div class="profecia-summary-item"><div class="profecia-summary-label">Máscaras binarias</div><div class="profecia-summary-value">{masks}</div></div>
            <div class="profecia-summary-item"><div class="profecia-summary-label">Target / variables</div><div class="profecia-summary-value">{state.target_name} / {len(state.variable_names)}</div></div>
            <div class="profecia-summary-item"><div class="profecia-summary-label">Celdas válidas</div><div class="profecia-summary-value">{stats['valid']:,} ({stats['pct']:.1f} %)</div></div>
        </div>
        <div class="profecia-info">Filtros categóricos: {categorical}</div>
        """

    def save_current_config(self) -> None:
        state = self.read_state()
        valid_mask, _, skipped_filters = self._compute_current_masks(state)
        validation = validate_state(
            state,
            lat=self.lat,
            lon=self.lon,
            valid_mask=valid_mask,
            available_variables=self.available_variables,
            binary_mask_catalog=self.binary_catalog,
            categorical_mask_catalog=self.categorical_catalog,
            skipped_filters=skipped_filters,
        )
        self.validation_html.value = validation.as_html()
        if not validation.ok:
            with self.log_output:
                print("No se ha guardado: la configuración contiene errores.")
            return

        self.config = update_config_from_state(self.config, state)
        save_config(self.config, self.config_path, backup=True)
        with self.log_output:
            print(f"Configuración guardada en: {self.config_path}")
            print(f"Copia previa: {self.config_path}.bak")

    def display(self) -> None:
        display(self.root)
