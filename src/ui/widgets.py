"""Factoría de widgets para la interfaz PROFECIA."""

from __future__ import annotations

from typing import Any, Callable

import ipywidgets as widgets

from .paths_io import flatten_predictor_groups
from .state import ProfeciaUIState


class ProfeciaWidgets:
    """Crea y mantiene los widgets de configuración."""

    def __init__(
        self,
        initial_state: ProfeciaUIState,
        *,
        binary_mask_catalog: dict[str, dict[str, Any]],
        categorical_mask_catalog: dict[str, dict[str, Any]],
        variable_groups: dict[str, list[str]],
        target_names: list[str] | None = None,
    ) -> None:
        self.initial_state = initial_state
        self.binary_mask_catalog = binary_mask_catalog
        self.categorical_mask_catalog = categorical_mask_catalog
        self.variable_groups = variable_groups
        self.target_names = target_names or ["LAI"]
        self.target_name = "LAI" if "LAI" in self.target_names else self.target_names[0]
        self.all_predictors = flatten_predictor_groups(self.variable_groups)

        self._callbacks: list[Callable[[], None]] = []

        self.temporal_resolution = widgets.ToggleButtons(
            options=[("Mensual", "monthly"), ("Anual", "annual")],
            value=initial_state.temporal_resolution,
            description="",
            button_style="",
            layout=widgets.Layout(width="100%"),
        )
        self.data_value_type = widgets.ToggleButtons(
            options=[("Real", "real"), ("Anomalía", "anomaly"), ("Tendencia", "trend")],
            value=initial_state.data_value_type,
            description="",
            layout=widgets.Layout(width="100%"),
        )

        year_options = list(range(1982, 2023))
        start = initial_state.start_year if initial_state.start_year in year_options else 1982
        end = initial_state.end_year_inclusive if initial_state.end_year_inclusive in year_options else 2022
        self.period = widgets.SelectionRangeSlider(
            options=year_options,
            index=(year_options.index(start), year_options.index(end)),
            description="",
            continuous_update=False,
            layout=widgets.Layout(width="100%"),
        )

        float_layout = widgets.Layout(width="120px")
        self.lat_min = widgets.BoundedFloatText(
            value=initial_state.lat_min,
            min=-90,
            max=90,
            step=0.5,
            description="",
            layout=float_layout,
            continuous_update=False,
        )
        self.lat_max = widgets.BoundedFloatText(
            value=initial_state.lat_max,
            min=-90,
            max=90,
            step=0.5,
            description="",
            layout=float_layout,
            continuous_update=False,
        )
        self.lon_min = widgets.BoundedFloatText(
            value=initial_state.lon_min,
            min=-180,
            max=180,
            step=0.5,
            description="",
            layout=float_layout,
            continuous_update=False,
        )
        self.lon_max = widgets.BoundedFloatText(
            value=initial_state.lon_max,
            min=-180,
            max=180,
            step=0.5,
            description="",
            layout=float_layout,
            continuous_update=False,
        )

        self.mask_checkboxes: dict[str, widgets.Checkbox] = {}
        for name, meta in self.binary_mask_catalog.items():
            label = str(meta.get("label", name))
            cb = widgets.Checkbox(
                value=name in initial_state.mask_names,
                description=f"{label}",
                indent=False,
                layout=widgets.Layout(width="100%"),
            )
            self.mask_checkboxes[name] = cb

        self.categorical_selectors: dict[str, widgets.SelectMultiple] = {}
        for name, meta in self.categorical_mask_catalog.items():
            classes = meta.get("classes", {}) or {}
            selected = tuple(int(v) for v in initial_state.categorical_filters.get(name, []) if int(v) in classes)
            selector = widgets.SelectMultiple(
                options=[(f"{class_id} · {label}", int(class_id)) for class_id, label in classes.items()],
                value=selected,
                description="",
                rows=min(max(len(classes), 3), 7),
                layout=widgets.Layout(width="100%"),
            )
            self.categorical_selectors[name] = selector

        self.variable_search = widgets.Text(
            value="",
            placeholder="Buscar predictor...",
            description="",
            layout=widgets.Layout(width="100%"),
        )
        self.variable_checkboxes: dict[str, widgets.Checkbox] = {}
        self.variable_group_boxes: list[widgets.VBox] = []
        self.variables_accordion = self._build_variables_accordion(initial_state.predictor_names)
        self.selected_chips = widgets.HTML(value="")
        self.target = widgets.HTML(value=self._target_html(self.target_name))

        self.auto_update = widgets.Checkbox(value=True, description="Autoactualizar vista previa", indent=False)
        self.btn_update = widgets.Button(description="Actualizar mapa", button_style="info", icon="refresh")
        self.btn_save = widgets.Button(description="Guardar data.toml", button_style="success", icon="save")

        self._wire_internal_events()
        self._update_selected_chips()

    def _target_html(self, target_name: str) -> str:
        return (
            '<div class="profecia-small-label">Target</div>'
            f'<div class="profecia-target-chip">{target_name}</div>'
        )

    def _build_variables_accordion(self, selected_predictors: list[str]) -> widgets.Accordion:
        children: list[widgets.VBox] = []
        titles: list[str] = []

        for group, variables in self.variable_groups.items():
            boxes: list[widgets.Checkbox] = []
            for var in variables:
                checkbox = widgets.Checkbox(
                    value=var in selected_predictors,
                    description=var,
                    indent=False,
                    layout=widgets.Layout(width="150px"),
                )
                self.variable_checkboxes[var] = checkbox
                boxes.append(checkbox)

            grid = widgets.GridBox(
                boxes,
                layout=widgets.Layout(
                    grid_template_columns="repeat(3, minmax(110px, 1fr))",
                    grid_gap="4px 12px",
                    width="100%",
                ),
            )
            box = widgets.VBox([grid])
            children.append(box)
            titles.append(group)
            self.variable_group_boxes.append(box)

        accordion = widgets.Accordion(children=children, selected_index=0 if children else None)
        for i, title in enumerate(titles):
            accordion.set_title(i, title)
        return accordion

    def _wire_internal_events(self) -> None:
        for widget in [
            self.temporal_resolution,
            self.data_value_type,
            self.period,
            self.lat_min,
            self.lat_max,
            self.lon_min,
            self.lon_max,
        ]:
            widget.observe(lambda change: self._emit_change(), names="value")

        for checkbox in self.mask_checkboxes.values():
            checkbox.observe(lambda change: self._emit_change(), names="value")

        for selector in self.categorical_selectors.values():
            selector.observe(lambda change: self._emit_change(), names="value")

        for checkbox in self.variable_checkboxes.values():
            checkbox.observe(lambda change: self._on_variable_change(), names="value")

        self.variable_search.observe(lambda change: self._filter_variables(), names="value")

    def register_on_change(self, callback: Callable[[], None]) -> None:
        self._callbacks.append(callback)

    def _emit_change(self) -> None:
        for callback in self._callbacks:
            callback()

    def _on_variable_change(self) -> None:
        self._update_selected_chips()
        self._emit_change()

    def _filter_variables(self) -> None:
        query = self.variable_search.value.strip().lower()
        for var, checkbox in self.variable_checkboxes.items():
            checkbox.layout.display = None if query in var.lower() else "none"

    def _update_selected_chips(self) -> None:
        selected = self.selected_predictors
        chips = "".join(f'<span class="profecia-chip">{v}</span>' for v in selected)
        if not chips:
            chips = '<span class="profecia-chip profecia-chip-empty">sin predictores</span>'
        self.selected_chips.value = (
            f'<div class="profecia-small-label">Predictores seleccionados ({len(selected)})</div>'
            f'<div class="profecia-chip-row">{chips}</div>'
        )

    @property
    def selected_binary_masks(self) -> list[str]:
        return [name for name, checkbox in self.mask_checkboxes.items() if checkbox.value]

    @property
    def selected_categorical_filters(self) -> dict[str, list[int]]:
        return {
            name: [int(v) for v in selector.value]
            for name, selector in self.categorical_selectors.items()
        }

    @property
    def selected_predictors(self) -> list[str]:
        return [var for var, checkbox in self.variable_checkboxes.items() if checkbox.value]

    @property
    def selected_variables(self) -> list[str]:
        return [self.target_name] + self.selected_predictors

    def read_state(self) -> ProfeciaUIState:
        start_year, end_year = self.period.value
        return ProfeciaUIState(
            temporal_resolution=self.temporal_resolution.value,
            data_value_type=self.data_value_type.value,
            start_year=int(start_year),
            end_year_inclusive=int(end_year),
            lat_min=float(self.lat_min.value),
            lat_max=float(self.lat_max.value),
            lon_min=float(self.lon_min.value),
            lon_max=float(self.lon_max.value),
            mask_names=self.selected_binary_masks,
            categorical_filters=self.selected_categorical_filters,
            variable_names=self.selected_variables,
            target_name=self.target_name,
            dtype=self.initial_state.dtype,
        )

    def roi_field(self, label: str, widget: widgets.Widget) -> widgets.VBox:
        return widgets.VBox(
            [widgets.HTML(f'<div class="profecia-roi-label">{label}</div>'), widget],
            layout=widgets.Layout(width="130px"),
        )

    def binary_mask_grid(self) -> widgets.GridBox:
        return widgets.GridBox(
            list(self.mask_checkboxes.values()),
            layout=widgets.Layout(
                grid_template_columns="repeat(2, minmax(180px, 1fr))",
                grid_gap="8px 18px",
                width="100%",
            ),
        )

    def categorical_filter_boxes(self) -> widgets.VBox:
        children: list[widgets.Widget] = []
        for name, selector in self.categorical_selectors.items():
            meta = self.categorical_mask_catalog[name]
            children.append(
                widgets.VBox(
                    [
                        widgets.HTML(
                            f'<div class="profecia-small-label">{meta.get("label", name)}</div>'
                            f'<div class="profecia-card-subtitle">{meta.get("description", "")} Sin selección = no filtrar.</div>'
                        ),
                        selector,
                    ],
                    layout=widgets.Layout(width="100%"),
                )
            )
        if not children:
            children.append(widgets.HTML('<div class="profecia-info">No hay máscaras categóricas definidas en paths.toml.</div>'))
        return widgets.VBox(children, layout=widgets.Layout(gap="10px", width="100%"))

    def action_bar(self) -> widgets.HBox:
        return widgets.HBox(
            [
                self.auto_update,
                self.btn_update,
                self.btn_save,
            ],
            layout=widgets.Layout(
                align_items="center",
                gap="12px",
                flex_flow="row wrap",
            ),
        )