"""Utilidades para variables dinámicas de PROFECIA.

Las variables disponibles ya no se declaran aquí de forma fija. La fuente de verdad
es config/paths.toml y este módulo solo ofrece funciones auxiliares para la UI.
"""

from __future__ import annotations

from collections import OrderedDict


def flatten_variable_groups(variable_groups: dict[str, list[str]]) -> list[str]:
    """Devuelve una lista de variables sin duplicados, preservando el orden."""
    out: list[str] = []
    for variables in variable_groups.values():
        for var in variables:
            if var not in out:
                out.append(var)
    return out


def sanitize_selected_variables(
    selected_variables: list[str],
    available_predictors: list[str],
    target_name: str = "LAI",
) -> list[str]:
    """Limpia una selección frente al catálogo actual.

    El target se mantiene siempre en primera posición. Los predictores que no existan
    en paths.toml se descartan. Esto solo afecta a configuraciones antiguas o editadas
    manualmente.
    """
    out = [target_name]
    for var in selected_variables:
        if var == target_name:
            continue
        if var in available_predictors and var not in out:
            out.append(var)
    return out


def selected_predictors_from_variable_names(variable_names: list[str], target_name: str = "LAI") -> list[str]:
    return [var for var in variable_names if var != target_name]


def empty_variable_groups() -> "OrderedDict[str, list[str]]":
    return OrderedDict()
