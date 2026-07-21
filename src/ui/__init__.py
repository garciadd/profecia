"""Interfaz interactiva para configurar datasets de PROFECIA."""

from .state import ProfeciaUIState

__all__ = ["ProfeciaConfiguratorUI", "ProfeciaUIState"]


def __getattr__(name: str):
    if name == "ProfeciaConfiguratorUI":
        from .layout import ProfeciaConfiguratorUI

        return ProfeciaConfiguratorUI

    raise AttributeError(name)
