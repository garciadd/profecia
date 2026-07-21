"""Optional provenance metadata for PROFECIA training runs."""

from .config import (
    CodeConfig, CroissantConfig, EnvironmentConfig, InputsConfig, LicensesConfig,
    ModelArtifactConfig, ReproducibilityConfig, UpstreamConfig, VerificationConfig,
)


def create_training_rocrate(*args, **kwargs):
    from .rocrate_writer import create_training_rocrate as implementation

    return implementation(*args, **kwargs)


def validate_rocrate_jsonld(*args, **kwargs):
    from .rocrate_writer import validate_rocrate_jsonld as implementation

    return implementation(*args, **kwargs)


__all__ = [
    "CroissantConfig",
    "CodeConfig",
    "ModelArtifactConfig",
    "LicensesConfig",
    "EnvironmentConfig",
    "UpstreamConfig",
    "InputsConfig",
    "VerificationConfig",
    "ReproducibilityConfig",
    "create_training_rocrate",
    "validate_rocrate_jsonld",
]
