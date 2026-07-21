from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelArtifactConfig:
    mode: str = "attached"
    download_url: str | None = None
    landing_page: str | None = None
    identifier: str | None = None


@dataclass(frozen=True)
class CroissantConfig:
    mode: str = "descriptive"
    include_split_manifest: bool = False
    split_manifest_format: str = "npz"


@dataclass(frozen=True)
class CodeConfig:
    mode: str = "reference"
    require_clean_repository: bool = False


@dataclass(frozen=True)
class LicensesConfig:
    metadata: str | None = "https://spdx.org/licenses/CC0-1.0.html"
    code: str | None = None
    model: str | None = None
    results: str | None = None
    derived_dataset: str | None = None


@dataclass(frozen=True)
class EnvironmentConfig:
    lock_file: Path | None = None
    container_image: str | None = None
    container_digest: str | None = None
    container_platform: str | None = None
    container_runtime: str | None = None
    python_version: str | None = None


@dataclass(frozen=True)
class UpstreamConfig:
    identifier: str | None = None
    landing_page: str | None = None
    metadata_url: str | None = None
    archive_url: str | None = None
    cache_dir: Path | None = None
    download_missing: bool = True
    verify_checksums: bool = True
    offline: bool = False


@dataclass(frozen=True)
class InputsConfig:
    require_upstream_checksum: bool = False
    allow_derived_local_inputs: bool = True


@dataclass(frozen=True)
class VerificationConfig:
    expected_results: str = "reproducibility/expected_results.json"
    fail_on_metric_difference: bool = True
    metric_absolute_tolerance: float = 0.0001


@dataclass(frozen=True)
class ReproducibilityConfig:
    """Resolved settings for the optional per-run RO-Crate."""

    create_rocrate: bool = False
    reproducible_run: bool = False
    output_dir: Path | None = None
    upstream_rocrate: str | None = None
    input_catalogue: Path | None = None
    model_artifact: ModelArtifactConfig = ModelArtifactConfig()
    croissant: CroissantConfig = CroissantConfig()
    code: CodeConfig = CodeConfig()
    licenses: LicensesConfig = LicensesConfig()
    environment: EnvironmentConfig = EnvironmentConfig()
    upstream: UpstreamConfig = UpstreamConfig()
    inputs: InputsConfig = InputsConfig()
    verification: VerificationConfig = VerificationConfig()

    @classmethod
    def from_mapping(cls, raw: dict[str, Any] | None, base_dir: Path) -> "ReproducibilityConfig":
        raw = raw or {}

        def path_value(name: str) -> Path | None:
            value = raw.get(name)
            if value in (None, ""):
                return None
            path = Path(str(value)).expanduser()
            return path if path.is_absolute() else (base_dir / path).resolve()

        upstream = raw.get("upstream_rocrate")
        if upstream:
            candidate = Path(str(upstream)).expanduser()
            if not _looks_like_remote(str(upstream)):
                upstream = str(candidate if candidate.is_absolute() else (base_dir / candidate).resolve())

        artifact_raw = raw.get("model_artifact") or {}
        artifact_mode = str(artifact_raw.get("mode", "attached")).lower().strip()
        if artifact_mode not in {"attached", "external", "metadata_only"}:
            raise ValueError("reproducibility.model_artifact.mode must be attached, external or metadata_only")
        croissant_raw = raw.get("croissant") or {}
        croissant_mode = str(croissant_raw.get("mode", "descriptive")).lower().strip()
        if croissant_mode not in {"descriptive", "materialized"}:
            raise ValueError("reproducibility.croissant.mode must be descriptive or materialized")
        manifest_format = str(croissant_raw.get("split_manifest_format", "npz")).lower().strip()
        if manifest_format != "npz":
            raise ValueError("reproducibility.croissant.split_manifest_format currently supports only npz")
        code_raw = raw.get("code") or {}
        code_mode = str(code_raw.get("mode", "reference")).lower().strip()
        if code_mode not in {"reference", "patch", "snapshot"}:
            raise ValueError("reproducibility.code.mode must be reference, patch or snapshot")
        licenses_raw = raw.get("licenses") or {}
        environment_raw = raw.get("environment") or {}
        upstream_raw = raw.get("upstream") or {}
        inputs_raw = raw.get("inputs") or {}
        verification_raw = raw.get("verification") or {}
        lock_file_value = environment_raw.get("lock_file")
        lock_file = None
        if lock_file_value:
            candidate = Path(str(lock_file_value)).expanduser()
            lock_file = candidate if candidate.is_absolute() else (base_dir / candidate).resolve()
        container_image = _optional_text(environment_raw.get("container_image"))
        container_digest = _optional_text(environment_raw.get("container_digest"))
        if bool(container_digest) != bool(container_image):
            raise ValueError(
                "reproducibility.environment.container_image and container_digest must be configured together"
            )
        cache_value = upstream_raw.get("cache_dir")
        cache_dir = None
        if cache_value:
            candidate = Path(str(cache_value)).expanduser()
            cache_dir = candidate if candidate.is_absolute() else (base_dir / candidate).resolve()
        nested_metadata = _optional_text(upstream_raw.get("metadata_url"))
        if nested_metadata:
            if not _looks_like_remote(nested_metadata):
                candidate = Path(nested_metadata).expanduser()
                nested_metadata = str(candidate if candidate.is_absolute() else (base_dir / candidate).resolve())
            upstream = nested_metadata

        return cls(
            create_rocrate=bool(raw.get("create_rocrate", False)),
            reproducible_run=bool(raw.get("reproducible_run", False)),
            output_dir=path_value("output_dir"),
            upstream_rocrate=str(upstream) if upstream else None,
            input_catalogue=path_value("input_catalogue"),
            model_artifact=ModelArtifactConfig(
                mode=artifact_mode,
                download_url=artifact_raw.get("download_url") or None,
                landing_page=artifact_raw.get("landing_page") or None,
                identifier=artifact_raw.get("identifier") or None,
            ),
            croissant=CroissantConfig(
                mode=croissant_mode,
                include_split_manifest=bool(croissant_raw.get("include_split_manifest", False)),
                split_manifest_format=manifest_format,
            ),
            code=CodeConfig(
                mode=code_mode,
                require_clean_repository=bool(code_raw.get("require_clean_repository", False)),
            ),
            licenses=LicensesConfig(
                metadata=_optional_text(licenses_raw.get("metadata", "https://spdx.org/licenses/CC0-1.0.html")),
                code=_optional_text(licenses_raw.get("code")),
                model=_optional_text(licenses_raw.get("model")),
                results=_optional_text(licenses_raw.get("results")),
                derived_dataset=_optional_text(licenses_raw.get("derived_dataset")),
            ),
            environment=EnvironmentConfig(
                lock_file=lock_file,
                container_image=container_image,
                container_digest=container_digest,
                container_platform=_optional_text(environment_raw.get("container_platform")),
                container_runtime=_optional_text(environment_raw.get("container_runtime")),
                python_version=_optional_text(environment_raw.get("python_version")),
            ),
            upstream=UpstreamConfig(
                identifier=_optional_text(upstream_raw.get("identifier")),
                landing_page=_optional_text(upstream_raw.get("landing_page")),
                metadata_url=nested_metadata,
                archive_url=_optional_text(upstream_raw.get("archive_url")),
                cache_dir=cache_dir,
                download_missing=bool(upstream_raw.get("download_missing", True)),
                verify_checksums=bool(upstream_raw.get("verify_checksums", True)),
                offline=bool(upstream_raw.get("offline", False)),
            ),
            inputs=InputsConfig(
                require_upstream_checksum=bool(inputs_raw.get("require_upstream_checksum", False)),
                allow_derived_local_inputs=bool(inputs_raw.get("allow_derived_local_inputs", True)),
            ),
            verification=VerificationConfig(
                expected_results=str(verification_raw.get("expected_results", "reproducibility/expected_results.json")),
                fail_on_metric_difference=bool(verification_raw.get("fail_on_metric_difference", True)),
                metric_absolute_tolerance=float(verification_raw.get("metric_absolute_tolerance", 0.0001)),
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "create_rocrate": self.create_rocrate,
            "reproducible_run": self.reproducible_run,
            "output_dir": self.output_dir,
            "upstream_rocrate": self.upstream_rocrate,
            "input_catalogue": self.input_catalogue,
            "model_artifact": self.model_artifact,
            "croissant": self.croissant,
            "code": self.code,
            "licenses": self.licenses,
            "environment": self.environment,
            "upstream": self.upstream,
            "inputs": self.inputs,
            "verification": self.verification,
        }


def _looks_like_remote(value: str) -> bool:
    lowered = value.lower().strip()
    return lowered.startswith(("http://", "https://", "doi:", "10."))


def _optional_text(value: Any) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None
