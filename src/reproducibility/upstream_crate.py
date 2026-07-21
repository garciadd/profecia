from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

LOGGER = logging.getLogger(__name__)


class UpstreamResolutionError(ValueError):
    """A required training input cannot be tied to upstream provenance."""


@dataclass(frozen=True)
class UpstreamReference:
    identifier: str
    metadata: dict[str, Any] | None
    metadata_url: str | None = None
    accessed_at: str | None = None
    metadata_sha256: str | None = None
    raw_metadata: bytes | None = None


@dataclass(frozen=True)
class InputSpec:
    key: str
    kind: str
    local_file: str
    upstream_entity_id: str | None = None
    netcdf_variable: str | None = None
    expected_sha256: str | None = None
    derived_from_upstream_entity_id: str | None = None
    derivation_type: str | None = None
    class_value: int | None = None
    class_label: str | None = None
    source_local_file: str | None = None
    role: str | None = None
    frequency: str | None = None
    required: bool = True


@dataclass(frozen=True)
class ResolvedInput:
    spec: InputSpec
    upstream_entity: dict[str, Any]
    match_method: str
    local_path: Path | None
    local_sha256: str | None
    source_local_sha256: str | None
    checksum_status: str
    derived_entity_id: str | None = None
    locally_modified_entity_id: str | None = None

    @property
    def training_entity_id(self) -> str:
        return self.derived_entity_id or self.locally_modified_entity_id or self.upstream_entity["@id"]

    @property
    def is_derived(self) -> bool:
        return self.derived_entity_id is not None


def load_upstream_rocrate(reference: str | None, timeout: int = 15) -> UpstreamReference | None:
    if not reference:
        return None
    reference = reference.strip()
    path = Path(reference).expanduser()
    if path.exists():
        metadata_path = path / "ro-crate-metadata.json" if path.is_dir() else path
        raw = metadata_path.read_bytes()
        metadata = json.loads(raw)
        return UpstreamReference(
            metadata_path.resolve().as_uri(), metadata,
            metadata_url=metadata_path.resolve().as_uri(),
            accessed_at=datetime.now(UTC).isoformat(),
            metadata_sha256=_sha256_file(metadata_path),
            raw_metadata=raw,
        )
    if reference.lower().startswith("doi:"):
        return UpstreamReference(f"https://doi.org/{reference[4:].strip()}", None)
    if reference.startswith("10."):
        return UpstreamReference(f"https://doi.org/{reference}", None)
    if reference.startswith(("http://", "https://")):
        try:
            metadata, metadata_url, checksum, raw = _load_remote_metadata_details(reference, timeout)
            return UpstreamReference(
                reference, metadata, metadata_url=metadata_url,
                accessed_at=datetime.now(UTC).isoformat(), metadata_sha256=checksum,
                raw_metadata=raw,
            )
        except Exception:
            LOGGER.warning("Could not read upstream RO-Crate metadata from %s", reference, exc_info=True)
            return UpstreamReference(reference, None)
    raise FileNotFoundError(f"Upstream RO-Crate not found: {reference}")


def catalogue_inputs(catalogue_path: Path | None) -> list[InputSpec]:
    """Build the complete required-input inventory from the data catalogue."""
    if catalogue_path is None:
        return []
    with open(catalogue_path, "rb") as handle:
        catalogue = tomllib.load(handle)
    from src.data.catalogue import FILE_MAP, MASK_MAP

    data = catalogue.get("data", {})
    variable_mappings = catalogue.get("variables", {})
    mask_mappings = catalogue.get("masks", {})
    specs: list[InputSpec] = []
    for raw_name in data.get("variable_names", []):
        name = str(raw_name).upper().strip()
        mapping = _mapping(variable_mappings, name)
        local_file = mapping.get("local_file") or FILE_MAP.get(name)
        if not local_file:
            raise UpstreamResolutionError(f"No local_file is configured for required variable {name}")
        specs.append(_input_spec(f"variable:{name}", "variable", local_file, mapping))
    for raw_name in data.get("mask_names", []):
        name = str(raw_name).lower().strip()
        mapping = _mapping(mask_mappings, name)
        local_file = mapping.get("local_file") or MASK_MAP.get(name)
        if not local_file:
            raise UpstreamResolutionError(f"No local_file is configured for required mask {name}")
        specs.append(_input_spec(f"mask:{name}", "mask", local_file, mapping))

    # Backward-compatible generic file catalogue when no [data] inventory exists.
    if not specs:
        generic: list[str] = []
        _walk_file_values(catalogue, generic)
        specs.extend(InputSpec(f"file:{name}", "file", name) for name in dict.fromkeys(generic))
    return specs


def resolve_catalogue_inputs(
    metadata: dict[str, Any] | None,
    specs: list[InputSpec],
    raw_dir: Path | None = None,
    mask_dir: Path | None = None,
) -> list[ResolvedInput]:
    """Resolve every required input; never returns a partially resolved set."""
    if not metadata:
        raise UpstreamResolutionError("Upstream ro-crate-metadata.json is unavailable; required inputs cannot be resolved")
    graph = [entity for entity in metadata.get("@graph", []) if isinstance(entity, dict) and entity.get("@id")]
    resolved: list[ResolvedInput] = []
    errors: list[str] = []
    for spec in specs:
        local_path = _local_path(spec, raw_dir, mask_dir)
        local_sha256 = _sha256_file(local_path) if local_path and local_path.is_file() else None
        source_path = (
            (mask_dir / spec.source_local_file).resolve()
            if spec.derived_from_upstream_entity_id and spec.source_local_file and mask_dir
            else None
        )
        source_local_sha256 = _sha256_file(source_path) if source_path and source_path.is_file() else None
        requested_id = spec.derived_from_upstream_entity_id or spec.upstream_entity_id
        try:
            if requested_id:
                entity = _entity_by_explicit_id(graph, requested_id, spec.key)
                method = "explicit_upstream_entity_id"
            else:
                entity, method = _fallback_entity(graph, spec, local_sha256)
            upstream_sha256 = _checksum(entity)
            expected = spec.expected_sha256 or source_local_sha256 or local_sha256
            if upstream_sha256 and expected:
                checksum_status = "verified" if upstream_sha256.lower() == expected.lower() else "mismatch"
                if checksum_status == "mismatch":
                    LOGGER.warning(
                        "Checksum mismatch for %s: local/configured checksum differs from upstream entity %s",
                        spec.key,
                        entity["@id"],
                    )
            else:
                checksum_status = "not_available"
            derived_id = f"data/masks/{PurePosixPath(spec.local_file).name}" if spec.derived_from_upstream_entity_id else None
            locally_modified_id = (
                f"urn:sha256:{local_sha256}" if checksum_status == "mismatch" and local_sha256 and not derived_id else None
            )
            resolved.append(
                ResolvedInput(
                    spec=spec,
                    upstream_entity=_portable_entity(entity),
                    match_method=method,
                    local_path=local_path,
                    local_sha256=local_sha256,
                    source_local_sha256=source_local_sha256,
                    checksum_status=checksum_status,
                    derived_entity_id=derived_id,
                    locally_modified_entity_id=locally_modified_id,
                )
            )
        except UpstreamResolutionError as exc:
            errors.append(str(exc))
    if errors:
        raise UpstreamResolutionError("Required upstream inputs could not be resolved:\n- " + "\n- ".join(errors))
    return resolved


def validate_resolved_coverage(cfg: dict[str, Any], resolved: list[ResolvedInput]) -> None:
    expected = {f"variable:{str(name).upper().strip()}" for name in cfg.get("variable_names", [])}
    expected.update(f"mask:{str(name).lower().strip()}" for name in cfg.get("data", {}).get("mask_names", []))
    actual = {item.spec.key for item in resolved}
    missing = sorted(expected - actual)
    if missing:
        raise UpstreamResolutionError(
            "RO-Crate input coverage is incomplete; used variables/masks without resolved provenance: "
            + ", ".join(missing)
        )


# Compatibility helpers retained for callers of the initial implementation.
def catalogue_input_names(catalogue_path: Path | None) -> list[str]:
    return [PurePosixPath(spec.local_file).name for spec in catalogue_inputs(catalogue_path)]


def find_upstream_entities(metadata: dict[str, Any] | None, filenames: list[str]) -> tuple[list[dict], list[str]]:
    specs = [InputSpec(f"file:{name}", "file", name) for name in filenames]
    found, missing = [], []
    for spec in specs:
        try:
            found.append(resolve_catalogue_inputs(metadata, [spec])[0].upstream_entity)
        except UpstreamResolutionError:
            missing.append(spec.local_file)
    return found, missing


def upstream_dataset_terms(metadata: dict[str, Any] | None) -> dict[str, Any]:
    if not metadata:
        return {}
    graph = metadata.get("@graph", [])
    root = next((entity for entity in graph if entity.get("@id") == "./"), None)
    if root is None:
        root = next((entity for entity in graph if "Dataset" in _as_types(entity.get("@type"))), None)
    return {key: root[key] for key in ("license", "creator") if root and key in root}


def _input_spec(key: str, kind: str, local_file: str, mapping: dict[str, Any]) -> InputSpec:
    class_value = mapping.get("class_value")
    return InputSpec(
        key=key,
        kind=kind,
        local_file=str(local_file),
        upstream_entity_id=mapping.get("upstream_entity_id"),
        netcdf_variable=mapping.get("netcdf_variable"),
        expected_sha256=mapping.get("sha256"),
        derived_from_upstream_entity_id=mapping.get("derived_from_upstream_entity_id"),
        derivation_type=mapping.get("derivation_type"),
        class_value=int(class_value) if class_value is not None else None,
        class_label=mapping.get("class_label"),
        source_local_file=mapping.get("source_local_file"),
        role=mapping.get("role"),
        frequency=mapping.get("frequency"),
        required=bool(mapping.get("required", True)),
    )


def _mapping(mappings: dict[str, Any], name: str) -> dict[str, Any]:
    for candidate in (name, name.upper(), name.lower()):
        value = mappings.get(candidate)
        if isinstance(value, dict):
            return value
    return {}


def _entity_by_explicit_id(graph: list[dict[str, Any]], requested: str, key: str) -> dict[str, Any]:
    normalized = requested.strip().rstrip("/")
    matches = [
        entity for entity in graph
        if entity["@id"].rstrip("/") == normalized
        or entity["@id"].rstrip("/").endswith(normalized.lstrip("/#"))
    ]
    if not matches:
        raise UpstreamResolutionError(f"{key}: configured upstream_entity_id does not exist: {requested}")
    if len(matches) > 1:
        raise UpstreamResolutionError(f"{key}: upstream_entity_id is ambiguous: {requested}")
    return matches[0]


def _fallback_entity(graph: list[dict[str, Any]], spec: InputSpec, local_sha256: str | None) -> tuple[dict, str]:
    checksum = spec.expected_sha256 or local_sha256
    if checksum:
        matches = [entity for entity in graph if (_checksum(entity) or "").lower() == checksum.lower()]
        if len(matches) == 1:
            return matches[0], "checksum"
    filename = PurePosixPath(spec.local_file.replace("\\", "/")).name
    matches = [entity for entity in graph if filename in _entity_names(entity)]
    if len(matches) == 1:
        return matches[0], "name_or_alias"
    if len(matches) > 1:
        raise UpstreamResolutionError(f"{spec.key}: filename/alias is ambiguous upstream: {filename}")
    raise UpstreamResolutionError(f"{spec.key}: no upstream entity found for required local file {filename}")


def _local_path(spec: InputSpec, raw_dir: Path | None, mask_dir: Path | None) -> Path | None:
    base = mask_dir if spec.kind == "mask" else raw_dir
    if base is None:
        return None
    return (base / spec.local_file).resolve()


def _checksum(entity: dict[str, Any]) -> str | None:
    for key in ("sha256", "checksum", "contentChecksum"):
        value = entity.get(key)
        if isinstance(value, str):
            return value.removeprefix("sha256:")
    return None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _entity_names(entity: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    for key in ("@id", "url", "contentUrl", "dcat:downloadURL", "name", "alternateName", "sameAs"):
        values = entity.get(key, [])
        if not isinstance(values, list):
            values = [values]
        for value in values:
            if isinstance(value, dict):
                value = value.get("@id")
            if isinstance(value, str):
                names.add(PurePosixPath(urlparse(value).path or value).name)
    return names


def _portable_entity(entity: dict[str, Any]) -> dict[str, Any]:
    keys = {
        "@id", "@type", "name", "alternateName", "description", "url", "contentUrl",
        "dcat:downloadURL", "encodingFormat", "contentSize", "sha256", "sha512", "md5",
        "checksum", "identifier", "sameAs", "variableMeasured",
        "license", "dcterms:license", "prov:wasDerivedFrom", "dcterms:source",
    }
    return {key: value for key, value in entity.items() if key in keys}


def _walk_file_values(value: Any, names: list[str]) -> None:
    if isinstance(value, dict):
        for nested in value.values():
            _walk_file_values(nested, names)
    elif isinstance(value, list):
        for nested in value:
            _walk_file_values(nested, names)
    elif isinstance(value, str) and Path(value).suffix.lower() in {".nc", ".nc4", ".npy", ".zarr"}:
        names.append(PurePosixPath(value.replace("\\", "/")).name)


def _load_remote_metadata(reference: str, timeout: int) -> dict[str, Any]:
    return _load_remote_metadata_details(reference, timeout)[0]


def _load_remote_metadata_details(reference: str, timeout: int) -> tuple[dict[str, Any], str, str, bytes]:
    request = Request(reference, headers={"Accept": "application/ld+json, application/json"})
    with urlopen(request, timeout=timeout) as response:  # noqa: S310
        content_type = response.headers.get_content_type()
        if content_type in {"application/json", "application/ld+json"}:
            raw = response.read()
            return json.loads(raw), response.geturl(), hashlib.sha256(raw).hexdigest(), raw
        link_header = response.headers.get("Link", "")
    linkset_url = _link_with_type(link_header, "application/linkset+json")
    if not linkset_url:
        raise ValueError(f"Remote resource is {content_type}, not RO-Crate JSON-LD")
    with urlopen(linkset_url, timeout=timeout) as response:  # noqa: S310
        linkset = json.load(response)
    for entry in linkset.get("linkset", []):
        for described_by in entry.get("describedby", []):
            if described_by.get("type") == "application/ld+json" and described_by.get("href"):
                with urlopen(described_by["href"], timeout=timeout) as response:  # noqa: S310
                    raw = response.read()
                    return json.loads(raw), response.geturl(), hashlib.sha256(raw).hexdigest(), raw
    raise ValueError("Remote linkset does not advertise an application/ld+json description")


def _link_with_type(header: str, media_type: str) -> str | None:
    for item in header.split(","):
        match = re.search(r"<([^>]+)>", item)
        if match and f'type="{media_type}"' in item:
            return match.group(1)
    return None


def _as_types(value: Any) -> list[str]:
    return value if isinstance(value, list) else [value]
