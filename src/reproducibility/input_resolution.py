from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .config import InputsConfig, UpstreamConfig
from .upstream_crate import ResolvedInput, UpstreamResolutionError


def resolve_reproduction_inputs(
    resolved: list[ResolvedInput], upstream: UpstreamConfig, inputs: InputsConfig,
    destination: Path, offline: bool | None = None,
) -> list[dict[str, Any]]:
    """Download/cache required upstream resources and verify their reproducibility contract."""
    offline = upstream.offline if offline is None else offline
    cache_dir = upstream.cache_dir or destination.parent / "cache" / "upstream"
    cache_dir.mkdir(parents=True, exist_ok=True)
    destination.mkdir(parents=True, exist_ok=True)
    report = []
    for item in resolved:
        entity = item.upstream_entity
        expected = _checksum(entity)
        url = _download_url(entity)
        if inputs.require_upstream_checksum and not expected:
            raise UpstreamResolutionError(f"{item.spec.key}: upstream entity has no SHA-256 checksum")
        if item.checksum_status == "mismatch" and not (
            item.is_derived and inputs.allow_derived_local_inputs
        ):
            raise UpstreamResolutionError(
                f"{item.spec.key}: local checksum differs from upstream without a documented derivation"
            )
        cache_name = _cache_name(entity, expected)
        cached = cache_dir / cache_name
        if not cached.is_file():
            if offline:
                raise UpstreamResolutionError(f"{item.spec.key}: required cached input is missing in offline mode: {cached}")
            if not upstream.download_missing:
                raise UpstreamResolutionError(f"{item.spec.key}: input is missing and download_missing=false")
            if not url:
                raise UpstreamResolutionError(f"{item.spec.key}: upstream entity has no direct download URL")
            _download(url, cached)
        actual = _sha256(cached)
        if upstream.verify_checksums and expected and actual.lower() != expected.lower():
            raise UpstreamResolutionError(
                f"{item.spec.key}: cached/downloaded SHA-256 mismatch ({actual} != {expected})"
            )
        if item.spec.netcdf_variable and cached.suffix.lower() in {".nc", ".nc4"}:
            _verify_netcdf_variable(cached, item.spec.netcdf_variable)
        resolved_name = item.spec.source_local_file if item.is_derived and item.spec.source_local_file else item.spec.local_file
        relative = Path(resolved_name.replace("\\", "/"))
        target = (destination / relative).resolve()
        allowed_root = destination.parent.resolve()
        if not target.is_relative_to(allowed_root):
            raise UpstreamResolutionError(f"{item.spec.key}: unsafe local_file path outside reproduction workspace")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            target.unlink()
        try:
            target.hardlink_to(cached)
        except OSError:
            shutil.copy2(cached, target)
        report.append({
            "key": item.spec.key, "role": item.spec.role, "required": item.spec.required,
            "frequency": item.spec.frequency, "local_file": item.spec.local_file,
            "upstream_entity_id": entity["@id"], "download_url": url,
            "cache_path": str(cached), "resolved_path": str(target),
            "match_method": item.match_method, "expected_sha256": expected,
            "actual_sha256": actual, "checksum_verified": bool(expected and actual.lower() == expected.lower()),
            "netcdf_variable": item.spec.netcdf_variable, "derived": item.is_derived,
        })
    return report


def write_input_resolution(crate_dir: Path, entries: list[dict[str, Any]]) -> Path:
    path = crate_dir / "reproducibility" / "input_resolution.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    portable = []
    for entry in entries:
        value = dict(entry)
        value.pop("cache_path", None)
        value.pop("resolved_path", None)
        portable.append(value)
    path.write_text(json.dumps({"inputs": portable}, indent=2) + "\n", encoding="utf-8")
    return path


def resolution_metadata(resolved: list[ResolvedInput]) -> list[dict[str, Any]]:
    """Describe resolution without downloading, suitable for the original training crate."""
    return [{
        "key": item.spec.key, "role": item.spec.role, "required": item.spec.required,
        "frequency": item.spec.frequency, "local_file": item.spec.local_file,
        "upstream_entity_id": item.upstream_entity["@id"],
        "download_url": _download_url(item.upstream_entity),
        "match_method": item.match_method, "upstream_sha256": _checksum(item.upstream_entity),
        "local_sha256": item.local_sha256, "checksum_status": item.checksum_status,
        "netcdf_variable": item.spec.netcdf_variable, "derived": item.is_derived,
        "derivation_type": item.spec.derivation_type,
        "source_local_file": item.spec.source_local_file,
        "class_value": item.spec.class_value, "class_label": item.spec.class_label,
    } for item in resolved]


def reconstruct_derived_inputs(
    resolved: list[ResolvedInput], downloaded: list[dict[str, Any]], output_dir: Path
) -> list[dict[str, Any]]:
    """Recreate supported local derived inputs and record input/output checksums."""
    import numpy as np

    by_key = {item["key"]: item for item in downloaded}
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for item in resolved:
        if not item.is_derived:
            continue
        if item.spec.derivation_type != "class_selection" or item.spec.class_value is None:
            raise UpstreamResolutionError(
                f"{item.spec.key}: required local derivation is unsupported or undocumented"
            )
        source = Path(by_key[item.spec.key]["resolved_path"])
        target = output_dir / PurePosixPath(item.spec.local_file.replace("\\", "/")).name
        values = np.load(source, allow_pickle=False)
        derived = values == item.spec.class_value
        np.save(target, derived)
        results.append({
            "key": item.spec.key, "function": "numpy.equal",
            "parameters": {"class_value": item.spec.class_value, "class_label": item.spec.class_label},
            "input": str(source), "input_sha256": _sha256(source),
            "output": str(target), "output_sha256": _sha256(target),
            "shape": list(derived.shape), "dtype": str(derived.dtype),
        })
    return results


def _download(url: str, target: Path) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise UpstreamResolutionError(f"Unsupported download URL: {url}")
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    try:
        with urlopen(Request(url, headers={"User-Agent": "PROFECIA-reproduce/1"}), timeout=60) as response:  # noqa: S310
            with partial.open("wb") as handle:
                shutil.copyfileobj(response, handle)
        partial.replace(target)
    finally:
        if partial.exists():
            partial.unlink()


def _download_url(entity: dict[str, Any]) -> str | None:
    for key in ("contentUrl", "dcat:downloadURL"):
        value = entity.get(key)
        if isinstance(value, dict):
            value = value.get("@id")
        if isinstance(value, str):
            parsed = urlparse(value)
            if parsed.scheme in {"http", "https"} and parsed.netloc:
                return value
    return None


def _checksum(entity: dict[str, Any]) -> str | None:
    for key in ("sha256", "checksum", "contentChecksum"):
        value = entity.get(key)
        if isinstance(value, str):
            return value.removeprefix("sha256:")
    return None


def _cache_name(entity: dict[str, Any], checksum: str | None) -> str:
    basename = PurePosixPath(urlparse(str(entity.get("@id", "input"))).path).name or "input"
    safe = "".join(character if character.isalnum() or character in ".-_" else "_" for character in basename)
    return f"{checksum[:16]}-{safe}" if checksum else safe


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_netcdf_variable(path: Path, variable: str) -> None:
    try:
        import xarray as xr
        with xr.open_dataset(path, decode_cf=False) as dataset:
            if variable not in dataset.variables:
                raise UpstreamResolutionError(f"{path.name}: expected NetCDF variable {variable!r} is absent")
    except UpstreamResolutionError:
        raise
    except Exception as exc:
        raise UpstreamResolutionError(f"Could not inspect NetCDF input {path}: {exc}") from exc
