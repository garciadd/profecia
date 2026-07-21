from __future__ import annotations

import argparse
import dataclasses
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import ReproducibilityConfig
from .input_resolution import reconstruct_derived_inputs, resolve_reproduction_inputs
from .run_metadata import file_facts
from .upstream_crate import catalogue_inputs, load_upstream_rocrate, resolve_catalogue_inputs
from .verify import verify_reproduction, write_reproduction_report


def reproduce(crate: Path, work_dir: Path, *, offline: bool = False, verify_only: bool = False,
              recompute_split: bool = False, use_container: bool = False,
              local_environment: bool = False) -> dict[str, Any]:
    """Resolve dependencies and reproduce or verify a packaged PROFECIA training run."""
    crate, work_dir = crate.resolve(), work_dir.resolve()
    _require(crate / "ro-crate-metadata.json", "original RO-Crate metadata")
    effective = json.loads(_require(crate / "config" / "effective_config.json", "effective configuration").read_text())
    settings = ReproducibilityConfig.from_mapping(effective.get("reproducibility"), crate)
    work_dir.mkdir(parents=True, exist_ok=True)

    if use_container or (settings.environment.container_image and not local_environment):
        return _run_container(crate, work_dir, settings, offline, verify_only, recompute_split)

    metadata_source = settings.upstream.metadata_url
    cached_metadata = work_dir / "cache" / "upstream" / "ro-crate-metadata.json"
    if offline and cached_metadata.is_file():
        metadata_source = str(cached_metadata)
    if offline and not cached_metadata.is_file() and not (metadata_source and Path(metadata_source).is_file()):
        raise ValueError("fatal error: upstream ro-crate-metadata.json is absent from the offline cache")
    upstream = load_upstream_rocrate(metadata_source)
    if not upstream or not upstream.metadata:
        raise ValueError("fatal error: upstream ro-crate-metadata.json could not be recovered")
    original_graph = json.loads((crate / "ro-crate-metadata.json").read_text()).get("@graph", [])
    upstream_entity = next((item for item in original_graph if str(item.get("@id", "")).endswith("#upstream-preprocessing-crate")), {})
    expected_metadata_sha = upstream_entity.get("sha256")
    if expected_metadata_sha and upstream.metadata_sha256 != expected_metadata_sha:
        raise ValueError("fatal error: upstream ro-crate-metadata.json checksum does not match the original run")
    if not offline:
        cached_metadata.parent.mkdir(parents=True, exist_ok=True)
        cached_metadata.write_bytes(upstream.raw_metadata or (json.dumps(upstream.metadata, indent=2) + "\n").encode())

    specs = catalogue_inputs(crate / "config" / "data.toml")
    resolved = resolve_catalogue_inputs(upstream.metadata, specs)
    runtime_upstream = dataclasses.replace(
        settings.upstream, cache_dir=work_dir / "cache" / "upstream" / "files", offline=offline
    )
    variables = [item for item in resolved if item.spec.kind != "mask"]
    masks = [item for item in resolved if item.spec.kind == "mask"]
    input_report = resolve_reproduction_inputs(
        variables, runtime_upstream, settings.inputs, work_dir / "inputs", offline
    )
    mask_report = resolve_reproduction_inputs(
        masks, runtime_upstream, settings.inputs, work_dir / "inputs" / "mask-sources", offline
    ) if masks else []
    derivations = reconstruct_derived_inputs(
        masks, mask_report, work_dir / "derived-inputs" / "masks"
    )
    derived_mask_dir = work_dir / "derived-inputs" / "masks"
    derived_mask_dir.mkdir(parents=True, exist_ok=True)
    by_key = {entry["key"]: entry for entry in mask_report}
    for item in masks:
        if not item.is_derived:
            shutil.copy2(by_key[item.spec.key]["resolved_path"], derived_mask_dir / Path(item.spec.local_file).name)
    resolution = {"inputs": [*input_report, *mask_report], "derived_inputs": derivations}
    (work_dir / "input_resolution.json").write_text(json.dumps(resolution, indent=2) + "\n", encoding="utf-8")
    _validate_environment(settings, crate)
    _validate_split(crate)

    if verify_only:
        report = {
            "passed": True, "reproduction_level": "functionally reproduced",
            "mode": "verify-only", "checks": [
                {"name": "upstream_release", "passed": True},
                {"name": "input_checksums", "passed": all(item["checksum_verified"] for item in input_report + mask_report if item["expected_sha256"])},
                {"name": "derived_transformations", "passed": len(derivations) == len([item for item in masks if item.is_derived])},
                {"name": "split_manifest", "passed": True},
                {"name": "environment", "passed": True},
            ],
        }
        write_reproduction_report(work_dir, report)
        return report

    env = os.environ.copy()
    env.update({
        "PROFECIA_TRAIN_CONFIG": str(crate / "config" / "train.toml"),
        "PROFECIA_EFFECTIVE_CONFIG": str(crate / "config" / "effective_config.json"),
        "PROFECIA_WORK_DIR": str(work_dir),
        "PROFECIA_SPLIT_MANIFEST": str(crate / "fair" / "split_manifest.npz"),
        "PROFECIA_REPRODUCTION_MODE": "1",
    })
    if recompute_split:
        env["PROFECIA_RECOMPUTE_SPLIT"] = "1"
    repository_root = Path(__file__).resolve().parents[2]
    subprocess.run(
        [sys.executable, str(repository_root / "scripts" / "workflow_step_by_step_configurable.py")],
        cwd=repository_root, env=env, check=True,
    )
    comparison = verify_reproduction(crate, work_dir)
    comparison["exact_split"] = not recompute_split
    comparison["actual"] = _actual_from_workflow(work_dir)
    write_reproduction_report(work_dir, comparison)
    _write_reproduced_crate(crate, work_dir, comparison, settings.upstream.identifier)
    if settings.verification.fail_on_metric_difference and not comparison["passed"]:
        raise ValueError("fatal error: reproduced metrics are outside configured tolerances")
    return comparison


def _validate_split(crate: Path) -> None:
    import hashlib
    import numpy as np
    metadata = json.loads(_require(crate / "fair" / "split_manifest.json", "split metadata").read_text())
    path = _require(crate / "fair" / "split_manifest.npz", "split manifest")
    if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["sha256"]:
        raise ValueError("fatal error: split manifest checksum mismatch")
    with np.load(path, allow_pickle=False) as payload:
        train, test = payload["train_indices"], payload["test_indices"]
    if len(np.unique(train)) != len(train) or len(np.unique(test)) != len(test):
        raise ValueError("fatal error: split manifest contains duplicate indices")
    if np.intersect1d(train, test).size:
        raise ValueError("fatal error: train and test overlap")


def _validate_environment(settings: ReproducibilityConfig, crate: Path) -> None:
    expected = settings.environment.python_version
    if expected and not platform.python_version().startswith(expected):
        raise ValueError(f"non-reproducible condition: Python {expected} is required")
    if settings.environment.lock_file:
        configured = Path(settings.environment.lock_file)
        packaged = list((crate / "environment" / "lock").glob("*")) if (crate / "environment" / "lock").is_dir() else []
        if not configured.is_file() and not packaged:
            raise ValueError("fatal error: configured environment lock file is missing")
    elif not settings.environment.container_image:
        inventory_path = crate / "environment" / "python-packages.json"
        inventory = json.loads(_require(inventory_path, "Python environment inventory").read_text())
        mismatches = []
        installed = {dist.metadata.get("Name", "").lower(): dist.version for dist in importlib.metadata.distributions()}
        for name, version in (inventory.get("packages") or {}).items():
            if installed.get(str(name).lower()) != str(version):
                mismatches.append(f"{name}=={version}")
        if mismatches:
            raise ValueError(
                "non-reproducible condition: installed environment differs from inventory: "
                + ", ".join(mismatches[:10])
            )


def _run_container(crate: Path, work: Path, settings: ReproducibilityConfig, offline: bool,
                   verify_only: bool, recompute: bool) -> dict[str, Any]:
    if not settings.environment.container_image or not settings.environment.container_digest:
        raise ValueError("fatal error: --container requires an image and immutable digest")
    runtime = settings.environment.container_runtime or "docker"
    if not shutil.which(runtime):
        raise ValueError(f"fatal error: container runtime is unavailable: {runtime}")
    image = f"{settings.environment.container_image}@{settings.environment.container_digest}"
    command = [runtime, "run", "--rm", "-v", f"{crate}:/crate:ro", "-v", f"{work}:/work", image,
               "python", "-m", "profecia.reproducibility.reproduce", "--crate", "/crate",
               "--work-dir", "/work", "--local-environment"]
    if offline: command.append("--offline")
    if verify_only: command.append("--verify-only")
    if recompute: command.append("--recompute-split")
    subprocess.run(command, check=True)
    report = json.loads((work / "reproduction_report.json").read_text())
    return report


def _actual_from_workflow(work: Path) -> dict[str, Any]:
    summaries = sorted(work.rglob("workflow_step_by_step_summary.json"))
    summary = json.loads(summaries[-1].read_text())
    dataset = summary.get("dataset_metadata") or {}
    return {"dataset": {"n_train": dataset.get("n_train"), "n_test": dataset.get("n_test")},
            "metrics": (summary.get("evaluation") or {}).get("global_metrics") or {}}


def _write_reproduced_crate(original: Path, work: Path, report: dict[str, Any], upstream_id: str | None) -> Path:
    crate = work / "reproduced-ro-crate"
    crate.mkdir(parents=True, exist_ok=False)
    copied: list[Path] = []
    for name in ("reproduction_report.json", "reproduction_report.md", "input_resolution.json"):
        source = work / name
        if source.is_file():
            target = crate / name
            shutil.copy2(source, target)
            copied.append(target)
    summaries = sorted(work.rglob("workflow_step_by_step_summary.json"))
    if summaries:
        target = crate / "metrics" / "workflow_summary.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(summaries[-1], target)
        copied.append(target)
    environment_path = crate / "environment" / "reproduction-environment.json"
    environment_path.parent.mkdir(parents=True, exist_ok=True)
    environment_path.write_text(json.dumps({
        "python_version": platform.python_version(), "platform": platform.platform(),
        "machine": platform.machine(), "hostname": platform.node(),
        "packages": {
            dist.metadata.get("Name", "unknown"): dist.version
            for dist in importlib.metadata.distributions() if dist.metadata.get("Name")
        },
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    copied.append(environment_path)
    run_id = f"urn:uuid:{uuid.uuid4()}"
    original_metadata = json.loads((original / "ro-crate-metadata.json").read_text())
    original_graph = original_metadata["@graph"]
    original_id = next(item for item in original_graph if item["@id"] == "./").get("identifier")
    original_code = next(
        (item for item in original_graph if str(item.get("@id", "")).endswith("#source-code")), {}
    )
    original_commit = original_code.get("version") or original_code.get("commitHash")
    repository_root = Path(__file__).resolve().parents[2]
    current_commit = _git_value(repository_root, ["rev-parse", "HEAD"])
    current_dirty = bool(_git_value(repository_root, ["status", "--porcelain"]))
    assessment_id = f"{run_id}#reproduction-assessment"
    action_id = f"{run_id}#reproduce-training-run"
    code_id = f"{run_id}#source-code"
    environment_id = f"{run_id}#reproduction-environment"
    sources = [{"@id": original_id}]
    if upstream_id: sources.append({"@id": upstream_id})
    graph = [
        {"@id": "ro-crate-metadata.json", "@type": "CreativeWork", "about": {"@id": "./"},
         "conformsTo": {"@id": "https://w3id.org/ro/crate/1.1"}},
        {"@id": "./", "@type": "Dataset", "identifier": run_id, "datePublished": datetime.now(UTC).isoformat(),
         "name": "Reproduced PROFECIA training run", "prov:wasDerivedFrom": sources,
         "dcterms:source": sources,
         "mentions": [{"@id": assessment_id}, {"@id": action_id}, {"@id": code_id}],
         "hasPart": [{"@id": path.relative_to(crate).as_posix()} for path in copied]},
        {"@id": assessment_id, "@type": "CreativeWork", "name": "PROFECIA reproduction assessment",
         "prov:wasDerivedFrom": {"@id": original_id}, "reproductionLevel": report["reproduction_level"],
         "passedChecks": report.get("passed_checks", []), "failedChecks": report.get("failed_checks", []),
         "metricDifferences": report.get("metric_checks", []),
         "originalCommit": original_commit, "reproducedCommit": current_commit,
         "sameCommit": bool(original_commit and original_commit == current_commit)},
        {"@id": action_id, "@type": "CreateAction", "name": "Reproduce a PROFECIA training run",
         "actionStatus": {"@id": "http://schema.org/CompletedActionStatus"},
         "object": sources, "prov:used": [*sources, {"@id": code_id}, {"@id": environment_id}],
         "result": [{"@id": assessment_id}, {"@id": "reproduction_report.json"}]},
        {"@id": code_id, "@type": "SoftwareSourceCode", "name": "PROFECIA source used for reproduction",
         "codeRepository": "https://github.com/garciadd/profecia", "version": current_commit,
         "url": f"https://github.com/garciadd/profecia/tree/{current_commit}" if current_commit else None,
         "git_dirty": current_dirty},
        {"@id": environment_id, "@type": "SoftwareApplication", "name": "Reproduction software environment",
         "softwareVersion": platform.python_version(), "operatingSystem": platform.platform(),
         "subjectOf": {"@id": "environment/reproduction-environment.json"}},
        {"@id": original_id, "@type": "Dataset", "name": "Original PROFECIA training run"},
    ]
    graph.extend({
        "@id": path.relative_to(crate).as_posix(), "@type": "File", **file_facts(path)
    } for path in copied)
    resolution_path = work / "input_resolution.json"
    if resolution_path.is_file():
        resolution = json.loads(resolution_path.read_text(encoding="utf-8"))
        for item in resolution.get("inputs", []):
            identifier = item.get("upstream_entity_id")
            if identifier and not any(entity.get("@id") == identifier for entity in graph):
                graph.append({
                    "@id": identifier, "@type": "File", "name": item.get("local_file"),
                    "sha256": item.get("actual_sha256"), "contentUrl": item.get("download_url"),
                })
    if upstream_id: graph.append({"@id": upstream_id, "@type": "Dataset", "name": "Upstream preprocessing release"})
    graph = [{key: value for key, value in entity.items() if value is not None} for entity in graph]
    metadata = {"@context": ["https://w3id.org/ro/crate/1.1/context", {"prov": "http://www.w3.org/ns/prov#", "dcterms": "http://purl.org/dc/terms/"}], "@graph": graph}
    (crate / "ro-crate-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return crate


def _git_value(repository: Path, arguments: list[str]) -> str | None:
    try:
        value = subprocess.run(
            ["git", *arguments], cwd=repository, check=True, capture_output=True, text=True
        ).stdout.strip()
        return value or None
    except (OSError, subprocess.CalledProcessError):
        return None


def _require(path: Path, label: str) -> Path:
    if not path.is_file():
        raise ValueError(f"fatal error: missing {label}: {path}")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reproduce a PROFECIA training RO-Crate")
    parser.add_argument("--crate", required=True, type=Path)
    parser.add_argument("--work-dir", type=Path, default=Path("reproduction"))
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--recompute-split", action="store_true")
    parser.add_argument("--container", action="store_true")
    parser.add_argument("--local-environment", action="store_true")
    args = parser.parse_args(argv)
    try:
        reproduce(args.crate, args.work_dir, offline=args.offline, verify_only=args.verify_only,
                  recompute_split=args.recompute_split, use_container=args.container,
                  local_environment=args.local_environment)
        return 0
    except Exception as exc:
        args.work_dir.mkdir(parents=True, exist_ok=True)
        write_reproduction_report(args.work_dir, {
            "passed": False, "reproduction_level": "not reproduced",
            "checks": [{"name": "fatal_error", "passed": False, "message": str(exc)}],
        })
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
