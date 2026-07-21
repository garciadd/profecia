from __future__ import annotations

import importlib.metadata
import json
import logging
import mimetypes
import platform
import re
import shutil
import subprocess
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote, urlparse

from .config import ReproducibilityConfig
from .croissant import build_croissant_descriptor
from .effective_config import write_effective_config
from .expected_results import write_expected_results
from .fair4ml import build_fair4ml_descriptor
from .input_resolution import resolution_metadata, write_input_resolution
from .run_metadata import (
    CREATOR, DATASET_LICENSE, build_run_metadata, ensure_run_identifiers,
    file_facts, write_split_manifest,
)
from .upstream_crate import (
    ResolvedInput,
    UpstreamResolutionError,
    catalogue_inputs,
    load_upstream_rocrate,
    resolve_catalogue_inputs,
    validate_resolved_coverage,
)


LOGGER = logging.getLogger(__name__)
RO_CRATE = "https://w3id.org/ro/crate/1.1"
PROCESS_RUN_CRATE = "https://w3id.org/ro/wfrun/process/0.5"
CANONICAL_REPOSITORY_URL = "https://github.com/garciadd/profecia"
SECRET_LINE = re.compile(
    r"^(?P<prefix>\s*[A-Za-z0-9_.-]*(?:password|passwd|token|secret|api_key)[A-Za-z0-9_.-]*\s*=\s*).*$",
    re.IGNORECASE,
)


def create_training_rocrate(
    cfg: dict[str, Any],
    summary: dict[str, Any],
    started_at: datetime,
    ended_at: datetime | None = None,
    command: list[str] | None = None,
    project_root: Path | None = None,
) -> Path | None:
    """Create one self-contained metadata crate for a completed training run."""
    repro = cfg.get("reproducibility")
    if isinstance(repro, ReproducibilityConfig):
        settings = repro
    else:
        settings = ReproducibilityConfig.from_mapping(repro, Path.cwd())
    if not settings.create_rocrate:
        return None

    ended_at = ended_at or datetime.now(UTC)
    project_root = (project_root or Path.cwd()).resolve()
    if settings.code.require_clean_repository:
        try:
            dirty_status = subprocess.run(
                ["git", "status", "--porcelain"], cwd=project_root, check=True,
                capture_output=True, text=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError) as exc:
            raise ValueError("require_clean_repository=true requires a readable Git working tree") from exc
        if dirty_status:
            raise ValueError(
                "Git repository is dirty but reproducibility.code.require_clean_repository=true"
            )
    # ``upstream_rocrate`` may point to a local metadata cache used while
    # packaging; the nested metadata_url remains the canonical retrieval URL.
    upstream = load_upstream_rocrate(settings.upstream_rocrate or settings.upstream.metadata_url)
    if settings.reproducible_run:
        if not settings.upstream.identifier:
            raise ValueError("reproducible_run=true requires reproducibility.upstream.identifier")
        if not settings.upstream.metadata_url:
            raise ValueError("reproducible_run=true requires reproducibility.upstream.metadata_url")
        if settings.code.mode != "reference":
            raise ValueError("reproducible_run=true requires code.mode=reference; source code is referenced, not copied")
    specs = catalogue_inputs(settings.input_catalogue)
    resolved_inputs = resolve_catalogue_inputs(
        upstream.metadata if upstream else None,
        specs,
        raw_dir=Path(cfg["raw_dir"]) if cfg.get("raw_dir") else None,
        mask_dir=Path(cfg["mask_dir"]) if cfg.get("mask_dir") else None,
    )
    validate_resolved_coverage(cfg, resolved_inputs)
    if settings.reproducible_run:
        for item in resolved_inputs:
            if item.spec.required and not (item.spec.upstream_entity_id or item.spec.derived_from_upstream_entity_id):
                raise UpstreamResolutionError(
                    f"{item.spec.key}: reproducible runs require an explicit upstream_entity_id"
                )
            if item.checksum_status == "mismatch" and not (
                item.is_derived and settings.inputs.allow_derived_local_inputs
            ):
                raise UpstreamResolutionError(
                    f"{item.spec.key}: checksum differs from upstream without a documented transformation"
                )
    execution_uuid = uuid.uuid4()
    run_id = f"{ended_at.strftime('%Y%m%dT%H%M%SZ')}-{execution_uuid.hex[:8]}"
    run = build_run_metadata(cfg, summary, base_id=f"urn:uuid:{execution_uuid}")
    run["licenses"] = {
        name: getattr(settings.licenses, name)
        for name in ("metadata", "code", "model", "results", "derived_dataset")
    }
    run["license"] = settings.licenses.derived_dataset
    base = settings.output_dir or (Path(cfg["model_dir"]) / "ro-crates")
    crate_dir = Path(base) / cfg.get("model_run_name", "training") / run_id / "ro-crate"
    crate_dir.mkdir(parents=True, exist_ok=False)

    model_source = _model_source(cfg, summary)
    copied = _copy_run_files(
        crate_dir, cfg, summary, project_root, started_at, model_source,
        code_mode=settings.code.mode,
    )
    copied.extend(_copy_intermediate_metadata(crate_dir, run))
    copied.extend([
        write_effective_config(crate_dir, cfg),
        write_input_resolution(crate_dir, resolution_metadata(resolved_inputs)),
    ])
    git = _git_metadata(project_root, crate_dir, settings.code.mode)
    if git.get("dirty") and settings.code.require_clean_repository:
        raise ValueError(
            "Git repository is dirty but reproducibility.code.require_clean_repository=true"
        )
    if git.get("dirty") and settings.code.mode == "reference":
        LOGGER.warning(
            "code.mode=reference with git_dirty=true cannot reproduce the exact executed state; "
            "use code.mode=patch or require a clean repository"
        )
    if git.get("diff_path"):
        copied.append(git["diff_path"])
    copied.extend(git.get("untracked_paths") or [])
    environment_path, environment_extra, software_requirements, container_entity = _write_environment(
        crate_dir, settings.environment
    )
    copied.extend([environment_path, *environment_extra])
    run["software_requirements"] = software_requirements
    run["container_entity"] = container_entity
    derived_entities, derived_paths = _materialize_derived_inputs(crate_dir, resolved_inputs)
    local_modified_entities = _local_modified_input_entities(run, resolved_inputs)
    derived_entities.extend(local_modified_entities)
    copied.extend(derived_paths)
    derived_by_id = {entity["@id"]: entity for entity in derived_entities}
    training_entities = []
    for item in resolved_inputs:
        training_entities.append(derived_by_id.get(item.training_entity_id, item.upstream_entity))
        if item.training_entity_id != item.upstream_entity["@id"]:
            training_entities.append(item.upstream_entity)
    training_entities = _unique_entities(training_entities)

    artifact, artifact_path = _prepare_model_artifact(
        crate_dir, settings.model_artifact, model_source, run["identifiers"]["model-artifact"]
    )
    if artifact_path:
        copied.append(artifact_path)

    split_manifest = None
    if settings.croissant.include_split_manifest:
        split_manifest = write_split_manifest(crate_dir, run, git)
        copied.extend([split_manifest["npz_path"], split_manifest["json_path"]])
    copied.append(write_expected_results(
        crate_dir, run, split_manifest, settings.verification.metric_absolute_tolerance
    ))
    reproduce_path = _write_reproduce_script(crate_dir, git)
    copied.append(reproduce_path)

    release_identifier = settings.upstream.identifier or (upstream.identifier if upstream else None)
    if release_identifier:
        run["upstream_entity_id"] = run["identifiers"]["upstream-preprocessing-crate"]

    fair_dir = crate_dir / "fair"
    fair_dir.mkdir(exist_ok=True)
    croissant_path = fair_dir / "dataset.croissant.json"
    croissant, materialized_paths = build_croissant_descriptor(
        run=run,
        upstream_identifier=release_identifier,
        upstream_entities=training_entities,
        published_at=ended_at.isoformat(),
        mode=settings.croissant.mode,
        crate_dir=crate_dir,
        model_data_dir=Path(cfg["model_data_dir"]) if cfg.get("model_data_dir") else None,
        split_manifest=split_manifest,
    )
    copied.extend(materialized_paths)
    validate_jsonld_document(croissant, "Croissant")
    _write_json(croissant_path, croissant)
    fair4ml_path = fair_dir / "model.fair4ml.json"
    fair4ml = build_fair4ml_descriptor(
        run=run,
        artifact=artifact,
        created_at=ended_at.isoformat(),
        config_ids=[
            *[f"../config/{path.name}" for path in _config_paths(cfg) if path.exists()],
            "../config/effective_config.json",
        ],
        split_manifest=split_manifest,
        git=git,
        result_ids=[
            f"../{_relative(path, crate_dir)}" for path in copied
            if path.is_file() and _relative(path, crate_dir).startswith(("metrics/", "predictions/", "figures/"))
        ],
    )
    validate_jsonld_document(fair4ml, "FAIR4ML")
    _write_json(fair4ml_path, fair4ml)
    copied.extend([croissant_path, fair4ml_path])
    validation_report = _write_validation_report(crate_dir, croissant, split_manifest)
    copied.append(validation_report)

    _write_readme(
        crate_dir, cfg, run_id, release_identifier, [],
        artifact=artifact, croissant_mode=settings.croissant.mode,
        code_mode=settings.code.mode, split_manifest=split_manifest,
        licenses=run["licenses"], git=git,
        reproducible_run=settings.reproducible_run,
        environment=settings.environment,
    )
    copied.append(crate_dir / "README.md")
    checksum_path = _write_checksum_manifest(crate_dir)
    copied.append(checksum_path)

    metadata = _build_rocrate_metadata(
        crate_dir=crate_dir,
        copied=copied,
        cfg=cfg,
        summary=summary,
        started_at=started_at,
        ended_at=ended_at,
        command=command or sys.argv,
        git=git,
        upstream_identifier=release_identifier,
        upstream_reference=upstream,
        upstream_settings=settings.upstream,
        resolved_inputs=resolved_inputs,
        derived_entities=derived_entities,
        input_catalogue_id=f"config/{settings.input_catalogue.name}" if settings.input_catalogue else None,
        run=run,
        artifact=artifact,
        fair4ml_model=fair4ml["@graph"][0],
        split_manifest=split_manifest,
        code_mode=settings.code.mode,
    )
    metadata_path = crate_dir / "ro-crate-metadata.json"
    _write_json(metadata_path, metadata)
    validate_rocrate_jsonld(metadata)
    _validate_input_links(metadata, croissant, resolved_inputs)
    _validate_specialized_metadata(metadata, croissant, fair4ml, artifact, crate_dir, run)
    _validate_jsonld_rdf(metadata, croissant, fair4ml)
    _validate_checksum_manifest(crate_dir)
    LOGGER.info("RO-Crate created and basically validated at %s", crate_dir)
    return crate_dir


def validate_rocrate_jsonld(document: dict[str, Any]) -> None:
    """Perform offline structural checks for RO-Crate JSON-LD."""
    if not isinstance(document.get("@context"), (str, list, dict)):
        raise ValueError("RO-Crate JSON-LD has no valid @context")
    graph = document.get("@graph")
    if not isinstance(graph, list) or not graph:
        raise ValueError("RO-Crate JSON-LD @graph must be a non-empty list")
    ids = [entity.get("@id") for entity in graph]
    if any(not isinstance(identifier, str) or not identifier for identifier in ids):
        raise ValueError("Every RO-Crate entity must have a non-empty @id")
    if len(ids) != len(set(ids)):
        raise ValueError("RO-Crate entity @id values must be unique")
    by_id = {entity["@id"]: entity for entity in graph}
    descriptor = by_id.get("ro-crate-metadata.json")
    root = by_id.get("./")
    if not descriptor or descriptor.get("about") != {"@id": "./"}:
        raise ValueError("RO-Crate metadata descriptor must be about ./")
    if not root or not _has_reference(root.get("conformsTo"), PROCESS_RUN_CRATE):
        raise ValueError("Root dataset does not declare Process Run Crate 0.5 conformance")
    actions = [entity for entity in graph if "CreateAction" in _types(entity)]
    if not actions or not actions[0].get("instrument"):
        raise ValueError("Process Run Crate requires a CreateAction with an instrument")


def validate_jsonld_document(document: dict[str, Any], label: str = "JSON-LD") -> None:
    """Check the minimum JSON-LD shape shared by auxiliary descriptors."""
    if not isinstance(document, dict) or "@context" not in document:
        raise ValueError(f"{label} document has no @context")
    if "@type" not in document and "@graph" not in document:
        raise ValueError(f"{label} document has neither @type nor @graph")
    if "@graph" in document:
        if not isinstance(document["@graph"], list) or not document["@graph"]:
            raise ValueError(f"{label} @graph must be a non-empty list")
        identifiers = [entity.get("@id") for entity in document["@graph"] if isinstance(entity, dict)]
        identifiers = [identifier for identifier in identifiers if identifier is not None]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError(f"{label} @id values must be unique")


def _build_rocrate_metadata(
    crate_dir: Path,
    copied: list[Path],
    cfg: dict[str, Any],
    summary: dict[str, Any],
    started_at: datetime,
    ended_at: datetime,
    command: list[str],
    git: dict[str, Any],
    upstream_identifier: str | None,
    resolved_inputs: list[ResolvedInput],
    derived_entities: list[dict[str, Any]],
    input_catalogue_id: str | None,
    run: dict[str, Any],
    artifact: dict[str, Any],
    fair4ml_model: dict[str, Any],
    split_manifest: dict[str, Any] | None = None,
    code_mode: str = "reference",
    upstream_reference: Any | None = None,
    upstream_settings: Any | None = None,
) -> dict[str, Any]:
    local_files = sorted({_relative(path, crate_dir) for path in copied if path.is_file()})
    ids = ensure_run_identifiers(run)
    run.setdefault("creator", CREATOR)
    run.setdefault("license", DATASET_LICENSE)
    licenses = run.get("licenses") or {
        "metadata": DATASET_LICENSE, "code": None, "model": None,
        "results": None, "derived_dataset": run.get("license"),
    }
    action_id = ids["train-model"]
    evaluation_action_id = ids["evaluate-model"]
    prepare_action_id = ids["prepare-dataset"]
    package_action_id = ids["package-rocrate"]
    training_input_ids = [item.training_entity_id for item in resolved_inputs]
    config_ids = [path for path in local_files if path.startswith("config/")]
    mask_ids = [item.training_entity_id for item in resolved_inputs if item.spec.kind == "mask"]
    processed_entities, partition_entities = _intermediate_entities(
        run, resolved_inputs, prepare_action_id
    )
    processed_ids = [entity["@id"] for entity in processed_entities]
    train_parts = [entity["@id"] for entity in partition_entities if entity["split"] == "train"]
    test_parts = [entity["@id"] for entity in partition_entities if entity["split"] == "test"]
    for entity in partition_entities:
        entity.pop("split", None)

    root = {
        "@id": "./",
        "@type": "Dataset",
        "name": f"PROFECIA training run: {cfg.get('model_run_name', 'model')}",
        "description": "Reproducibility package for one PROFECIA dataset preparation, model training and evaluation run.",
        "identifier": run["base_id"],
        "creator": {"@id": run["creator"]["@id"]},
        "license": {"@id": licenses["metadata"]} if licenses.get("metadata") else None,
        "version": git.get("commit"),
        "codeRepository": git.get("repository_url"),
        "isBasedOn": {"@id": git.get("commit_url")} if git.get("commit_url") else None,
        "datePublished": ended_at.isoformat(),
        "conformsTo": {"@id": PROCESS_RUN_CRATE},
        "hasPart": [{"@id": path} for path in local_files],
        "mentions": [
            {"@id": prepare_action_id}, {"@id": action_id}, {"@id": evaluation_action_id},
            {"@id": package_action_id}, {"@id": ids["source-code"]},
            {"@id": ids["resolve-upstream"]}, {"@id": ids["download-inputs"]},
            {"@id": ids["derive-inputs"]},
            {"@id": ids["apply-split"]},
            {"@id": ids["verify-reproduction"]},
            {"@id": ids["ml-dataset"]}, {"@id": ids["training-dataset"]},
            {"@id": ids["test-dataset"]}, {"@id": ids["ml-model"]},
            {"@id": ids["model-evaluation"]},
            {"@id": ids["reproduction-command"]},
        ],
    }
    root = {key: value for key, value in root.items() if value is not None}
    upstream_entity_id = ids["upstream-preprocessing-crate"] if upstream_identifier else None
    if upstream_entity_id:
        root["dcterms:source"] = {"@id": upstream_entity_id}
        root["prov:wasDerivedFrom"] = {"@id": upstream_entity_id}

    prepare_objects = [{"@id": identifier} for identifier in training_input_ids + config_ids]
    prepare_action = {
        "@id": prepare_action_id,
        "@type": "CreateAction",
        "name": "Prepare and partition the PROFECIA ML dataset",
        "description": "Transform configured NetCDF variables to intermediate arrays, apply masks, and create train/test partitions.",
        "actionStatus": {"@id": "http://schema.org/CompletedActionStatus"},
        "instrument": {"@id": ids["training-application"]},
        "object": prepare_objects,
        "prov:used": prepare_objects,
        "result": [
            *({"@id": identifier} for identifier in processed_ids),
            {"@id": ids["ml-dataset"]}, {"@id": ids["training-dataset"]},
            {"@id": ids["test-dataset"]},
            *([{"@id": ids["split-manifest"]}] if split_manifest else []),
        ],
    }
    resolve_upstream_action = {
        "@id": ids["resolve-upstream"], "@type": "CreateAction",
        "name": "Resolve the immutable upstream preprocessing RO-Crate",
        "actionStatus": {"@id": "http://schema.org/CompletedActionStatus"},
        "instrument": {"@id": ids["rocrate-generator"]},
        "object": {"@id": upstream_entity_id} if upstream_entity_id else [],
        "prov:used": {"@id": upstream_entity_id} if upstream_entity_id else [],
        "result": {"@id": "reproducibility/input_resolution.json"},
    }
    download_action = {
        "@id": ids["download-inputs"], "@type": "CreateAction",
        "name": "Download and verify required upstream inputs",
        "actionStatus": {"@id": "http://schema.org/CompletedActionStatus"},
        "instrument": {"@id": ids["reproduction-command"]},
        "object": [{"@id": item.upstream_entity["@id"]} for item in resolved_inputs],
        "prov:used": [{"@id": item.upstream_entity["@id"]} for item in resolved_inputs],
        "result": [{"@id": item.training_entity_id} for item in resolved_inputs],
    }
    derived_input_ids = [
        item.training_entity_id for item in resolved_inputs
        if item.training_entity_id != item.upstream_entity["@id"]
    ]
    derive_inputs_action = {
        "@id": ids["derive-inputs"], "@type": "CreateAction",
        "name": "Generate documented local derived inputs",
        "description": (
            "Apply the explicitly configured local input transformations."
            if derived_input_ids else "No local input transformation was required for this run."
        ),
        "actionStatus": {"@id": "http://schema.org/CompletedActionStatus"},
        "instrument": {"@id": ids["training-application"]},
        "object": [{"@id": item.upstream_entity["@id"]} for item in resolved_inputs],
        "prov:used": [{"@id": item.upstream_entity["@id"]} for item in resolved_inputs],
        "result": [{"@id": identifier} for identifier in derived_input_ids],
    }
    apply_split_action = {
        "@id": ids["apply-split"], "@type": "CreateAction",
        "name": "Apply the exact persisted train/test split",
        "actionStatus": {"@id": "http://schema.org/CompletedActionStatus"},
        "instrument": {"@id": ids["reproduction-command"]},
        "object": [
            {"@id": ids["ml-dataset"]},
            *([{"@id": ids["split-manifest"]}] if split_manifest else []),
        ],
        "prov:used": [
            {"@id": ids["ml-dataset"]},
            *([{"@id": ids["split-manifest"]}] if split_manifest else []),
        ],
        "result": [{"@id": ids["training-dataset"]}, {"@id": ids["test-dataset"]}],
    }
    verification_objects = [
        {"@id": "reproducibility/expected_results.json"},
        *([{"@id": ids["split-manifest"]}] if split_manifest else []),
    ]
    verify_reproduction_action = {
        "@id": ids["verify-reproduction"], "@type": "CreateAction",
        "name": "Verify a future reproduction against expected results",
        "actionStatus": {"@id": "http://schema.org/PotentialActionStatus"},
        "instrument": {"@id": ids["reproduction-command"]},
        "object": verification_objects,
        "prov:used": verification_objects,
        "result": {"@id": ids["reproduction-report"]},
    }
    training_objects = [{"@id": ids["training-dataset"]}]
    result_files = [
        path for path in local_files
        if path.startswith(("metrics/", "predictions/", "figures/", "model/"))
        and path not in {"fair/dataset.croissant.json", "fair/model.fair4ml.json"}
    ]
    action: dict[str, Any] = {
        "@id": action_id,
        "@type": "CreateAction",
        "name": "Train a PROFECIA model",
        "description": _shell_join(command),
        "startTime": started_at.isoformat(),
        "endTime": ended_at.isoformat(),
        "actionStatus": {"@id": "http://schema.org/CompletedActionStatus"},
        "instrument": {"@id": ids["training-application"]},
        "object": training_objects,
        "result": [
            {"@id": ids["ml-model"]}, {"@id": artifact["main_id"]},
        ],
        "prov:used": training_objects,
    }
    mlflow_run_id = summary.get("train_info", {}).get("mlflow_run_id")
    if mlflow_run_id:
        action["subjectOf"] = {"@id": ids["mlflow-run"]}
        root["mentions"].append({"@id": ids["mlflow-run"]})

    evaluation_objects = [{"@id": ids["ml-model"]}, {"@id": ids["test-dataset"]}]
    evaluation_action = {
        "@id": evaluation_action_id, "@type": "CreateAction",
        "name": "Evaluate a PROFECIA model on the test partition",
        "actionStatus": {"@id": "http://schema.org/CompletedActionStatus"},
        "instrument": {"@id": ids["training-application"]},
        "object": evaluation_objects, "prov:used": evaluation_objects,
        "result": [
            {"@id": ids["model-evaluation"]},
            *({"@id": path} for path in result_files if path != artifact["main_id"]),
        ],
        "endTime": ended_at.isoformat(),
    }

    repository = git.get("repository_url")
    commit = git.get("commit")
    commit_url = git.get("commit_url")
    branch = git.get("branch")
    branch_url = git.get("branch_url")
    dirty = bool(git.get("dirty"))
    common_code_properties = [
        {"@type": "PropertyValue", "name": "git_branch", "value": branch, "url": branch_url},
        {"@type": "PropertyValue", "name": "git_commit", "value": commit, "url": commit_url},
        {"@type": "PropertyValue", "name": "git_dirty", "value": dirty},
    ]
    software: dict[str, Any] = {
        "@id": ids["training-application"],
        "@type": "SoftwareApplication",
        "name": "PROFECIA configurable training workflow",
        "runtimePlatform": f"Python {platform.python_version()}",
        "softwareRequirements": [
            {"@id": identifier} for identifier in run.get("software_requirements", ["environment/python-packages.json"])
        ],
        "isBasedOn": {"@id": ids["source-code"]},
        "additionalProperty": common_code_properties,
    }
    if commit:
        software["softwareVersion"] = commit
        software["identifier"] = commit
    if commit_url:
        software["url"] = commit_url
    if repository:
        software["codeRepository"] = repository
    code = {
        "@id": ids["source-code"],
        "@type": "SoftwareSourceCode",
        "name": "PROFECIA source code state",
        "version": commit,
        "codeRepository": repository,
        "identifier": commit,
        "url": commit_url,
        "additionalProperty": common_code_properties,
    }
    if commit_url:
        code["sameAs"] = {"@id": commit_url}
    if licenses.get("code"):
        code["license"] = {"@id": licenses["code"]}
        software["license"] = {"@id": licenses["code"]}
    if dirty and git.get("diff_path"):
        code["hasPart"] = [{"@id": "code/git-diff.patch"}]
        code["description"] = (
            "The immutable Git commit identifies the base source. Because git_dirty=true, "
            "code/git-diff.patch plus that commit describe the source state actually executed."
        )
    if code_mode == "snapshot":
        code["hasPart"] = [{"@id": path} for path in local_files if path.startswith("code/")]
    elif code_mode == "patch" and git.get("untracked_paths"):
        code.setdefault("hasPart", []).extend(
            {"@id": _relative(path, crate_dir)} for path in git["untracked_paths"]
        )
    code.setdefault("additionalProperty", []).append(
        {"@type": "PropertyValue", "name": "code_packaging_mode", "value": code_mode}
    )
    dataset = {
        "@id": ids["ml-dataset"],
        "@type": "Dataset",
        "name": "Concrete masked and partitioned ML dataset",
        "subjectOf": {"@id": "fair/dataset.croissant.json"},
        "hasPart": [{"@id": ids["training-dataset"]}, {"@id": ids["test-dataset"]}],
        "prov:wasDerivedFrom": [{"@id": identifier} for identifier in processed_ids + mask_ids],
        "prov:wasGeneratedBy": {"@id": prepare_action_id},
        "additionalProperty": _dataset_properties(cfg, summary, resolved_inputs),
    }
    if upstream_entity_id:
        dataset["dcterms:source"] = {"@id": upstream_entity_id}
    if licenses.get("derived_dataset"):
        dataset["license"] = {"@id": licenses["derived_dataset"]}
    else:
        dataset["conditionsOfAccess"] = "License for the derived ML dataset has not yet been established."
    if split_manifest:
        dataset["subjectOf"] = [dataset["subjectOf"], {"@id": ids["split-manifest"]}]
    training_dataset = {
        "@id": ids["training-dataset"], "@type": ["Dataset", "cr:Dataset"],
        "name": "PROFECIA training partition", "isPartOf": {"@id": ids["ml-dataset"]},
        "hasPart": [{"@id": identifier} for identifier in train_parts],
        "prov:wasDerivedFrom": [{"@id": identifier} for identifier in processed_ids + mask_ids],
        "prov:wasGeneratedBy": {"@id": prepare_action_id},
        "additionalProperty": {"@type": "PropertyValue", "name": "number_of_observations", "value": run["dataset"].get("n_train")},
    }
    test_dataset = {
        "@id": ids["test-dataset"], "@type": ["Dataset", "cr:Dataset"],
        "name": "PROFECIA test partition", "isPartOf": {"@id": ids["ml-dataset"]},
        "hasPart": [{"@id": identifier} for identifier in test_parts],
        "prov:wasDerivedFrom": [{"@id": identifier} for identifier in processed_ids + mask_ids],
        "prov:wasGeneratedBy": {"@id": prepare_action_id},
        "additionalProperty": {"@type": "PropertyValue", "name": "number_of_observations", "value": run["dataset"].get("n_test")},
    }
    if split_manifest:
        for split_dataset in (training_dataset, test_dataset):
            split_dataset["subjectOf"] = {"@id": ids["split-manifest"]}
            split_dataset["hasPart"].append({"@id": ids["split-manifest"]})
    for split_dataset in (training_dataset, test_dataset):
        if licenses.get("derived_dataset"):
            split_dataset["license"] = {"@id": licenses["derived_dataset"]}
        else:
            split_dataset["conditionsOfAccess"] = "License for the derived ML dataset has not yet been established."
    model_entity = {
        "@id": ids["ml-model"],
        "@type": ["CreativeWork", "fair4ml:MLModel"],
        "name": run["train"].get("model_name", "Trained model"),
        "subjectOf": {"@id": "fair/model.fair4ml.json"},
        "prov:wasGeneratedBy": {"@id": action_id},
        "fair4ml:trainedOn": {"@id": ids["training-dataset"]},
        "fair4ml:testedOn": {"@id": ids["test-dataset"]},
        "fair4ml:evaluatedWith": {"@id": ids["model-evaluation"]},
        "fair4ml:modelCategory": "Supervised learning",
        "fair4ml:mlTask": "Regression",
        "fair4ml:modelRisksBiasLimitations": fair4ml_model["fair4ml:modelRisksBiasLimitations"],
        "fair4ml:usageInstructions": fair4ml_model["fair4ml:usageInstructions"],
        "additionalProperty": fair4ml_model["additionalProperty"],
        "associatedMedia": {"@id": artifact["main_id"]},
        "isBasedOn": [
            {"@id": ids["ml-dataset"]}, {"@id": ids["source-code"]},
            *({"@id": identifier} for identifier in config_ids),
        ],
        "softwareRequirements": {"@id": ids["training-application"]},
    }
    if licenses.get("model"):
        model_entity["license"] = {"@id": licenses["model"]}
    else:
        model_entity["conditionsOfAccess"] = "License for the trained model has not yet been established."
    if split_manifest:
        model_entity["subjectOf"] = [model_entity["subjectOf"], {"@id": ids["split-manifest"]}]
    evaluation_entity = {
        "@id": ids["model-evaluation"], "@type": ["CreativeWork", "fair4ml:MLModelEvaluation"],
        "fair4ml:evaluatedMLModel": {"@id": ids["ml-model"]},
        "fair4ml:evaluationDataset": {"@id": ids["test-dataset"]},
        "fair4ml:evaluationSoftware": {"@id": ids["training-application"]},
        "fair4ml:evaluationMetrics": _metric_values(run.get("metrics") or {}),
        "prov:wasGeneratedBy": {"@id": evaluation_action_id},
    }
    package_objects = [
        {"@id": action_id}, {"@id": evaluation_action_id},
        {"@id": ids["ml-dataset"]}, {"@id": ids["ml-model"]},
        *({"@id": path} for path in local_files if path.startswith("metadata/")),
    ]
    package_action = {
        "@id": package_action_id, "@type": "CreateAction",
        "name": "Generate Croissant, FAIR4ML and RO-Crate metadata",
        "actionStatus": {"@id": "http://schema.org/CompletedActionStatus"},
        "instrument": {"@id": ids["rocrate-generator"]},
        "object": package_objects,
        "prov:used": package_objects,
        "result": [
            {"@id": "fair/dataset.croissant.json"},
            {"@id": "fair/model.fair4ml.json"},
            {"@id": "ro-crate-metadata.json"},
        ],
    }
    packaging_software = {
        "@id": ids["rocrate-generator"], "@type": "SoftwareApplication",
        "name": "PROFECIA reproducibility metadata generator",
        "isBasedOn": {"@id": ids["source-code"]},
        "softwareVersion": commit,
    }

    graph: list[dict[str, Any]] = [
        {
            "@id": "ro-crate-metadata.json", "@type": "CreativeWork",
            "conformsTo": {"@id": RO_CRATE}, "about": {"@id": "./"},
            **({"license": {"@id": licenses["metadata"]}} if licenses.get("metadata") else {}),
        },
        root,
        {"@id": PROCESS_RUN_CRATE, "@type": "CreativeWork", "name": "Process Run Crate", "version": "0.5"},
        prepare_action,
        resolve_upstream_action,
        download_action,
        derive_inputs_action,
        apply_split_action,
        verify_reproduction_action,
        action,
        evaluation_action,
        package_action,
        software,
        packaging_software,
        code,
        {
            "@id": ids["reproduction-command"], "@type": "SoftwareApplication",
            "name": "PROFECIA RO-Crate reproduction command",
            "executableName": "./reproduce.sh",
            "softwareRequirements": [{"@id": identifier} for identifier in run.get("software_requirements", [])],
            "isBasedOn": {"@id": ids["source-code"]},
        },
        {
            "@id": ids["reproduction-report"], "@type": "CreativeWork",
            "name": "Reproduction report produced by a future execution",
            "conditionsOfAccess": "Not available until reproduce.sh is executed.",
        },
        dataset,
        training_dataset,
        test_dataset,
        model_entity,
        evaluation_entity,
    ]
    graph.extend(_file_entity(path, crate_dir) for path in local_files)
    for entity in graph:
        if "File" in _types(entity) and entity["@id"].startswith(("metrics/", "predictions/", "figures/")):
            if licenses.get("results"):
                entity["license"] = {"@id": licenses["results"]}
            else:
                entity["conditionsOfAccess"] = "License for generated results has not yet been established."
    specialized = {
        "fair/dataset.croissant.json": {
            "@id": "fair/dataset.croissant.json", "@type": ["File", "CreativeWork"],
            "encodingFormat": 'application/ld+json; profile="http://mlcommons.org/croissant/1.1"',
            "conformsTo": {"@id": "http://mlcommons.org/croissant/1.1"},
            "about": {"@id": ids["ml-dataset"]}, "prov:wasGeneratedBy": {"@id": package_action_id},
            **({"license": {"@id": licenses["metadata"]}} if licenses.get("metadata") else {}),
        },
        "fair/model.fair4ml.json": {
            "@id": "fair/model.fair4ml.json", "@type": ["File", "CreativeWork"],
            "encodingFormat": "application/ld+json",
            "conformsTo": {"@id": "https://w3id.org/fair4ml/0.1.0"},
            "about": {"@id": ids["ml-model"]}, "prov:wasGeneratedBy": {"@id": package_action_id},
            **({"license": {"@id": licenses["metadata"]}} if licenses.get("metadata") else {}),
        },
    }
    graph = [{**entity, **specialized[entity["@id"]]} if entity["@id"] in specialized else entity for entity in graph]
    graph.extend(processed_entities)
    graph.extend(partition_entities)
    artifact_entity = dict(artifact["entity"])
    artifact_entity["@id"] = artifact["main_id"]
    artifact_entity["prov:wasGeneratedBy"] = {"@id": action_id}
    if licenses.get("model"):
        artifact_entity["license"] = {"@id": licenses["model"]}
    graph = [artifact_entity if entity["@id"] == artifact["main_id"] else entity for entity in graph]
    if artifact["main_id"] not in {entity["@id"] for entity in graph}:
        graph.append(artifact_entity)
    derived_by_id = {entity["@id"]: entity for entity in derived_entities}
    graph = [derived_by_id.get(entity["@id"], entity) for entity in graph]
    patch_entity = next((entity for entity in graph if entity["@id"] == "code/git-diff.patch"), None)
    if patch_entity:
        patch_entity["about"] = {"@id": ids["source-code"]}
        patch_entity["description"] = "Patch against the recorded Git commit for the executed dirty source state."
    if upstream_entity_id:
        upstream_root = _upstream_root(upstream_reference.metadata if upstream_reference else None)
        upstream_entity = {
            "@id": upstream_entity_id, "@type": "Dataset",
            "name": "Upstream preprocessing RO-Crate release",
            "identifier": upstream_identifier,
            "url": getattr(upstream_settings, "landing_page", None) or upstream_identifier,
            "subjectOf": {"@id": getattr(upstream_settings, "metadata_url", None) or getattr(upstream_reference, "metadata_url", None)},
            "dateModified": upstream_root.get("datePublished") or upstream_root.get("dateModified"),
            "version": upstream_root.get("version"),
            "dateAccessed": getattr(upstream_reference, "accessed_at", None),
            "sha256": getattr(upstream_reference, "metadata_sha256", None),
            "conformsTo": {"@id": RO_CRATE},
        }
        graph.append({key: value for key, value in upstream_entity.items() if value is not None})
        metadata_url = getattr(upstream_settings, "metadata_url", None) or getattr(upstream_reference, "metadata_url", None)
        if metadata_url:
            graph.append({
                "@id": metadata_url, "@type": "CreativeWork",
                "name": "Upstream ro-crate-metadata.json",
                "sha256": getattr(upstream_reference, "metadata_sha256", None),
                "about": {"@id": upstream_entity_id},
            })
    existing_ids = {entity["@id"] for entity in graph}
    upstream_entities = _unique_entities(item.upstream_entity for item in resolved_inputs)
    graph.extend(entity for entity in upstream_entities if entity["@id"] not in existing_ids)
    derivation_actions = _derivation_actions(
        resolved_inputs, input_catalogue_id, ids["training-application"]
    )
    derivation_actions.extend(_local_input_transformation_actions(
        resolved_inputs, input_catalogue_id, ids["training-application"], run["base_id"]
    ))
    graph.extend(derivation_actions)
    root["mentions"].extend({"@id": entity["@id"]} for entity in derivation_actions)
    if mlflow_run_id:
        graph.append(_mlflow_entity(cfg, summary, mlflow_run_id, ids["mlflow-run"]))
    if run.get("cv", {}).get("enabled"):
        cv = run["cv"]
        graph.append({
            "@id": ids["cross-validation"], "@type": "CreateAction",
            "name": f"Internal {cv.get('cv_class', 'cross-validation')}",
            "description": "Internal validation folds over the training partition; no fixed validation dataset was persisted.",
            "instrument": {"@id": ids["training-application"]},
            "object": {"@id": ids["training-dataset"]},
            "result": {"@id": ids["ml-model"]},
            "additionalProperty": [
                {"@type": "PropertyValue", "name": key, "value": cv.get(key)}
                for key in ("cv_class", "n_splits", "n_iter", "scoring", "best_score")
            ],
        })
        root["mentions"].append({"@id": ids["cross-validation"]})
    graph.append(run["creator"])
    for license_id in dict.fromkeys(value for value in licenses.values() if value):
        graph.append({"@id": license_id, "@type": "CreativeWork", "name": "Configured license"})
    if run.get("container_entity"):
        graph.append(run["container_entity"])
    if split_manifest:
        manifest_entity = _file_entity("fair/split_manifest.npz", crate_dir)
        manifest_entity.update({
            "@id": ids["split-manifest"],
            "name": "Exact train/test split manifest",
            "contentUrl": "fair/split_manifest.npz",
            "encodingFormat": "application/x-npz",
            "about": [
                {"@id": ids["ml-dataset"]}, {"@id": ids["training-dataset"]},
                {"@id": ids["test-dataset"]},
            ],
            "prov:wasGeneratedBy": {"@id": prepare_action_id},
            "subjectOf": {"@id": "fair/split_manifest.json"},
            "additionalProperty": [
                {"@type": "PropertyValue", "name": key, "value": value}
                for key, value in split_manifest["metadata"].items()
                if key not in {"identifier", "sha256", "contentSize"}
            ],
        })
        graph.append(manifest_entity)
    return {
        "@context": [
            "https://w3id.org/ro/crate/1.1/context",
            "https://w3id.org/ro/terms/workflow-run/context",
            {
                "prov": "http://www.w3.org/ns/prov#", "dcterms": "http://purl.org/dc/terms/",
                "cr": "http://mlcommons.org/croissant/", "fair4ml": "https://w3id.org/fair4ml#",
                "codemeta": "https://w3id.org/codemeta/",
            },
        ],
        "@graph": graph,
    }


def _intermediate_entities(
    run: dict[str, Any], resolved_inputs: list[ResolvedInput], action_id: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    upstream_by_name = {
        item.spec.key.split(":", 1)[1].upper(): item.training_entity_id
        for item in resolved_inputs
        if item.spec.kind == "variable"
    }
    processed = []
    for item in run.get("processed_variables", []):
        entity: dict[str, Any] = {
            "@id": item["entity_id"],
            "@type": "MediaObject",
            "name": f"Intermediate processed array for {item['name']}",
            "encodingFormat": "application/x-npy",
            "conditionsOfAccess": "Local intermediate array; intentionally not included in this RO-Crate.",
            "isAccessibleForFree": False,
            "variableMeasured": item["name"],
            "prov:wasGeneratedBy": {"@id": action_id},
            "additionalProperty": [
                {"@type": "PropertyValue", "name": "shape", "value": item.get("shape")},
                {"@type": "PropertyValue", "name": "dtype", "value": item.get("dtype")},
                {"@type": "PropertyValue", "name": "units", "value": item.get("units")},
                {"@type": "PropertyValue", "name": "grid", "value": item.get("grid")},
                {"@type": "PropertyValue", "name": "temporal_grid", "value": item.get("temporal_grid")},
                {"@type": "PropertyValue", "name": "processing", "value": item.get("processing")},
            ],
        }
        path = item.get("path")
        if path and path.is_file():
            entity.update(file_facts(path))
        upstream_id = upstream_by_name.get(str(item["name"]).upper())
        if upstream_id:
            entity["prov:wasDerivedFrom"] = {"@id": upstream_id}
        processed.append(entity)

    processed_ids = [entity["@id"] for entity in processed]
    partition = []
    for item in run.get("partition_arrays", []):
        partition.append(
            {
                "@id": item["entity_id"],
                "@type": "MediaObject",
                "name": item["path"].name,
                "encodingFormat": "application/x-npy",
                "contentSize": item["contentSize"],
                "sha256": item["sha256"],
                "conditionsOfAccess": "Local train/test intermediate; intentionally not included in this RO-Crate.",
                "isAccessibleForFree": False,
                "prov:wasDerivedFrom": [{"@id": identifier} for identifier in processed_ids],
                "prov:wasGeneratedBy": {"@id": action_id},
                "split": item["split"],
                "additionalProperty": {"@type": "PropertyValue", "name": "array_role", "value": item["role"]},
            }
        )
    return processed, partition


def _metric_values(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"@type": "PropertyValue", "name": name, "value": value, "measurementTechnique": name}
        for name, value in metrics.items()
        if isinstance(value, (int, float))
    ]


def _copy_run_files(
    crate_dir: Path,
    cfg: dict[str, Any],
    summary: dict[str, Any],
    project_root: Path,
    started_at: datetime,
    model_source: Path | None,
    code_mode: str = "reference",
) -> list[Path]:
    copied: list[Path] = []
    summary_copy = crate_dir / "metrics" / "workflow_summary.json"
    _write_json(summary_copy, _portable_metadata(summary))
    copied.append(summary_copy)
    for source in _config_paths(cfg):
        if source.exists():
            target = crate_dir / "config" / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            _copy_sanitized_toml(source, target)
            copied.append(target)

    if code_mode == "snapshot":
        try:
            tracked = subprocess.run(
                ["git", "ls-files"], cwd=project_root, check=True,
                capture_output=True, text=True,
            ).stdout.splitlines()
        except (OSError, subprocess.CalledProcessError) as exc:
            raise ValueError("code.mode=snapshot requires a readable Git working tree") from exc
        for relative in tracked:
            source = project_root / relative
            if not source.is_file():
                continue
            target = crate_dir / "code" / "snapshot" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if source.suffix.lower() == ".toml":
                _copy_sanitized_toml(source, target)
            else:
                shutil.copy2(source, target)
            copied.append(target)

    log_path = summary.get("log_path")
    if log_path and Path(log_path).is_file():
        source = Path(log_path)
        target = crate_dir / "logs" / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied.append(target)

    mappings: list[tuple[Path, str]] = []
    if cfg.get("model_artifacts_dir"):
        mappings.append((Path(cfg["model_artifacts_dir"]), "model"))
    if cfg.get("model_figures_dir"):
        mappings.append((Path(cfg["model_figures_dir"]), "figures"))
    if cfg.get("model_dir"):
        mappings.append((Path(cfg["model_dir"]) / "reports", "reports"))
    for source_dir, category in mappings:
        if not source_dir.exists():
            continue
        for source in source_dir.rglob("*"):
            if not source.is_file() or source.suffix.lower() in {".nc", ".nc4"}:
                continue
            if model_source and source.resolve() == model_source.resolve():
                continue
            # Model directories are reused by the existing pipeline. Exclude
            # stale artifacts left by earlier runs from this per-run crate.
            if source.stat().st_mtime < started_at.timestamp() - 2:
                continue
            destination_category = category
            if category == "reports":
                destination_category = "predictions" if "prediction" in source.name.lower() else "metrics"
            target = crate_dir / destination_category / source.relative_to(source_dir)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            copied.append(target)
    return copied


def _copy_intermediate_metadata(crate_dir: Path, run: dict[str, Any]) -> list[Path]:
    copied = []
    names = {
        "processed_metadata": "processed_metadata.json",
        "processed_run_config": "processed_run_config.json",
        "dataset_metadata": "dataset_metadata.json",
        "split_metadata": "split_metadata.json",
        "train_info": "train_info.json",
    }
    for key, source in run.get("metadata_files", {}).items():
        target = crate_dir / "metadata" / names[key]
        target.parent.mkdir(parents=True, exist_ok=True)
        value = json.loads(source.read_text(encoding="utf-8"))
        _write_json(target, _portable_metadata(value))
        copied.append(target)
    return copied


def _model_source(cfg: dict[str, Any], summary: dict[str, Any]) -> Path | None:
    value = (summary.get("saved_paths") or {}).get("model_path")
    if value:
        return Path(value)
    if cfg.get("model_artifacts_dir"):
        candidate = Path(cfg["model_artifacts_dir"]) / "model.joblib"
        if candidate.is_file():
            return candidate
    return None


def _prepare_model_artifact(
    crate_dir: Path, settings: Any, source: Path | None, contextual_id: str = "#model-artifact"
) -> tuple[dict[str, Any], Path | None]:
    mode = settings.mode
    if mode in {"attached", "external"} and (source is None or not source.is_file()):
        raise ValueError(f"model_artifact.mode={mode} requires the generated model file")
    facts = file_facts(source) if source and source.is_file() else {}
    if mode == "attached":
        target = crate_dir / "model" / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        entity = {
            "@id": f"model/{target.name}",
            "@type": "File",
            "name": "Trained PROFECIA model",
            "encodingFormat": "application/octet-stream",
            **facts,
        }
        return {
            "mode": mode,
            "main_id": entity["@id"],
            "fair_id": f"../{entity['@id']}",
            "entity": entity,
        }, target
    if mode == "external":
        download_url = settings.download_url
        parsed = urlparse(download_url or "")
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("model_artifact.mode=external requires an absolute HTTP(S) download_url")
        entity = {
            "@id": download_url,
            "@type": "File",
            "name": "Trained PROFECIA model",
            "encodingFormat": "application/octet-stream",
            "contentUrl": download_url,
            **facts,
        }
        if settings.landing_page:
            entity["url"] = settings.landing_page
        if settings.identifier:
            entity["identifier"] = settings.identifier
        return {"mode": mode, "main_id": download_url, "fair_id": download_url, "entity": entity}, None
    entity = {
        "@id": contextual_id,
        "@type": "CreativeWork",
        "name": source.name if source else "Trained PROFECIA model artifact",
        "alternateName": "Trained PROFECIA model artifact (not available)",
        "encodingFormat": "application/octet-stream",
        "conditionsOfAccess": "The generated model artifact was intentionally not attached or published.",
        "isAccessibleForFree": False,
        "additionalProperty": {"@type": "PropertyValue", "name": "reusable", "value": False},
        **facts,
    }
    return {
        "mode": mode,
        "main_id": entity["@id"],
        "fair_id": entity["@id"],
        "entity": entity,
    }, None


def _materialize_derived_inputs(
    crate_dir: Path, resolved_inputs: list[ResolvedInput]
) -> tuple[list[dict[str, Any]], list[Path]]:
    entities: list[dict[str, Any]] = []
    paths: list[Path] = []
    for item in resolved_inputs:
        if not item.is_derived:
            continue
        if item.local_path is None or not item.local_path.is_file():
            raise UpstreamResolutionError(
                f"{item.spec.key}: derived mask file is required but missing: {item.local_path}"
            )
        _validate_derived_class_mask(item)
        target = crate_dir / item.training_entity_id
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item.local_path, target)
        paths.append(target)
        entity: dict[str, Any] = {
            "@id": item.training_entity_id,
            "@type": "File",
            "name": _path_name(item.spec.local_file),
            "encodingFormat": "application/x-npy",
            "contentSize": target.stat().st_size,
            "sha256": item.local_sha256,
            "prov:wasDerivedFrom": {"@id": item.upstream_entity["@id"]},
            "prov:wasGeneratedBy": {"@id": f"#derive-{item.spec.key.replace(':', '-')}"},
            "variableMeasured": item.spec.class_label or item.spec.key.split(":", 1)[-1],
        }
        entities.append(entity)
    return entities, paths


def _local_modified_input_entities(
    run: dict[str, Any], resolved_inputs: list[ResolvedInput]
) -> list[dict[str, Any]]:
    processed = {str(item.get("name", "")).upper(): item for item in run.get("processed_variables", [])}
    entities = []
    for item in resolved_inputs:
        if not item.locally_modified_entity_id:
            continue
        name = item.spec.key.split(":", 1)[-1]
        metadata = processed.get(name.upper(), {})
        entity = {
            "@id": item.locally_modified_entity_id, "@type": "MediaObject",
            "name": f"Local input state used for {item.spec.key}",
            "encodingFormat": item.upstream_entity.get("encodingFormat") or "application/x-netcdf",
            "sha256": item.local_sha256,
            "conditionsOfAccess": "Local derived input used by the original run; not distributed in this crate.",
            "prov:wasDerivedFrom": {"@id": item.upstream_entity["@id"]},
            "prov:wasGeneratedBy": {"@id": f"{run['base_id']}#transform-{_slug(item.spec.key)}"},
            "variableMeasured": item.spec.netcdf_variable or name,
            "additionalProperty": [
                {"@type": "PropertyValue", "name": "upstream_sha256", "value": item.upstream_entity.get("sha256")},
                {"@type": "PropertyValue", "name": "dimensions", "value": metadata.get("shape")},
                {"@type": "PropertyValue", "name": "grid", "value": metadata.get("grid")},
                {"@type": "PropertyValue", "name": "temporal_grid", "value": metadata.get("temporal_grid")},
                {"@type": "PropertyValue", "name": "processing", "value": metadata.get("processing")},
            ],
        }
        if item.local_path and item.local_path.is_file():
            entity["dateModified"] = datetime.fromtimestamp(item.local_path.stat().st_mtime, UTC).isoformat()
            entity["contentSize"] = str(item.local_path.stat().st_size)
        entities.append(entity)
    return entities


def _validate_derived_class_mask(item: ResolvedInput) -> None:
    """Verify a declared class-selection derivation against the local arrays."""
    if item.spec.derivation_type != "class_selection":
        return
    if item.spec.class_value is None or not item.spec.source_local_file or item.local_path is None:
        raise UpstreamResolutionError(
            f"{item.spec.key}: class_selection requires class_value, source_local_file and local_file"
        )
    source_path = item.local_path.parent / item.spec.source_local_file
    if not source_path.is_file():
        raise UpstreamResolutionError(
            f"{item.spec.key}: categorical source mask is missing: {source_path}"
        )
    try:
        import numpy as np

        source = np.load(source_path, allow_pickle=False)
        derived = np.load(item.local_path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise UpstreamResolutionError(
            f"{item.spec.key}: could not validate the declared class-selection mask derivation: {exc}"
        ) from exc
    expected = source == item.spec.class_value
    if derived.shape != expected.shape or not np.array_equal(derived, expected):
        raise UpstreamResolutionError(
            f"{item.spec.key}: derived mask does not equal source mask == {item.spec.class_value}"
        )


def _derivation_actions(
    resolved_inputs: list[ResolvedInput], input_catalogue_id: str | None,
    training_application_id: str = "#training-application",
) -> list[dict[str, Any]]:
    actions = []
    for item in resolved_inputs:
        if not item.is_derived:
            continue
        objects = [{"@id": item.upstream_entity["@id"]}]
        if input_catalogue_id:
            objects.append({"@id": input_catalogue_id})
        actions.append(
            {
                "@id": f"#derive-{item.spec.key.replace(':', '-')}",
                "@type": "CreateAction",
                "name": f"Derive {item.spec.key}",
                "description": (
                    f"Select class {item.spec.class_value} ({item.spec.class_label}) from the categorical "
                    "land-cover mask to create the binary exclusion mask."
                ),
                "instrument": {"@id": training_application_id},
                "object": objects,
                "prov:used": objects,
                "result": {"@id": item.training_entity_id},
                "actionStatus": {"@id": "http://schema.org/CompletedActionStatus"},
            }
        )
    return actions


def _local_input_transformation_actions(
    resolved_inputs: list[ResolvedInput], input_catalogue_id: str | None,
    training_application_id: str, base_id: str,
) -> list[dict[str, Any]]:
    actions = []
    for item in resolved_inputs:
        if not item.locally_modified_entity_id:
            continue
        objects = [{"@id": item.upstream_entity["@id"]}]
        if input_catalogue_id:
            objects.append({"@id": input_catalogue_id})
        actions.append({
            "@id": f"{base_id}#transform-{_slug(item.spec.key)}", "@type": "CreateAction",
            "name": f"Prepare local input for {item.spec.key}",
            "description": "Apply the recorded PROFECIA loading and preprocessing configuration to the upstream resource.",
            "instrument": {"@id": training_application_id}, "object": objects,
            "prov:used": objects, "result": {"@id": item.locally_modified_entity_id},
            "actionStatus": {"@id": "http://schema.org/CompletedActionStatus"},
        })
    return actions


def _validate_input_links(
    metadata: dict[str, Any], croissant: dict[str, Any], resolved_inputs: list[ResolvedInput]
) -> None:
    graph = {entity["@id"]: entity for entity in metadata["@graph"]}
    prepare = next(
        entity for entity in metadata["@graph"]
        if entity.get("@type") == "CreateAction"
        and entity.get("name") == "Prepare and partition the PROFECIA ML dataset"
    )
    expected = {item.training_entity_id for item in resolved_inputs}
    object_ids = _reference_ids(prepare.get("object"))
    used_ids = _reference_ids(prepare.get("prov:used"))
    processed_ids = {
        entity_id for entity_id in graph
        if "#processed-array-" in entity_id
    }
    training_id = next(identifier for identifier in graph if identifier.endswith("#training-dataset"))
    test_id = next(identifier for identifier in graph if identifier.endswith("#test-dataset"))
    training_derived = _reference_ids(graph[training_id].get("prov:wasDerivedFrom"))
    test_derived = _reference_ids(graph[test_id].get("prov:wasDerivedFrom"))
    distribution_ids = {item.get("@id") for item in croissant.get("distribution", [])}
    downloadable_expected = {
        identifier
        for item in resolved_inputs
        for identifier in (
            ([item.training_entity_id] if item.is_derived else [])
            + ([item.upstream_entity["@id"]] if _has_direct_download_url(item.upstream_entity) else [])
        )
    }
    failures = {
        "dataset-preparation.object": expected - object_ids,
        "dataset-preparation.prov:used": expected - used_ids,
        "#training-dataset.prov:wasDerivedFrom": processed_ids - training_derived,
        "#test-dataset.prov:wasDerivedFrom": processed_ids - test_derived,
        "Croissant.distribution": {
            identifier for identifier in downloadable_expected
            if identifier not in distribution_ids and f"../{identifier}" not in distribution_ids
        },
    }
    failures = {name: sorted(values) for name, values in failures.items() if values}
    if failures:
        raise ValueError(f"RO-Crate input-link validation failed: {failures}")
    croissant_by_id = {item.get("@id"): item for item in croissant.get("distribution", [])}
    for resolved in resolved_inputs:
        if resolved.is_derived:
            continue
        upstream_url = resolved.upstream_entity.get("contentUrl") or resolved.upstream_entity.get("dcat:downloadURL")
        if isinstance(upstream_url, dict):
            upstream_url = upstream_url.get("@id")
        parsed_upstream = urlparse(str(upstream_url or ""))
        if parsed_upstream.scheme not in {"http", "https"} or not parsed_upstream.netloc:
            continue
        identifier = resolved.upstream_entity["@id"]
        candidate = croissant_by_id.get(identifier) or croissant_by_id.get(f"../{identifier}")
        if not candidate or candidate.get("contentUrl") != upstream_url:
            raise ValueError(
                f"Croissant must preserve the exact upstream download URL for {resolved.spec.key}"
            )


def _validate_specialized_metadata(
    metadata: dict[str, Any],
    croissant: dict[str, Any],
    fair4ml: dict[str, Any],
    artifact: dict[str, Any],
    crate_dir: Path,
    run: dict[str, Any],
) -> None:
    if croissant.get("dct:conformsTo") != "http://mlcommons.org/croissant/1.1":
        raise ValueError("Croissant descriptor does not declare Croissant 1.1")
    fair_graph = {entity["@id"]: entity for entity in fair4ml.get("@graph", [])}
    ids = run["identifiers"]
    model = fair_graph.get(ids["ml-model"])
    if not model or not _has_reference(model.get("conformsTo"), "https://w3id.org/fair4ml/0.1.0"):
        raise ValueError("FAIR4ML model does not declare FAIR4ML 0.1.0")
    graph = {entity["@id"]: entity for entity in metadata["@graph"]}
    for required in (
        ids["ml-dataset"], ids["training-dataset"], ids["test-dataset"],
        ids["ml-model"], ids["model-evaluation"],
    ):
        if required not in graph:
            raise ValueError(f"RO-Crate graph is missing required entity {required}")
    if any(identifier.endswith("#validation-dataset") for identifier in graph):
        raise ValueError("A validation dataset must not be asserted unless a fixed validation split exists")
    training = next(
        entity for entity in metadata["@graph"]
        if entity.get("name") == "Train a PROFECIA model"
    )
    if "fair/dataset.croissant.json" in _reference_ids(training.get("object")):
        raise ValueError("Croissant metadata must not be an input of model training")
    if artifact["mode"] == "attached" and not (crate_dir / artifact["main_id"]).is_file():
        raise ValueError("Attached model artifact is missing from the RO-Crate")
    if artifact["mode"] == "external" and not artifact["main_id"].startswith(("http://", "https://")):
        raise ValueError("External model artifact must use its absolute download URL as @id")
    if artifact["mode"] == "metadata_only" and graph[artifact["main_id"]].get("isAccessibleForFree") is not False:
        raise ValueError("Metadata-only artifact must be explicitly unavailable")
    graph_ids = set(graph)
    for entity in metadata["@graph"]:
        for reference in _walk_references(entity):
            if (reference.startswith("#") or reference.startswith(run["base_id"] + "#")) and reference not in graph_ids:
                raise ValueError(f"Broken internal RO-Crate reference: {reference}")
    for entity in metadata["@graph"]:
        if "File" not in _types(entity):
            continue
        identifier = entity["@id"]
        if identifier.startswith(("http://", "https://", "urn:")):
            continue
        path = crate_dir / identifier
        if path.is_file():
            facts = file_facts(path)
            if entity.get("sha256") != facts["sha256"] or str(entity.get("contentSize")) != facts["contentSize"]:
                raise ValueError(f"Local File entity has missing or invalid size/checksum: {identifier}")
    croissant_refs = set(_walk_all_ids(croissant))
    fair_ids = set(fair_graph)
    for core in (
        ids["ml-dataset"], ids["training-dataset"], ids["test-dataset"],
        ids["training-application"],
    ):
        if core != croissant.get("@id") and core not in croissant_refs:
            raise ValueError(f"Croissant does not use the shared identifier {core}")
    for core in (
        ids["ml-dataset"], ids["training-dataset"], ids["test-dataset"],
        ids["ml-model"], ids["model-evaluation"], ids["training-application"],
    ):
        if core not in fair_ids:
            raise ValueError(f"FAIR4ML does not use the shared identifier {core}")
    if model.get("fair4ml:trainedOn") != {"@id": ids["training-dataset"]}:
        raise ValueError("FAIR4ML trainedOn must reference the training dataset")
    if model.get("fair4ml:testedOn") != {"@id": ids["test-dataset"]}:
        raise ValueError("FAIR4ML testedOn must reference the test dataset")
    fair_dir = crate_dir / "fair"
    for file_object in croissant.get("distribution", []):
        content_url = file_object.get("contentUrl")
        if not content_url:
            raise ValueError(f"Croissant FileObject has no contentUrl: {file_object.get('@id')}")
        parsed = urlparse(str(content_url))
        if parsed.scheme:
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise ValueError(f"Croissant remote FileObject has invalid URL: {content_url}")
        else:
            resolved = (fair_dir / str(content_url)).resolve()
            if not resolved.is_file() or not resolved.is_relative_to(crate_dir.resolve()):
                raise ValueError(f"Croissant local FileObject does not resolve inside the crate: {content_url}")
    manifest_path = fair_dir / "split_manifest.npz"
    if manifest_path.is_file():
        _validate_split_manifest(crate_dir, run)


def _validate_split_manifest(crate_dir: Path, run: dict[str, Any]) -> None:
    import numpy as np

    from .run_metadata import decode_observation_indices

    metadata = json.loads((crate_dir / "fair" / "split_manifest.json").read_text(encoding="utf-8"))
    path = crate_dir / "fair" / "split_manifest.npz"
    if file_facts(path)["sha256"] != metadata.get("sha256"):
        raise ValueError("Split manifest SHA-256 does not match split_manifest.json")
    with np.load(path, allow_pickle=False) as payload:
        train = payload["train_indices"]
        test = payload["test_indices"]
    train_unique = np.unique(train)
    test_unique = np.unique(test)
    if len(train_unique) != len(train) or len(test_unique) != len(test):
        raise ValueError("Split manifest partitions contain duplicate indices")
    if np.intersect1d(train_unique, test_unique, assume_unique=True).size:
        raise ValueError("Split manifest train and test partitions overlap")
    if len(train) != int(metadata["n_train"]) or len(test) != int(metadata["n_test"]):
        raise ValueError("Split manifest sizes do not match its metadata")
    if len(train) + len(test) != int(metadata["n_selected"]):
        raise ValueError("Split manifest union does not match selected observations")
    total = int(metadata["time_size"]) * int(metadata["latitude_size"]) * int(metadata["longitude_size"])
    if (train.size and int(train.max()) >= total) or (test.size and int(test.max()) >= total):
        raise ValueError("Split manifest contains an out-of-range index")
    decoded = decode_observation_indices(
        np.concatenate((train[: min(1000, len(train))], test[: min(1000, len(test))])), metadata
    )
    if (
        np.any(decoded["latitude_idx"] >= int(metadata["latitude_size"]))
        or np.any(decoded["longitude_idx"] >= int(metadata["longitude_size"]))
    ):
        raise ValueError("Split manifest indexing formula does not reconstruct valid coordinates")


def _validate_jsonld_rdf(*documents: dict[str, Any]) -> None:
    try:
        from rdflib import Graph
    except ImportError as exc:
        raise ValueError("rdflib is required to validate generated JSON-LD") from exc
    for document in documents:
        Graph().parse(data=json.dumps(document), format="json-ld")


def _validate_checksum_manifest(crate_dir: Path) -> None:
    manifest = crate_dir / "checksums.sha256"
    entries = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        entries[relative] = digest
    expected = {
        _relative(path, crate_dir) for path in crate_dir.rglob("*")
        if path.is_file() and path not in {manifest, crate_dir / "ro-crate-metadata.json"}
    }
    if set(entries) != expected:
        raise ValueError("checksums.sha256 does not cover every non-circular physical crate file")
    for relative, digest in entries.items():
        if file_facts(crate_dir / relative)["sha256"] != digest:
            raise ValueError(f"checksums.sha256 mismatch for {relative}")


def _walk_references(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        if set(value) == {"@id"} and isinstance(value["@id"], str):
            yield value["@id"]
        else:
            for nested in value.values():
                yield from _walk_references(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _walk_references(nested)


def _walk_all_ids(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        if isinstance(value.get("@id"), str):
            yield value["@id"]
        for nested in value.values():
            yield from _walk_all_ids(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _walk_all_ids(nested)


def _unique_entities(entities: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return list({entity["@id"]: entity for entity in entities}.values())


def _has_direct_download_url(entity: dict[str, Any]) -> bool:
    for key in ("contentUrl", "dcat:downloadURL"):
        value = entity.get(key)
        if isinstance(value, dict):
            value = value.get("@id")
        parsed = urlparse(str(value or ""))
        if parsed.scheme in {"http", "https"} and parsed.netloc:
            return True
    return False


def _reference_ids(value: Any) -> set[str]:
    values = value if isinstance(value, list) else [value]
    return {item["@id"] for item in values if isinstance(item, dict) and item.get("@id")}


def _path_name(value: str) -> str:
    return Path(value.replace("\\", "/")).name


def _slug(value: Any) -> str:
    return "".join(character.lower() if character.isalnum() else "-" for character in str(value)).strip("-")


def _git_metadata(project_root: Path, crate_dir: Path, mode: str = "patch") -> dict[str, Any]:
    def git(*args: str) -> str | None:
        try:
            return subprocess.run(
                ["git", *args], cwd=project_root, check=True, capture_output=True, text=True
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    scope = ("src", "scripts", "config", "pyproject.toml", "requirements.txt")
    tracked_diff = git("diff", "--binary", "HEAD", "--", *scope) or ""
    untracked = (git("ls-files", "--others", "--exclude-standard", "--", *scope) or "").splitlines()
    untracked_diffs: list[str] = []
    for relative_path in untracked:
        process = subprocess.run(
            ["git", "diff", "--no-index", "--binary", "/dev/null", relative_path],
            cwd=project_root,
            capture_output=True,
            text=True,
        )
        if process.returncode in {0, 1} and process.stdout:
            untracked_diffs.append(process.stdout.strip())
    diff = "\n".join(part for part in [tracked_diff.strip(), *untracked_diffs] if part)
    dirty = bool(git("status", "--porcelain"))
    diff_path = None
    if dirty and mode == "patch":
        diff_path = crate_dir / "code" / "git-diff.patch"
        diff_path.parent.mkdir(parents=True, exist_ok=True)
        if not diff:
            diff = "# Dirty worktree; no changes in the executed src/scripts/config scope.\n"
        diff_path.write_text(_redact_patch_secrets(diff) + "\n", encoding="utf-8")
    remote = git("config", "--get", "remote.origin.url")
    repository_url = _repository_url(remote) or CANONICAL_REPOSITORY_URL
    commit = git("rev-parse", "HEAD")
    branch = git("branch", "--show-current")
    copied_untracked = []
    if mode == "patch":
        for relative in untracked:
            source = project_root / relative
            if not source.is_file():
                continue
            target = crate_dir / "code" / "untracked" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if source.suffix.lower() == ".toml":
                _copy_sanitized_toml(source, target)
            else:
                shutil.copy2(source, target)
            copied_untracked.append(target)
    return {
        "commit": commit,
        "branch": branch,
        "remote": remote,
        "repository_url": repository_url,
        "commit_url": f"{repository_url}/tree/{commit}" if repository_url and commit else None,
        "branch_url": f"{repository_url}/tree/{quote(branch, safe='/')}" if repository_url and branch else None,
        "author_name": git("show", "-s", "--format=%an", "HEAD"),
        "dirty": dirty,
        "diff_path": diff_path,
        "untracked_paths": copied_untracked,
        "code_mode": mode,
    }


def _repository_url(remote: str | None) -> str | None:
    if not remote:
        return None
    value = remote.strip().removesuffix(".git")
    ssh_match = re.fullmatch(r"git@github\.com:(.+)", value)
    if ssh_match:
        return f"https://github.com/{ssh_match.group(1)}"
    if value.startswith("http://github.com/"):
        return "https://" + value.removeprefix("http://")
    return value


def _write_environment(
    crate_dir: Path, settings: Any
) -> tuple[Path, list[Path], list[str], dict[str, Any] | None]:
    path = crate_dir / "environment" / "python-packages.json"
    payload = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": {dist.metadata.get("Name", "unknown"): dist.version for dist in importlib.metadata.distributions()},
    }
    _write_json(path, payload)
    copied: list[Path] = []
    requirements = ["environment/python-packages.json"]
    if settings.lock_file:
        source = Path(settings.lock_file)
        if not source.is_file():
            raise ValueError(f"Configured reproducibility environment lock file does not exist: {source}")
        target = crate_dir / "environment" / "lock" / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied.append(target)
        requirements.append(_relative(target, crate_dir))
    container = None
    if settings.container_image:
        immutable_id = f"{settings.container_image}@{settings.container_digest}"
        container = {
            "@id": immutable_id,
            "@type": "SoftwareApplication",
            "name": settings.container_image,
            "identifier": immutable_id,
            "softwareVersion": settings.container_digest,
            "runtimePlatform": settings.container_runtime,
            "operatingSystem": settings.container_platform,
            "additionalProperty": [
                {"@type": "PropertyValue", "name": "container_image", "value": settings.container_image},
                {"@type": "PropertyValue", "name": "container_digest", "value": settings.container_digest},
                {"@type": "PropertyValue", "name": "platform", "value": settings.container_platform},
                {"@type": "PropertyValue", "name": "runtime", "value": settings.container_runtime},
            ],
        }
        requirements.append(immutable_id)
    return path, copied, requirements, container


def _dataset_properties(
    cfg: dict[str, Any], summary: dict[str, Any], resolved_inputs: list[ResolvedInput]
) -> list[dict[str, Any]]:
    metadata = summary.get("dataset_metadata", {})
    values = {
        "variables": cfg.get("variable_names", []),
        "target": cfg.get("target_name"),
        "masks": cfg.get("data", {}).get("mask_names", []),
        "split_mode": cfg.get("split_mode"),
        "train_fraction": cfg.get("train_fraction"),
        "test_fraction": cfg.get("test_fraction"),
        "split_seed": cfg.get("seed"),
        "split_metadata": metadata.get("split_metadata", {}),
        "n_train": metadata.get("n_train"),
        "n_test": metadata.get("n_test"),
        "upstream_files_not_matched": [],
        "input_resolution": [
            {
                "key": item.spec.key,
                "local_file": item.spec.local_file,
                "training_entity_id": item.training_entity_id,
                "upstream_entity_id": item.upstream_entity["@id"],
                "match_method": item.match_method,
                "local_sha256": item.local_sha256,
                "source_local_sha256": item.source_local_sha256,
                "upstream_sha256": item.upstream_entity.get("sha256"),
                "checksum_status": item.checksum_status,
                "netcdf_variable": item.spec.netcdf_variable,
                "derived": item.is_derived,
            }
            for item in resolved_inputs
        ],
    }
    return [{"@type": "PropertyValue", "name": key, "value": value} for key, value in values.items()]


def _mlflow_entity(
    cfg: dict[str, Any], summary: dict[str, Any], run_id: str, identifier: str = "#mlflow-run"
) -> dict[str, Any]:
    mlflow = cfg.get("mlflow", {})
    tracking_uri = mlflow.get("tracking_uri")
    entity = {
        "@id": identifier,
        "@type": "CreativeWork",
        "name": f"MLflow run {run_id}",
        "identifier": run_id,
        "additionalProperty": [
            {"@type": "PropertyValue", "name": "experiment", "value": summary.get("train_info", {}).get("mlflow_experiment")},
            {"@type": "PropertyValue", "name": "tracking_uri", "value": tracking_uri},
        ],
    }
    if tracking_uri:
        entity["url"] = tracking_uri.rstrip("/")
    return entity


def _config_paths(cfg: dict[str, Any]) -> list[Path]:
    paths = [cfg.get("config_path"), cfg.get("data_config_path")]
    repro = cfg.get("reproducibility")
    if isinstance(repro, ReproducibilityConfig):
        paths.append(repro.input_catalogue)
    return list(dict.fromkeys(Path(path) for path in paths if path))


def _copy_sanitized_toml(source: Path, target: Path) -> None:
    lines = []
    portable_directories = {
        "main_dir": "${WORK_DIR}", "raw_dir": "${WORK_DIR}/inputs",
        "processed_base_dir": "${WORK_DIR}/processed",
        "mask_dir": "${WORK_DIR}/derived-inputs/masks",
        "model_base_dir": "${WORK_DIR}/outputs",
    }
    for line in source.read_text(encoding="utf-8").splitlines():
        match = SECRET_LINE.match(line)
        if match:
            lines.append(f'{match.group("prefix")} "***REDACTED***"')
            continue
        path_match = re.match(r'^\s*([A-Za-z0-9_.-]+)\s*=\s*["\'](/[^"\']*)["\']\s*(#.*)?$', line)
        if path_match and path_match.group(1) in portable_directories:
            comment = f" {path_match.group(3)}" if path_match.group(3) else ""
            lines.append(f'{path_match.group(1)} = "{portable_directories[path_match.group(1)]}"{comment}')
        else:
            lines.append(line)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _portable_metadata(value: Any, key: str = "") -> Any:
    """Remove machine-specific absolute paths from copied diagnostic JSON."""
    if isinstance(value, dict):
        return {name: _portable_metadata(item, str(name)) for name, item in value.items()}
    if isinstance(value, list):
        return [_portable_metadata(item, key) for item in value]
    if isinstance(value, str) and value.startswith("/"):
        name = Path(value).name
        if key in {"processed_dir", "array_path"}:
            return f"${{WORK_DIR}}/processed/{name}" if name else "${WORK_DIR}/processed"
        if key in {"model_data_dir"}:
            return "${WORK_DIR}/model-data"
        if key.endswith("_path"):
            return f"${{WORK_DIR}}/outputs/{name}"
        return f"${{ORIGINAL_WORKSPACE}}/{name}" if name else "${ORIGINAL_WORKSPACE}"
    return value


def _redact_patch_secrets(diff: str) -> str:
    """Redact TOML-style secrets in added or removed diff lines."""
    pattern = re.compile(
        r"^(?P<prefix>[+\- ]*\s*[A-Za-z0-9_.-]*(?:password|passwd|token|secret|api_key)"
        r"[A-Za-z0-9_.-]*\s*=\s*)(?P<quote>[\"']).*?(?P=quote)",
        re.IGNORECASE | re.MULTILINE,
    )
    return pattern.sub(lambda match: f'{match.group("prefix")}\"***REDACTED***\"', diff)


def _file_entity(relative_path: str, crate_dir: Path | None = None) -> dict[str, Any]:
    entity: dict[str, Any] = {"@id": relative_path, "@type": "File", "name": Path(relative_path).name}
    media_type, _ = mimetypes.guess_type(relative_path)
    if media_type:
        entity["encodingFormat"] = media_type
    path = crate_dir / relative_path if crate_dir else None
    if path and path.is_file():
        entity.update(file_facts(path))
    return entity


def _write_readme(
    crate_dir: Path,
    cfg: dict[str, Any],
    run_id: str,
    upstream: str | None,
    unmatched: list[str],
    artifact: dict[str, Any],
    croissant_mode: str,
    code_mode: str = "reference",
    split_manifest: dict[str, Any] | None = None,
    licenses: dict[str, str | None] | None = None,
    git: dict[str, Any] | None = None,
    reproducible_run: bool = False,
    environment: Any | None = None,
) -> None:
    licenses = licenses or {}
    git = git or {}
    lines = [
        f"# PROFECIA training RO-Crate: {run_id}",
        "",
        "This crate describes one completed training and evaluation run.",
        "Large upstream NetCDF inputs are referenced, not copied.",
        "Intermediate NPY arrays are described by metadata and checksums, not copied.",
        "",
        f"- Model run: `{cfg.get('model_run_name')}`",
        f"- Upstream preprocessing crate: `{upstream or 'not configured'}`",
        "- Profiles: RO-Crate 1.1 and Process Run Crate 0.5",
        "- ML dataset metadata: `fair/dataset.croissant.json` (Croissant 1.1)",
        "- Model metadata: `fair/model.fair4ml.json` (FAIR4ML 0.1.0 vocabulary)",
        f"- Model artifact mode: `{artifact['mode']}`",
        f"- Croissant mode: `{croissant_mode}`",
        f"- Code mode: `{code_mode}`",
        f"- Exact split manifest: `{'fair/split_manifest.npz' if split_manifest else 'disabled'}`",
        "- Metadata license: `" + str(licenses.get("metadata") or "not established") + "`",
        "- Code license: `" + str(licenses.get("code") or "not established") + "`",
        "- Model license: `" + str(licenses.get("model") or "not established") + "`",
        "- Results license: `" + str(licenses.get("results") or "not established") + "`",
        "- Derived ML dataset license: `" + str(licenses.get("derived_dataset") or "not established") + "`",
        "",
        "The metadata license does not change or replace the licenses of upstream data resources.",
    ]
    if artifact["mode"] == "attached":
        lines.append(f"- Reuse: load `{artifact['main_id']}` with joblib using the recorded environment.")
    elif artifact["mode"] == "external":
        lines.append(f"- Reuse: download `{artifact['main_id']}` and verify its SHA-256 checksum before loading.")
    else:
        lines.append("- Reuse: unavailable; this crate contains model metadata only.")
    if code_mode == "reference" and git.get("dirty"):
        lines.extend([
            "",
            "WARNING: code.mode=reference was used with a dirty Git repository. The recorded commit does not fully describe the executed source state; use patch mode or require a clean repository for a release.",
        ])
    if unmatched:
        lines.extend(["", "Upstream catalogue entries not matched: " + ", ".join(unmatched)])
    if reproducible_run:
        container = (
            f"{environment.container_image}@{environment.container_digest}"
            if environment and environment.container_image else "not configured"
        )
        lines.extend([
            "", "## Overview", "", "This executable crate can resolve its upstream release, recreate the exact split, train, evaluate and assess a reproduction.",
            "", "## Upstream preprocessing dependency", "", f"Immutable release: `{upstream or 'not configured'}`.",
            "", "## Requirements", "", "A POSIX shell, Python, Git, network access unless using a complete cache, and the recorded environment requirements.",
            "", "## Reproduce with a container", "", f"Configured immutable image: `{container}`.", "", "```bash", "./reproduce.sh --container --work-dir ./reproduction", "```",
            "", "## Reproduce with a local environment", "", "```bash", "./reproduce.sh --local-environment --work-dir ./reproduction", "```",
            "", "## Offline reproduction", "", "```bash", "./reproduce.sh --offline --work-dir ./reproduction", "```",
            "", "## Input resolution", "", "See `reproducibility/input_resolution.json`. Missing required entities or checksum failures are fatal.",
            "", "## Dataset transformations", "", "Derived inputs are reconstructed only from transformations explicitly recorded in the input catalogue.",
            "", "## Exact train/test reconstruction", "", "The persisted `fair/split_manifest.npz` is applied directly. Use `--recompute-split` only for a non-exact experiment.",
            "", "## Expected results", "", "Expected metrics and tolerances are recorded in `reproducibility/expected_results.json`.",
            "", "## Verification", "", "```bash", "python -m profecia.reproducibility.verify --original-crate . --reproduced-run ./reproduction", "```",
            "", "## Model availability", "", f"Artifact mode: `{artifact['mode']}`. Metadata-only models are not distributed but can be regenerated.",
            "", "## Licenses", "", "Metadata, code, model, results and derived dataset licenses are independent; see the summary above.",
            "", "## Known limitations", "", "Numerical reproducibility may depend on platform and library implementations; metric tolerances define acceptance.",
        ])
    (crate_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_checksum_manifest(crate_dir: Path) -> Path:
    path = crate_dir / "checksums.sha256"
    lines = []
    for source in sorted(crate_dir.rglob("*")):
        if not source.is_file() or source == path or source.name == "ro-crate-metadata.json":
            continue
        lines.append(f"{file_facts(source)['sha256']}  {_relative(source, crate_dir)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_reproduce_script(crate_dir: Path, git: dict[str, Any]) -> Path:
    repository = CANONICAL_REPOSITORY_URL
    raw_commit = str(git.get("commit") or "")
    commit = raw_commit if re.fullmatch(r"[0-9a-fA-F]{40}", raw_commit) else "HEAD"
    script = f'''#!/bin/sh
set -eu
CRATE_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
WORK_DIR=./reproduction
OFFLINE=0
PREV=""
for ARG in "$@"; do
  if [ "$PREV" = "--work-dir" ]; then WORK_DIR=$ARG; fi
  if [ "$ARG" = "--offline" ]; then OFFLINE=1; fi
  PREV=$ARG
done
if ! python -c "import profecia.reproducibility" >/dev/null 2>&1; then
  if [ "$OFFLINE" = "1" ]; then
    echo "fatal error: PROFECIA commit {commit} is not installed and offline mode forbids retrieval" >&2
    exit 2
  fi
  BOOTSTRAP="$WORK_DIR/.profecia-bootstrap"
  python -m venv "$BOOTSTRAP"
  "$BOOTSTRAP/bin/python" -m pip install "git+{repository}@{commit}"
  PYTHON="$BOOTSTRAP/bin/python"
else
  PYTHON=python
fi
exec "$PYTHON" -m profecia.reproducibility.reproduce --crate "$CRATE_DIR" "$@"
'''
    path = crate_dir / "reproduce.sh"
    path.write_text(script, encoding="utf-8")
    path.chmod(0o755)
    return path


def _upstream_root(metadata: dict[str, Any] | None) -> dict[str, Any]:
    if not metadata:
        return {}
    graph = metadata.get("@graph", [])
    return next((item for item in graph if item.get("@id") == "./"), {})


def _write_validation_report(
    crate_dir: Path, croissant: dict[str, Any], split_manifest: dict[str, Any] | None
) -> Path:
    properties = {
        item.get("name"): item.get("value")
        for item in croissant.get("additionalProperty", [])
        if isinstance(item, dict)
    }
    payload = {
        "status": "validated_during_rocrate_generation",
        "croissant_downloadable_resources": properties.get("croissant_downloadable_resources", []),
        "croissant_semantic_only_resources": properties.get("croissant_semantic_only_resources", []),
        "split_manifest_included": split_manifest is not None,
        "checks": [
            "jsonld_rdf_parse", "shared_identifiers", "split_uniqueness",
            "split_disjointness", "split_union_size", "split_index_range",
            "coordinate_reconstruction", "split_checksum", "croissant_local_paths",
            "croissant_remote_urls", "metadata_only_artifact", "license_policy",
            "physical_file_checksum_coverage",
        ],
    }
    path = crate_dir / "metadata" / "reproducibility_validation.json"
    _write_json(path, payload)
    return path


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _types(entity: dict[str, Any]) -> list[str]:
    value = entity.get("@type", [])
    return value if isinstance(value, list) else [value]


def _has_reference(value: Any, identifier: str) -> bool:
    values = value if isinstance(value, list) else [value]
    return any(item == identifier or (isinstance(item, dict) and item.get("@id") == identifier) for item in values)


def _shell_join(command: list[str]) -> str:
    import shlex

    return shlex.join(str(item) for item in command)
