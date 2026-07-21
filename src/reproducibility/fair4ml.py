from __future__ import annotations

import math
from typing import Any

from .run_metadata import ensure_run_identifiers


FAIR4ML_VERSION = "https://w3id.org/fair4ml/0.1.0"
MODEL_INTENDED_USE = "Estimate annual global Leaf Area Index from climatic, soil and anthropogenic predictors."
MODEL_LIMITATIONS = (
    "Evaluation is limited to the configured test partition, variables, masks, spatial grid and temporal range; "
    "performance outside that scope has not been established."
)


def build_fair4ml_descriptor(
    run: dict[str, Any],
    artifact: dict[str, Any],
    created_at: str,
    config_ids: list[str] | None = None,
    split_manifest: dict[str, Any] | None = None,
    git: dict[str, Any] | None = None,
    result_ids: list[str] | None = None,
) -> dict[str, Any]:
    metrics = _metrics(run.get("metrics") or {})
    train = run.get("train") or {}
    mode = artifact["mode"]
    ids = ensure_run_identifiers(run)
    git = git or {}
    licenses = run.get("licenses") or {}
    artifact_reference = {"@id": artifact["fair_id"]}
    usage = (
        "Load the attached joblib artifact with the recorded Python environment and provide predictors "
        f"in this order: {', '.join(run.get('features') or [])}."
        if mode == "attached"
        else (
            "Download the model artifact from the recorded content URL, verify its SHA-256 checksum, "
            "then load it with joblib using the recorded Python environment."
            if mode == "external"
            else "The trained artifact is not available; this metadata-only model cannot be reused for inference."
        )
    )
    model: dict[str, Any] = {
        "@id": ids["ml-model"],
        "@type": ["CreativeWork", "fair4ml:MLModel"],
        "conformsTo": {"@id": FAIR4ML_VERSION},
        "name": train.get("model_name") or train.get("model_name_requested") or "Trained PROFECIA model",
        "dateCreated": created_at,
        "creator": run["creator"],
        "fair4ml:trainedOn": {"@id": ids["training-dataset"]},
        "fair4ml:testedOn": {"@id": ids["test-dataset"]},
        "fair4ml:evaluatedWith": {"@id": ids["model-evaluation"]},
        "fair4ml:modelCategory": ["Supervised learning", _algorithm_category(train)],
        "fair4ml:mlTask": "Regression",
        "fair4ml:intendedUse": MODEL_INTENDED_USE,
        "fair4ml:modelRisksBiasLimitations": MODEL_LIMITATIONS,
        "fair4ml:usageInstructions": usage,
        "associatedMedia": artifact_reference,
        "isBasedOn": [
            {"@id": ids["ml-dataset"]}, {"@id": ids["source-code"]},
            *([{"@id": run["upstream_entity_id"]}] if run.get("upstream_entity_id") else []),
            *({"@id": identifier} for identifier in (config_ids or [])),
        ],
        "prov:wasGeneratedBy": {"@id": ids["train-model"]},
        "potentialAction": {
            "@type": "CreateAction", "name": "Reproduce PROFECIA training",
            "target": "./reproduce.sh", "instrument": {"@id": ids["reproduction-command"]},
        },
        "codeRepository": git.get("repository_url"),
        "version": git.get("commit"),
        "softwareRequirements": [
            {"@id": f"../{identifier}" if not identifier.startswith(("http://", "https://", "urn:")) else identifier}
            for identifier in run.get("software_requirements", ["environment/python-packages.json"])
        ],
        "fair4ml:trainingSoftware": {"@id": ids["training-application"]},
        "additionalProperty": [
            {"@type": "PropertyValue", "name": "algorithm", "value": train.get("model_name")},
            {"@type": "PropertyValue", "name": "library", "value": "scikit-learn"},
            {"@type": "PropertyValue", "name": "hyperparameters", "value": train.get("model_params", {})},
            {"@type": "PropertyValue", "name": "random_seed", "value": train.get("random_state")},
            {"@type": "PropertyValue", "name": "split_strategy", "value": run.get("dataset", {}).get("split_mode")},
            {"@type": "PropertyValue", "name": "n_train", "value": run.get("dataset", {}).get("n_train")},
            {"@type": "PropertyValue", "name": "n_test", "value": run.get("dataset", {}).get("n_test")},
            {"@type": "PropertyValue", "name": "predictors", "value": run.get("features", [])},
            {"@type": "PropertyValue", "name": "target", "value": run.get("target")},
            {"@type": "PropertyValue", "name": "masks", "value": run.get("masks", [])},
            {"@type": "PropertyValue", "name": "artifact_mode", "value": mode},
            {"@type": "PropertyValue", "name": "reusable", "value": mode != "metadata_only"},
        ],
    }
    model = {key: value for key, value in model.items() if value is not None}
    if licenses.get("model"):
        model["license"] = {"@id": licenses["model"]}
    else:
        model["conditionsOfAccess"] = "License for the trained model has not yet been established."
    if split_manifest:
        model["subjectOf"] = [{"@id": ids["split-manifest"]}]
    subjects = model.get("subjectOf", [])
    if not isinstance(subjects, list):
        subjects = [subjects]
    subjects.append({"@id": "../reproducibility/expected_results.json"})
    model["subjectOf"] = subjects
    if run.get("mlflow_run_id"):
        subjects = model.get("subjectOf", [])
        if not isinstance(subjects, list):
            subjects = [subjects]
        subjects.append({"@id": ids["mlflow-run"]})
        model["subjectOf"] = subjects

    evaluation = {
        "@id": ids["model-evaluation"],
        "@type": ["CreativeWork", "fair4ml:MLModelEvaluation"],
        "conformsTo": {"@id": FAIR4ML_VERSION},
        "fair4ml:evaluatedMLModel": {"@id": ids["ml-model"]},
        "fair4ml:evaluationDataset": {"@id": ids["test-dataset"]},
        "fair4ml:evaluationSoftware": {"@id": ids["training-application"]},
        "fair4ml:evaluationMetrics": metrics,
        "fair4ml:evaluationResults": "Metrics computed on the persisted PROFECIA test partition.",
        "fair4ml:extrinsicEvaluation": False,
    }
    if result_ids:
        evaluation["subjectOf"] = [{"@id": identifier} for identifier in result_ids]
    graph = [
        model, evaluation, _dataset_reference("full", run),
        _dataset_reference("training", run), _dataset_reference("test", run),
    ]
    graph.append(_artifact_reference_entity(artifact))
    if split_manifest:
        graph.append({
            "@id": ids["split-manifest"], "@type": "MediaObject",
            "name": "Exact PROFECIA train/test split manifest",
            "contentUrl": "split_manifest.npz", "encodingFormat": "application/x-npz",
            "sha256": split_manifest["metadata"]["sha256"],
            "contentSize": split_manifest["metadata"]["contentSize"],
        })
    graph.extend([
        {
            "@id": ids["training-application"], "@type": "SoftwareApplication",
            "name": "PROFECIA configurable training workflow",
            "softwareRequirements": model["softwareRequirements"],
            "isBasedOn": {"@id": ids["source-code"]},
            "version": git.get("commit"), "codeRepository": git.get("repository_url"),
        },
        {
            "@id": ids["source-code"], "@type": "SoftwareSourceCode",
            "name": "PROFECIA source code state", "version": git.get("commit"),
            "codeRepository": git.get("repository_url"), "url": git.get("commit_url"),
        },
        {
            "@id": ids["reproduction-command"], "@type": "SoftwareApplication",
            "name": "PROFECIA reproduction command", "executableName": "./reproduce.sh",
            "isBasedOn": {"@id": ids["source-code"]},
        },
    ])
    if run.get("upstream_entity_id"):
        graph.append({
            "@id": run["upstream_entity_id"], "@type": "Dataset",
            "name": "Upstream preprocessing RO-Crate release",
        })
    if run.get("mlflow_run_id"):
        graph.append(
            {
                "@id": ids["mlflow-run"],
                "@type": "CreativeWork",
                "identifier": run["mlflow_run_id"],
                "name": f"MLflow run {run['mlflow_run_id']}",
            }
        )
    return {
        "@context": {
            "@vocab": "https://schema.org/",
            "fair4ml": "https://w3id.org/fair4ml#",
            "cr": "http://mlcommons.org/croissant/",
            "codemeta": "https://w3id.org/codemeta/",
        },
        "@graph": graph,
    }


def _metrics(values: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "@type": "PropertyValue",
            "name": name,
            "value": value,
            "measurementTechnique": name,
        }
        for name, value in values.items()
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    ]


def _algorithm_category(train: dict[str, Any]) -> str:
    name = str(train.get("model_name") or train.get("model_name_requested") or "Regression model")
    if "randomforest" in name.replace("_", "").lower() or name.lower() in {"rf", "random forest"}:
        return "Random forest regression"
    return f"{name} regression"


def _dataset_reference(kind: str, run: dict[str, Any]) -> dict[str, Any]:
    ids = run["identifiers"]
    license_id = (run.get("licenses") or {}).get("derived_dataset") or run.get("license")
    if kind == "full":
        entity = {
            "@id": ids["ml-dataset"], "@type": ["Dataset", "cr:Dataset"],
            "name": "PROFECIA concrete ML dataset",
            "hasPart": [{"@id": ids["training-dataset"]}, {"@id": ids["test-dataset"]}],
        }
    else:
        entity = {
        "@id": ids[f"{kind}-dataset"],
        "@type": ["Dataset", "cr:Dataset"],
        "name": f"PROFECIA {kind} partition",
        "isPartOf": {"@id": ids["ml-dataset"]},
        "additionalProperty": {
            "@type": "PropertyValue",
            "name": "number_of_observations",
            "value": run.get("dataset", {}).get("n_train" if kind == "training" else "n_test"),
        },
        }
    if license_id:
        entity["license"] = {"@id": license_id}
    else:
        entity["conditionsOfAccess"] = "License for the derived ML dataset has not yet been established."
    return entity


def _artifact_reference_entity(artifact: dict[str, Any]) -> dict[str, Any]:
    entity = dict(artifact["entity"])
    entity["@id"] = artifact["fair_id"]
    if artifact["mode"] == "attached":
        entity["contentUrl"] = artifact["fair_id"]
    return entity
