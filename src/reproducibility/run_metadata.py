from __future__ import annotations

import hashlib
import importlib.metadata
import json
import uuid
from pathlib import Path
from typing import Any


CREATOR = {
    "@id": "https://orcid.org/0000-0001-9462-4831",
    "@type": "Person",
    "name": "Fernando Aguilar Gomez",
}
DATASET_LICENSE = "https://spdx.org/licenses/CC0-1.0.html"


def ensure_run_identifiers(run: dict[str, Any]) -> dict[str, str]:
    """Return the stable cross-descriptor identifiers, adding them for legacy callers."""
    base_id = str(run.setdefault("base_id", f"urn:uuid:{uuid.uuid4()}"))
    names = (
        "ml-dataset", "training-dataset", "test-dataset", "ml-model",
        "model-evaluation", "training-application", "rocrate-generator",
        "split-manifest", "prepare-dataset", "train-model", "evaluate-model",
        "package-rocrate", "source-code", "mlflow-run", "cross-validation",
        "model-artifact",
        "upstream-preprocessing-crate", "resolve-upstream", "download-inputs",
        "derive-inputs", "apply-split", "reproduction-command", "reproduction-assessment",
        "verify-reproduction", "reproduction-report",
    )
    identifiers = run.setdefault("identifiers", {})
    for name in names:
        identifiers.setdefault(name, f"{base_id}#{name}")
    return identifiers


def build_run_metadata(
    cfg: dict[str, Any], summary: dict[str, Any], base_id: str | None = None
) -> dict[str, Any]:
    """Load persisted run metadata and reconcile it with the workflow summary."""
    if not _valid_orcid(CREATOR["@id"]):
        raise ValueError(f"Invalid configured ORCID: {CREATOR['@id']}")
    base_id = base_id or f"urn:uuid:{uuid.uuid4()}"
    seed = {"base_id": base_id}
    identifiers = ensure_run_identifiers(seed)
    paths = _metadata_paths(cfg, summary)
    loaded = {name: _load_json(path) for name, path in paths.items() if path and path.is_file()}

    summary_dataset = dict(summary.get("dataset_metadata") or {})
    dataset = dict(loaded.get("dataset_metadata") or summary_dataset)
    split = dict(loaded.get("split_metadata") or dataset.get("split_metadata") or {})
    dataset["split_metadata"] = split
    summary_train = dict(summary.get("train_info") or {})
    train = dict(loaded.get("train_info") or summary_train)
    evaluation = dict(summary.get("evaluation") or {})

    _check_equal("dataset feature_names", dataset.get("feature_names"), summary_dataset.get("feature_names"))
    _check_equal("dataset target", dataset.get("target"), summary_dataset.get("target"))
    _check_equal("dataset n_train", dataset.get("n_train"), summary_dataset.get("n_train"))
    _check_equal("dataset n_test", dataset.get("n_test"), summary_dataset.get("n_test"))
    _check_equal("split mode", split.get("split_mode"), dataset.get("split_mode"))
    _check_equal("model name", train.get("model_name"), summary_train.get("model_name"))
    _check_equal("model params", train.get("model_params"), summary_train.get("model_params"))
    _check_equal("random seed", train.get("random_state"), summary_train.get("random_state"))

    processed = dict(loaded.get("processed_metadata") or {})
    processed_variables = []
    for name, value in (processed.get("variables") or {}).items():
        value = value if isinstance(value, dict) else {}
        array_path = Path(value.get("array_path", "")) if value.get("array_path") else None
        processed_variables.append(
            {
                "name": str(name),
                "entity_id": f"{base_id}#processed-array-{_slug(name)}",
                "path": array_path,
                "load_metadata": value.get("load_metadata") or {},
                "processing": value.get("processing") or {},
                "grid": value.get("grid") or {},
                "temporal_grid": value.get("temporal_grid") or {},
                "shape": value.get("final_shape"),
                "dtype": value.get("final_dtype"),
                "units": value.get("final_units"),
            }
        )

    partition_arrays = _partition_arrays(cfg)
    for item in partition_arrays:
        item["entity_id"] = f"{base_id}{item['entity_id']}"
    metrics = {
        name: value
        for name, value in (evaluation.get("global_metrics") or {}).items()
        if isinstance(value, (int, float))
    }
    cv = train.get("hyperparameter_search") or summary_train.get("hyperparameter_search") or {"enabled": False}
    return {
        "dataset": dataset,
        "split": split,
        "train": train,
        "evaluation": evaluation,
        "metrics": metrics,
        "cv": cv,
        "processed": processed,
        "processed_variables": processed_variables,
        "partition_arrays": partition_arrays,
        "metadata_files": {name: path for name, path in paths.items() if path and path.is_file()},
        "creator": CREATOR,
        "license": DATASET_LICENSE,
        "features": list(dataset.get("feature_names") or cfg.get("predictor_names") or []),
        "target": dataset.get("target") or cfg.get("target_name"),
        "masks": list(cfg.get("data", {}).get("mask_names") or []),
        "mlflow_run_id": train.get("mlflow_run_id") or summary_train.get("mlflow_run_id"),
        "mlflow_experiment": train.get("mlflow_experiment") or summary_train.get("mlflow_experiment"),
        "base_id": base_id,
        "identifiers": identifiers,
    }


def file_facts(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {"contentSize": str(path.stat().st_size), "sha256": digest.hexdigest()}


def _metadata_paths(cfg: dict[str, Any], summary: dict[str, Any]) -> dict[str, Path | None]:
    processed_dir = _optional_path(summary.get("processed_dir") or cfg.get("processed_dir"))
    data_dir = _optional_path(summary.get("model_data_dir") or cfg.get("model_data_dir"))
    train_info = _optional_path((summary.get("saved_paths") or {}).get("train_info_path"))
    if train_info is None and cfg.get("model_artifacts_dir"):
        train_info = Path(cfg["model_artifacts_dir"]) / "train_info.json"
    return {
        "processed_metadata": processed_dir / "metadata.json" if processed_dir else None,
        "processed_run_config": processed_dir / "run_config.json" if processed_dir else None,
        "dataset_metadata": data_dir / "dataset_metadata.json" if data_dir else None,
        "split_metadata": data_dir / "split_metadata.json" if data_dir else None,
        "train_info": train_info,
    }


def _partition_arrays(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    data_dir = _optional_path(cfg.get("model_data_dir"))
    if not data_dir:
        return []
    result = []
    for split in ("train", "test"):
        for stem in ("X", "y", "pixel_id", "lat_idx", "lon_idx", "time_idx"):
            path = data_dir / f"{stem}_{split}.npy"
            if path.is_file():
                result.append(
                    {
                        "split": split,
                        "role": stem,
                        "path": path,
                        "entity_id": f"#npy-{stem.lower().replace('_', '-')}-{split}",
                        **file_facts(path),
                    }
                )
    return result


def write_split_manifest(
    crate_dir: Path, run: dict[str, Any], git: dict[str, Any]
) -> dict[str, Any]:
    """Persist exact train/test membership as global C-order observation indices."""
    import numpy as np

    arrays = {(item["split"], item["role"]): item["path"] for item in run["partition_arrays"]}
    latitude_size = int(run["dataset"].get("latitude_size") or 0)
    longitude_size = int(run["dataset"].get("longitude_size") or 0)
    time_size = int(run["dataset"].get("n_time") or 0)
    if not latitude_size or not longitude_size:
        raise ValueError("Split manifest requires latitude_size and longitude_size in dataset_metadata.json")

    indices: dict[str, Any] = {}
    for split in ("train", "test"):
        required = [(split, role) for role in ("time_idx", "lat_idx", "lon_idx")]
        missing = [f"{key[1]}_{split}.npy" for key in required if key not in arrays]
        if missing:
            raise ValueError("Split manifest requires persisted trace arrays: " + ", ".join(missing))
        time_idx = np.load(arrays[(split, "time_idx")], mmap_mode="r", allow_pickle=False).astype(np.uint64)
        lat_idx = np.load(arrays[(split, "lat_idx")], mmap_mode="r", allow_pickle=False).astype(np.uint64)
        lon_idx = np.load(arrays[(split, "lon_idx")], mmap_mode="r", allow_pickle=False).astype(np.uint64)
        if not (len(time_idx) == len(lat_idx) == len(lon_idx)):
            raise ValueError(f"Trace arrays for {split} have inconsistent lengths")
        if (
            (time_size and np.any(time_idx >= time_size))
            or np.any(lat_idx >= latitude_size)
            or np.any(lon_idx >= longitude_size)
        ):
            raise ValueError(f"Trace arrays for {split} contain coordinates outside the documented grid")
        indices[split] = (time_idx * latitude_size + lat_idx) * longitude_size + lon_idx

    train_unique = np.unique(indices["train"])
    test_unique = np.unique(indices["test"])
    if len(train_unique) != len(indices["train"]) or len(test_unique) != len(indices["test"]):
        raise ValueError("Split manifest contains duplicate observation identifiers within a partition")
    overlap = np.intersect1d(train_unique, test_unique, assume_unique=True)
    if overlap.size:
        raise ValueError(f"Train/test split manifest overlaps in {overlap.size} observations")
    expected_train = int(run["dataset"].get("n_train") or 0)
    expected_test = int(run["dataset"].get("n_test") or 0)
    if len(train_unique) != expected_train or len(test_unique) != expected_test:
        raise ValueError(
            "Split manifest sizes disagree with dataset_metadata.json: "
            f"train={len(train_unique)}/{expected_train}, test={len(test_unique)}/{expected_test}"
        )
    selected = run["split"].get("n_selected_observations")
    union_size = len(train_unique) + len(test_unique)
    if selected is not None and union_size != int(selected):
        raise ValueError("Train/test union does not match n_selected_observations")

    fair_dir = crate_dir / "fair"
    fair_dir.mkdir(parents=True, exist_ok=True)
    npz_path = fair_dir / "split_manifest.npz"
    np.savez_compressed(
        npz_path,
        train_indices=indices["train"],
        test_indices=indices["test"],
    )
    facts = file_facts(npz_path)
    grid = _manifest_grid(run)
    metadata = {
        "identifier": run["identifiers"]["split-manifest"],
        "format": "npz",
        "index_formula": "(time_idx * latitude_size + lat_idx) * longitude_size + lon_idx",
        "dimension_order": ["time", "latitude", "longitude"],
        "flatten_order": "C",
        "sample_id": "deterministic global grid observation index",
        "algorithm": run["dataset"].get("split_mode"),
        "seed": run["split"].get("seed"),
        "train_fraction_requested": run["split"].get("train_fraction_requested"),
        "test_fraction_requested": run["split"].get("test_fraction_requested"),
        "variables": [*run.get("features", []), run.get("target")],
        "n_total_possible": run["split"].get("n_total_observations") or (
            time_size * latitude_size * longitude_size if time_size else None
        ),
        "n_valid": run["split"].get("n_valid_observations"),
        "n_selected": union_size,
        "n_train": len(train_unique),
        "n_test": len(test_unique),
        "time_size": time_size or None,
        **grid,
        "time_values": run["dataset"].get("time_values"),
        "numpy_version": importlib.metadata.version("numpy"),
        "scikit_learn_version": importlib.metadata.version("scikit-learn"),
        "sha256": facts["sha256"],
        "contentSize": facts["contentSize"],
        "code_version": git.get("commit"),
        "git_dirty": bool(git.get("dirty")),
        "code_patch": "../code/git-diff.patch" if git.get("diff_path") else None,
    }
    metadata.update({
        "n_total_observations_possible": metadata["n_total_possible"],
        "n_valid_observations": metadata["n_valid"],
        "n_selected_observations": metadata["n_selected"],
        "n_train_observations": metadata["n_train"],
        "n_test_observations": metadata["n_test"],
    })
    json_path = fair_dir / "split_manifest.json"
    json_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"npz_path": npz_path, "json_path": json_path, "metadata": metadata}


def unflatten_observation_indices(
    flat_indices: Any, latitude_size: int, longitude_size: int, time_size: int | None = None
) -> tuple[Any, Any, Any]:
    """Convert global C-order indices into time, latitude and longitude indices."""
    import numpy as np

    flat = np.asarray(flat_indices, dtype=np.uint64)
    plane_size = int(latitude_size) * int(longitude_size)
    if plane_size <= 0:
        raise ValueError("latitude_size and longitude_size must be positive")
    time_idx, within_time = np.divmod(flat, plane_size)
    latitude_idx, longitude_idx = np.divmod(within_time, int(longitude_size))
    if time_size is not None and np.any(time_idx >= int(time_size)):
        raise ValueError("Flat observation index is outside the documented time dimension")
    return time_idx, latitude_idx, longitude_idx


def decode_observation_indices(flat_indices: Any, manifest: dict[str, Any]) -> dict[str, Any]:
    """Decode flat indices and, when metadata permits, return physical coordinates."""
    import numpy as np

    time_idx, latitude_idx, longitude_idx = unflatten_observation_indices(
        flat_indices,
        int(manifest["latitude_size"]),
        int(manifest["longitude_size"]),
        int(manifest["time_size"]) if manifest.get("time_size") else None,
    )
    result = {
        "time_idx": time_idx,
        "latitude_idx": latitude_idx,
        "longitude_idx": longitude_idx,
    }
    time_values = manifest.get("time_values")
    if time_values:
        result["time"] = np.asarray(time_values, dtype=object)[time_idx.astype(int)]
    if manifest.get("latitude_min") is not None and manifest.get("latitude_step") is not None:
        result["latitude"] = float(manifest["latitude_min"]) + latitude_idx * float(manifest["latitude_step"])
    if manifest.get("longitude_min") is not None and manifest.get("longitude_step") is not None:
        result["longitude"] = float(manifest["longitude_min"]) + longitude_idx * float(manifest["longitude_step"])
    return result


def _manifest_grid(run: dict[str, Any]) -> dict[str, Any]:
    fields = {
        "lat_min": "latitude_min", "lat_max": "latitude_max",
        "latitude_resolution_deg": "latitude_step", "latitude_order": "latitude_order",
        "lon_min": "longitude_min", "lon_max": "longitude_max",
        "longitude_resolution_deg": "longitude_step", "longitude_order": "longitude_order",
    }
    grids = [item.get("grid") or {} for item in run.get("processed_variables", [])]
    populated = [grid for grid in grids if grid]
    if populated:
        reference = populated[0]
        for grid in populated[1:]:
            for source_key in fields:
                if grid.get(source_key) is not None and reference.get(source_key) != grid.get(source_key):
                    raise ValueError(f"Processed variables disagree on grid field {source_key}")
    else:
        reference = {}
    return {
        target_key: reference.get(source_key)
        for source_key, target_key in fields.items()
    } | {
        "latitude_size": int(run["dataset"].get("latitude_size") or reference.get("latitude_size") or 0),
        "longitude_size": int(run["dataset"].get("longitude_size") or reference.get("longitude_size") or 0),
    }


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Intermediate metadata must be a JSON object: {path}")
    return value


def _check_equal(label: str, persisted: Any, summary: Any) -> None:
    if persisted is not None and summary is not None and persisted != summary:
        raise ValueError(
            f"Inconsistent {label} between persisted intermediate metadata and workflow summary: "
            f"{persisted!r} != {summary!r}"
        )


def _optional_path(value: Any) -> Path | None:
    return Path(value) if value not in (None, "") else None


def _slug(value: Any) -> str:
    return "".join(char.lower() if str(char).isalnum() else "-" for char in str(value)).strip("-")


def _valid_orcid(value: str) -> bool:
    compact = value.rstrip("/").rsplit("/", 1)[-1].replace("-", "").upper()
    if len(compact) != 16 or not compact[:15].isdigit() or compact[-1] not in "0123456789X":
        return False
    total = 0
    for character in compact[:15]:
        total = (total + int(character)) * 2
    remainder = (12 - total % 11) % 11
    check = "X" if remainder == 10 else str(remainder)
    return compact[-1] == check
