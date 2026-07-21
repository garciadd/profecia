from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import Any

from .run_metadata import ensure_run_identifiers, file_facts


CROISSANT_VERSION = "http://mlcommons.org/croissant/1.1"
CROISSANT_CONTEXT = {
    "@vocab": "https://schema.org/",
    "sc": "https://schema.org/",
    "cr": "http://mlcommons.org/croissant/",
    "dct": "http://purl.org/dc/terms/",
    "prov": "http://www.w3.org/ns/prov#",
}


def build_croissant_descriptor(
    run: dict[str, Any],
    upstream_identifier: str | None,
    upstream_entities: list[dict[str, Any]],
    published_at: str,
    mode: str = "descriptive",
    crate_dir: Path | None = None,
    model_data_dir: Path | None = None,
    split_manifest: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[Path]]:
    """Build a Croissant 1.1 descriptor without claiming unsupported NetCDF extraction."""
    created: list[Path] = []
    ids = ensure_run_identifiers(run)
    fair_dir = crate_dir / "fair" if crate_dir else None
    distributions = [_file_object(entity, fair_dir) for entity in upstream_entities]
    distributions = [item for item in distributions if item is not None]
    semantic_only = [
        entity for entity in upstream_entities
        if not _direct_remote_url(entity)
        and not _included_local_entity(entity, fair_dir)
    ]
    record_sets: list[dict[str, Any]] = []
    split_distributions: dict[str, list[dict[str, str]]] = {"train": [], "test": []}

    manifest_objects: list[dict[str, Any]] = []
    if split_manifest:
        for name, description in (
            ("split_manifest.npz", "Compressed arrays containing the exact flattened observation indices assigned to the training and test partitions."),
            ("split_manifest.json", "Metadata required to interpret and decode the exact train/test split manifest."),
        ):
            path = split_manifest["npz_path"] if name.endswith(".npz") else split_manifest["json_path"]
            manifest_objects.append({
                "@id": name,
                "@type": "cr:FileObject",
                "name": "Exact PROFECIA train/test split manifest" if name.endswith(".npz") else "PROFECIA split manifest metadata",
                "description": description,
                "contentUrl": name,
                "encodingFormat": "application/x-npz" if name.endswith(".npz") else "application/json",
                "identifier": split_manifest["metadata"]["identifier"],
                "prov:wasGeneratedBy": {"@id": ids["prepare-dataset"]},
                **file_facts(path),
            })
        distributions.extend(manifest_objects)
    if crate_dir and (crate_dir / "config" / "effective_config.json").is_file():
        effective_path = crate_dir / "config" / "effective_config.json"
        distributions.append({
            "@id": "../config/effective_config.json", "@type": "cr:FileObject",
            "name": "Effective PROFECIA training configuration",
            "contentUrl": "../config/effective_config.json", "encodingFormat": "application/json",
            **file_facts(effective_path),
        })

    if mode == "materialized":
        if crate_dir is None or model_data_dir is None:
            raise ValueError("Croissant materialized mode requires crate_dir and model_data_dir")
        materialized, record_sets, created = _materialize_parquet(
            run=run, crate_dir=crate_dir, model_data_dir=model_data_dir
        )
        distributions.extend(materialized)
        split_distributions = {
            "train": [{"@id": "../data/ml/train.parquet"}],
            "test": [{"@id": "../data/ml/test.parquet"}],
        }

    dataset = run["dataset"]
    split = run["split"]
    training = _split_dataset(
        ids["training-dataset"], ids["ml-dataset"], "training",
        dataset.get("n_train"), split_distributions["train"], split_manifest,
    )
    testing = _split_dataset(
        ids["test-dataset"], ids["ml-dataset"], "test",
        dataset.get("n_test"), split_distributions["test"], split_manifest,
    )
    descriptor: dict[str, Any] = {
        "@context": CROISSANT_CONTEXT,
        "@id": ids["ml-dataset"],
        "@type": "Dataset",
        "dct:conformsTo": CROISSANT_VERSION,
        "name": "PROFECIA concrete training and test dataset",
        "description": (
            "Dataset selected, masked and partitioned for one PROFECIA model run. "
            + (
                "Train and test observations are materialized as Parquet."
                if mode == "materialized"
                else "The descriptor is descriptive: intermediate NPY arrays are not distributed or declared loadable."
            )
        ),
        "url": "../",
        "creator": run["creator"],
        "datePublished": published_at,
        "distribution": distributions,
        "hasPart": [training, testing],
        "variableMeasured": _variable_descriptions(run),
        "prov:wasDerivedFrom": [{"@id": entity["@id"]} for entity in upstream_entities],
        "prov:wasGeneratedBy": {"@id": ids["prepare-dataset"]},
        "prov:wasAttributedTo": {"@id": ids["training-application"]},
        "additionalProperty": [
            {"@type": "PropertyValue", "name": "croissant_mode", "value": mode},
            {"@type": "PropertyValue", "name": "reconstruction_command", "value": "./reproduce.sh"},
            {"@type": "PropertyValue", "name": "split_mode", "value": dataset.get("split_mode")},
            {"@type": "PropertyValue", "name": "split_metadata", "value": split},
            {"@type": "PropertyValue", "name": "n_train", "value": dataset.get("n_train")},
            {"@type": "PropertyValue", "name": "n_test", "value": dataset.get("n_test")},
            {"@type": "PropertyValue", "name": "target", "value": run.get("target")},
            {
                "@type": "PropertyValue", "name": "local_transformations",
                "value": [
                    {"variable": item.get("name"), "processing": item.get("processing"), "grid": item.get("grid")}
                    for item in run.get("processed_variables", [])
                ],
            },
            {"@type": "PropertyValue", "name": "masks", "value": run.get("masks", [])},
            {"@type": "PropertyValue", "name": "temporal_resolution", "value": dataset.get("temporal_resolution_inferred")},
            {"@type": "PropertyValue", "name": "time_start", "value": dataset.get("time_start")},
            {"@type": "PropertyValue", "name": "time_end", "value": dataset.get("time_end")},
            {"@type": "PropertyValue", "name": "latitude_size", "value": dataset.get("latitude_size")},
            {"@type": "PropertyValue", "name": "longitude_size", "value": dataset.get("longitude_size")},
            {"@type": "PropertyValue", "name": "spatial_resolution", "value": dataset.get("spatial_resolution") or dataset.get("resolution")},
            {
                "@type": "PropertyValue", "name": "croissant_downloadable_resources",
                "value": [item["@id"] for item in distributions],
            },
            {
                "@type": "PropertyValue", "name": "croissant_semantic_only_resources",
                "value": [item["@id"] for item in semantic_only],
            },
        ],
    }
    derived_license = (run.get("licenses") or {}).get("derived_dataset") or run.get("license")
    if derived_license:
        descriptor["license"] = derived_license
    else:
        descriptor["conditionsOfAccess"] = "License for the derived ML dataset has not yet been established."
    if split_manifest:
        descriptor["subjectOf"] = [{"@id": item["@id"]} for item in manifest_objects]
        descriptor["additionalProperty"].append(
            {"@type": "PropertyValue", "name": "split_manifest", "value": split_manifest["metadata"]}
        )
    if upstream_identifier:
        descriptor["isBasedOn"] = {"@id": upstream_identifier}
        descriptor["dct:source"] = {"@id": upstream_identifier}
    if record_sets:
        descriptor["recordSet"] = record_sets
    return descriptor, created


def _file_object(entity: dict[str, Any], fair_dir: Path | None = None) -> dict[str, Any] | None:
    content_url = _direct_remote_url(entity)
    identifier = str(entity.get("@id", ""))
    local_entity = _included_local_entity(entity, fair_dir)
    if not content_url and local_entity:
        content_url = f"../{identifier}"
    if not content_url:
        return None
    item: dict[str, Any] = {
        "@type": "cr:FileObject",
        "@id": f"../{identifier}" if local_entity else identifier,
        "name": entity.get("name") or entity.get("alternateName") or Path(identifier).name,
        "contentUrl": content_url,
    }
    for key in ("encodingFormat", "contentSize", "sha256", "md5", "variableMeasured", "prov:wasDerivedFrom"):
        if entity.get(key) is not None:
            item[key] = str(entity[key]) if key == "contentSize" else entity[key]
    if entity.get("license") is not None:
        item["license"] = entity["license"]
    if "encodingFormat" not in item:
        media_type, _ = mimetypes.guess_type(str(item["name"]))
        item["encodingFormat"] = media_type or "application/octet-stream"
    return item


def _direct_remote_url(entity: dict[str, Any]) -> str | None:
    from urllib.parse import urlparse

    for key in ("contentUrl", "dcat:downloadURL"):
        value = entity.get(key)
        if isinstance(value, dict):
            value = value.get("@id")
        if isinstance(value, str):
            parsed = urlparse(value)
            if parsed.scheme in {"http", "https"} and parsed.netloc:
                return value
    return None


def _included_local_entity(entity: dict[str, Any], fair_dir: Path | None) -> bool:
    identifier = str(entity.get("@id", ""))
    return bool(
        fair_dir and identifier.startswith(("data/", "model/"))
        and (fair_dir / ".." / identifier).resolve().is_file()
    )


def _split_dataset(
    identifier: str,
    parent_id: str,
    kind: str,
    size: Any,
    distribution: list[dict[str, str]],
    split_manifest: dict[str, Any] | None,
) -> dict[str, Any]:
    entity: dict[str, Any] = {
        "@id": identifier,
        "@type": "Dataset",
        "name": f"PROFECIA {kind} partition",
        "isPartOf": {"@id": parent_id},
        "additionalProperty": {"@type": "PropertyValue", "name": "number_of_observations", "value": size},
    }
    if distribution:
        entity["distribution"] = distribution
    if split_manifest:
        entity["subjectOf"] = [
            {"@id": "split_manifest.npz"}, {"@id": "split_manifest.json"},
        ]
        entity["hasPart"] = {"@id": "split_manifest.npz"}
    return entity


def _variable_descriptions(run: dict[str, Any]) -> list[dict[str, Any]]:
    processed = {item["name"]: item for item in run.get("processed_variables", [])}
    values = []
    for name in [*run.get("features", []), run.get("target")]:
        if not name:
            continue
        item = processed.get(name, {})
        values.append(
            {
                "@type": "PropertyValue",
                "name": name,
                "value": "target" if name == run.get("target") else "predictor",
                "unitText": item.get("units") or None,
                "additionalProperty": [
                    {"@type": "PropertyValue", "name": "dtype", "value": item.get("dtype")},
                    {"@type": "PropertyValue", "name": "shape", "value": item.get("shape")},
                ],
            }
        )
    return values


def _materialize_parquet(
    run: dict[str, Any], crate_dir: Path, model_data_dir: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[Path]]:
    import numpy as np
    import pandas as pd

    output_dir = crate_dir / "data" / "ml"
    output_dir.mkdir(parents=True, exist_ok=True)
    distributions, record_sets, paths = [], [], []
    features = list(run.get("features") or [])
    target = str(run.get("target") or "target")
    for split in ("train", "test"):
        X = np.load(model_data_dir / f"X_{split}.npy", mmap_mode="r", allow_pickle=False)
        y = np.load(model_data_dir / f"y_{split}.npy", mmap_mode="r", allow_pickle=False).reshape(-1)
        if X.shape != (len(y), len(features)):
            raise ValueError(f"Cannot materialize {split}: X/y/features shapes are inconsistent")
        frame = pd.DataFrame(np.asarray(X), columns=features)
        frame[target] = np.asarray(y)
        trace_columns = []
        for stem in ("pixel_id", "lat_idx", "lon_idx", "time_idx"):
            source = model_data_dir / f"{stem}_{split}.npy"
            if source.is_file():
                values = np.load(source, mmap_mode="r", allow_pickle=False).reshape(-1)
                if len(values) != len(frame):
                    raise ValueError(f"Cannot materialize {split}: {source.name} row count is inconsistent")
                frame[stem] = np.asarray(values)
                trace_columns.append(stem)
        if all(column in frame for column in ("time_idx", "lat_idx", "lon_idx")):
            n_lat = int(run["dataset"].get("latitude_size") or 0)
            n_lon = int(run["dataset"].get("longitude_size") or 0)
            if not n_lat or not n_lon:
                raise ValueError("Cannot create sample_id without latitude_size and longitude_size")
            frame["sample_id"] = (
                (frame["time_idx"].astype("uint64") * n_lat + frame["lat_idx"].astype("uint64"))
                * n_lon + frame["lon_idx"].astype("uint64")
            )
            trace_columns.append("sample_id")
        frame["split"] = split
        trace_columns.append("split")
        path = output_dir / f"{split}.parquet"
        frame.to_parquet(path, index=False)
        paths.append(path)
        identifier = f"../data/ml/{split}.parquet"
        distributions.append(
            {
                "@type": "cr:FileObject",
                "@id": identifier,
                "name": path.name,
                "contentUrl": identifier,
                "encodingFormat": "application/vnd.apache.parquet",
                **file_facts(path),
            }
        )
        fields = []
        for column in [*features, target, *trace_columns]:
            dtype = "sc:Text" if column == "split" else (
                "sc:Float" if column in [*features, target] else "sc:Integer"
            )
            semantic = "cr:Label" if column == target else (
                "cr:Feature" if column in features else ("cr:Split" if column == "split" else None)
            )
            data_type = [dtype, semantic] if semantic else dtype
            fields.append(
                {
                    "@type": "cr:Field",
                    "@id": f"{split}-records/{column}",
                    "name": column,
                    "dataType": data_type,
                    "source": {
                        "fileObject": {"@id": identifier},
                        "extract": {"column": column},
                    },
                }
            )
        record_sets.append(
            {
                "@type": "cr:RecordSet",
                "@id": f"{split}-records",
                "name": f"{split} observations",
                "field": fields,
            }
        )
    return distributions, record_sets, paths
