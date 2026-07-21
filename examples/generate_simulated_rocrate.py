#!/usr/bin/env python3
"""Generate a tiny executable PROFECIA training RO-Crate without real NetCDF data."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from src.reproducibility.config import (
    CodeConfig,
    CroissantConfig,
    EnvironmentConfig,
    InputsConfig,
    ModelArtifactConfig,
    ReproducibilityConfig,
    UpstreamConfig,
)
from src.reproducibility.rocrate_writer import create_training_rocrate


def generate(output_dir: Path, project_root: Path) -> tuple[Path, Path]:
    output_dir = output_dir.resolve()
    fixture = output_dir / "simulated-fixture"
    raw = fixture / "raw"
    processed = fixture / "processed"
    model_dir = fixture / "model"
    data_dir = model_dir / "data"
    artifacts = model_dir / "artifacts"
    reports = model_dir / "reports"
    for directory in (raw, processed, data_dir, artifacts, reports):
        directory.mkdir(parents=True, exist_ok=True)

    lai = raw / "lai.npy"
    t2m = raw / "t2m.npy"
    np.save(lai, np.arange(8, dtype=np.float32).reshape(2, 2, 2))
    np.save(t2m, np.linspace(270, 277, 8, dtype=np.float32).reshape(2, 2, 2))
    upstream_metadata = fixture / "upstream-ro-crate-metadata.json"
    upstream_graph = [
        {"@id": "./", "@type": "Dataset", "name": "Simulated preprocessing release", "version": "1.0.0"},
        _upstream_file("https://example.org/profecia/lai.npy", lai, "target"),
        _upstream_file("https://example.org/profecia/t2m.npy", t2m, "predictor"),
    ]
    upstream_metadata.write_text(json.dumps({
        "@context": "https://w3id.org/ro/crate/1.1/context", "@graph": upstream_graph,
    }, indent=2) + "\n", encoding="utf-8")

    data_config = fixture / "data.toml"
    data_config.write_text('''[project]
main_dir = "."
raw_dir = "raw"
processed_base_dir = "processed"
mask_dir = "masks"

[data]
temporal_resolution = "annual"
mask_names = []
start_year = 2000
end_year_inclusive = 2001
dtype = "float32"
variable_names = ["LAI", "T2M"]
target_name = "LAI"

[variables.LAI]
role = "target"
local_file = "lai.npy"
upstream_entity_id = "https://example.org/profecia/lai.npy"
frequency = "annual"
required = true

[variables.T2M]
role = "predictor"
local_file = "t2m.npy"
upstream_entity_id = "https://example.org/profecia/t2m.npy"
frequency = "annual"
required = true
''', encoding="utf-8")
    train_config = fixture / "train.toml"
    train_config.write_text('''[project]
data_config = "data.toml"

[split]
mode = "random_observation"
train_fraction = 0.5
test_fraction = 0.5
seed = 42

[model]
name = "rf"
scaler = "none"
random_state = 42

[model.params]
n_estimators = 2

[reproducibility]
create_rocrate = true
reproducible_run = true
''', encoding="utf-8")

    np.save(processed / "LAI.npy", np.load(lai))
    np.save(processed / "T2M.npy", np.load(t2m))
    grid = {
        "lat_min": -0.5, "lat_max": 0.0, "lon_min": 10.0, "lon_max": 10.5,
        "latitude_resolution_deg": 0.5, "longitude_resolution_deg": 0.5,
        "latitude_order": "ascending", "longitude_order": "ascending",
    }
    (processed / "metadata.json").write_text(json.dumps({"variables": {
        name: {"array_path": str(processed / f"{name}.npy"), "final_shape": [2, 2, 2],
               "final_dtype": "float32", "final_units": unit, "grid": grid}
        for name, unit in (("LAI", "m2/m2"), ("T2M", "K"))
    }}, indent=2) + "\n", encoding="utf-8")
    (processed / "run_config.json").write_text(
        '{"variable_names":["LAI","T2M"],"target_name":"LAI"}\n', encoding="utf-8"
    )

    split_rows = {"train": np.array([0, 3, 4, 7]), "test": np.array([1, 2, 5, 6])}
    for split, indices in split_rows.items():
        time_idx, within = np.divmod(indices, 4)
        lat_idx, lon_idx = np.divmod(within, 2)
        np.save(data_dir / f"X_{split}.npy", np.ones((4, 1), dtype=np.float32))
        np.save(data_dir / f"y_{split}.npy", np.ones(4, dtype=np.float32))
        np.save(data_dir / f"pixel_id_{split}.npy", lat_idx * 2 + lon_idx)
        np.save(data_dir / f"time_idx_{split}.npy", time_idx)
        np.save(data_dir / f"lat_idx_{split}.npy", lat_idx)
        np.save(data_dir / f"lon_idx_{split}.npy", lon_idx)

    dataset_metadata = {
        "feature_names": ["T2M"], "target": "LAI", "split_mode": "random_observation",
        "n_train": 4, "n_test": 4, "n_time": 2, "latitude_size": 2, "longitude_size": 2,
        "time_values": ["2000", "2001"], "split_metadata": {"seed": 42},
    }
    train_info = {
        "model_name": "RandomForestRegressor", "model_params": {"n_estimators": 2},
        "random_state": 42, "scaler_name": "none", "hyperparameter_search": {"enabled": False},
    }
    (data_dir / "dataset_metadata.json").write_text(json.dumps(dataset_metadata), encoding="utf-8")
    (data_dir / "split_metadata.json").write_text(
        '{"seed":42,"split_mode":"random_observation","n_selected_observations":8}', encoding="utf-8"
    )
    (artifacts / "train_info.json").write_text(json.dumps(train_info), encoding="utf-8")
    (artifacts / "model.joblib").write_bytes(b"simulated model artifact metadata source")
    (reports / "summary_metrics.csv").write_text("r2,rmse,mae\n0.9,0.25,0.16\n", encoding="utf-8")

    settings = ReproducibilityConfig(
        create_rocrate=True, reproducible_run=True, output_dir=output_dir / "crates",
        upstream_rocrate=str(upstream_metadata), input_catalogue=data_config,
        upstream=UpstreamConfig(
            identifier="https://doi.org/10.0000/profecia.simulated.1",
            landing_page="https://example.org/profecia/simulated-release-1",
            metadata_url="https://example.org/profecia/simulated-release-1/ro-crate-metadata.json",
            cache_dir=Path("./cache/upstream"), verify_checksums=True,
        ),
        inputs=InputsConfig(require_upstream_checksum=True, allow_derived_local_inputs=True),
        environment=EnvironmentConfig(lock_file=project_root / "requirements.txt"),
        code=CodeConfig(mode="reference", require_clean_repository=False),
        croissant=CroissantConfig(mode="descriptive", include_split_manifest=True),
        model_artifact=ModelArtifactConfig(mode="metadata_only"),
    )
    cfg = {
        "config_path": str(train_config), "data_config_path": str(data_config),
        "data": {"mask_names": [], "target_name": "LAI"}, "raw_dir": raw,
        "mask_dir": fixture / "masks", "processed_dir": processed,
        "model_dir": model_dir, "model_data_dir": data_dir,
        "model_artifacts_dir": artifacts, "model_figures_dir": model_dir / "figures",
        "variable_names": ["LAI", "T2M"], "predictor_names": ["T2M"], "target_name": "LAI",
        "split_mode": "random_observation", "train_fraction": 0.5, "test_fraction": 0.5,
        "seed": 42, "random_state": 42, "model_name": "rf", "model_params": {"n_estimators": 2},
        "model_run_name": "rf_simulated", "mlflow": {"enabled": False},
        "reproducibility": settings,
    }
    summary = {
        "processed_dir": str(processed), "model_data_dir": str(data_dir),
        "dataset_metadata": dataset_metadata, "train_info": train_info,
        "evaluation": {"global_metrics": {"r2": 0.9, "rmse": 0.25, "mae": 0.16}},
        "saved_paths": {
            "model_path": str(artifacts / "model.joblib"),
            "train_info_path": str(artifacts / "train_info.json"),
        },
    }
    crate = create_training_rocrate(
        cfg, summary, datetime(2026, 7, 21, 12, 0, tzinfo=UTC),
        datetime(2026, 7, 21, 12, 1, tzinfo=UTC),
        ["python", "scripts/workflow_step_by_step_configurable.py"], project_root=project_root,
    )
    if crate is None:
        raise RuntimeError("RO-Crate generation was unexpectedly disabled")
    zip_path = Path(shutil.make_archive(str(output_dir / "profecia-simulated-training-run"), "zip", crate))
    return crate, zip_path


def _upstream_file(identifier: str, path: Path, role: str) -> dict:
    return {
        "@id": identifier, "@type": "File", "name": path.name,
        "contentUrl": identifier, "encodingFormat": "application/x-npy",
        "contentSize": str(path.stat().st_size),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "additionalProperty": {"@type": "PropertyValue", "name": "role", "value": role},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/simulated-rocrate"))
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    crate, archive = generate(args.output_dir, project_root)
    print(crate)
    print(archive)


if __name__ == "__main__":
    main()
