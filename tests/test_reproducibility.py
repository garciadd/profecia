from __future__ import annotations

import json
import hashlib
import subprocess
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path

from src.reproducibility.config import (
    CodeConfig, CroissantConfig, EnvironmentConfig, LicensesConfig,
    ModelArtifactConfig, ReproducibilityConfig,
)
from src.reproducibility.croissant import build_croissant_descriptor
from src.reproducibility.fair4ml import build_fair4ml_descriptor
from src.reproducibility.run_metadata import (
    CREATOR, DATASET_LICENSE, build_run_metadata, ensure_run_identifiers,
    decode_observation_indices, write_split_manifest,
)
from src.reproducibility.rocrate_writer import (
    _build_rocrate_metadata,
    _prepare_model_artifact,
    _redact_patch_secrets,
    _write_environment,
    create_training_rocrate,
    validate_rocrate_jsonld,
)
from src.reproducibility.upstream_crate import (
    InputSpec,
    UpstreamResolutionError,
    resolve_catalogue_inputs,
)


class ReproducibilityTests(unittest.TestCase):
    def test_disabled_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = {"model_dir": Path(tmp), "reproducibility": ReproducibilityConfig()}
            result = create_training_rocrate(cfg, {}, datetime.now(UTC), project_root=Path(tmp))
        self.assertIsNone(result)

    def test_license_and_environment_configuration(self) -> None:
        settings = ReproducibilityConfig.from_mapping(
            {
                "licenses": {
                    "metadata": DATASET_LICENSE, "code": "", "derived_dataset": "",
                },
                "environment": {
                    "container_image": "registry.example/profecia:1",
                    "container_digest": "sha256:abc",
                },
                "code": {"require_clean_repository": True},
            },
            Path("/tmp"),
        )
        self.assertIsInstance(settings.licenses, LicensesConfig)
        self.assertEqual(settings.licenses.metadata, DATASET_LICENSE)
        self.assertIsNone(settings.licenses.derived_dataset)
        self.assertTrue(settings.code.require_clean_repository)
        self.assertEqual(settings.environment.container_digest, "sha256:abc")
        with self.assertRaisesRegex(ValueError, "configured together"):
            ReproducibilityConfig.from_mapping(
                {"environment": {"container_image": "registry.example/profecia:latest"}}, Path("/tmp")
            )

    def test_simulated_run_references_upstream_and_copies_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_string:
            tmp = Path(tmp_string)
            model_dir = tmp / "run"
            artifacts = model_dir / "artifacts"
            figures = model_dir / "figures"
            reports = model_dir / "reports"
            logs = model_dir / "logs"
            processed_dir = tmp / "processed"
            data_dir = model_dir / "data"
            for directory in (artifacts, figures, reports, logs, processed_dir, data_dir):
                directory.mkdir(parents=True)
            (artifacts / "model.joblib").write_bytes(b"simulated model")
            (artifacts / "train_info.json").write_text("{}", encoding="utf-8")
            (figures / "evaluation.png").write_bytes(b"png")
            (reports / "summary_metrics.csv").write_text("rmse\n0.5\n", encoding="utf-8")
            (reports / "prediction_sample.csv").write_text("y_true,y_pred\n1,1.1\n", encoding="utf-8")
            (logs / "workflow.log").write_text("completed", encoding="utf-8")
            import numpy as np

            np.save(processed_dir / "T2M.npy", np.arange(10, dtype=np.float32))
            np.save(processed_dir / "LAI.npy", np.arange(10, dtype=np.float32))
            for split, rows in (("train", 8), ("test", 2)):
                np.save(data_dir / f"X_{split}.npy", np.ones((rows, 1), dtype=np.float32))
                np.save(data_dir / f"y_{split}.npy", np.ones(rows, dtype=np.float32))
                for stem in ("pixel_id", "lat_idx", "lon_idx", "time_idx"):
                    offset = 8 if split == "test" and stem == "time_idx" else 0
                    np.save(data_dir / f"{stem}_{split}.npy", np.arange(rows, dtype=np.int32) + offset)
            (processed_dir / "metadata.json").write_text(
                json.dumps({"variables": {
                    "T2M": {"array_path": str(processed_dir / "T2M.npy"), "final_shape": [10], "final_dtype": "float32", "final_units": "K"},
                    "LAI": {"array_path": str(processed_dir / "LAI.npy"), "final_shape": [10], "final_dtype": "float32", "final_units": "m2/m2"},
                }}), encoding="utf-8",
            )
            (processed_dir / "run_config.json").write_text('{"variable_names":["LAI","T2M"]}', encoding="utf-8")

            train_config = tmp / "train.toml"
            train_config.write_text('[mlflow]\ntracking_password = "do-not-copy"\n', encoding="utf-8")
            catalogue = tmp / "paths.toml"
            catalogue.write_text('[inputs]\nsource = "large-input.nc"\n', encoding="utf-8")
            upstream = tmp / "upstream"
            upstream.mkdir()
            upstream_metadata = {
                "@context": "https://w3id.org/ro/crate/1.1/context",
                "@graph": [
                    {
                        "@id": "https://data.example/files/large-input.nc",
                        "@type": "File",
                        "name": "large-input.nc",
                        "contentUrl": "https://download.example/large-input.nc",
                        "sha256": "abc123",
                    }
                ],
            }
            (upstream / "ro-crate-metadata.json").write_text(json.dumps(upstream_metadata), encoding="utf-8")

            cfg = {
                "config_path": str(train_config),
                "data_config_path": str(catalogue),
                "data": {"mask_names": [], "target_name": "LAI"},
                "variable_names": [],
                "target_name": "LAI",
                "split_mode": "random_observation",
                "train_fraction": 0.8,
                "test_fraction": 0.2,
                "seed": 42,
                "model_run_name": "rf_test",
                "model_dir": model_dir,
                "model_artifacts_dir": artifacts,
                "model_figures_dir": figures,
                "processed_dir": processed_dir,
                "model_data_dir": data_dir,
                "mlflow": {"enabled": False},
                "reproducibility": ReproducibilityConfig(
                    create_rocrate=True,
                    output_dir=tmp / "crates",
                    upstream_rocrate=str(upstream),
                    input_catalogue=catalogue,
                    croissant=CroissantConfig(
                        mode="descriptive", include_split_manifest=True,
                    ),
                ),
            }
            summary = {
                "dataset_metadata": {
                    "feature_names": ["T2M"],
                    "target": "LAI",
                    "split_mode": "random_observation",
                    "n_train": 8,
                    "n_test": 2,
                    "n_time": 10,
                    "latitude_size": 10,
                    "longitude_size": 10,
                    "split_metadata": {"seed": 42},
                },
                "train_info": {
                    "model_name": "RandomForestRegressor",
                    "model_params": {"n_estimators": 2},
                    "random_state": 42,
                    "scaler_name": "none",
                    "mlflow_run_id": None,
                    "hyperparameter_search": {
                        "enabled": True, "cv_class": "KFold", "n_splits": 3,
                        "n_iter": 2, "scoring": "neg_root_mean_squared_error", "best_score": -0.5,
                    },
                },
                "evaluation": {"global_metrics": {"rmse": 0.5, "r2": 0.8}},
            }
            (data_dir / "dataset_metadata.json").write_text(
                json.dumps(summary["dataset_metadata"]), encoding="utf-8"
            )
            (data_dir / "split_metadata.json").write_text(
                '{"seed":42,"n_selected_observations":10}', encoding="utf-8"
            )
            (artifacts / "train_info.json").write_text(
                json.dumps(summary["train_info"]), encoding="utf-8"
            )
            summary["processed_dir"] = str(processed_dir)
            summary["model_data_dir"] = str(data_dir)
            summary["saved_paths"] = {
                "model_path": str(artifacts / "model.joblib"),
                "train_info_path": str(artifacts / "train_info.json"),
            }
            crate = create_training_rocrate(
                cfg,
                summary,
                datetime(2026, 1, 1, tzinfo=UTC),
                datetime(2026, 1, 1, 0, 1, tzinfo=UTC),
                ["python", "train.py", "--small"],
                project_root=tmp,
            )
            self.assertIsNotNone(crate)
            assert crate is not None
            self.assertTrue((crate / "model" / "model.joblib").exists())
            self.assertTrue((crate / "metrics" / "summary_metrics.csv").exists())
            self.assertTrue((crate / "predictions" / "prediction_sample.csv").exists())
            self.assertTrue((crate / "figures" / "evaluation.png").exists())
            self.assertFalse(any(crate.rglob("*.nc")))
            self.assertFalse(any(path for path in crate.rglob("*.npy")))
            self.assertTrue((crate / "fair" / "split_manifest.npz").is_file())
            self.assertTrue((crate / "metadata" / "processed_metadata.json").exists())
            self.assertTrue((crate / "metadata" / "dataset_metadata.json").exists())
            self.assertIn("***REDACTED***", (crate / "config" / "train.toml").read_text(encoding="utf-8"))
            self.assertNotIn("do-not-copy", (crate / "config" / "train.toml").read_text(encoding="utf-8"))

            metadata = json.loads((crate / "ro-crate-metadata.json").read_text(encoding="utf-8"))
            validate_rocrate_jsonld(metadata)
            graph = {entity["@id"]: entity for entity in metadata["@graph"]}
            upstream_id = "https://data.example/files/large-input.nc"
            self.assertEqual(graph[upstream_id]["contentUrl"], "https://download.example/large-input.nc")
            self.assertEqual(graph[upstream_id]["sha256"], "abc123")
            action = next(entity for entity in metadata["@graph"] if entity.get("@type") == "CreateAction")
            self.assertIn({"@id": upstream_id}, action["prov:used"])
            by_suffix = lambda suffix: next(entity for key, entity in graph.items() if key.endswith(suffix))
            self.assertTrue(any(key.endswith("#training-dataset") for key in graph))
            self.assertTrue(any(key.endswith("#test-dataset") for key in graph))
            self.assertFalse(any(key.endswith("#validation-dataset") for key in graph))
            self.assertTrue(any(key.endswith("#cross-validation") for key in graph))
            self.assertNotIn("contentUrl", by_suffix("#processed-array-t2m"))
            ml_dataset = by_suffix("#ml-dataset")
            model = by_suffix("#ml-model")
            self.assertNotIn("license", ml_dataset)
            self.assertIn("not yet been established", ml_dataset["conditionsOfAccess"])
            self.assertNotIn("license", model)
            self.assertIn("not yet been established", model["conditionsOfAccess"])
            self.assertEqual(graph["./"]["license"], {"@id": DATASET_LICENSE})

            croissant = json.loads((crate / "fair" / "dataset.croissant.json").read_text(encoding="utf-8"))
            self.assertEqual(croissant["dct:conformsTo"], "http://mlcommons.org/croissant/1.1")
            self.assertEqual(croissant["distribution"][0]["sha256"], "abc123")
            self.assertNotIn("recordSet", croissant)
            fair4ml = json.loads((crate / "fair" / "model.fair4ml.json").read_text(encoding="utf-8"))
            self.assertIn("fair4ml:MLModel", fair4ml["@graph"][0]["@type"])
            metrics = {
                item["name"]: item["value"]
                for item in fair4ml["@graph"][1]["fair4ml:evaluationMetrics"]
            }
            self.assertEqual(metrics, summary["evaluation"]["global_metrics"])
            from rdflib import Graph

            Graph().parse(data=json.dumps(croissant), format="json-ld")
            Graph().parse(data=json.dumps(fair4ml), format="json-ld")

    def test_basic_validation_rejects_duplicate_ids(self) -> None:
        invalid = {
            "@context": "https://w3id.org/ro/crate/1.1/context",
            "@graph": [{"@id": "./"}, {"@id": "./"}],
        }
        with self.assertRaisesRegex(ValueError, "unique"):
            validate_rocrate_jsonld(invalid)

    def test_git_patch_secrets_are_redacted(self) -> None:
        patch = '+tracking_password = "sensitive"\n-api_token = "old-token"\n+enabled = true'
        redacted = _redact_patch_secrets(patch)
        self.assertNotIn("sensitive", redacted)
        self.assertNotIn("old-token", redacted)
        self.assertEqual(redacted.count("***REDACTED***"), 2)

    def test_acceptance_git_aliases_derivation_and_complete_links(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_string:
            tmp = Path(tmp_string)
            self._init_git_repository(tmp)
            raw_dir, mask_dir = tmp / "raw", tmp / "masks"
            raw_dir.mkdir()
            mask_dir.mkdir()
            local_files = {
                "T2M": "t2m_1982_2022_monthly_0.5deg.nc",
                "TP": "tp_1982_2022_monthly_0.5deg.nc",
                "SSRD": "ssrd_1982_2022_monthly_0.5deg.nc",
            }
            canonical = {
                "T2M": "era5_single_levels_t2m_monthly_0.5deg.nc",
                "TP": "era5_single_levels_tp_monthly_0.5deg.nc",
                "SSRD": "era5_single_levels_ssrd_monthly_0.5deg.nc",
            }
            upstream_graph = []
            for name, local_name in local_files.items():
                content = f"{name}-content".encode()
                (raw_dir / local_name).write_bytes(content)
                entity_id = f"https://example.org/crate#referenced-data/monthly/{canonical[name]}"
                upstream_graph.append(
                    {
                        "@id": entity_id,
                        "@type": "File",
                        "name": f"monthly/{canonical[name]}",
                        "dcat:downloadURL": f"https://download.example/{canonical[name]}",
                        "sha256": hashlib.sha256(content).hexdigest(),
                        "variableMeasured": {"@id": f"https://example.org/variable/{name.lower()}"},
                    }
                )

            import numpy as np

            categorical_path = mask_dir / "landcover_mask_0p5_7classes.npy"
            binary_path = mask_dir / "landcover_snow_ice_0p5deg.npy"
            categorical_values = np.array([[10, 100], [30, 100]], dtype=np.int16)
            np.save(categorical_path, categorical_values)
            np.save(binary_path, categorical_values == 100)
            categorical = categorical_path.read_bytes()
            mask_source_id = "https://example.org/resources/landcover_mask_0p5_7classes.npy"
            upstream_graph.append(
                {
                    "@id": mask_source_id,
                    "@type": "File",
                    "name": "resources/source_data/masks/landcover_mask_0p5_7classes.npy",
                    "contentUrl": "https://download.example/landcover_mask_0p5_7classes.npy",
                    "sha256": hashlib.sha256(categorical).hexdigest(),
                }
            )
            upstream = tmp / "upstream"
            upstream.mkdir()
            (upstream / "ro-crate-metadata.json").write_text(
                json.dumps({"@context": "https://w3id.org/ro/crate/1.1/context", "@graph": upstream_graph}),
                encoding="utf-8",
            )
            catalogue = tmp / "config" / "data.toml"
            catalogue.parent.mkdir()
            catalogue.write_text(
                """
[data]
variable_names = ["T2M", "TP", "SSRD"]
mask_names = ["landcover_snow_ice"]
target_name = "T2M"

[variables.T2M]
local_file = "t2m_1982_2022_monthly_0.5deg.nc"
upstream_entity_id = "referenced-data/monthly/era5_single_levels_t2m_monthly_0.5deg.nc"
netcdf_variable = "t2m"
[variables.TP]
local_file = "tp_1982_2022_monthly_0.5deg.nc"
upstream_entity_id = "referenced-data/monthly/era5_single_levels_tp_monthly_0.5deg.nc"
netcdf_variable = "tp"
[variables.SSRD]
local_file = "ssrd_1982_2022_monthly_0.5deg.nc"
upstream_entity_id = "referenced-data/monthly/era5_single_levels_ssrd_monthly_0.5deg.nc"
netcdf_variable = "ssrd"
[masks.landcover_snow_ice]
local_file = "landcover_snow_ice_0p5deg.npy"
source_local_file = "landcover_mask_0p5_7classes.npy"
derived_from_upstream_entity_id = "https://example.org/resources/landcover_mask_0p5_7classes.npy"
derivation_type = "class_selection"
class_value = 100
class_label = "Snow/Ice"
""".strip()
                + "\n",
                encoding="utf-8",
            )
            train_config = tmp / "config" / "train.toml"
            train_config.write_text("[reproducibility]\ncreate_rocrate = true\n", encoding="utf-8")
            model_dir = tmp / "model-run"
            artifacts, figures, reports = model_dir / "artifacts", model_dir / "figures", model_dir / "reports"
            for directory in (artifacts, figures, reports):
                directory.mkdir(parents=True)
            (artifacts / "model.joblib").write_bytes(b"model")
            cfg = {
                "config_path": str(train_config), "data_config_path": str(catalogue),
                "raw_dir": raw_dir, "mask_dir": mask_dir,
                "data": {"mask_names": ["landcover_snow_ice"], "target_name": "T2M"},
                "variable_names": ["T2M", "TP", "SSRD"], "target_name": "T2M",
                "split_mode": "random_observation", "train_fraction": 0.8, "test_fraction": 0.2, "seed": 42,
                "model_run_name": "acceptance", "model_dir": model_dir,
                "model_artifacts_dir": artifacts, "model_figures_dir": figures,
                "mlflow": {"enabled": False},
                "reproducibility": ReproducibilityConfig(
                    create_rocrate=True, output_dir=tmp / "crates",
                    upstream_rocrate=str(upstream), input_catalogue=catalogue,
                    code=CodeConfig(mode="patch"),
                ),
            }
            summary = {
                "dataset_metadata": {"feature_names": ["TP", "SSRD"], "target": "T2M", "n_train": 8, "n_test": 2},
                "train_info": {"model_name": "RF", "model_params": {}, "random_state": 42},
                "evaluation": {"global_metrics": {"rmse": 0.5}},
            }
            crate = create_training_rocrate(
                cfg, summary, datetime(2026, 1, 1, tzinfo=UTC), datetime.now(UTC),
                ["python", "train.py"], project_root=tmp,
            )
            assert crate is not None
            metadata = json.loads((crate / "ro-crate-metadata.json").read_text())
            graph = {entity["@id"]: entity for entity in metadata["@graph"]}
            commit = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=tmp, check=True, capture_output=True, text=True
            ).stdout.strip()
            repository = "https://github.com/garciadd/profecia"
            source_id = next(key for key in graph if key.endswith("#source-code"))
            application_id = next(key for key in graph if key.endswith("#training-application"))
            for entity_id in (source_id, application_id):
                entity = graph[entity_id]
                self.assertEqual(entity["codeRepository"], repository)
                self.assertEqual(entity["identifier"], commit)
                self.assertEqual(entity["url"], f"{repository}/tree/{commit}")
                properties = {item["name"]: item for item in entity["additionalProperty"]}
                self.assertEqual(properties["git_branch"]["value"], "feature/ro-crate")
                self.assertEqual(properties["git_branch"]["url"], f"{repository}/tree/feature/ro-crate")
                self.assertTrue(properties["git_dirty"]["value"])
            self.assertIn({"@id": "code/git-diff.patch"}, graph[source_id]["hasPart"])
            self.assertEqual(graph["code/git-diff.patch"]["about"], {"@id": source_id})

            expected_direct = {
                f"https://example.org/crate#referenced-data/monthly/{name}" for name in canonical.values()
            }
            derived_id = "data/masks/landcover_snow_ice_0p5deg.npy"
            expected_used = expected_direct | {derived_id}
            for entity_id in expected_direct:
                self.assertIn("dcat:downloadURL", graph[entity_id])
                self.assertIn("variableMeasured", graph[entity_id])
            training = next(e for e in metadata["@graph"] if e.get("name") == "Train a PROFECIA model")
            preparation = next(
                e for e in metadata["@graph"]
                if e.get("name") == "Prepare and partition the PROFECIA ML dataset"
            )
            refs = lambda values: {item["@id"] for item in values}
            self.assertTrue(expected_used <= refs(preparation["object"]))
            self.assertTrue(expected_used <= refs(preparation["prov:used"]))
            train_dataset_id = next(key for key in graph if key.endswith("#training-dataset"))
            self.assertEqual(training["object"], [{"@id": train_dataset_id}])
            self.assertNotIn({"@id": "fair/dataset.croissant.json"}, training["object"])
            croissant = json.loads((crate / "fair" / "dataset.croissant.json").read_text())
            expected_croissant = {
                f"../{identifier}" if identifier.startswith("data/") else identifier
                for identifier in expected_used
            }
            self.assertTrue(expected_croissant <= {item["@id"] for item in croissant["distribution"]})
            self.assertEqual(graph[derived_id]["prov:wasDerivedFrom"], {"@id": mask_source_id})
            derive_action = graph["#derive-mask-landcover_snow_ice"]
            self.assertEqual(derive_action["result"], {"@id": derived_id})
            self.assertIn({"@id": "config/data.toml"}, derive_action["object"])
            ml_dataset_id = next(key for key in graph if key.endswith("#ml-dataset"))
            properties = {item["name"]: item["value"] for item in graph[ml_dataset_id]["additionalProperty"]}
            self.assertEqual(properties["upstream_files_not_matched"], [])

    def test_wrong_explicit_upstream_id_is_an_error(self) -> None:
        metadata = {"@graph": [{"@id": "https://example.org/real.nc", "name": "real.nc"}]}
        spec = InputSpec("variable:T2M", "variable", "local.nc", upstream_entity_id="missing/entity.nc")
        with self.assertRaisesRegex(UpstreamResolutionError, "does not exist"):
            resolve_catalogue_inputs(metadata, [spec])

    def test_model_artifact_modes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_string:
            tmp = Path(tmp_string)
            source = tmp / "source.joblib"
            source.write_bytes(b"trained-model")

            attached, attached_path = _prepare_model_artifact(
                tmp / "attached", ModelArtifactConfig(mode="attached"), source
            )
            self.assertTrue(attached_path and attached_path.is_file())
            self.assertEqual(attached["main_id"], "model/source.joblib")
            self.assertIn("sha256", attached["entity"])

            external, external_path = _prepare_model_artifact(
                tmp / "external",
                ModelArtifactConfig(
                    mode="external",
                    download_url="https://models.example/model.joblib",
                    landing_page="https://models.example/records/1",
                    identifier="https://doi.org/10.example/model",
                ),
                source,
            )
            self.assertIsNone(external_path)
            self.assertEqual(external["main_id"], "https://models.example/model.joblib")
            self.assertFalse((tmp / "external" / "model" / "source.joblib").exists())

            metadata_only, metadata_path = _prepare_model_artifact(
                tmp / "metadata", ModelArtifactConfig(mode="metadata_only"), source
            )
            self.assertIsNone(metadata_path)
            self.assertFalse(metadata_only["entity"]["isAccessibleForFree"])
            self.assertFalse(metadata_only["entity"]["additionalProperty"]["value"])

    def test_materialized_croissant_uses_parquet_not_npy(self) -> None:
        import numpy as np
        from rdflib import Graph

        with tempfile.TemporaryDirectory() as tmp_string:
            tmp = Path(tmp_string)
            data_dir, crate = tmp / "arrays", tmp / "crate"
            data_dir.mkdir()
            crate.mkdir()
            for split, rows in (("train", 3), ("test", 2)):
                np.save(data_dir / f"X_{split}.npy", np.arange(rows * 2, dtype=np.float32).reshape(rows, 2))
                np.save(data_dir / f"y_{split}.npy", np.arange(rows, dtype=np.float32))
                np.save(data_dir / f"pixel_id_{split}.npy", np.arange(rows, dtype=np.int32))
            run = {
                "dataset": {"n_train": 3, "n_test": 2, "split_mode": "random_observation"},
                "split": {"seed": 42}, "features": ["T2M", "TP"], "target": "LAI",
                "masks": ["land"], "creator": CREATOR, "license": DATASET_LICENSE,
                "processed_variables": [],
            }
            descriptor, paths = build_croissant_descriptor(
                run, "https://example.org/upstream", [], "2026-01-01T00:00:00Z",
                mode="materialized", crate_dir=crate, model_data_dir=data_dir,
            )
            self.assertEqual({path.name for path in paths}, {"train.parquet", "test.parquet"})
            self.assertTrue(all(path.is_file() for path in paths))
            self.assertFalse(any(crate.rglob("*.npy")))
            self.assertEqual(len(descriptor["recordSet"]), 2)
            sources = [
                field["source"]["fileObject"]["@id"]
                for records in descriptor["recordSet"] for field in records["field"]
            ]
            self.assertTrue(all(source.endswith(".parquet") for source in sources))
            field_names = {
                field["name"] for records in descriptor["recordSet"] for field in records["field"]
            }
            self.assertIn("split", field_names)
            Graph().parse(data=json.dumps(descriptor), format="json-ld")

    def test_split_manifest_is_exact_and_shared_across_descriptors(self) -> None:
        import numpy as np

        with tempfile.TemporaryDirectory() as tmp_string:
            tmp = Path(tmp_string)
            arrays = tmp / "arrays"
            crate = tmp / "crate"
            arrays.mkdir()
            crate.mkdir()
            traces = {
                "train": {"time_idx": [0, 0, 1], "lat_idx": [0, 1, 0], "lon_idx": [0, 1, 0]},
                "test": {"time_idx": [1, 1], "lat_idx": [0, 1], "lon_idx": [1, 0]},
            }
            partition_arrays = []
            for split, values in traces.items():
                for role, data in values.items():
                    path = arrays / f"{role}_{split}.npy"
                    np.save(path, np.asarray(data, dtype=np.int32))
                    partition_arrays.append({"split": split, "role": role, "path": path})
            run = {
                "dataset": {
                    "n_train": 3, "n_test": 2, "n_time": 2,
                    "latitude_size": 2, "longitude_size": 2,
                    "split_mode": "random_observation",
                },
                "split": {"seed": 42, "n_selected_observations": 5},
                "partition_arrays": partition_arrays,
                "features": ["T2M"], "target": "LAI", "masks": [],
                "creator": CREATOR, "license": DATASET_LICENSE,
                "processed_variables": [], "train": {"model_name": "RF"},
                "metrics": {"rmse": 0.5},
            }
            ids = ensure_run_identifiers(run)
            run["processed_variables"] = [{
                "name": "T2M",
                "grid": {
                    "lat_min": -90.0, "lat_max": -89.5,
                    "lon_min": -180.0, "lon_max": -179.5,
                    "latitude_resolution_deg": 0.5, "longitude_resolution_deg": 0.5,
                    "latitude_order": "ascending", "longitude_order": "ascending",
                },
            }]
            run["dataset"]["time_values"] = ["1982", "1983"]
            manifest = write_split_manifest(
                crate, run, {"commit": "a" * 40, "dirty": True, "diff_path": Path("patch")}
            )
            with np.load(manifest["npz_path"]) as payload:
                np.testing.assert_array_equal(payload["train_indices"], [0, 3, 4])
                np.testing.assert_array_equal(payload["test_indices"], [5, 6])
            self.assertEqual(manifest["metadata"]["code_version"], "a" * 40)
            self.assertEqual(manifest["metadata"]["flatten_order"], "C")
            decoded = decode_observation_indices([0, 7], manifest["metadata"])
            self.assertEqual(decoded["time"].tolist(), ["1982", "1983"])
            self.assertEqual(decoded["latitude"].tolist(), [-90.0, -89.5])
            self.assertEqual(decoded["longitude"].tolist(), [-180.0, -179.5])
            croissant, _ = build_croissant_descriptor(
                run, None, [], "2026-01-01T00:00:00Z", split_manifest=manifest
            )
            artifact = {
                "mode": "metadata_only", "main_id": ids["model-artifact"],
                "fair_id": ids["model-artifact"],
                "entity": {"@id": ids["model-artifact"], "@type": "CreativeWork"},
            }
            fair = build_fair4ml_descriptor(
                run, artifact, "2026-01-01T00:00:00Z", split_manifest=manifest
            )
            self.assertEqual(croissant["@id"], ids["ml-dataset"])
            self.assertEqual(croissant["hasPart"][0]["@id"], ids["training-dataset"])
            fair_ids = {entity["@id"] for entity in fair["@graph"]}
            self.assertTrue({ids["ml-model"], ids["model-evaluation"], ids["ml-dataset"]} <= fair_ids)

    def test_croissant_classifies_remote_semantic_and_local_resources(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_string:
            crate = Path(tmp_string)
            local = crate / "data" / "masks" / "derived.npy"
            local.parent.mkdir(parents=True)
            local.write_bytes(b"npy")
            run = {
                "dataset": {"n_train": 1, "n_test": 1}, "split": {},
                "features": ["T2M"], "target": "LAI", "masks": [],
                "creator": CREATOR, "license": None, "processed_variables": [],
            }
            remote_url = "https://data.example/T2M.nc"
            entities = [
                {"@id": "https://example.org/T2M", "name": "T2M.nc", "contentUrl": remote_url},
                {"@id": "https://example.org/semantic-only", "name": "TP.nc"},
                {"@id": "data/masks/derived.npy", "name": "derived.npy"},
            ]
            descriptor, _ = build_croissant_descriptor(
                run, None, entities, "2026-01-01T00:00:00Z", crate_dir=crate
            )
            distribution = {item["@id"]: item for item in descriptor["distribution"]}
            self.assertEqual(distribution["https://example.org/T2M"]["contentUrl"], remote_url)
            self.assertNotIn("https://example.org/semantic-only", distribution)
            self.assertEqual(distribution["../data/masks/derived.npy"]["contentUrl"], "../data/masks/derived.npy")
            for item in distribution.values():
                if not item["contentUrl"].startswith(("http://", "https://")):
                    self.assertTrue((crate / "fair" / item["contentUrl"]).resolve().is_file())
            properties = {item["name"]: item["value"] for item in descriptor["additionalProperty"]}
            self.assertIn("https://example.org/semantic-only", properties["croissant_semantic_only_resources"])

    def test_environment_lock_and_immutable_container_are_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_string:
            tmp = Path(tmp_string)
            lock = tmp / "requirements.lock"
            lock.write_text("numpy==2.0.0\n", encoding="utf-8")
            env, copied, requirements, container = _write_environment(
                tmp / "crate",
                EnvironmentConfig(
                    lock_file=lock, container_image="registry.example/profecia:1",
                    container_digest="sha256:abc", container_platform="linux/amd64",
                    container_runtime="docker",
                ),
            )
            self.assertTrue(env.is_file())
            self.assertEqual(len(copied), 1)
            self.assertIn("environment/lock/requirements.lock", requirements)
            assert container is not None
            self.assertEqual(container["@id"], "registry.example/profecia:1@sha256:abc")

    def test_require_clean_repository_rejects_dirty_worktree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_string:
            tmp = Path(tmp_string)
            self._init_git_repository(tmp)
            cfg = {
                "model_dir": tmp / "model", "model_run_name": "dirty",
                "reproducibility": ReproducibilityConfig(
                    create_rocrate=True, output_dir=tmp / "crates",
                    code=CodeConfig(mode="reference", require_clean_repository=True),
                ),
            }
            with self.assertRaisesRegex(ValueError, "repository is dirty"):
                create_training_rocrate(cfg, {}, datetime.now(UTC), project_root=tmp)
            self.assertFalse((tmp / "crates").exists())

    def test_external_model_is_connected_without_local_reference(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_string:
            tmp = Path(tmp_string)
            source = tmp / "source.joblib"
            source.write_bytes(b"model")
            artifact, _ = _prepare_model_artifact(
                tmp,
                ModelArtifactConfig(mode="external", download_url="https://models.example/model.joblib"),
                source,
            )
            fair_model = {
                "fair4ml:modelRisksBiasLimitations": "Test scope only.",
                "fair4ml:usageInstructions": "Download and verify.",
                "additionalProperty": [],
            }
            metadata = _build_rocrate_metadata(
                crate_dir=tmp, copied=[], cfg={"data": {"mask_names": []}},
                summary={"train_info": {}, "dataset_metadata": {}},
                started_at=datetime.now(UTC), ended_at=datetime.now(UTC), command=["python", "train.py"],
                git={}, upstream_identifier=None, resolved_inputs=[], derived_entities=[],
                input_catalogue_id=None,
                run={"dataset": {}, "train": {}, "metrics": {}, "cv": {"enabled": False}},
                artifact=artifact, fair4ml_model=fair_model,
            )
            graph = {entity["@id"]: entity for entity in metadata["@graph"]}
            url = "https://models.example/model.joblib"
            model_id = next(key for key in graph if key.endswith("#ml-model"))
            self.assertEqual(graph[model_id]["associatedMedia"], {"@id": url})
            training = next(entity for entity in metadata["@graph"] if entity.get("name") == "Train a PROFECIA model")
            self.assertIn({"@id": url}, training["result"])
            self.assertNotIn("model/model.joblib", graph)

    def test_intermediate_metadata_mismatch_is_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_string:
            tmp = Path(tmp_string)
            data_dir = tmp / "data"
            data_dir.mkdir()
            (data_dir / "dataset_metadata.json").write_text(
                json.dumps({"feature_names": ["TP"], "target": "LAI", "n_train": 8, "n_test": 2}),
                encoding="utf-8",
            )
            cfg = {"model_data_dir": data_dir, "data": {"mask_names": []}}
            summary = {"dataset_metadata": {"feature_names": ["T2M"], "target": "LAI", "n_train": 8, "n_test": 2}}
            with self.assertRaisesRegex(ValueError, "Inconsistent dataset feature_names"):
                build_run_metadata(cfg, summary)

    @staticmethod
    def _init_git_repository(root: Path) -> None:
        subprocess.run(["git", "init", "-b", "feature/ro-crate"], cwd=root, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test Author"], cwd=root, check=True)
        subprocess.run(["git", "config", "user.email", "test@example.org"], cwd=root, check=True)
        subprocess.run(
            ["git", "remote", "add", "origin", "https://github.com/garciadd/profecia.git"], cwd=root, check=True
        )
        source = root / "src" / "app.py"
        source.parent.mkdir()
        source.write_text("VERSION = 1\n", encoding="utf-8")
        subprocess.run(["git", "add", "src/app.py"], cwd=root, check=True)
        subprocess.run(["git", "commit", "-m", "initial"], cwd=root, check=True, capture_output=True)
        source.write_text("VERSION = 2\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
