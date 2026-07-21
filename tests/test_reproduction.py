from __future__ import annotations

import hashlib
import json
import tempfile
import threading
import unittest
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import xarray as xr

from src.reproducibility.config import InputsConfig, UpstreamConfig
from src.reproducibility.effective_config import build_effective_config
from src.reproducibility.expected_results import build_expected_results, compare_results
from src.reproducibility.input_resolution import reconstruct_derived_inputs, resolve_reproduction_inputs
from src.reproducibility.reproduce import _write_reproduced_crate, reproduce
from src.reproducibility.upstream_crate import InputSpec, ResolvedInput, UpstreamResolutionError
from src.project_config import resolve_effective_train_config


class ReproductionTests(unittest.TestCase):
    def test_download_cache_offline_and_checksum(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_string:
            tmp = Path(tmp_string)
            served = tmp / "served"
            served.mkdir()
            payload = served / "input.npy"
            np.save(payload, np.arange(3))
            checksum = _sha(payload)
            server = ThreadingHTTPServer(
                ("127.0.0.1", 0), partial(SimpleHTTPRequestHandler, directory=str(served))
            )
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                url = f"http://127.0.0.1:{server.server_port}/input.npy"
                resolved = [_resolved("variable:T2M", "input.npy", url, checksum)]
                upstream = UpstreamConfig(cache_dir=tmp / "cache", verify_checksums=True)
                first = resolve_reproduction_inputs(resolved, upstream, InputsConfig(True, True), tmp / "run1")
                self.assertTrue(first[0]["checksum_verified"])
                second = resolve_reproduction_inputs(
                    resolved, upstream, InputsConfig(True, True), tmp / "run2", offline=True
                )
                self.assertTrue(Path(second[0]["resolved_path"]).is_file())
                bad = [_resolved("variable:T2M", "bad.npy", url, "0" * 64)]
                with self.assertRaisesRegex(UpstreamResolutionError, "SHA-256 mismatch"):
                    resolve_reproduction_inputs(bad, upstream, InputsConfig(True, True), tmp / "bad")
            finally:
                server.shutdown()
                thread.join()
                server.server_close()

    def test_documented_derived_input_with_different_checksum(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_string:
            tmp = Path(tmp_string)
            source = tmp / "categorical.npy"
            np.save(source, np.array([[100, 20], [20, 100]], dtype=np.int16))
            spec = InputSpec(
                "mask:snow", "mask", "snow.npy", derived_from_upstream_entity_id="source.npy",
                derivation_type="class_selection", class_value=100, class_label="Snow/Ice",
                source_local_file="categorical.npy",
            )
            item = ResolvedInput(
                spec, {"@id": "source.npy", "sha256": _sha(source)}, "explicit", None,
                "different", _sha(source), "verified", derived_entity_id="data/masks/snow.npy",
            )
            result = reconstruct_derived_inputs(
                [item], [{"key": "mask:snow", "resolved_path": str(source)}], tmp / "derived"
            )
            self.assertEqual(result[0]["function"], "numpy.equal")
            np.testing.assert_array_equal(np.load(tmp / "derived" / "snow.npy"), [[True, False], [False, True]])

    def test_exact_split_reconstruction(self) -> None:
        from src.training.dataset import masks_from_split_manifest

        with tempfile.TemporaryDirectory() as tmp_string:
            tmp = Path(tmp_string)
            target = xr.DataArray(
                np.zeros((2, 2, 2)), dims=("time", "latitude", "longitude"),
                coords={"time": ["1982", "1983"], "latitude": [-90, -89.5], "longitude": [-180, -179.5]},
            )
            np.savez_compressed(tmp / "split_manifest.npz", train_indices=[0, 7], test_indices=[1, 6])
            (tmp / "split_manifest.json").write_text(json.dumps({
                "time_size": 2, "latitude_size": 2, "longitude_size": 2,
                "seed": 42, "sha256": _sha(tmp / "split_manifest.npz"),
            }))
            train, test, metadata = masks_from_split_manifest(target, tmp / "split_manifest.npz")
            self.assertEqual(int(train.sum()), 2)
            self.assertEqual(int(test.sum()), 2)
            self.assertFalse(bool((train & test).any()))
            self.assertFalse(metadata["recomputed"])

    def test_effective_config_and_metric_tolerances(self) -> None:
        effective = build_effective_config({
            "raw_dir": Path("/private/data"), "tracking_password": "secret",
            "seed": 42, "reproducibility": {},
        })
        self.assertEqual(effective["raw_dir"], "${WORK_DIR}/inputs")
        self.assertEqual(effective["tracking_password"], "***REDACTED***")
        expected = build_expected_results(
            {"dataset": {"n_train": 8, "n_test": 2}, "metrics": {"rmse": 0.5}}, None, 0.01
        )
        passed = compare_results(expected, {"dataset": {"n_train": 8, "n_test": 2}, "metrics": {"rmse": 0.505}})
        failed = compare_results(expected, {"dataset": {"n_train": 8, "n_test": 2}, "metrics": {"rmse": 0.6}})
        self.assertEqual(passed["reproduction_level"], "numerically reproduced")
        self.assertEqual(failed["reproduction_level"], "not reproduced")

    def test_effective_config_is_restored_for_the_reproduction_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_string:
            tmp = Path(tmp_string)
            crate = tmp / "crate"
            (crate / "config").mkdir(parents=True)
            effective_path = crate / "config" / "effective_config.json"
            effective_path.write_text(json.dumps({
                "effective_config_format": "profecia-reproducibility-1",
                "work_dir_variable": "WORK_DIR",
                "raw_dir": "${WORK_DIR}/inputs",
                "model_dir": "${WORK_DIR}/outputs/model",
                "data": {"raw_dir": "${WORK_DIR}/inputs"},
                "reproducibility": {
                    "input_catalogue": "config/data.toml",
                    "upstream": {"metadata_url": "https://example.org/ro-crate-metadata.json"},
                },
            }), encoding="utf-8")
            cfg = resolve_effective_train_config(effective_path, tmp / "reproduction")
            self.assertEqual(cfg["raw_dir"], (tmp / "reproduction" / "inputs").resolve())
            self.assertEqual(cfg["data"]["raw_dir"], cfg["raw_dir"])
            self.assertEqual(cfg["reproducibility"].input_catalogue, crate / "config" / "data.toml")

    def test_offline_verify_only_and_reproduced_crate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_string:
            tmp = Path(tmp_string)
            crate, work = tmp / "crate", tmp / "work"
            for directory in (crate / "config", crate / "fair", crate / "environment"):
                directory.mkdir(parents=True)
            upstream_data = tmp / "tiny.npy"
            np.save(upstream_data, np.arange(2))
            data_sha = _sha(upstream_data)
            upstream_metadata = crate / "upstream.json"
            upstream_metadata.write_text(json.dumps({"@graph": [{
                "@id": "tiny.npy", "@type": "File", "name": "tiny.npy",
                "contentUrl": "https://example.invalid/tiny.npy", "sha256": data_sha,
            }]}))
            metadata_sha = _sha(upstream_metadata)
            (crate / "config" / "data.toml").write_text('[inputs]\nsource = "tiny.npy"\n')
            (crate / "config" / "effective_config.json").write_text(json.dumps({
                "reproducibility": {
                    "upstream": {"identifier": "https://doi.org/10.example/upstream", "metadata_url": str(upstream_metadata)},
                    "inputs": {"require_upstream_checksum": True},
                    "environment": {},
                }
            }))
            (crate / "environment" / "python-packages.json").write_text('{"packages":{}}')
            np.savez_compressed(crate / "fair" / "split_manifest.npz", train_indices=[0], test_indices=[1])
            split_sha = _sha(crate / "fair" / "split_manifest.npz")
            (crate / "fair" / "split_manifest.json").write_text(json.dumps({"sha256": split_sha}))
            original_id = "urn:uuid:original"
            (crate / "ro-crate-metadata.json").write_text(json.dumps({"@graph": [
                {"@id": "./", "identifier": original_id},
                {"@id": f"{original_id}#upstream-preprocessing-crate", "sha256": metadata_sha},
            ]}))
            cache = work / "cache" / "upstream" / "files"
            cache.mkdir(parents=True)
            (cache / f"{data_sha[:16]}-tiny.npy").write_bytes(upstream_data.read_bytes())
            report = reproduce(crate, work, offline=True, verify_only=True, local_environment=True)
            self.assertTrue(report["passed"])
            self.assertTrue((work / "reproduction_report.md").is_file())
            report.update({"passed_checks": ["all"], "failed_checks": [], "metric_checks": []})
            reproduced = _write_reproduced_crate(
                crate, work, report, "https://doi.org/10.example/upstream"
            )
            graph = json.loads((reproduced / "ro-crate-metadata.json").read_text())["@graph"]
            root = next(item for item in graph if item["@id"] == "./")
            sources = {item["@id"] for item in root["prov:wasDerivedFrom"]}
            self.assertEqual(sources, {original_id, "https://doi.org/10.example/upstream"})


def _resolved(key: str, filename: str, url: str, checksum: str) -> ResolvedInput:
    return ResolvedInput(
        InputSpec(key, "variable", filename, netcdf_variable=None),
        {"@id": url, "contentUrl": url, "sha256": checksum},
        "explicit_upstream_entity_id", None, None, None, "not_available",
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    unittest.main()
