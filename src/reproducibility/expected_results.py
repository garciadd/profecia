from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def build_expected_results(
    run: dict[str, Any], split_manifest: dict[str, Any] | None, absolute_tolerance: float = 0.0001
) -> dict[str, Any]:
    metrics = {
        name: {"expected": value, "absolute_tolerance": absolute_tolerance}
        for name, value in (run.get("metrics") or {}).items()
        if isinstance(value, (int, float)) and math.isfinite(float(value)) and name != "n_samples"
    }
    return {
        "format": "profecia-expected-results-1",
        "dataset": {
            "n_train": run.get("dataset", {}).get("n_train"),
            "n_test": run.get("dataset", {}).get("n_test"),
            "split_manifest_sha256": (
                split_manifest["metadata"]["sha256"] if split_manifest else None
            ),
        },
        "metrics": metrics,
        "reproduction_levels": [
            "exactly reproduced", "numerically reproduced",
            "functionally reproduced", "not reproduced",
        ],
    }


def write_expected_results(
    crate_dir: Path, run: dict[str, Any], split_manifest: dict[str, Any] | None,
    absolute_tolerance: float = 0.0001,
) -> Path:
    path = crate_dir / "reproducibility" / "expected_results.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(build_expected_results(run, split_manifest, absolute_tolerance), indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def compare_results(expected: dict[str, Any], actual: dict[str, Any]) -> dict[str, Any]:
    checks = []
    for key in ("n_train", "n_test"):
        wanted = expected.get("dataset", {}).get(key)
        got = actual.get("dataset", {}).get(key)
        checks.append({"name": key, "passed": wanted == got, "expected": wanted, "actual": got})
    metric_checks = []
    for name, specification in expected.get("metrics", {}).items():
        got = actual.get("metrics", {}).get(name)
        wanted = specification["expected"]
        tolerance = specification.get("absolute_tolerance", 0.0)
        difference = None if got is None else abs(float(got) - float(wanted))
        metric_checks.append({
            "name": name, "passed": difference is not None and difference <= tolerance,
            "expected": wanted, "actual": got, "absolute_difference": difference,
            "absolute_tolerance": tolerance,
        })
    passed = all(item["passed"] for item in [*checks, *metric_checks])
    exact = passed and expected.get("artifacts_sha256") == actual.get("artifacts_sha256") and bool(expected.get("artifacts_sha256"))
    return {
        "reproduction_level": "exactly reproduced" if exact else ("numerically reproduced" if passed else "not reproduced"),
        "passed": passed,
        "checks": checks,
        "metric_checks": metric_checks,
    }
