from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .expected_results import compare_results


def verify_reproduction(original_crate: Path, reproduced_run: Path) -> dict[str, Any]:
    expected_path = original_crate / "reproducibility" / "expected_results.json"
    if not expected_path.is_file():
        raise ValueError(f"fatal error: expected results are missing: {expected_path}")
    expected = json.loads(expected_path.read_text(encoding="utf-8"))
    actual = _load_actual_results(reproduced_run)
    comparison = compare_results(expected, actual)
    metadata = _read_json(original_crate / "ro-crate-metadata.json")
    graph = metadata.get("@graph", []) if metadata else []
    upstream = next(
        (item for item in graph if str(item.get("@id", "")).endswith("#upstream-preprocessing-crate")), None
    )
    source_code = next((item for item in graph if str(item.get("@id", "")).endswith("#source-code")), {})
    commit = str(source_code.get("version") or source_code.get("commitHash") or "")
    split_ok, split_message = _verify_split(original_crate, expected)
    inputs_ok, inputs_message = _verify_inputs(original_crate, reproduced_run)
    jsonld_ok = all(
        _read_json(original_crate / relative) is not None
        for relative in ("ro-crate-metadata.json", "fair/dataset.croissant.json", "fair/model.fair4ml.json")
    )
    environment_ok = (
        (original_crate / "environment" / "python-packages.json").is_file()
        or any((original_crate / "environment" / "lock").glob("*"))
        or any(str(item.get("@id", "")).startswith(("docker://", "oci:")) for item in graph)
    )
    checks = [
        {"name": "original_rocrate", "passed": bool(metadata)},
        {"name": "upstream_release", "passed": bool(upstream and upstream.get("identifier"))},
        {"name": "input_checksums", "passed": inputs_ok, "message": inputs_message},
        {"name": "effective_config", "passed": _read_json(original_crate / "config" / "effective_config.json") is not None},
        {"name": "git_commit", "passed": len(commit) == 40, "actual": commit or None},
        {"name": "environment", "passed": environment_ok},
        {"name": "split_manifest", "passed": split_ok, "message": split_message},
        {"name": "jsonld_descriptors", "passed": jsonld_ok},
        *comparison["checks"], *comparison["metric_checks"],
    ]
    comparison["passed_checks"] = [item["name"] for item in checks if item["passed"]]
    comparison["failed_checks"] = [item["name"] for item in checks if not item["passed"]]
    comparison["checks"] = checks
    comparison["passed"] = not comparison["failed_checks"]
    if not comparison["passed"]:
        comparison["reproduction_level"] = "not reproduced"
    return comparison


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def _verify_split(crate: Path, expected: dict[str, Any]) -> tuple[bool, str]:
    npz = crate / "fair" / "split_manifest.npz"
    metadata = _read_json(crate / "fair" / "split_manifest.json")
    if not npz.is_file() or not metadata:
        return False, "split manifest is missing"
    actual = hashlib.sha256(npz.read_bytes()).hexdigest()
    declared = metadata.get("sha256")
    wanted = expected.get("dataset", {}).get("split_manifest_sha256")
    passed = bool(declared and actual == declared and (not wanted or actual == wanted))
    return passed, "verified" if passed else "split manifest checksum mismatch"


def _verify_inputs(crate: Path, reproduced: Path) -> tuple[bool, str]:
    original = _read_json(crate / "reproducibility" / "input_resolution.json")
    runtime = _read_json(reproduced / "input_resolution.json")
    if not original:
        return False, "original input resolution is missing"
    entries = original.get("inputs") or []
    unresolved = [item.get("key") for item in entries if not item.get("upstream_entity_id")]
    undocumented = [
        item.get("key") for item in entries
        if item.get("checksum_status") == "mismatch" and not item.get("derived")
    ]
    if unresolved or undocumented:
        return False, "unresolved or undocumented inputs: " + ", ".join(map(str, unresolved + undocumented))
    if runtime:
        failed = [
            item.get("key") for item in runtime.get("inputs", [])
            if item.get("expected_sha256") and not item.get("checksum_verified")
        ]
        if failed:
            return False, "runtime checksum failures: " + ", ".join(map(str, failed))
    return True, "verified"


def write_reproduction_report(output_dir: Path, report: dict[str, Any]) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "reproduction_report.json"
    md_path = output_dir / "reproduction_report.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# PROFECIA reproduction report", "",
        f"- Result: **{report.get('reproduction_level', 'not reproduced')}**",
        f"- Passed: `{bool(report.get('passed'))}`", "", "## Checks", "",
    ]
    for check in report.get("checks", []):
        lines.append(f"- {'PASS' if check.get('passed') else 'FAIL'} — {check.get('name')}")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


def _load_actual_results(path: Path) -> dict[str, Any]:
    report = path / "reproduction_report.json"
    if report.is_file():
        value = json.loads(report.read_text(encoding="utf-8"))
        if "actual" in value:
            return value["actual"]
    summaries = sorted(path.rglob("workflow_step_by_step_summary.json"))
    if not summaries:
        raise ValueError(f"fatal error: no reproduced workflow summary found under {path}")
    summary = json.loads(summaries[-1].read_text(encoding="utf-8"))
    dataset = summary.get("dataset_metadata") or {}
    return {
        "dataset": {"n_train": dataset.get("n_train"), "n_test": dataset.get("n_test")},
        "metrics": (summary.get("evaluation") or {}).get("global_metrics") or {},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify a PROFECIA reproduction")
    parser.add_argument("--original-crate", required=True, type=Path)
    parser.add_argument("--reproduced-run", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        report = verify_reproduction(args.original_crate.resolve(), args.reproduced_run.resolve())
        write_reproduction_report(args.reproduced_run, report)
        return 0 if report["passed"] else 1
    except Exception as exc:
        write_reproduction_report(args.reproduced_run, {
            "passed": False, "reproduction_level": "not reproduced",
            "checks": [{"name": "fatal_error", "passed": False, "message": str(exc)}],
        })
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
