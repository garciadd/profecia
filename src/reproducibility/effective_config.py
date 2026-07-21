from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path
from typing import Any


SECRET_KEY = re.compile(r"password|passwd|token|secret|api_key", re.IGNORECASE)
PORTABLE_PATHS = {
    "raw_dir": "${WORK_DIR}/inputs",
    "mask_dir": "${WORK_DIR}/derived-inputs/masks",
    "processed_dir": "${WORK_DIR}/processed",
    "model_data_dir": "${WORK_DIR}/model-data",
    "model_dir": "${WORK_DIR}/outputs",
    "model_artifacts_dir": "${WORK_DIR}/outputs/artifacts",
    "model_figures_dir": "${WORK_DIR}/outputs/figures",
}


def build_effective_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Return the resolved, redacted and workspace-portable training configuration."""
    payload = _jsonable(cfg)
    payload["config_path"] = "config/train.toml"
    payload["data_config_path"] = "config/data.toml"
    for key, value in PORTABLE_PATHS.items():
        if key in payload:
            payload[key] = value
    data = payload.get("data")
    if isinstance(data, dict):
        for key in ("raw_dir", "mask_dir", "output_dir", "processed_base_dir"):
            if key in data:
                data[key] = PORTABLE_PATHS.get(key, "${WORK_DIR}/" + key.replace("_dir", ""))
    reproducibility = payload.get("reproducibility")
    if isinstance(reproducibility, dict):
        reproducibility["output_dir"] = "${WORK_DIR}/outputs/ro-crates"
        reproducibility["input_catalogue"] = "config/data.toml"
        upstream = reproducibility.get("upstream")
        if isinstance(upstream, dict):
            upstream["cache_dir"] = "${WORK_DIR}/cache/upstream"
            if upstream.get("metadata_url"):
                # The legacy field can be a machine-local packaging cache. The
                # canonical nested URL is the portable reproduction source.
                reproducibility["upstream_rocrate"] = None
        environment = reproducibility.get("environment")
        if isinstance(environment, dict) and environment.get("lock_file"):
            environment["lock_file"] = "environment/lock/" + Path(str(environment["lock_file"])).name
    payload["effective_config_format"] = "profecia-reproducibility-1"
    payload["work_dir_variable"] = "WORK_DIR"
    return payload


def write_effective_config(crate_dir: Path, cfg: dict[str, Any]) -> Path:
    path = crate_dir / "config" / "effective_config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build_effective_config(cfg), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _jsonable(value: Any, key: str = "") -> Any:
    if SECRET_KEY.search(key):
        return "***REDACTED***"
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value), key)
    if isinstance(value, dict):
        return {str(k): _jsonable(v, str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item, key) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
