import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.preprocessing import RobustScaler, StandardScaler

from src.models.tabular_models import build_model, get_model_name


# SCALING

def build_scaler(scaler_name: str | None):
    """
    Crea un scaler opcional.

    scaler_name:
    - None / "none"   -> sin escalado
    - "standard"      -> StandardScaler
    - "robust"        -> RobustScaler
    """
    if scaler_name is None:
        return None

    scaler_name = scaler_name.lower().strip()

    if scaler_name == "none":
        return None
    if scaler_name == "standard":
        return StandardScaler()
    if scaler_name == "robust":
        return RobustScaler()

    raise ValueError(
        f"Scaler no soportado: '{scaler_name}'. "
        "Usa uno de: None, 'none', 'standard', 'robust'."
    )


# DATA LOADING

def load_train_arrays(
    input_dir: str | Path,
    mmap_mode: str | None = "r",
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """
    Carga X_train, y_train y dataset_metadata desde un directorio tabular exportado.
    """
    input_dir = Path(input_dir)

    X_path = input_dir / "X_train.npy"
    y_path = input_dir / "y_train.npy"
    meta_path = input_dir / "dataset_metadata.json"

    if not X_path.exists():
        raise FileNotFoundError(f"No existe: {X_path}")
    if not y_path.exists():
        raise FileNotFoundError(f"No existe: {y_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"No existe: {meta_path}")

    X_train = np.load(X_path, mmap_mode=mmap_mode)
    y_train = np.load(y_path, mmap_mode=mmap_mode)

    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    if X_train.ndim != 2:
        raise ValueError(f"X_train debe ser 2D. Recibido: {X_train.shape}")

    if y_train.ndim != 1:
        y_train = np.asarray(y_train).reshape(-1)

    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError("X_train e y_train no tienen el mismo número de filas.")

    return X_train, y_train, metadata


# TRAIN

def train_tabular_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_name: str = "rf",
    scaler_name: str | None = None,
    random_state: int = 42,
    **model_kwargs,
) -> dict[str, Any]:
    """
    Entrena un modelo tabular opcionalmente con escalado.

    Returns
    -------
    dict con:
    - model
    - scaler
    - X_train_used
    - train_info
    """
    if X_train.ndim != 2:
        raise ValueError(f"X_train debe ser 2D. Recibido: {X_train.shape}")

    y_train = np.asarray(y_train).reshape(-1)

    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError("X_train e y_train no tienen el mismo número de filas.")

    scaler = build_scaler(scaler_name)

    if scaler is not None:
        X_train_used = scaler.fit_transform(X_train)
        X_train_used = X_train_used.astype(np.float32, copy=False)
    else:
        X_train_used = X_train

    model = build_model(
        model_name=model_name,
        random_state=random_state,
        **model_kwargs,
    )

    model.fit(X_train_used, y_train)

    train_info = {
        "model_name_requested": model_name,
        "model_name": get_model_name(model),
        "model_params": model.get_params(),
        "scaler_name": scaler_name if scaler_name is not None else "none",
        "random_state": int(random_state),
        "n_rows_train": int(X_train.shape[0]),
        "n_features": int(X_train.shape[1]),
        "X_dtype": str(X_train.dtype),
        "y_dtype": str(y_train.dtype),
    }

    return {
        "model": model,
        "scaler": scaler,
        "X_train_used": X_train_used,
        "train_info": train_info,
    }


# SAVE / LOAD

def save_trained_pipeline(
    output_dir: str | Path,
    model,
    scaler,
    train_info: dict[str, Any],
    dataset_metadata: dict[str, Any],
    prefix: str,
) -> dict[str, str | None]:
    """
    Guarda modelo, scaler y metadata de entrenamiento.

    Guarda:
    - {prefix}_model.joblib
    - {prefix}_scaler.joblib (si aplica)
    - {prefix}_train_info.json
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / f"{prefix}_model.joblib"
    scaler_path = output_dir / f"{prefix}_scaler.joblib"
    train_info_path = output_dir / f"{prefix}_train_info.json"

    joblib.dump(model, model_path)

    if scaler is not None:
        joblib.dump(scaler, scaler_path)
        scaler_path_str = str(scaler_path)
    else:
        scaler_path_str = None

    payload = {
        # info de entrenamiento
        "prefix": prefix,
        "model_name_requested": train_info.get("model_name_requested"),
        "model_name": train_info.get("model_name"),
        "model_params": train_info.get("model_params"),
        "scaler_name": train_info.get("scaler_name"),
        "random_state": train_info.get("random_state"),
        "n_rows_train": train_info.get("n_rows_train"),
        "n_features": train_info.get("n_features"),
        "X_dtype": train_info.get("X_dtype"),
        "y_dtype": train_info.get("y_dtype"),

        # info del dataset tabular
        "feature_names": dataset_metadata.get("feature_names"),
        "target": dataset_metadata.get("target"),
        "dataset_n_train": dataset_metadata.get("n_train"),
        "dataset_n_test": dataset_metadata.get("n_test"),
        "dataset_prefix": dataset_metadata.get("prefix"),
        "temporal_resolution_inferred": dataset_metadata.get("temporal_resolution_inferred"),
        "time_start": dataset_metadata.get("time_start"),
        "time_end": dataset_metadata.get("time_end"),
        "n_time": dataset_metadata.get("n_time"),
        "latitude_size": dataset_metadata.get("latitude_size"),
        "longitude_size": dataset_metadata.get("longitude_size"),

        # rutas
        "model_path": str(model_path),
        "scaler_path": scaler_path_str,
    }

    with open(train_info_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return {
        "model_path": str(model_path),
        "scaler_path": scaler_path_str,
        "train_info_path": str(train_info_path),
    }


def load_trained_pipeline(
    model_path: str | Path,
    scaler_path: str | Path | None = None,
):
    """
    Carga modelo y scaler.
    """
    model = joblib.load(model_path)

    scaler = None
    if scaler_path is not None:
        scaler = joblib.load(scaler_path)

    return model, scaler