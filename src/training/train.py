import json
import logging
from pathlib import Path
from typing import Any

import joblib
import os
import numpy as np
from sklearn.preprocessing import RobustScaler, StandardScaler

from src.models.tabular_models import build_model, get_model_name


LOGGER = logging.getLogger(__name__)


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
    return_trace: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]] | tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    """
    Carga X_train, y_train y dataset_metadata desde un directorio tabular exportado.
    Si return_trace=True, tambien carga las trazas por observacion de train.
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

    if not return_trace:
        return X_train, y_train, metadata

    trace = {
        "pixel_id_train": np.load(input_dir / "pixel_id_train.npy", mmap_mode=mmap_mode),
        "lat_idx_train": np.load(input_dir / "lat_idx_train.npy", mmap_mode=mmap_mode),
        "lon_idx_train": np.load(input_dir / "lon_idx_train.npy", mmap_mode=mmap_mode),
        "time_idx_train": np.load(input_dir / "time_idx_train.npy", mmap_mode=mmap_mode),
    }

    for name, values in trace.items():
        if values.shape[0] != X_train.shape[0]:
            raise ValueError(f"{name} no tiene el mismo numero de filas que X_train.")

    return X_train, y_train, trace, metadata


def load_test_arrays(
    input_dir: str | Path,
    mmap_mode: str | None = "r",
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    input_dir = Path(input_dir)

    X_test = np.load(input_dir / "X_test.npy", mmap_mode=mmap_mode)
    y_test = np.load(input_dir / "y_test.npy", mmap_mode=mmap_mode)
    trace = {
        "pixel_id_test": np.load(input_dir / "pixel_id_test.npy", mmap_mode=mmap_mode),
        "lat_idx_test": np.load(input_dir / "lat_idx_test.npy", mmap_mode=mmap_mode),
        "lon_idx_test": np.load(input_dir / "lon_idx_test.npy", mmap_mode=mmap_mode),
        "time_idx_test": np.load(input_dir / "time_idx_test.npy", mmap_mode=mmap_mode),
    }

    with open(input_dir / "dataset_metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return X_test, y_test, trace, metadata


# TRAIN

def train_tabular_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_name: str = "rf",
    scaler_name: str | None = None,
    random_state: int = 42,
    mlflow_config: dict | None = None,
    log_shap: bool = True,
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

    # Keep operational/config kwargs from leaking into sklearn estimators.
    for reserved_key in ("mlflow_config", "log_shap"):
        model_kwargs.pop(reserved_key, None)

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

    # --- MLflow logging (optional) ---------------------------------
    mlflow_run_id = None
    mlflow_experiment = None
    if mlflow_config is not None:
        LOGGER.info(
            "MLflow logging enabled for model=%s experiment=%s tracking_uri=%s log_shap=%s",
            model_name,
            mlflow_config.get("experiment_name", "default"),
            mlflow_config.get("tracking_uri"),
            log_shap,
        )
        try:
            import mlflow
            import mlflow.sklearn
            # auth via env vars if provided
            tracking_uri = mlflow_config.get("tracking_uri")
            username = mlflow_config.get("user")
            password = mlflow_config.get("password")
            experiment_name = mlflow_config.get("experiment_name", "default")
            run_name = mlflow_config.get("run_name")

            if username:
                os.environ.setdefault("MLFLOW_TRACKING_USERNAME", str(username))
            if password:
                os.environ.setdefault("MLFLOW_TRACKING_PASSWORD", str(password))
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)

            mlflow.set_experiment(experiment_name)
            mlflow_experiment = experiment_name
            LOGGER.info(
                "MLflow configured with experiment=%s run_name=%s",
                experiment_name,
                run_name,
            )

            with mlflow.start_run(run_name=run_name) as run:
                mlflow_run_id = run.info.run_id
                LOGGER.info("Started MLflow run: %s", mlflow_run_id)
                # log params
                mlflow.log_param("model_name_requested", model_name)
                mlflow.log_param("scaler_name", scaler_name if scaler_name is not None else "none")
                mlflow.log_param("random_state", int(random_state))
                mlflow.log_param("n_rows_train", int(X_train.shape[0]))
                mlflow.log_param("n_features", int(X_train.shape[1]))
                
                LOGGER.info(
                    "Logging MLflow params for model=%s rows=%s features=%s",
                    model_name,
                    int(X_train.shape[0]),
                    int(X_train.shape[1]),
                )
                # model params
                try:
                    for k, v in model.get_params().items():
                        mlflow.log_param(f"model_param_{k}", str(v))
                    LOGGER.info("Logged %s model hyperparameters to MLflow", len(model.get_params()))
                except Exception:
                    LOGGER.exception("Failed to log model hyperparameters to MLflow")

                # predictions on train for metrics
                try:
                    y_pred = model.predict(X_train_used)
                    LOGGER.info("Computed train predictions for MLflow metrics")
                except Exception:
                    y_pred = None
                    LOGGER.exception("Failed to compute train predictions for MLflow metrics")

                # determine task type
                is_classification = False
                try:
                    uniq = np.unique(y_train)
                    if np.issubdtype(y_train.dtype, np.integer) and uniq.size <= 20:
                        is_classification = True
                    LOGGER.info(
                        "Detected task type for MLflow metrics: %s",
                        "classification" if is_classification else "regression",
                    )
                except Exception:
                    is_classification = False
                    LOGGER.exception("Failed to infer task type; defaulting to regression metrics")

                # compute metrics
                if y_pred is not None:
                    try:
                        if is_classification:
                            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                            acc = float(accuracy_score(y_train, y_pred))
                            prec = float(precision_score(y_train, y_pred, average="weighted", zero_division=0))
                            rec = float(recall_score(y_train, y_pred, average="weighted", zero_division=0))
                            f1 = float(f1_score(y_train, y_pred, average="weighted", zero_division=0))
                            mlflow.log_metric("train_accuracy", acc)
                            mlflow.log_metric("train_precision_weighted", prec)
                            mlflow.log_metric("train_recall_weighted", rec)
                            mlflow.log_metric("train_f1_weighted", f1)
                            LOGGER.info(
                                "Logged classification metrics to MLflow: accuracy=%.6f f1=%.6f",
                                acc,
                                f1,
                            )
                        else:
                            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                            mse = float(mean_squared_error(y_train, y_pred))
                            rmse = float(np.sqrt(mse))
                            mae = float(mean_absolute_error(y_train, y_pred))
                            r2 = float(r2_score(y_train, y_pred))
                            mlflow.log_metric("train_mse", mse)
                            mlflow.log_metric("train_rmse", rmse)
                            mlflow.log_metric("train_mae", mae)
                            mlflow.log_metric("train_r2", r2)
                            LOGGER.info(
                                "Logged regression metrics to MLflow: rmse=%.6f mae=%.6f r2=%.6f",
                                rmse,
                                mae,
                                r2,
                            )
                    except Exception:
                        LOGGER.exception("Failed to compute or log train metrics to MLflow")
                else:
                    LOGGER.warning("Skipping MLflow metric logging because train predictions are unavailable")

                # log model and scaler as artifacts
                try:
                    mlflow.sklearn.log_model(model, name="model")
                    LOGGER.info("Logged sklearn model artifact to MLflow")
                except Exception:
                    LOGGER.warning("mlflow.sklearn.log_model failed for model; trying joblib artifact fallback")
                    try:
                        # fallback: save joblib and log as artifact
                        #tmp_model = Path("/tmp") / f"model_{mlflow_run_id}.joblib"
                        #joblib.dump(model, tmp_model)
                        #mlflow.log_artifact(str(tmp_model))
                        LOGGER.info("SKIPPED: Logged model artifact via joblib fallback: %s", tmp_model)
                    except Exception:
                        LOGGER.exception("Failed to log model artifact to MLflow, including fallback")

                if scaler is not None:
                    try:
                        mlflow.sklearn.log_model(scaler, name="scaler")
                        LOGGER.info("Logged scaler artifact to MLflow")
                    except Exception:
                        LOGGER.warning("mlflow.sklearn.log_model failed for scaler; trying joblib artifact fallback")
                        try:
                            #tmp_scaler = Path("/tmp") / f"scaler_{mlflow_run_id}.joblib"
                            #joblib.dump(scaler, tmp_scaler)
                            #mlflow.log_artifact(str(tmp_scaler))
                            LOGGER.info("SKIPPED: Logged scaler artifact via joblib fallback: %s", tmp_scaler)
                        except Exception:
                            LOGGER.exception("Failed to log scaler artifact to MLflow, including fallback")
                else:
                    LOGGER.info("No scaler configured; skipping scaler artifact logging")

                # SHAP explanations (optional)
                if log_shap:
                    LOGGER.info("SHAP logging enabled; attempting explainer generation")
                    try:
                        import shap
                        import matplotlib.pyplot as plt
                        # use fast explainer when possible
                        try:
                            expl = shap.Explainer(model, X_train_used)
                            shap_values = expl(X_train_used)
                            LOGGER.info("Computed SHAP values with shap.Explainer")
                        except Exception:
                            LOGGER.warning("shap.Explainer failed; trying TreeExplainer fallback")
                            # fallback to TreeExplainer or KernelExplainer
                            try:
                                expl = shap.TreeExplainer(model)
                                shap_values = expl.shap_values(X_train_used)
                                LOGGER.info("Computed SHAP values with shap.TreeExplainer")
                            except Exception:
                                shap_values = None
                                LOGGER.exception("Failed to compute SHAP values with available explainers")

                        if shap_values is not None:
                            # compute mean absolute importance per feature
                            try:
                                import numpy as _np
                                if hasattr(shap_values, "values"):
                                    vals = shap_values.values
                                else:
                                    vals = shap_values
                                # handle multilevel output
                                if isinstance(vals, list):
                                    vals_arr = _np.mean(_np.abs(_np.vstack([_np.asarray(v) for v in vals])), axis=0)
                                else:
                                    vals_arr = _np.mean(_np.abs(vals), axis=0)

                                feat_names = [f"f{i}" for i in range(vals_arr.shape[-1])]
                                # bar plot
                                fig, ax = plt.subplots(figsize=(8, 4))
                                ax.bar(range(len(vals_arr)), vals_arr)
                                ax.set_xticks(range(len(vals_arr)))
                                ax.set_xticklabels(feat_names, rotation=90)
                                ax.set_ylabel("mean |SHAP value|")
                                ax.set_title("Feature importance (SHAP)")
                                shap_path = Path("/tmp") / f"shap_importance_{mlflow_run_id}.png"
                                fig.tight_layout()
                                fig.savefig(shap_path, dpi=150, bbox_inches='tight')
                                plt.close(fig)
                                mlflow.log_artifact(str(shap_path))
                                LOGGER.info("Logged SHAP importance plot to MLflow: %s", shap_path)
                            except Exception:
                                LOGGER.exception("Failed to create or log SHAP importance plot")
                        else:
                            LOGGER.warning("Skipping SHAP artifact logging because no SHAP values were computed")
                    except Exception:
                        LOGGER.exception("Failed during SHAP logging setup or execution")
                else:
                    LOGGER.info("SHAP logging disabled for this run")

                LOGGER.info("Completed MLflow logging for run %s", mlflow_run_id)

        except Exception:
            # if MLflow not installed or any error, continue silently
            mlflow_run_id = None
            mlflow_experiment = None
            LOGGER.exception("MLflow logging failed; continuing without MLflow artifacts")
    else:
        LOGGER.info("MLflow logging disabled because mlflow_config is None")

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
        "mlflow_run_id": mlflow_run_id,
        "mlflow_experiment": mlflow_experiment,
    }

    return {
        "model": model,
        "scaler": scaler,
        "X_train_used": X_train_used,
        "train_info": train_info,
        "mlflow_run_id": mlflow_run_id,
        "mlflow_experiment": mlflow_experiment,
    }


# SAVE / LOAD

def save_trained_pipeline(
    output_dir: str | Path,
    model,
    scaler,
    train_info: dict[str, Any],
    dataset_metadata: dict[str, Any],
    prefix: str | None = None,
) -> dict[str, str | None]:
    """
    Guarda modelo, scaler y metadata de entrenamiento.

    Por defecto usa la convencion nueva:
    - model.joblib
    - scaler.joblib (si aplica)
    - train_info.json

    Si se pasa prefix, conserva compatibilidad con artefactos antiguos:
    - {prefix}_model.joblib
    - {prefix}_scaler.joblib
    - {prefix}_train_info.json
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if prefix:
        model_path = output_dir / f"{prefix}_model.joblib"
        scaler_path = output_dir / f"{prefix}_scaler.joblib"
        train_info_path = output_dir / f"{prefix}_train_info.json"
    else:
        model_path = output_dir / "model.joblib"
        scaler_path = output_dir / "scaler.joblib"
        train_info_path = output_dir / "train_info.json"

    joblib.dump(model, model_path)

    if scaler is not None:
        joblib.dump(scaler, scaler_path)
        scaler_path_str = str(scaler_path)
    else:
        scaler_path_str = None

    payload = {
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
        "hyperparameter_search": train_info.get("hyperparameter_search"),
        "mlflow_run_id": train_info.get("mlflow_run_id"),
        "mlflow_experiment": train_info.get("mlflow_experiment"),

        # info del dataset tabular
        "feature_names": dataset_metadata.get("feature_names"),
        "target": dataset_metadata.get("target"),
        "dataset_n_train": dataset_metadata.get("n_train"),
        "dataset_n_test": dataset_metadata.get("n_test"),
        "split_mode": dataset_metadata.get("split_mode"),
        "split_metadata": dataset_metadata.get("split_metadata"),
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
