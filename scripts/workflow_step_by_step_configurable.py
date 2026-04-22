#!/usr/bin/env python3

"""
Versión script del notebook `notebooks/workflow_step_by_step.ipynb`.

La intención de este fichero es que puedas leerlo igual que el notebook:
1. Configuración editable.
2. Preprocesado y guardado de variables.
3. Carga de procesados y split espacial.
4. Exportación tabular.
5. Entrenamiento de varios modelos.
6. Evaluación y logging opcional en MLflow.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from pprint import pprint

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr


# ---------------------------------------------------------------------------
# Paso 0. Imports locales del repo
# ---------------------------------------------------------------------------

# Este bloque replica la primera celda del notebook: permite ejecutar el script
# desde la raíz del repo o desde cualquier otro directorio.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data import io
from src.data.eda import load_processed_dataset
from src.data.io import load_mask
from src.evaluation.regression import (
    build_pixel_timeseries_dataframe,
    build_prediction_dataframe,
    compute_global_regression_metrics,
    compute_pixel_regression_metrics,
    rank_best_and_worst_pixels,
    summarize_pixel_metrics,
)
from src.split.export_tabular import export_train_test_split
from src.split.spatial import make_stratified_spatial_split, save_spatial_split
from src.training.train import save_trained_pipeline, train_tabular_model


# ---------------------------------------------------------------------------
# Paso 1. Configuración editable
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelConfig:
    """Un modelo concreto que se entrenará sobre el mismo split."""

    run_name: str
    model_name: str
    params: dict
    scaler_name: str | None = None
    random_state: int = 42
    log_shap: bool = False


@dataclass(frozen=True)
class WorkflowConfig:
    """Todos los parámetros que en el notebook estaban repartidos por celdas."""

    experiment_name: str = "Test_automatic_1"
    raw_dir: Path = Path("/srv/read_only_data/final_datasets/monthly")
    mask_dir: Path = Path("/srv/volumes/ferag/masks")
    workflow_root: Path = Path("/srv/volumes/ferag/worflow_runs")

    variable_names: list[str] = field(
        default_factory=lambda: ["LAI", "SM1", "SM2", "TP", "T2M", "SSRD", "VPD"]
    )
    target_name: str = "LAI"
    predictor_names: list[str] = field(
        default_factory=lambda: ["SM1", "SM2", "TP", "T2M", "SSRD", "VPD"]
    )

    temporal_resolution: str = "monthly"
    start_year: int = 2000
    end_year_inclusive: int = 2022
    dtype: str = "float32"
    roi: io.ROI | None = None
    #roi: io.ROI | None = io.ROI(
    #    lat_min=36.0,
    #    lat_max=44.0,
    #    lon_min=-10.0,
    #    lon_max=3.5,
    #) #Península ibérica

    # Estas son las máscaras de filtrado del notebook. En `io.build_combined_filter_mask`
    # `land` se usa como tierra, mientras `ebf` y `bs` se invierten para excluir esas zonas.
    mask_names: list[str] = field(default_factory=lambda: ["land", "ebf"])

    # Estas máscaras se añaden como predictores one-hot: una columna por clase.
    # Ejemplo: landcover -> landcover__forest, landcover__cropland, etc.
    predictor_masks: list[str] = field(default_factory=lambda: ["landcover"])
    #predictor_masks = []
    predictor_mask_ignore_codes: tuple[int, ...] = (0,)

    # Split espacial estratificado, igual que en la sección de split del notebook.
    split_mask_name: str = "landcover"
    pixel_fraction: float = 0.30
    test_fraction: float = 0.10
    min_valid_fraction: float = 0.0
    seed: int = 42
    subset_seed: int | None = 42
    ignore_codes: tuple[int, ...] = (0,)

    save_output: bool = True
    mmap_mode: str | None = "r"

    # MLflow se configura por variables de entorno para evitar credenciales hardcodeadas.
    mlflow_enabled: bool = True
    mlflow_tracking_uri: str | None = os.getenv("MLFLOW_TRACKING_URI")
    mlflow_username: str | None = os.getenv("MLFLOW_TRACKING_USERNAME")
    mlflow_password: str | None = os.getenv("MLFLOW_TRACKING_PASSWORD")
    mlflow_experiment_name: str | None = None

    # Lista de modelos: se ejecutan todos, uno detrás de otro, dentro del mismo workflow.
    models: list[ModelConfig] = field(
        default_factory=lambda: [
            # ModelConfig(
            #     run_name="rf_200_depth20_leaf5",
            #     model_name="rf",
            #     params={
            #         "n_estimators": 200,
            #         "max_depth": 20,
            #         "min_samples_leaf": 5,
            #         "max_features": "sqrt",
            #         "n_jobs": -1,
            #         "bootstrap": True,
            #         "max_samples": 0.3,
            #     },
            # ),
            # ModelConfig(
            #     run_name="rf_800_depth25_leaf35",
            #     model_name="rf",
            #     params={
            #         "n_estimators": 800,
            #         "max_depth": 25,
            #         "min_samples_leaf": 5,
            #         "max_features": 0.5,
            #         "n_jobs": -1,
            #         "bootstrap": True,
            #         "max_samples": 0.7,
            #     },
            # ),
            # ModelConfig(
            #     run_name="rf_800_depth50_leaf35",
            #     model_name="rf",
            #     params={
            #         "n_estimators": 800,
            #         "max_depth": 50,
            #         "min_samples_leaf": 5,
            #         "max_features": 0.5,
            #         "n_jobs": -1,
            #         "bootstrap": True,
            #         "max_samples": 0.7,
            #     },
            # ),
            # ModelConfig(
            #     run_name="hgb_lr0p1_iter200",
            #     model_name="hgb",
            #     params={
            #         "learning_rate": 0.1,
            #         "max_iter": 200,
            #         "max_depth": None,
            #         "min_samples_leaf": 20,
            #     },
            # ),
            # ModelConfig(
            #     run_name="hgb_lr0p01_iter200",
            #     model_name="hgb",
            #     params={
            #         "learning_rate": 0.1,
            #         "max_iter": 200,
            #         "max_depth": None,
            #         "min_samples_leaf": 20,
            #     },
            # ),
            ModelConfig(
                 run_name="mlp_lr1e-3_iter300",
                 model_name="mlp",
                 params={
                    "hidden_layer_sizes": (128, 64),
                    "activation": "relu",
                    "solver": "adam",
                    "alpha": 1e-4,
                    "learning_rate": "adaptive",
                    "learning_rate_init": 1e-3,
                    "max_iter": 300,
                    "early_stopping": True,
                    "validation_fraction": 0.1,
                 },
            ),
            ModelConfig(
                 run_name="mlp_lr1e-2_iter500",
                 model_name="mlp",
                 params={
                    "hidden_layer_sizes": (128, 64),
                    "activation": "relu",
                    "solver": "adam",
                    "alpha": 1e-4,
                    "learning_rate": "adaptive",
                    "learning_rate_init": 1e-2,
                    "max_iter": 500,
                    "early_stopping": True,
                    "validation_fraction": 0.1,
                 },
            ),

            
        ]
    )


CONFIG = WorkflowConfig()


# ---------------------------------------------------------------------------
# Paso 2. Helpers pequeños, equivalentes a celdas auxiliares del notebook
# ---------------------------------------------------------------------------

def normalize_mask_names(mask_names: list[str]) -> list[str]:
    """Valida y normaliza nombres de máscaras contra `io.MASK_MAP`."""

    out = [m.lower().strip() for m in mask_names]
    invalid = [m for m in out if m not in io.MASK_MAP]
    if invalid:
        raise ValueError(
            f"Máscaras no soportadas: {invalid}. Disponibles: {list(io.MASK_MAP)}"
        )
    return out


def slugify(value: object) -> str:
    """Convierte un valor de configuración en token usable en rutas y MLflow."""

    return str(value).strip().lower().replace(" ", "_").replace(".", "p")


def build_processed_run_name(config: WorkflowConfig) -> str:
    """Replica `build_run_name`, añadiendo si hay máscaras predictoras."""

    mask_tokens = normalize_mask_names(config.mask_names) or ["nomask"]
    pred_mask_tokens = normalize_mask_names(config.predictor_masks)
    tokens = [*mask_tokens, config.temporal_resolution]
    if pred_mask_tokens:
        tokens.append("predmask_" + "_".join(pred_mask_tokens))
    return "_".join(slugify(token) for token in tokens)


def build_model_run_name(config: WorkflowConfig, model_config: ModelConfig) -> str:
    """Construye nombres únicos por modelo, como hacía el notebook con `model_run_name`."""

    return f"{build_processed_run_name(config)}_{config.split_mask_name}_{slugify(model_config.run_name)}"


def build_mlflow_experiment_name(config: WorkflowConfig) -> str:
    """Nombre explicativo del experimento MLflow derivado de la configuración."""

    if config.mlflow_experiment_name:
        return config.mlflow_experiment_name
    masks = "+".join(normalize_mask_names(config.mask_names)) or "nomask"
    pred_masks = "+".join(normalize_mask_names(config.predictor_masks)) or "none"
    return (
        f"profecia_{config.experiment_name}/"
        f"{config.temporal_resolution}/"
        f"filters={masks}/"
        f"pred_masks={pred_masks}/"
        f"split={config.split_mask_name}"
    )


def load_mask_metadata(mask_dir: Path) -> dict:
    """Carga `mask_metadata.json` si existe; si no, devuelve un diccionario vacío."""

    path = mask_dir / "mask_metadata.json"
    if not path.exists():
        return {"masks": {}}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_mask_labels(mask_metadata: dict, mask_name: str) -> dict[int, str]:
    """Devuelve etiquetas de una máscara categórica usando el metadata del notebook."""

    labels = mask_metadata.get("masks", {}).get(mask_name, {}).get("labels", {})
    return {int(k): str(v) for k, v in labels.items()}


def build_mask_onehot_predictors(
    config: WorkflowConfig,
    mask_metadata: dict,
    reference: xr.DataArray,
) -> dict[str, xr.DataArray]:
    """
    Convierte máscaras categóricas en predictores one-hot 3D.

    `export_train_test_split` espera predictores con dims
    `(time, latitude, longitude)`, así que cada clase de la máscara se replica
    a todos los tiempos, exactamente como en la celda one-hot del notebook.
    """

    predictors: dict[str, xr.DataArray] = {}
    for mask_name in normalize_mask_names(config.predictor_masks):
        mask_da = load_mask(
            mask_dir=config.mask_dir,
            mask_name=mask_name,
            latitude=reference.latitude.values,
            longitude=reference.longitude.values,
        )
        labels = get_mask_labels(mask_metadata, mask_name)
        values = np.asarray(mask_da.values)
        finite = np.isfinite(values)
        classes = sorted(
            int(code)
            for code in np.unique(values[finite])
            if int(code) not in config.predictor_mask_ignore_codes
        )

        for code in classes:
            label = labels.get(code, f"class_{code}")
            feature_name = f"{mask_name}__{slugify(label)}"
            static = (values == code).astype(np.float32)
            repeated = np.broadcast_to(static, reference.shape).astype(np.float32, copy=True)
            predictors[feature_name] = xr.DataArray(
                repeated,
                coords=reference.coords,
                dims=reference.dims,
                name=feature_name,
            )

    return predictors


def save_json(path: Path, payload: dict) -> None:
    """Guarda JSON con el mismo estilo de metadata del notebook."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(io._to_jsonable(payload), f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Paso 3. Preprocesado y guardado de variables
# ---------------------------------------------------------------------------

def preprocess_variables(config: WorkflowConfig) -> tuple[Path, dict]:
    """Equivale a las celdas de carga de grid, máscaras y `load_and_save_variable`."""

    processed_run_name = build_processed_run_name(config)
    output_dir = config.workflow_root / config.experiment_name / "processed" / processed_run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    mask_names = normalize_mask_names(config.mask_names)
    predictor_masks = normalize_mask_names(config.predictor_masks)

    print("\n=== Paso 3. Preprocesado ===")
    print("processed_run_name:", processed_run_name)
    print("output_dir:", output_dir)
    print("mask_names:", mask_names if mask_names else ["nomask"])
    print("predictor_masks:", predictor_masks if predictor_masks else ["none"])

    # Igual que el notebook, cargamos LAI primero para fijar grid y validar máscaras.
    lai_raw, _ = io.load_netcdf(
        base_dir=config.raw_dir,
        variable=config.target_name,
        roi=config.roi,
        start_year=config.start_year,
        end_year_inclusive=config.end_year_inclusive,
        dtype=config.dtype,
    )
    latitude = lai_raw.latitude.values
    longitude = lai_raw.longitude.values

    masks = {
        name: io.load_mask(config.mask_dir, name, latitude, longitude)
        for name in mask_names
    }
    for name, da in masks.items():
        print("mask:", name, da.shape, da.dtype)

    lai_ref = io.aggregate_time(
        da=lai_raw,
        variable_name=config.target_name,
        temporal_resolution=config.temporal_resolution,
    )
    combined_mask, mask_info = io.build_combined_filter_mask(
        da=lai_ref,
        masks=masks if masks else None,
    )
    print("mask_info:")
    pprint(mask_info)

    # Este bucle conserva la celda principal de preprocesado del notebook.
    results = {}
    for variable in config.variable_names:
        print(f"Procesando {variable} ...")
        results |= io.load_and_save_variable(
            raw_dir=config.raw_dir,
            output_dir=output_dir,
            variable=variable,
            mask_dir=config.mask_dir,
            mask_names=mask_names,
            roi=config.roi,
            start_year=config.start_year,
            end_year_inclusive=config.end_year_inclusive,
            dtype=config.dtype,
            temporal_resolution=config.temporal_resolution,
            save_output=config.save_output,
        )
    print("Variables procesadas:", list(results.keys()))

    io.save_processed_metadata(
        output_dir=output_dir,
        variable_results=results,
        temporal_resolution=config.temporal_resolution,
        roi=config.roi,
        start_year=config.start_year,
        end_year_inclusive=config.end_year_inclusive,
        dtype=config.dtype,
    )

    run_config = {
        "processed_run_name": processed_run_name,
        "output_dir": str(output_dir),
        "raw_dir": str(config.raw_dir),
        "mask_dir": str(config.mask_dir),
        "temporal_resolution": config.temporal_resolution,
        "start_year": config.start_year,
        "end_year_inclusive": config.end_year_inclusive,
        "dtype": config.dtype,
        "roi": config.roi,
        "variable_names": config.variable_names,
        "predictor_names": config.predictor_names,
        "mask_names": mask_names,
        "predictor_masks": predictor_masks,
        "combined_filter_info": mask_info,
    }
    save_json(output_dir / "run_config.json", run_config)

    return output_dir, mask_info


# ---------------------------------------------------------------------------
# Paso 4. Carga de procesados y split espacial
# ---------------------------------------------------------------------------

def build_spatial_split(config: WorkflowConfig, processed_dir: Path) -> tuple[dict, dict, Path]:
    """Equivale a las celdas donde se carga LAI, máscara categórica y se crea el split."""

    print("\n=== Paso 4. Split espacial ===")
    mask_metadata = load_mask_metadata(config.mask_dir)
    labels = get_mask_labels(mask_metadata, config.split_mask_name)

    data_dict, metadata = load_processed_dataset(
        input_dir=processed_dir,
        variable_names=config.variable_names,
        reference_variable=config.target_name,
    )
    target = data_dict[config.target_name]

    predictors = {name: data_dict[name] for name in config.predictor_names}
    predictors.update(build_mask_onehot_predictors(config, mask_metadata, target))

    print("Variables cargadas:", list(data_dict.keys()))
    print("Predictores tabulares:", list(predictors.keys()))
    print("Shape target:", target.shape)

    category_mask = load_mask(
        mask_dir=config.mask_dir,
        mask_name=config.split_mask_name,
        latitude=target.latitude.values,
        longitude=target.longitude.values,
    )
    print("split_mask_name:", config.split_mask_name)
    print("labels:", labels)

    split_result = make_stratified_spatial_split(
        lai=target,
        category_mask=category_mask,
        test_fraction=config.test_fraction,
        seed=config.seed,
        ignore_codes=config.ignore_codes,
        split_name=f"{slugify(config.pixel_fraction)}pct_pixel_fraction",
        category_labels=labels,
        pixel_fraction=config.pixel_fraction,
        subset_seed=config.subset_seed,
        min_valid_fraction=config.min_valid_fraction,
    )

    train_mask = split_result["train_mask"]
    test_mask = split_result["test_mask"]
    selected_pixel_mask = split_result["selected_pixel_mask"]
    split_metadata = split_result["metadata"]

    assert train_mask.shape == (target.sizes["latitude"], target.sizes["longitude"])
    assert test_mask.shape == (target.sizes["latitude"], target.sizes["longitude"])
    assert not np.any(train_mask.values & test_mask.values), "Train y test no deben solaparse"
    assert np.array_equal(
        train_mask.values | test_mask.values,
        selected_pixel_mask.values.astype(bool),
    ), "Train + test no cubren exactamente los píxeles seleccionados"

    split_dir = (
        config.workflow_root
        / config.experiment_name
        / "splits"
        / build_processed_run_name(config)
        / config.split_mask_name
    )
    save_spatial_split(
        output_dir=split_dir,
        train_mask=train_mask,
        test_mask=test_mask,
        metadata=split_metadata,
        prefix=config.split_mask_name,
        selected_pixel_mask=split_result["selected_pixel_mask"],
        eligible_pixel_mask=split_result["eligible_pixel_mask"],
        valid_pixel_mask=split_result["valid_pixel_mask"],
    )

    print("Split guardado en:", split_dir)
    print("Selected pixels:", int(selected_pixel_mask.values.sum()))
    print("Train pixels   :", int(train_mask.values.sum()))
    print("Test pixels    :", int(test_mask.values.sum()))

    bundle = {
        "data_dict": data_dict,
        "metadata": metadata,
        "target": target,
        "predictors": predictors,
        "category_mask": category_mask,
        "split_result": split_result,
        "mask_metadata": mask_metadata,
    }
    return bundle, split_metadata, split_dir


# ---------------------------------------------------------------------------
# Paso 5. Exportación tabular
# ---------------------------------------------------------------------------

def export_tabular(config: WorkflowConfig, split_bundle: dict, split_dir: Path) -> dict:
    """Equivale a la celda `export_train_test_split` del notebook."""

    print("\n=== Paso 5. Export tabular ===")
    metadata = export_train_test_split(
        target=split_bundle["target"],
        predictors=split_bundle["predictors"],
        train_mask=split_bundle["split_result"]["train_mask"],
        test_mask=split_bundle["split_result"]["test_mask"],
        output_dir=split_dir,
        prefix=config.split_mask_name,
    )
    print("X features:", metadata["feature_names"])
    print("n_train:", metadata["n_train"])
    print("n_test :", metadata["n_test"])
    return metadata


# ---------------------------------------------------------------------------
# Paso 6. Entrenamiento y evaluación de un modelo
# ---------------------------------------------------------------------------

def train_and_evaluate_one_model(
    config: WorkflowConfig,
    model_config: ModelConfig,
    split_bundle: dict,
    split_dir: Path,
) -> dict:
    """Equivale a las celdas de entrenamiento, predicción, métricas y figuras."""

    model_run_name = build_model_run_name(config, model_config)
    model_dir = (
        config.workflow_root
        / config.experiment_name
        / "models"
        / build_processed_run_name(config)
        / config.split_mask_name
        / model_run_name
    )
    eval_dir = model_dir / "evaluation" / "step_by_step_script"
    model_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Paso 6. Modelo ===")
    print("model_run_name:", model_run_name)
    print("model_dir:", model_dir)
    print("model_name:", model_config.model_name)
    pprint(model_config.params)

    X_train = np.load(split_dir / "X_train.npy", mmap_mode=config.mmap_mode)
    y_train = np.load(split_dir / "y_train.npy", mmap_mode=config.mmap_mode)
    X_test = np.load(split_dir / "X_test.npy", mmap_mode=config.mmap_mode)
    y_test = np.load(split_dir / "y_test.npy", mmap_mode=config.mmap_mode)
    pixel_id_test = np.load(split_dir / "pixel_id_test.npy", mmap_mode=config.mmap_mode)
    lat_idx_test = np.load(split_dir / "lat_idx_test.npy", mmap_mode=config.mmap_mode)
    lon_idx_test = np.load(split_dir / "lon_idx_test.npy", mmap_mode=config.mmap_mode)
    time_idx_test = np.load(split_dir / "time_idx_test.npy", mmap_mode=config.mmap_mode)

    with open(split_dir / "dataset_metadata.json", "r", encoding="utf-8") as f:
        dataset_metadata = json.load(f)

    mlflow_config = None
    if config.mlflow_enabled:
        mlflow_config = {
            "tracking_uri": config.mlflow_tracking_uri,
            "user": config.mlflow_username,
            "password": config.mlflow_password,
            "experiment_name": build_mlflow_experiment_name(config),
            "run_name": model_run_name,
        }

    train_result = train_tabular_model(
        X_train=X_train,
        y_train=y_train,
        model_name=model_config.model_name,
        scaler_name=model_config.scaler_name,
        random_state=model_config.random_state,
        mlflow_config=mlflow_config,
        log_shap=model_config.log_shap,
        **model_config.params,
    )

    saved_paths = save_trained_pipeline(
        output_dir=model_dir,
        model=train_result["model"],
        scaler=train_result["scaler"],
        train_info=train_result["train_info"],
        dataset_metadata=dataset_metadata,
        prefix=model_run_name,
    )

    if train_result["scaler"] is not None:
        X_test_used = train_result["scaler"].transform(X_test).astype(np.float32, copy=False)
    else:
        X_test_used = X_test

    target = split_bundle["target"]
    y_pred_test = np.asarray(train_result["model"].predict(X_test_used)).reshape(-1)
    prediction_df = build_prediction_dataframe(
        y_true=y_test,
        y_pred=y_pred_test,
        pixel_id=pixel_id_test,
        lat_idx=lat_idx_test,
        lon_idx=lon_idx_test,
        time_idx=time_idx_test,
        latitude_values=target.latitude.values,
        longitude_values=target.longitude.values,
        time_values=pd.to_datetime(dataset_metadata["time_values"]),
    )
    prediction_df["residual"] = prediction_df["y_true"] - prediction_df["y_pred"]

    global_metrics = compute_global_regression_metrics(y_true=y_test, y_pred=y_pred_test)
    global_metrics["bias"] = float(np.mean(y_pred_test - np.asarray(y_test).reshape(-1)))
    global_metrics["pearson_r"] = float(
        np.corrcoef(np.asarray(y_test).reshape(-1), y_pred_test)[0, 1]
    )

    pixel_metrics_df = compute_pixel_regression_metrics(
        y_true=y_test,
        y_pred=y_pred_test,
        pixel_id=pixel_id_test,
        lat_idx=lat_idx_test,
        lon_idx=lon_idx_test,
        latitude=target.latitude.values[np.asarray(lat_idx_test).astype(int)],
        longitude=target.longitude.values[np.asarray(lon_idx_test).astype(int)],
    )
    pixel_metrics_summary = summarize_pixel_metrics(pixel_metrics_df)

    add_category_columns(
        config=config,
        split_bundle=split_bundle,
        prediction_df=prediction_df,
        pixel_metrics_df=pixel_metrics_df,
    )

    paths = save_evaluation_outputs(
        model_run_name=model_run_name,
        eval_dir=eval_dir,
        prediction_df=prediction_df,
        pixel_metrics_df=pixel_metrics_df,
        global_metrics=global_metrics,
        pixel_metrics_summary=pixel_metrics_summary,
    )

    log_evaluation_to_mlflow(
        mlflow_run_id=train_result.get("mlflow_run_id"),
        global_metrics=global_metrics,
        pixel_metrics_summary=pixel_metrics_summary,
        paths=paths,
    )

    print("Global metrics:")
    pprint(global_metrics)
    print("Saved model paths:")
    pprint(saved_paths)

    return {
        "model_run_name": model_run_name,
        "global_metrics": global_metrics,
        "pixel_metrics_summary": pixel_metrics_summary,
        "saved_paths": saved_paths,
        "evaluation_paths": paths,
        "mlflow_run_id": train_result.get("mlflow_run_id"),
    }


def add_category_columns(
    config: WorkflowConfig,
    split_bundle: dict,
    prediction_df: pd.DataFrame,
    pixel_metrics_df: pd.DataFrame,
) -> None:
    """Añade al DataFrame las clases de la máscara del split, como hacía el notebook."""

    labels = get_mask_labels(split_bundle["mask_metadata"], config.split_mask_name)
    category_mask = split_bundle["category_mask"]

    for df in (prediction_df, pixel_metrics_df):
        df[f"{config.split_mask_name}_code"] = category_mask.values[
            df["lat_idx"].to_numpy(dtype=int),
            df["lon_idx"].to_numpy(dtype=int),
        ].astype(int)
        df[f"{config.split_mask_name}_label"] = df[f"{config.split_mask_name}_code"].map(
            labels
        ).fillna("unknown")


def save_evaluation_outputs(
    model_run_name: str,
    eval_dir: Path,
    prediction_df: pd.DataFrame,
    pixel_metrics_df: pd.DataFrame,
    global_metrics: dict,
    pixel_metrics_summary: dict,
) -> dict[str, Path]:
    """Guarda CSV y figuras, siguiendo las celdas de evaluación del notebook."""

    summary_df = pd.DataFrame(
        [
            {
                "model_run_name": model_run_name,
                **global_metrics,
                **{f"pixel_{k}": v for k, v in pixel_metrics_summary.items()},
            }
        ]
    )
    summary_path = eval_dir / f"{model_run_name}_summary_metrics.csv"
    pixel_metrics_path = eval_dir / f"{model_run_name}_pixel_metrics.csv"
    prediction_sample_path = eval_dir / f"{model_run_name}_prediction_sample.csv"
    summary_df.to_csv(summary_path, index=False)
    pixel_metrics_df.to_csv(pixel_metrics_path, index=False)
    prediction_df.head(20_000).to_csv(prediction_sample_path, index=False)

    global_hexbin_path = eval_dir / f"{model_run_name}_hexbin_global.png"
    residual_hist_path = eval_dir / f"{model_run_name}_residual_hist.png"
    best_worst_pixels_path = eval_dir / f"{model_run_name}_best_worst_pixels_hexbin.png"

    fig, ax = plt.subplots(figsize=(7, 7))
    hb = ax.hexbin(prediction_df["y_true"], prediction_df["y_pred"], gridsize=60, mincnt=1)
    xy_min = min(prediction_df["y_true"].min(), prediction_df["y_pred"].min())
    xy_max = max(prediction_df["y_true"].max(), prediction_df["y_pred"].max())
    ax.plot([xy_min, xy_max], [xy_min, xy_max], "--", color="black", linewidth=1)
    ax.set_xlabel("LAI real")
    ax.set_ylabel("LAI predicho")
    ax.set_title(f"Test global hexbin | {model_run_name}")
    fig.colorbar(hb, ax=ax, label="N.º de muestras")
    fig.tight_layout()
    fig.savefig(global_hexbin_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(prediction_df["residual"], bins=60, color="#CC6677", alpha=0.85)
    ax.axvline(0.0, linestyle="--", color="black", linewidth=1)
    ax.set_title(f"Histograma de residuales | {model_run_name}")
    ax.set_xlabel("Residual = real - predicho")
    ax.set_ylabel("Frecuencia")
    fig.tight_layout()
    fig.savefig(residual_hist_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    ranking = rank_best_and_worst_pixels(pixel_metrics_df, metric="r2", top_k=4)
    selected_pixels = pd.concat(
        [ranking["best"].assign(group="best"), ranking["worst"].assign(group="worst")],
        ignore_index=True,
    )
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    for ax, (_, row) in zip(axes.ravel(), selected_pixels.iterrows()):
        ts_df = build_pixel_timeseries_dataframe(prediction_df, pixel_id=int(row["pixel_id"]))
        if len(ts_df) >= 20:
            hb = ax.hexbin(ts_df["y_true"], ts_df["y_pred"], gridsize=20, mincnt=1)
            fig.colorbar(hb, ax=ax, shrink=0.75)
        else:
            ax.scatter(ts_df["y_true"], ts_df["y_pred"], s=20)
        xy_min = min(ts_df["y_true"].min(), ts_df["y_pred"].min())
        xy_max = max(ts_df["y_true"].max(), ts_df["y_pred"].max())
        ax.plot([xy_min, xy_max], [xy_min, xy_max], "--", color="black")
        ax.set_title(f"{row['group']} | pixel {int(row['pixel_id'])} | R2={row['r2']:.3f}")
        ax.set_xlabel("Real")
        ax.set_ylabel("Predicho")
    fig.tight_layout()
    fig.savefig(best_worst_pixels_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    paths = {
        "summary": summary_path,
        "pixel_metrics": pixel_metrics_path,
        "prediction_sample": prediction_sample_path,
        "global_hexbin": global_hexbin_path,
        "residual_hist": residual_hist_path,
        "best_worst_pixels": best_worst_pixels_path,
    }

    label_column = next((c for c in pixel_metrics_df.columns if c.endswith("_label")), None)
    if label_column:
        category_metrics_path = eval_dir / f"{model_run_name}_{label_column}_metrics.csv"
        (
            pixel_metrics_df.groupby(label_column, dropna=False)[["r2", "rmse", "mae"]]
            .mean(numeric_only=True)
            .sort_values("r2", ascending=False)
            .to_csv(category_metrics_path)
        )
        paths["category_metrics"] = category_metrics_path

    return paths


def log_evaluation_to_mlflow(
    mlflow_run_id: str | None,
    global_metrics: dict,
    pixel_metrics_summary: dict,
    paths: dict[str, Path],
) -> None:
    """Añade métricas y artefactos de evaluación al mismo run MLflow del entrenamiento."""

    if not mlflow_run_id:
        print("No hay MLflow run id; evaluación guardada sólo en disco.")
        return

    import mlflow

    with mlflow.start_run(run_id=mlflow_run_id):
        for metric_name, metric_value in global_metrics.items():
            if np.isfinite(metric_value):
                mlflow.log_metric(f"test_{metric_name}", float(metric_value))
        for metric_name, metric_value in pixel_metrics_summary.items():
            if isinstance(metric_value, (int, float)) and np.isfinite(metric_value):
                mlflow.log_metric(f"pixel_{metric_name}", float(metric_value))
        for path in paths.values():
            mlflow.log_artifact(str(path), artifact_path="evaluation")

    print("Evaluation artifacts logged to MLflow run:", mlflow_run_id)


# ---------------------------------------------------------------------------
# Paso 7. Ejecución completa, en el mismo orden del notebook
# ---------------------------------------------------------------------------

def main() -> int:
    """Ejecuta todos los pasos en orden."""

    processed_dir, mask_info = preprocess_variables(CONFIG)
    split_bundle, split_metadata, split_dir = build_spatial_split(CONFIG, processed_dir)
    tabular_metadata = export_tabular(CONFIG, split_bundle, split_dir)

    model_summaries = []
    for model_config in CONFIG.models:
        model_summaries.append(
            train_and_evaluate_one_model(
                config=CONFIG,
                model_config=model_config,
                split_bundle=split_bundle,
                split_dir=split_dir,
            )
        )

    summary_path = (
        CONFIG.workflow_root
        / CONFIG.experiment_name
        / "step_by_step_script_summary.json"
    )
    save_json(
        summary_path,
        {
            "processed_dir": str(processed_dir),
            "split_dir": str(split_dir),
            "mask_info": mask_info,
            "tabular_metadata": tabular_metadata,
            "model_summaries": model_summaries,
        },
    )
    print("\nWorkflow terminado.")
    print("Resumen:", summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
