from __future__ import annotations

import json
import logging
import os
import shutil
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

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
from src.training.train import (
    load_train_arrays,
    save_trained_pipeline,
    train_tabular_model,
)


LOGGER = logging.getLogger(__name__)


SUPPORTED_DATA_EXTENSIONS = {".nc", ".nc4", ".netcdf", ".h5", ".hdf5", ".zarr"}


@dataclass(slots=True)
class VariableSource:
    """
    Describe la procedencia de una variable raw.

    El objetivo es que el script no infiera rutas ni nombres de variables
    internamente: cada experimento define explícitamente qué fichero cargar.
    """

    path: Path
    data_var: str | None = None
    description: str = ""


@dataclass(slots=True)
class MaskSource:
    """
    Describe una máscara utilizable en el workflow.

    `kind="boolean"`:
    - se interpreta como filtro binario
    - si `keep_values` es `None`, cualquier valor finito y distinto de cero
      se considera True

    `kind="categorical"`:
    - se conserva el valor original para splits/agrupaciones
    """

    path: Path
    kind: Literal["boolean", "categorical"] = "boolean"
    keep_values: tuple[int | float, ...] | None = None
    description: str = ""
    labels: dict[int, str] | None = None


@dataclass(slots=True)
class SpatialSplitConfig:
    """
    Configuración del split espacial.

    La opción más importante para tu caso es `reuse_saved_masks`: si existen
    máscaras guardadas para el experimento, se reutilizan para mantener el mismo
    train/test entre corridas.
    """

    category_mask_name: str
    split_name: str = "spatial_stratified_split"
    prefix: str | None = None
    test_fraction: float = 0.10
    pixel_fraction: float = 1.0
    ignore_codes: tuple[int, ...] = (0,)
    min_valid_fraction: float = 0.0
    seed: int = 42
    subset_seed: int | None = None
    reuse_saved_masks: bool = True
    reuse_exported_arrays: bool = True


@dataclass(slots=True)
class ModelRunConfig:
    """
    Configuración de una variante de modelo.

    Un mismo experimento puede entrenar varias variantes sobre el mismo
    preprocesado y el mismo split.
    """

    run_name: str
    model_name: Literal["rf", "hgb", "mlp"]
    scaler_name: str | None = None
    random_state: int = 42
    params: dict[str, Any] = field(default_factory=dict)
    save_predictions_sample: bool = True


@dataclass(slots=True)
class MlflowTrackingConfig:
    """
    Ajustes para el tracking en MLflow.
    """

    enabled: bool = False
    tracking_uri: str | None = None
    experiment_name: str = "profecia-lai"
    tags: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class PlotConfig:
    """
    Parámetros de las figuras de control.
    """

    enabled: bool = True
    scatter_sample_size: int = 20_000
    n_example_pixels: int = 6
    top_feature_count: int = 15


@dataclass(slots=True)
class ExperimentPaths:
    """
    Directorios usados por un experimento.

    Separar rutas por tipo de artefacto hace más fácil limpiar solo lo temporal
    sin tocar modelos, métricas o figuras.
    """

    intermediate_dir: Path
    split_dir: Path
    model_dir: Path
    report_dir: Path


@dataclass(slots=True)
class ExperimentConfig:
    """
    Configuración completa de un experimento end-to-end.
    """

    name: str
    paths: ExperimentPaths
    raw_variables: dict[str, VariableSource]
    target_name: str
    predictor_names: list[str]
    temporal_resolution: Literal["monthly", "annual"]
    start_year: int | None = None
    end_year_inclusive: int | None = None
    dtype: str = "float32"
    roi: dict[str, float] | None = None
    mask_sources: dict[str, MaskSource] = field(default_factory=dict)
    filter_mask_names: list[str] = field(default_factory=list)
    split: SpatialSplitConfig | None = None
    model_runs: list[ModelRunConfig] = field(default_factory=list)
    mlflow: MlflowTrackingConfig = field(default_factory=MlflowTrackingConfig)
    plots: PlotConfig = field(default_factory=PlotConfig)
    lat_block_size: int = 10
    mmap_mode: str | None = "r"
    save_processed_netcdf: bool = True
    cleanup_intermediate_after_models: bool = False


def validate_experiment_config(config: ExperimentConfig) -> None:
    """
    Valida la coherencia de la configuración antes de ejecutar nada.
    """

    if config.target_name not in config.raw_variables:
        raise ValueError(
            f"El target '{config.target_name}' no está definido en raw_variables."
        )

    if not config.predictor_names:
        raise ValueError("Debes indicar al menos una variable predictora.")

    missing_predictors = sorted(set(config.predictor_names) - set(config.raw_variables))
    if missing_predictors:
        raise ValueError(
            f"Las variables predictoras {missing_predictors} no existen en raw_variables."
        )

    if config.target_name in config.predictor_names:
        raise ValueError("El target no debe incluirse dentro de predictor_names.")

    if config.temporal_resolution not in {"monthly", "annual"}:
        raise ValueError(
            "temporal_resolution debe ser 'monthly' o 'annual'."
        )

    if config.split is None:
        raise ValueError("Debes definir una configuración de split espacial.")

    if config.split.category_mask_name not in config.mask_sources:
        raise ValueError(
            f"La máscara categórica '{config.split.category_mask_name}' no existe en mask_sources."
        )

    for mask_name in config.filter_mask_names:
        if mask_name not in config.mask_sources:
            raise ValueError(
                f"La máscara de filtrado '{mask_name}' no existe en mask_sources."
            )
        if config.mask_sources[mask_name].kind != "boolean":
            raise ValueError(
                f"La máscara '{mask_name}' debe ser de tipo boolean para usarla como filtro."
            )

    if not config.model_runs:
        raise ValueError("Debes definir al menos una variante de modelo.")


def run_experiment(config: ExperimentConfig) -> dict[str, Any]:
    """
    Ejecuta un experimento completo:

    1. carga y alinea variables raw
    2. aplica agregación temporal
    3. carga máscaras y construye el filtro combinado
    4. genera o reutiliza el split espacial
    5. exporta datos tabulares train/test
    6. entrena una o varias variantes de modelo
    7. evalúa y guarda métricas, figuras, modelos y artefactos de MLflow
    """

    validate_experiment_config(config)
    layout = prepare_output_layout(config)

    LOGGER.info("============================================================")
    LOGGER.info("Iniciando experimento '%s'", config.name)
    LOGGER.info("Intermedios   : %s", layout["intermediate_dir"])
    LOGGER.info("Split         : %s", layout["split_dir"])
    LOGGER.info("Modelos       : %s", layout["model_dir"])
    LOGGER.info("Reportes      : %s", layout["report_dir"])

    save_json(layout["report_dir"] / "effective_config.json", config)

    data_bundle = load_and_prepare_data(config, layout)
    split_bundle = prepare_or_load_split(config, layout, data_bundle)
    tabular_bundle = prepare_or_load_tabular_export(config, layout, data_bundle, split_bundle)

    model_summaries: list[dict[str, Any]] = []
    for model_run in config.model_runs:
        LOGGER.info("------------------------------------------------------------")
        LOGGER.info(
            "Entrenando variante '%s' (%s)",
            model_run.run_name,
            model_run.model_name,
        )
        summary = train_and_evaluate_model(
            config=config,
            model_run=model_run,
            layout=layout,
            data_bundle=data_bundle,
            split_bundle=split_bundle,
            tabular_bundle=tabular_bundle,
        )
        model_summaries.append(summary)

    if config.cleanup_intermediate_after_models:
        cleanup_intermediate_artifacts(layout)

    experiment_summary = {
        "experiment_name": config.name,
        "target_name": config.target_name,
        "predictor_names": config.predictor_names,
        "temporal_resolution": config.temporal_resolution,
        "model_runs": model_summaries,
        "paths": {key: str(value) for key, value in layout.items()},
    }
    save_json(layout["report_dir"] / "experiment_summary.json", experiment_summary)

    LOGGER.info("Experimento '%s' completado.", config.name)
    return experiment_summary


def prepare_output_layout(config: ExperimentConfig) -> dict[str, Path]:
    """
    Crea y devuelve el árbol de salida del experimento.
    """

    processed_dir = config.paths.intermediate_dir / "processed"
    tabular_dir = config.paths.intermediate_dir / "tabular"
    figures_dir = config.paths.report_dir / "figures"
    metrics_dir = config.paths.report_dir / "metrics"

    layout = {
        "intermediate_dir": config.paths.intermediate_dir,
        "processed_dir": processed_dir,
        "tabular_dir": tabular_dir,
        "split_dir": config.paths.split_dir,
        "model_dir": config.paths.model_dir,
        "report_dir": config.paths.report_dir,
        "figures_dir": figures_dir,
        "metrics_dir": metrics_dir,
    }

    for path in layout.values():
        path.mkdir(parents=True, exist_ok=True)

    return layout


def load_and_prepare_data(
    config: ExperimentConfig,
    layout: dict[str, Path],
) -> dict[str, Any]:
    """
    Carga las variables raw, aplica recortes temporales/espaciales, alinea grillas,
    agrega en el tiempo y construye las máscaras de filtrado.
    """

    ordered_names = [config.target_name, *config.predictor_names]
    loaded_raw: dict[str, xr.DataArray] = {}
    processed_data: dict[str, xr.DataArray] = {}
    variable_summaries: dict[str, dict[str, Any]] = {}

    reference: xr.DataArray | None = None

    for variable_name in ordered_names:
        LOGGER.info("Cargando variable raw '%s'", variable_name)
        source = config.raw_variables[variable_name]
        raw_da = load_variable_from_source(
            source=source,
            variable_name=variable_name,
            start_year=config.start_year,
            end_year_inclusive=config.end_year_inclusive,
            dtype=config.dtype,
            roi=config.roi,
        )

        prepared_da = aggregate_temporally(
            da=raw_da,
            temporal_resolution=config.temporal_resolution,
        )

        if reference is None:
            reference = prepared_da
        else:
            prepared_da = align_to_reference(prepared_da, reference, variable_name)

        loaded_raw[variable_name] = prepared_da
        processed_data[variable_name] = prepared_da
        variable_summaries[variable_name] = summarize_dataarray(prepared_da)

        if config.save_processed_netcdf:
            save_dataarray_netcdf(
                prepared_da,
                layout["processed_dir"] / f"{variable_name}.nc",
            )

    if reference is None:
        raise RuntimeError("No se ha podido cargar ninguna variable.")

    LOGGER.info("Cargando máscaras definidas en la configuración")
    masks = load_masks(config.mask_sources, reference)
    filter_masks = {
        name: masks[name]
        for name in config.filter_mask_names
    }
    combined_mask, mask_info = build_combined_filter_mask(reference, filter_masks)

    if combined_mask is not None and int(combined_mask.values.sum()) == 0:
        active_masks = ", ".join(config.filter_mask_names) or "ninguna"
        raise ValueError(
            "La combinación de máscaras no deja píxeles válidos. "
            f"Máscaras activas: [{active_masks}]. "
            "Las máscaras booleanas se combinan con AND; revisa si has usado "
            "máscaras mutuamente excluyentes, por ejemplo 'ebf' y 'bs'."
        )

    if combined_mask is not None:
        for variable_name, da in processed_data.items():
            processed_data[variable_name] = da.where(combined_mask)

    data_metadata = {
        "experiment_name": config.name,
        "temporal_resolution": config.temporal_resolution,
        "start_year": config.start_year,
        "end_year_inclusive": config.end_year_inclusive,
        "dtype": config.dtype,
        "roi": config.roi,
        "target_name": config.target_name,
        "predictor_names": config.predictor_names,
        "variable_summaries": variable_summaries,
        "mask_info": mask_info,
    }
    save_json(layout["report_dir"] / "data_preparation_summary.json", data_metadata)

    if config.plots.enabled:
        save_preprocessing_figures(
            config=config,
            output_dir=layout["figures_dir"] / "preprocessing",
            raw_data=loaded_raw,
            processed_data=processed_data,
            masks=masks,
            combined_mask=combined_mask,
        )

    return {
        "raw_data": loaded_raw,
        "processed_data": processed_data,
        "masks": masks,
        "combined_mask": combined_mask,
        "metadata": data_metadata,
    }


def prepare_or_load_split(
    config: ExperimentConfig,
    layout: dict[str, Path],
    data_bundle: dict[str, Any],
) -> dict[str, Any]:
    """
    Genera o reutiliza las máscaras de split.
    """

    assert config.split is not None

    prefix = split_prefix(config)
    split_dir = layout["split_dir"]

    train_mask_path = split_dir / f"{prefix}_train_mask.npy"
    test_mask_path = split_dir / f"{prefix}_test_mask.npy"
    metadata_path = split_dir / f"{prefix}_split_metadata.json"

    target = data_bundle["processed_data"][config.target_name]
    category_mask = data_bundle["masks"][config.split.category_mask_name]

    if (
        config.split.reuse_saved_masks
        and train_mask_path.exists()
        and test_mask_path.exists()
        and metadata_path.exists()
    ):
        LOGGER.info("Reutilizando split guardado previamente en %s", split_dir)
        split_bundle = load_saved_split(
            split_dir=split_dir,
            prefix=prefix,
            reference=category_mask,
        )
    else:
        LOGGER.info("Construyendo un split espacial nuevo")
        split_result = make_stratified_spatial_split(
            lai=target,
            category_mask=category_mask,
            test_fraction=config.split.test_fraction,
            seed=config.split.seed,
            ignore_codes=config.split.ignore_codes,
            split_name=config.split.split_name,
            category_labels=config.mask_sources[
                config.split.category_mask_name
            ].labels,
            pixel_fraction=config.split.pixel_fraction,
            subset_seed=config.split.subset_seed,
            min_valid_fraction=config.split.min_valid_fraction,
        )

        save_spatial_split(
            output_dir=split_dir,
            train_mask=split_result["train_mask"],
            test_mask=split_result["test_mask"],
            metadata=split_result["metadata"],
            prefix=prefix,
            selected_pixel_mask=split_result["selected_pixel_mask"],
            eligible_pixel_mask=split_result["eligible_pixel_mask"],
            valid_pixel_mask=split_result["valid_pixel_mask"],
        )
        split_bundle = split_result

    save_json(layout["report_dir"] / "split_summary.json", split_bundle["metadata"])

    if config.plots.enabled:
        save_split_figure(
            output_path=layout["figures_dir"] / "split" / "split_overview.png",
            category_mask=category_mask,
            split_bundle=split_bundle,
            category_labels=config.mask_sources[config.split.category_mask_name].labels,
        )

    return split_bundle


def prepare_or_load_tabular_export(
    config: ExperimentConfig,
    layout: dict[str, Path],
    data_bundle: dict[str, Any],
    split_bundle: dict[str, Any],
) -> dict[str, Any]:
    """
    Genera o reutiliza la exportación tabular train/test.
    """

    tabular_dir = layout["tabular_dir"]
    required_paths = [
        tabular_dir / "X_train.npy",
        tabular_dir / "y_train.npy",
        tabular_dir / "X_test.npy",
        tabular_dir / "y_test.npy",
        tabular_dir / "pixel_id_test.npy",
        tabular_dir / "lat_idx_test.npy",
        tabular_dir / "lon_idx_test.npy",
        tabular_dir / "time_idx_test.npy",
        tabular_dir / "dataset_metadata.json",
    ]

    if config.split and config.split.reuse_exported_arrays and all(
        path.exists() for path in required_paths
    ):
        LOGGER.info("Reutilizando exportación tabular existente en %s", tabular_dir)
        metadata = json.loads((tabular_dir / "dataset_metadata.json").read_text())
    else:
        LOGGER.info("Exportando train/test a formato tabular")
        metadata = export_train_test_split(
            target=data_bundle["processed_data"][config.target_name],
            predictors={
                name: data_bundle["processed_data"][name]
                for name in config.predictor_names
            },
            train_mask=split_bundle["train_mask"],
            test_mask=split_bundle["test_mask"],
            output_dir=tabular_dir,
            prefix=f"{config.name}_{split_prefix(config)}",
            lat_block_size=config.lat_block_size,
        )

    save_json(layout["report_dir"] / "tabular_export_summary.json", metadata)
    return {
        "tabular_dir": tabular_dir,
        "dataset_metadata": metadata,
    }


def train_and_evaluate_model(
    config: ExperimentConfig,
    model_run: ModelRunConfig,
    layout: dict[str, Path],
    data_bundle: dict[str, Any],
    split_bundle: dict[str, Any],
    tabular_bundle: dict[str, Any],
) -> dict[str, Any]:
    """
    Entrena, evalúa y registra una variante de modelo.
    """

    model_dir = layout["model_dir"] / model_run.run_name
    report_dir = layout["report_dir"] / model_run.run_name
    metrics_dir = report_dir / "metrics"
    figures_dir = report_dir / "figures"
    model_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    save_json(report_dir / "model_run_config.json", model_run)

    X_train, y_train, dataset_metadata = load_train_arrays(
        input_dir=tabular_bundle["tabular_dir"],
        mmap_mode=config.mmap_mode,
    )
    X_test = np.load(tabular_bundle["tabular_dir"] / "X_test.npy", mmap_mode=config.mmap_mode)
    y_test = np.load(tabular_bundle["tabular_dir"] / "y_test.npy", mmap_mode=config.mmap_mode)
    pixel_id_test = np.load(
        tabular_bundle["tabular_dir"] / "pixel_id_test.npy",
        mmap_mode=config.mmap_mode,
    )
    lat_idx_test = np.load(
        tabular_bundle["tabular_dir"] / "lat_idx_test.npy",
        mmap_mode=config.mmap_mode,
    )
    lon_idx_test = np.load(
        tabular_bundle["tabular_dir"] / "lon_idx_test.npy",
        mmap_mode=config.mmap_mode,
    )
    time_idx_test = np.load(
        tabular_bundle["tabular_dir"] / "time_idx_test.npy",
        mmap_mode=config.mmap_mode,
    )

    LOGGER.info(
        "Shapes tabulares | X_train=%s y_train=%s X_test=%s y_test=%s",
        X_train.shape,
        y_train.shape,
        X_test.shape,
        y_test.shape,
    )

    mlflow = configure_mlflow(config.mlflow)
    run_name = f"{config.name}__{model_run.run_name}"
    run_context = mlflow.start_run(run_name=run_name) if mlflow else nullcontext()

    with run_context:
        if mlflow:
            mlflow.set_tags(
                {
                    "experiment_name": config.name,
                    "model_run_name": model_run.run_name,
                    "target_name": config.target_name,
                    "temporal_resolution": config.temporal_resolution,
                    **config.mlflow.tags,
                }
            )
            mlflow.log_params(flatten_params("experiment", config))
            mlflow.log_params(flatten_params("model", model_run))

        LOGGER.info("Entrenando el modelo sobre train")
        train_result = train_tabular_model(
            X_train=X_train,
            y_train=y_train,
            model_name=model_run.model_name,
            scaler_name=model_run.scaler_name,
            random_state=model_run.random_state,
            **model_run.params,
        )
        train_result.pop("X_train_used", None)

        saved_paths = save_trained_pipeline(
            output_dir=model_dir,
            model=train_result["model"],
            scaler=train_result["scaler"],
            train_info=train_result["train_info"],
            dataset_metadata=dataset_metadata,
            prefix=model_run.run_name,
        )

        if train_result["scaler"] is not None:
            LOGGER.info("Aplicando scaler a test antes de predecir")
            X_test_used = train_result["scaler"].transform(X_test)
            X_test_used = X_test_used.astype(np.float32, copy=False)
        else:
            X_test_used = X_test

        LOGGER.info("Generando predicciones sobre test")
        y_pred = train_result["model"].predict(X_test_used)

        prediction_df = build_prediction_dataframe(
            y_true=y_test,
            y_pred=y_pred,
            pixel_id=pixel_id_test,
            lat_idx=lat_idx_test,
            lon_idx=lon_idx_test,
            time_idx=time_idx_test,
            latitude_values=data_bundle["processed_data"][config.target_name].latitude.values,
            longitude_values=data_bundle["processed_data"][config.target_name].longitude.values,
            time_values=pd.to_datetime(dataset_metadata["time_values"]),
        )
        prediction_df["residual"] = prediction_df["y_true"] - prediction_df["y_pred"]

        global_metrics = compute_global_regression_metrics(y_true=y_test, y_pred=y_pred)
        residual_summary = summarize_residuals(prediction_df)
        pixel_metrics_df = compute_pixel_regression_metrics(
            y_true=y_test,
            y_pred=y_pred,
            pixel_id=pixel_id_test,
            lat_idx=lat_idx_test,
            lon_idx=lon_idx_test,
            latitude=data_bundle["processed_data"][config.target_name].latitude.values[lat_idx_test],
            longitude=data_bundle["processed_data"][config.target_name].longitude.values[lon_idx_test],
        )
        pixel_metrics_summary = summarize_pixel_metrics(pixel_metrics_df)
        time_metrics_df = compute_group_regression_metrics(prediction_df, "time")
        year_metrics_df = compute_group_regression_metrics_from_time(
            prediction_df,
            group_name="year",
            extractor=lambda frame: frame["time"].dt.year,
        )

        extra_tables: dict[str, pd.DataFrame] = {
            "time_metrics": time_metrics_df,
            "year_metrics": year_metrics_df,
            "pixel_metrics": pixel_metrics_df,
        }

        if config.temporal_resolution == "monthly":
            month_metrics_df = compute_group_regression_metrics_from_time(
                prediction_df,
                group_name="calendar_month",
                extractor=lambda frame: frame["time"].dt.month,
            )
            extra_tables["calendar_month_metrics"] = month_metrics_df

        for mask_name, mask_source in config.mask_sources.items():
            if mask_source.kind != "categorical":
                continue
            category_df = compute_category_metrics(
                prediction_df=prediction_df,
                category_mask=data_bundle["masks"][mask_name],
                labels=mask_source.labels,
                output_column_prefix=mask_name,
            )
            extra_tables[f"{mask_name}_metrics"] = category_df

        save_json(metrics_dir / "global_metrics.json", global_metrics)
        save_json(metrics_dir / "residual_summary.json", residual_summary)
        save_json(metrics_dir / "pixel_metrics_summary.json", pixel_metrics_summary)
        save_json(metrics_dir / "saved_paths.json", saved_paths)
        for table_name, df in extra_tables.items():
            save_dataframe(df, metrics_dir / f"{table_name}.csv")

        if model_run.save_predictions_sample:
            prediction_df.head(2_000).to_csv(
                metrics_dir / "prediction_sample.csv",
                index=False,
            )

        monitoring = collect_model_monitoring_artifacts(
            model=train_result["model"],
            feature_names=config.predictor_names,
            output_dir=metrics_dir,
            top_feature_count=config.plots.top_feature_count,
        )

        if config.plots.enabled:
            create_evaluation_figures(
                output_dir=figures_dir,
                model=train_result["model"],
                prediction_df=prediction_df,
                pixel_metrics_df=pixel_metrics_df,
                time_metrics_df=time_metrics_df,
                target_name=config.target_name,
                plot_config=config.plots,
            )

        if mlflow:
            log_metrics_to_mlflow(mlflow, "test", global_metrics)
            log_metrics_to_mlflow(mlflow, "residual", residual_summary)
            log_metrics_to_mlflow(mlflow, "pixel_summary", pixel_metrics_summary)
            log_metrics_to_mlflow(mlflow, "model_monitoring", monitoring["scalar_metrics"])
            log_directory_to_mlflow(mlflow, metrics_dir, artifact_root="metrics")
            log_directory_to_mlflow(mlflow, figures_dir, artifact_root="figures")
            if hasattr(mlflow, "sklearn"):
                mlflow.sklearn.log_model(train_result["model"], artifact_path="model")

        model_summary = {
            "run_name": model_run.run_name,
            "model_name": model_run.model_name,
            "global_metrics": global_metrics,
            "residual_summary": residual_summary,
            "pixel_metrics_summary": pixel_metrics_summary,
            "artifacts": {
                "model_dir": str(model_dir),
                "report_dir": str(report_dir),
            },
        }
        save_json(report_dir / "model_run_summary.json", model_summary)

        return model_summary


def configure_mlflow(config: MlflowTrackingConfig):
    """
    Inicializa MLflow si está habilitado.
    """

    if not config.enabled:
        return None

    try:
        import mlflow
    except ImportError as exc:
        raise RuntimeError(
            "MLflow está habilitado en la configuración, pero no está instalado."
        ) from exc

    if config.tracking_uri:
        mlflow.set_tracking_uri(config.tracking_uri)
    mlflow.set_experiment(config.experiment_name)
    return mlflow


def load_variable_from_source(
    source: VariableSource,
    variable_name: str,
    start_year: int | None,
    end_year_inclusive: int | None,
    dtype: str,
    roi: dict[str, float] | None,
) -> xr.DataArray:
    """
    Carga una variable desde un fichero explícito y la normaliza a
    dims `(time, latitude, longitude)`.
    """

    path = Path(source.path)
    if not path.exists():
        raise FileNotFoundError(f"No existe el fichero de la variable '{variable_name}': {path}")

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_DATA_EXTENSIONS:
        raise ValueError(
            f"Formato no soportado para '{variable_name}': {path.suffix}. "
            f"Usa uno de {sorted(SUPPORTED_DATA_EXTENSIONS)}."
        )

    if suffix == ".zarr":
        ds = xr.open_zarr(path)
    else:
        ds = xr.open_dataset(path)

    data_var = source.data_var or variable_name
    if data_var not in ds.data_vars:
        available = list(ds.data_vars)
        if len(available) == 1:
            data_var = available[0]
        else:
            raise ValueError(
                f"No existe la variable '{data_var}' dentro de {path}. Disponibles: {available}"
            )

    da = ds[data_var].rename(variable_name)
    da = normalize_dataarray(da, variable_name)

    if start_year is not None:
        da = da.where(da["time"].dt.year >= start_year, drop=True)
    if end_year_inclusive is not None:
        da = da.where(da["time"].dt.year <= end_year_inclusive, drop=True)
    if da.sizes["time"] == 0:
        raise ValueError(
            f"La variable '{variable_name}' se quedó sin datos tras filtrar por años."
        )

    if roi:
        da = apply_roi(da, roi)

    da = da.astype(dtype)
    da = da.sortby("time").sortby("latitude").sortby("longitude")
    return da


def normalize_dataarray(da: xr.DataArray, variable_name: str) -> xr.DataArray:
    """
    Renombra dims/coordenadas habituales para trabajar siempre con
    `time`, `latitude` y `longitude`.
    """

    rename_map: dict[str, str] = {}
    for old_name, new_name in {
        "lat": "latitude",
        "latitude": "latitude",
        "y": "latitude",
        "lon": "longitude",
        "longitude": "longitude",
        "x": "longitude",
        "time": "time",
    }.items():
        if old_name in da.dims and old_name != new_name:
            rename_map[old_name] = new_name
        if old_name in da.coords and old_name != new_name:
            rename_map[old_name] = new_name

    if rename_map:
        da = da.rename(rename_map)

    required_dims = ("time", "latitude", "longitude")
    if tuple(da.dims) != required_dims:
        raise ValueError(
            f"La variable '{variable_name}' debe tener dims {required_dims}. "
            f"Recibido: {da.dims}"
        )

    da["time"] = pd.to_datetime(da["time"].values)
    return da


def aggregate_temporally(
    da: xr.DataArray,
    temporal_resolution: str,
) -> xr.DataArray:
    """
    Agrega la serie temporal a resolución mensual o anual.
    """

    if temporal_resolution == "monthly":
        aggregated = da.resample(time="MS").mean(skipna=True)
        return aggregated.rename(da.name)

    if temporal_resolution == "annual":
        grouped = da.groupby("time.year").mean(dim="time", skipna=True)
        years = grouped["year"].values
        aggregated = grouped.rename({"year": "time"}).assign_coords(
            time=pd.to_datetime([f"{int(year)}-01-01" for year in years])
        )
        return aggregated.rename(da.name)

    raise ValueError(
        f"temporal_resolution no soportada: {temporal_resolution}"
    )


def align_to_reference(
    da: xr.DataArray,
    reference: xr.DataArray,
    variable_name: str,
) -> xr.DataArray:
    """
    Comprueba que una variable comparte exactamente la misma rejilla y eje
    temporal que la referencia.
    """

    if da.shape != reference.shape:
        raise ValueError(
            f"La variable '{variable_name}' no coincide con la referencia. "
            f"{da.shape} != {reference.shape}"
        )

    for coord_name in ("time", "latitude", "longitude"):
        if not np.array_equal(da[coord_name].values, reference[coord_name].values):
            raise ValueError(
                f"La variable '{variable_name}' no comparte la coordenada '{coord_name}' "
                "con la referencia."
            )

    return da


def apply_roi(
    da: xr.DataArray,
    roi: dict[str, float],
) -> xr.DataArray:
    """
    Aplica un recorte rectangular opcional.

    ROI esperado:
    {
        "latitude_min": ...,
        "latitude_max": ...,
        "longitude_min": ...,
        "longitude_max": ...,
    }
    """

    lat_min = roi.get("latitude_min", float(da.latitude.min()))
    lat_max = roi.get("latitude_max", float(da.latitude.max()))
    lon_min = roi.get("longitude_min", float(da.longitude.min()))
    lon_max = roi.get("longitude_max", float(da.longitude.max()))

    da = da.sel(
        latitude=slice(min(lat_min, lat_max), max(lat_min, lat_max)),
        longitude=slice(min(lon_min, lon_max), max(lon_min, lon_max)),
    )
    return da


def load_masks(
    mask_sources: dict[str, MaskSource],
    reference: xr.DataArray,
) -> dict[str, xr.DataArray]:
    """
    Carga todas las máscaras declaradas y las alinea a la rejilla de referencia.
    """

    masks: dict[str, xr.DataArray] = {}
    for mask_name, mask_source in mask_sources.items():
        LOGGER.info("Cargando máscara '%s' desde %s", mask_name, mask_source.path)
        mask_da = load_single_mask(mask_name, mask_source, reference)
        masks[mask_name] = mask_da
    return masks


def load_single_mask(
    mask_name: str,
    source: MaskSource,
    reference: xr.DataArray,
) -> xr.DataArray:
    """
    Carga una máscara `.npy` o NetCDF y la normaliza a dims 2D.
    """

    path = Path(source.path)
    if not path.exists():
        raise FileNotFoundError(f"No existe la máscara '{mask_name}': {path}")

    if path.suffix.lower() == ".npy":
        values = np.load(path)
        if values.shape != (reference.sizes["latitude"], reference.sizes["longitude"]):
            raise ValueError(
                f"La máscara '{mask_name}' tiene shape {values.shape} y no coincide "
                f"con la referencia {(reference.sizes['latitude'], reference.sizes['longitude'])}."
            )
        mask_da = xr.DataArray(
            values,
            coords={
                "latitude": reference["latitude"].values,
                "longitude": reference["longitude"].values,
            },
            dims=("latitude", "longitude"),
            name=mask_name,
        )
    else:
        if path.suffix.lower() == ".zarr":
            ds = xr.open_zarr(path)
        else:
            ds = xr.open_dataset(path)

        data_var = next(iter(ds.data_vars))
        mask_da = ds[data_var].rename(mask_name)

        rename_map: dict[str, str] = {}
        if "lat" in mask_da.dims:
            rename_map["lat"] = "latitude"
        if "lon" in mask_da.dims:
            rename_map["lon"] = "longitude"
        if rename_map:
            mask_da = mask_da.rename(rename_map)

        if tuple(mask_da.dims) != ("latitude", "longitude"):
            raise ValueError(
                f"La máscara '{mask_name}' debe ser 2D con dims ('latitude', 'longitude'). "
                f"Recibido: {mask_da.dims}"
            )

        mask_da = mask_da.sortby("latitude").sortby("longitude")
        if mask_da.shape != (reference.sizes["latitude"], reference.sizes["longitude"]):
            raise ValueError(
                f"La máscara '{mask_name}' no coincide con la shape espacial de la referencia."
            )
        if not np.allclose(mask_da["latitude"].values, reference["latitude"].values):
            raise ValueError(
                f"La máscara '{mask_name}' no comparte la coordenada latitude con la referencia."
            )
        if not np.allclose(mask_da["longitude"].values, reference["longitude"].values):
            raise ValueError(
                f"La máscara '{mask_name}' no comparte la coordenada longitude con la referencia."
            )
        mask_da = mask_da.assign_coords(
            latitude=reference["latitude"].values,
            longitude=reference["longitude"].values,
        )

    if source.kind == "boolean":
        if source.keep_values is not None:
            bool_values = np.isin(mask_da.values, np.asarray(source.keep_values))
        else:
            bool_values = np.isfinite(mask_da.values) & (mask_da.values != 0)
        return xr.DataArray(
            bool_values,
            coords=mask_da.coords,
            dims=mask_da.dims,
            name=mask_name,
        )

    return mask_da


def build_combined_filter_mask(
    reference: xr.DataArray,
    masks: dict[str, xr.DataArray],
) -> tuple[xr.DataArray | None, dict[str, Any]]:
    """
    Construye una máscara combinada AND a partir de las máscaras booleanas activas.
    """

    if not masks:
        return None, {"active_masks": [], "combined_true_pixels": None}

    combined = np.ones(
        (reference.sizes["latitude"], reference.sizes["longitude"]),
        dtype=bool,
    )
    per_mask_summary: list[dict[str, Any]] = []

    for mask_name, mask_da in masks.items():
        values = np.asarray(mask_da.values).astype(bool)
        combined &= values
        per_mask_summary.append(
            {
                "mask_name": mask_name,
                "true_pixels": int(values.sum()),
                "false_pixels": int((~values).sum()),
                "true_fraction": float(values.mean()),
            }
        )

    combined_da = xr.DataArray(
        combined,
        coords={
            "latitude": reference["latitude"].values,
            "longitude": reference["longitude"].values,
        },
        dims=("latitude", "longitude"),
        name="combined_filter_mask",
    )
    mask_info = {
        "active_masks": list(masks.keys()),
        "per_mask_summary": per_mask_summary,
        "combined_true_pixels": int(combined.sum()),
        "combined_true_fraction": float(combined.mean()),
    }
    return combined_da, mask_info


def load_saved_split(
    split_dir: Path,
    prefix: str,
    reference: xr.DataArray,
) -> dict[str, Any]:
    """
    Reconstruye el split guardado desde disco.
    """

    def make_mask(name: str) -> xr.DataArray:
        path = split_dir / f"{prefix}_{name}.npy"
        return xr.DataArray(
            np.load(path).astype(bool),
            coords=reference.coords,
            dims=reference.dims,
            name=name,
        )

    metadata = json.loads((split_dir / f"{prefix}_split_metadata.json").read_text())
    bundle: dict[str, Any] = {
        "train_mask": make_mask("train_mask"),
        "test_mask": make_mask("test_mask"),
        "metadata": metadata,
    }

    for optional_name in (
        "selected_pixel_mask",
        "eligible_pixel_mask",
        "valid_pixel_mask",
    ):
        path = split_dir / f"{prefix}_{optional_name}.npy"
        if path.exists():
            bundle[optional_name] = xr.DataArray(
                np.load(path).astype(bool),
                coords=reference.coords,
                dims=reference.dims,
                name=optional_name,
            )

    return bundle


def split_prefix(config: ExperimentConfig) -> str:
    """
    Prefijo estable de los artefactos de split.
    """

    assert config.split is not None
    return config.split.prefix or config.split.category_mask_name


def summarize_dataarray(da: xr.DataArray) -> dict[str, Any]:
    """
    Resume una variable para logging y control.
    """

    return {
        "name": da.name,
        "shape": [int(size) for size in da.shape],
        "dtype": str(da.dtype),
        "time_start": str(pd.to_datetime(da["time"].values[0])),
        "time_end": str(pd.to_datetime(da["time"].values[-1])),
        "latitude_size": int(da.sizes["latitude"]),
        "longitude_size": int(da.sizes["longitude"]),
        "nan_fraction": float(np.isnan(da.values).mean()),
        "mean": safe_nanmean(da.values),
        "std": safe_nanstd(da.values),
    }


def save_dataarray_netcdf(da: xr.DataArray, path: Path) -> None:
    """
    Guarda una variable procesada para inspección posterior.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    da.to_netcdf(path)


def compute_group_regression_metrics(
    prediction_df: pd.DataFrame,
    group_column: str,
) -> pd.DataFrame:
    """
    Calcula métricas por grupo para capturar variación temporal o categórica.
    """

    if group_column not in prediction_df.columns:
        raise ValueError(f"No existe la columna de agrupación '{group_column}'.")

    rows: list[dict[str, Any]] = []
    for group_value, sub in prediction_df.groupby(group_column, sort=True):
        y_true = sub["y_true"].to_numpy()
        y_pred = sub["y_pred"].to_numpy()
        row = {
            group_column: group_value,
            "n_samples": int(len(sub)),
            "r2": safe_r2(y_true, y_pred),
            "rmse": safe_rmse(y_true, y_pred),
            "mae": safe_mae(y_true, y_pred),
            "bias": float(np.mean(y_pred - y_true)) if len(sub) else np.nan,
        }
        rows.append(row)

    return pd.DataFrame(rows)


def compute_group_regression_metrics_from_time(
    prediction_df: pd.DataFrame,
    group_name: str,
    extractor,
) -> pd.DataFrame:
    """
    Conveniencia para derivar agrupaciones desde la columna temporal.
    """

    df = prediction_df.copy()
    df[group_name] = extractor(df)
    return compute_group_regression_metrics(df, group_name)


def compute_category_metrics(
    prediction_df: pd.DataFrame,
    category_mask: xr.DataArray,
    labels: dict[int, str] | None,
    output_column_prefix: str,
) -> pd.DataFrame:
    """
    Calcula métricas por categoría espacial usando una máscara categórica.
    """

    df = prediction_df.copy()
    codes = category_mask.values[
        df["lat_idx"].to_numpy().astype(int),
        df["lon_idx"].to_numpy().astype(int),
    ]
    df[f"{output_column_prefix}_code"] = codes
    metrics_df = compute_group_regression_metrics(df, f"{output_column_prefix}_code")
    if labels:
        metrics_df[f"{output_column_prefix}_label"] = metrics_df[
            f"{output_column_prefix}_code"
        ].map(lambda value: labels.get(int(value), f"class_{int(value)}"))
    return metrics_df


def summarize_residuals(prediction_df: pd.DataFrame) -> dict[str, Any]:
    """
    Resume la distribución de residuos.
    """

    residuals = prediction_df["residual"].to_numpy()
    return {
        "mean": float(np.mean(residuals)),
        "median": float(np.median(residuals)),
        "std": float(np.std(residuals)),
        "abs_mean": float(np.mean(np.abs(residuals))),
        "p05": float(np.quantile(residuals, 0.05)),
        "p95": float(np.quantile(residuals, 0.95)),
    }


def safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    R2 robusto a grupos muy pequeños o sin varianza.
    """

    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    if len(y_true) < 2 or np.allclose(np.var(y_true), 0.0):
        return np.nan

    centered = y_true - y_true.mean()
    denominator = float(np.sum(centered ** 2))
    if denominator == 0.0:
        return np.nan

    residual = y_true - y_pred
    numerator = float(np.sum(residual ** 2))
    return float(1.0 - numerator / denominator)


def safe_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    RMSE seguro para grupos potencialmente vacíos.
    """

    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    if len(y_true) == 0:
        return np.nan
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def safe_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    MAE seguro para grupos potencialmente vacíos.
    """

    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    if len(y_true) == 0:
        return np.nan
    return float(np.mean(np.abs(y_true - y_pred)))


def safe_nanmean(values: np.ndarray) -> float:
    """
    Media ignorando NaN sin lanzar warnings cuando todo es NaN.
    """

    finite = np.isfinite(values)
    if not finite.any():
        return np.nan
    return float(np.nanmean(values))


def safe_nanstd(values: np.ndarray) -> float:
    """
    Desviación típica ignorando NaN sin lanzar warnings cuando todo es NaN.
    """

    finite = np.isfinite(values)
    if not finite.any():
        return np.nan
    return float(np.nanstd(values))


def collect_model_monitoring_artifacts(
    model,
    feature_names: list[str],
    output_dir: Path,
    top_feature_count: int,
) -> dict[str, Any]:
    """
    Extrae señales intermedias del modelo y las guarda como artefactos.

    Se registran solo atributos disponibles en el estimador concreto.
    """

    scalar_metrics: dict[str, float] = {}

    if hasattr(model, "oob_score_") and model.oob_score_ is not None:
        scalar_metrics["oob_score"] = float(model.oob_score_)

    if hasattr(model, "best_validation_score_") and model.best_validation_score_ is not None:
        scalar_metrics["best_validation_score"] = float(model.best_validation_score_)

    if hasattr(model, "n_iter_"):
        scalar_metrics["n_iter"] = float(model.n_iter_)

    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_)
        feature_df = pd.DataFrame(
            {
                "feature": feature_names,
                "importance": importances.astype(float),
            }
        ).sort_values("importance", ascending=False)
        save_dataframe(feature_df, output_dir / "feature_importances.csv")

        fig, ax = plt.subplots(figsize=(10, 5))
        feature_df.head(top_feature_count).plot(
            kind="bar",
            x="feature",
            y="importance",
            legend=False,
            ax=ax,
            color="#4477AA",
        )
        ax.set_title("Top feature importances")
        ax.set_xlabel("Feature")
        ax.set_ylabel("Importance")
        fig.tight_layout()
        save_figure(fig, output_dir / "feature_importances.png")

    for attribute_name in ("loss_curve_", "validation_scores_", "train_score_", "validation_score_"):
        if not hasattr(model, attribute_name):
            continue
        values = getattr(model, attribute_name)
        if values is None:
            continue
        values_array = np.asarray(values).reshape(-1)
        if values_array.size == 0:
            continue

        curve_df = pd.DataFrame(
            {
                "iteration": np.arange(1, values_array.size + 1, dtype=int),
                "value": values_array.astype(float),
            }
        )
        save_dataframe(curve_df, output_dir / f"{attribute_name}.csv")

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(curve_df["iteration"], curve_df["value"], color="#CC6677")
        ax.set_title(attribute_name)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Value")
        fig.tight_layout()
        save_figure(fig, output_dir / f"{attribute_name}.png")

    return {"scalar_metrics": scalar_metrics}


def create_evaluation_figures(
    output_dir: Path,
    model,
    prediction_df: pd.DataFrame,
    pixel_metrics_df: pd.DataFrame,
    time_metrics_df: pd.DataFrame,
    target_name: str,
    plot_config: PlotConfig,
) -> None:
    """
    Genera figuras de evaluación para comparar variantes de modelo.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    save_scatter_figure(
        prediction_df,
        output_dir / "test_scatter.png",
        plot_config.scatter_sample_size,
        target_name,
    )
    save_residual_histogram(prediction_df, output_dir / "residual_histogram.png")
    save_time_metrics_figure(time_metrics_df, output_dir / "time_metrics.png")
    save_pixel_metric_map(pixel_metrics_df, output_dir / "pixel_r2_map.png")
    save_example_pixel_timeseries(
        prediction_df=prediction_df,
        pixel_metrics_df=pixel_metrics_df,
        output_path=output_dir / "example_pixel_timeseries.png",
        n_example_pixels=plot_config.n_example_pixels,
        target_name=target_name,
    )

    if hasattr(model, "feature_importances_"):
        LOGGER.info("Se han generado artefactos de feature importance para el modelo.")


def save_preprocessing_figures(
    config: ExperimentConfig,
    output_dir: Path,
    raw_data: dict[str, xr.DataArray],
    processed_data: dict[str, xr.DataArray],
    masks: dict[str, xr.DataArray],
    combined_mask: xr.DataArray | None,
) -> None:
    """
    Guarda visualizaciones del preprocesado para facilitar la auditoría.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    save_mask_figure(masks, combined_mask, output_dir / "masks.png")
    save_missing_fraction_figure(processed_data, output_dir / "missing_fraction.png")
    save_target_summary_figure(
        raw_target=raw_data[config.target_name],
        processed_target=processed_data[config.target_name],
        output_path=output_dir / "target_summary.png",
        target_name=config.target_name,
    )
    save_correlation_figure(
        target=processed_data[config.target_name],
        predictors={name: processed_data[name] for name in config.predictor_names},
        output_path=output_dir / "predictor_correlations.png",
    )


def save_mask_figure(
    masks: dict[str, xr.DataArray],
    combined_mask: xr.DataArray | None,
    output_path: Path,
) -> None:
    """
    Dibuja todas las máscaras activas junto con la máscara combinada.
    """

    items = list(masks.items())
    if combined_mask is not None:
        items.append(("combined_filter", combined_mask))

    n_items = len(items)
    if n_items == 0:
        return

    fig, axes = plt.subplots(1, n_items, figsize=(5 * n_items, 4), squeeze=False)
    for ax, (name, mask_da) in zip(axes.ravel(), items):
        image = ax.imshow(mask_da.values, aspect="auto", cmap="viridis")
        ax.set_title(name)
        ax.set_xlabel("longitude index")
        ax.set_ylabel("latitude index")
        fig.colorbar(image, ax=ax, shrink=0.8)

    fig.tight_layout()
    save_figure(fig, output_path)


def save_missing_fraction_figure(
    processed_data: dict[str, xr.DataArray],
    output_path: Path,
) -> None:
    """
    Guarda un mapa de fracción de missing por variable.
    """

    n_items = len(processed_data)
    cols = min(3, n_items)
    rows = int(np.ceil(n_items / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows), squeeze=False)

    for ax, (name, da) in zip(axes.ravel(), processed_data.items()):
        missing_fraction = da.isnull().mean(dim="time")
        image = ax.imshow(missing_fraction.values, aspect="auto", cmap="magma", vmin=0.0, vmax=1.0)
        ax.set_title(f"Missing fraction | {name}")
        fig.colorbar(image, ax=ax, shrink=0.8)

    for ax in axes.ravel()[n_items:]:
        ax.axis("off")

    fig.tight_layout()
    save_figure(fig, output_path)


def save_target_summary_figure(
    raw_target: xr.DataArray,
    processed_target: xr.DataArray,
    output_path: Path,
    target_name: str,
) -> None:
    """
    Resume visualmente el target antes y después del filtrado.
    """

    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    raw_mean = raw_target.mean(dim="time", skipna=True)
    processed_mean = processed_target.mean(dim="time", skipna=True)
    temporal_mean = processed_target.mean(dim=("latitude", "longitude"), skipna=True)

    image0 = axes[0].imshow(raw_mean.values, aspect="auto", cmap="viridis")
    axes[0].set_title(f"{target_name} mean | raw")
    fig.colorbar(image0, ax=axes[0], shrink=0.8)

    image1 = axes[1].imshow(processed_mean.values, aspect="auto", cmap="viridis")
    axes[1].set_title(f"{target_name} mean | filtrado")
    fig.colorbar(image1, ax=axes[1], shrink=0.8)

    axes[2].plot(pd.to_datetime(temporal_mean["time"].values), temporal_mean.values, color="#4477AA")
    axes[2].set_title(f"{target_name} media espacial")
    axes[2].set_xlabel("time")
    axes[2].set_ylabel(target_name)

    fig.tight_layout()
    save_figure(fig, output_path)


def save_correlation_figure(
    target: xr.DataArray,
    predictors: dict[str, xr.DataArray],
    output_path: Path,
) -> None:
    """
    Dibuja la correlación global target-predictor tras el preprocesado.
    """

    rows = []
    target_values = target.values.reshape(-1)
    for name, predictor in predictors.items():
        predictor_values = predictor.values.reshape(-1)
        valid = np.isfinite(target_values) & np.isfinite(predictor_values)
        if valid.sum() < 2:
            corr = np.nan
        else:
            corr = float(np.corrcoef(target_values[valid], predictor_values[valid])[0, 1])
        rows.append({"predictor": name, "correlation": corr})

    corr_df = pd.DataFrame(rows).sort_values("correlation", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(corr_df["predictor"], corr_df["correlation"], color="#228833")
    ax.set_title("Correlación global predictor-target")
    ax.set_xlabel("Predictor")
    ax.set_ylabel("Correlation")
    fig.tight_layout()
    save_figure(fig, output_path)


def save_split_figure(
    output_path: Path,
    category_mask: xr.DataArray,
    split_bundle: dict[str, Any],
    category_labels: dict[int, str] | None,
) -> None:
    """
    Dibuja la máscara categórica y el reparto selected/train/test.
    """

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    image0 = axes[0, 0].imshow(category_mask.values, aspect="auto", cmap="tab20")
    axes[0, 0].set_title("Category mask")
    fig.colorbar(image0, ax=axes[0, 0], shrink=0.8)

    if "selected_pixel_mask" in split_bundle:
        image1 = axes[0, 1].imshow(split_bundle["selected_pixel_mask"].values, aspect="auto", cmap="Greens")
        axes[0, 1].set_title("Selected pixels")
        fig.colorbar(image1, ax=axes[0, 1], shrink=0.8)
    else:
        axes[0, 1].axis("off")

    image2 = axes[1, 0].imshow(split_bundle["train_mask"].values, aspect="auto", cmap="Blues")
    axes[1, 0].set_title("Train mask")
    fig.colorbar(image2, ax=axes[1, 0], shrink=0.8)

    image3 = axes[1, 1].imshow(split_bundle["test_mask"].values, aspect="auto", cmap="Oranges")
    axes[1, 1].set_title("Test mask")
    fig.colorbar(image3, ax=axes[1, 1], shrink=0.8)

    if category_labels:
        labels_text = "\n".join(
            f"{code}: {label}"
            for code, label in sorted(category_labels.items())
        )
        fig.text(0.99, 0.02, labels_text, ha="right", va="bottom", fontsize=8)

    fig.tight_layout()
    save_figure(fig, output_path)


def save_scatter_figure(
    prediction_df: pd.DataFrame,
    output_path: Path,
    sample_size: int,
    target_name: str,
) -> None:
    """
    Scatter/hexbin real vs predicho.
    """

    sample_df = sample_dataframe(prediction_df, sample_size)
    fig, ax = plt.subplots(figsize=(6, 6))

    if len(sample_df) >= 500:
        hb = ax.hexbin(
            sample_df["y_true"].values,
            sample_df["y_pred"].values,
            gridsize=35,
            mincnt=1,
            cmap="viridis",
        )
        fig.colorbar(hb, ax=ax, shrink=0.8)
    else:
        ax.scatter(
            sample_df["y_true"].values,
            sample_df["y_pred"].values,
            s=10,
            alpha=0.5,
            color="#4477AA",
        )

    xy_min = min(sample_df["y_true"].min(), sample_df["y_pred"].min())
    xy_max = max(sample_df["y_true"].max(), sample_df["y_pred"].max())
    ax.plot([xy_min, xy_max], [xy_min, xy_max], linestyle="--", color="black")
    ax.set_xlabel(f"{target_name} real")
    ax.set_ylabel(f"{target_name} predicho")
    ax.set_title("Test scatter")
    fig.tight_layout()
    save_figure(fig, output_path)


def save_residual_histogram(
    prediction_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Histograma de residuos.
    """

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(prediction_df["residual"].values, bins=50, color="#CC6677", alpha=0.85)
    ax.axvline(0.0, color="black", linestyle="--")
    ax.set_title("Residual histogram")
    ax.set_xlabel("Residual")
    ax.set_ylabel("Count")
    fig.tight_layout()
    save_figure(fig, output_path)


def save_time_metrics_figure(
    time_metrics_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Evolución temporal de R2/RMSE.
    """

    if time_metrics_df.empty:
        return

    time_metrics_df = time_metrics_df.sort_values("time")
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    time_values = pd.to_datetime(time_metrics_df["time"])
    axes[0].plot(time_values, time_metrics_df["r2"], color="#4477AA")
    axes[0].set_ylabel("R2")
    axes[0].set_title("Métricas por instante temporal")

    axes[1].plot(time_values, time_metrics_df["rmse"], color="#CC6677")
    axes[1].set_ylabel("RMSE")
    axes[1].set_xlabel("time")

    fig.tight_layout()
    save_figure(fig, output_path)


def save_pixel_metric_map(
    pixel_metrics_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Mapa disperso del R2 por píxel.
    """

    if pixel_metrics_df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    scatter = ax.scatter(
        pixel_metrics_df["longitude"],
        pixel_metrics_df["latitude"],
        c=pixel_metrics_df["r2"],
        s=12,
        cmap="viridis",
    )
    ax.set_title("R2 por píxel")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    fig.colorbar(scatter, ax=ax, shrink=0.8)
    fig.tight_layout()
    save_figure(fig, output_path)


def save_example_pixel_timeseries(
    prediction_df: pd.DataFrame,
    pixel_metrics_df: pd.DataFrame,
    output_path: Path,
    n_example_pixels: int,
    target_name: str,
) -> None:
    """
    Dibuja series temporales de algunos píxeles buenos y malos.
    """

    if pixel_metrics_df.empty:
        return

    ranking = rank_best_and_worst_pixels(
        pixel_metrics_df=pixel_metrics_df,
        metric="r2",
        top_k=max(1, n_example_pixels // 2),
    )
    selected = pd.concat(
        [
            ranking["best"].assign(group="best"),
            ranking["worst"].assign(group="worst"),
        ],
        ignore_index=True,
    )

    if selected.empty:
        return

    cols = min(3, len(selected))
    rows = int(np.ceil(len(selected) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 3.8 * rows), squeeze=False)

    for ax, (_, row) in zip(axes.ravel(), selected.iterrows()):
        ts_df = build_pixel_timeseries_dataframe(
            prediction_df=prediction_df,
            pixel_id=int(row["pixel_id"]),
        )
        ax.plot(ts_df["time"], ts_df["y_true"], label="real", color="#4477AA")
        ax.plot(ts_df["time"], ts_df["y_pred"], label="predicho", color="#CC6677")
        ax.set_title(f"{row['group']} | pixel {int(row['pixel_id'])} | R2={row['r2']:.3f}")
        ax.set_xlabel("time")
        ax.set_ylabel(target_name)
        ax.legend()

    for ax in axes.ravel()[len(selected):]:
        ax.axis("off")

    fig.tight_layout()
    save_figure(fig, output_path)


def sample_dataframe(df: pd.DataFrame, sample_size: int) -> pd.DataFrame:
    """
    Submuestrea filas solo para graficar.
    """

    if len(df) <= sample_size:
        return df
    return df.sample(sample_size, random_state=42)


def save_figure(fig, path: Path) -> None:
    """
    Guarda y libera una figura.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_json(path: Path, payload: Any) -> None:
    """
    Guarda un objeto en JSON con conversión segura de tipos.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_to_jsonable(payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    """
    Guarda una tabla CSV.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def flatten_params(prefix: str, payload: Any) -> dict[str, str | float | int | bool]:
    """
    Convierte dataclasses/dicts en pares clave-valor aptos para MLflow params.
    """

    flat: dict[str, str | float | int | bool] = {}

    def visit(current_prefix: str, value: Any) -> None:
        if is_dataclass(value):
            visit(current_prefix, asdict(value))
            return
        if isinstance(value, dict):
            for key, inner_value in value.items():
                visit(f"{current_prefix}.{key}", inner_value)
            return
        if isinstance(value, (list, tuple, set)):
            flat[current_prefix] = json.dumps(_to_jsonable(value), ensure_ascii=False)
            return
        if isinstance(value, Path):
            flat[current_prefix] = str(value)
            return
        if isinstance(value, (str, int, float, bool)) or value is None:
            flat[current_prefix] = "None" if value is None else value
            return
        flat[current_prefix] = str(value)

    visit(prefix, payload)
    return flat


def log_metrics_to_mlflow(mlflow, prefix: str, metrics: dict[str, Any]) -> None:
    """
    Envía solo métricas escalares válidas a MLflow.
    """

    scalar_metrics: dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, (np.integer, int)):
            scalar_metrics[f"{prefix}.{key}"] = int(value)
        elif isinstance(value, (np.floating, float)):
            if np.isfinite(value):
                scalar_metrics[f"{prefix}.{key}"] = float(value)
    if scalar_metrics:
        mlflow.log_metrics(scalar_metrics)


def log_directory_to_mlflow(mlflow, directory: Path, artifact_root: str) -> None:
    """
    Registra todos los ficheros de un directorio como artefactos.
    """

    if not directory.exists():
        return

    for path in sorted(directory.rglob("*")):
        if path.is_file():
            relative_parent = path.parent.relative_to(directory)
            artifact_path = (
                artifact_root
                if str(relative_parent) == "."
                else f"{artifact_root}/{relative_parent.as_posix()}"
            )
            mlflow.log_artifact(str(path), artifact_path=artifact_path)


def cleanup_intermediate_artifacts(layout: dict[str, Path]) -> None:
    """
    Elimina solo los artefactos temporales del workflow.
    """

    for key in ("processed_dir", "tabular_dir"):
        path = layout[key]
        if path.exists():
            LOGGER.info("Eliminando intermedios temporales en %s", path)
            shutil.rmtree(path)


def _to_jsonable(value: Any) -> Any:
    """
    Convierte tipos de numpy/pandas/dataclasses a algo serializable.
    """

    if is_dataclass(value):
        return _to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _to_jsonable(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(inner) for inner in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Timestamp):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


__all__ = [
    "ExperimentConfig",
    "ExperimentPaths",
    "MaskSource",
    "MlflowTrackingConfig",
    "ModelRunConfig",
    "PlotConfig",
    "SpatialSplitConfig",
    "VariableSource",
    "run_experiment",
    "validate_experiment_config",
]
