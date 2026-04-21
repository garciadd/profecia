#!/usr/bin/env python3

"""
Workflow configurable para preparar datos, crear/reutilizar el split espacial,
entrenar varias variantes de modelo y registrar artefactos en disco y MLflow.

Este fichero está pensado para editarse directamente, sin flags.

Cómo usarlo:
1. Ajusta las rutas de datos raw y máscaras en la sección "Configuración editable".
2. Define una o varias entradas dentro de `EXPERIMENTS`.
3. Para cada experimento puedes cambiar:
   - rutas de entrada/salida
   - resolución temporal y rango de años
   - variables predictoras
   - máscaras de filtrado
   - máscara categórica del split
   - modelos e hiperparámetros
4. Ejecuta el script y revisa logs, figuras, métricas, modelos y artefactos MLflow.

La intención es que cada `ExperimentConfig` sea autocontenido y reproducible.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.workflow import (
    ExperimentConfig,
    ExperimentPaths,
    MaskSource,
    MlflowTrackingConfig,
    ModelRunConfig,
    PlotConfig,
    SpatialSplitConfig,
    VariableSource,
    run_experiment,
    validate_experiment_config,
)


LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuración editable
# ---------------------------------------------------------------------------

# Cambia estas rutas base para apuntar a tus datos reales.
RAW_DATA_ROOT = "/srv/read_only_data/final_datasets/monthly"
MASK_ROOT = "/srv/volumes/ferag/masks"
WORKFLOW_ROOT = "/srv/volumes/ferag/workflow_runs"

# Si prefieres que siga con el siguiente experimento, usa False.
STOP_ON_ERROR = True

# Configuración editable de MLflow
# ---------------------------------------------------------------------------
#
# Edita estos valores directamente aquí. El script los exporta a variables de
# entorno para que MLflow las use durante la ejecución.
#
# Si algún valor se deja como cadena vacía, no se exporta.
MLFLOW_TRACKING_URI = "https://mlflow.cloud.ai4eosc.eu/"
MLFLOW_TRACKING_USERNAME = "aguilarf@ifca.unican.es"
MLFLOW_TRACKING_PASSWORD = "saintGenis@17"


def export_mlflow_environment() -> None:
    """
    Exporta la configuración de MLflow a variables de entorno del proceso.

    Esto permite centralizar la edición en este fichero y mantener el resto del
    código desacoplado de credenciales embebidas.
    """

    if MLFLOW_TRACKING_URI.strip():
        os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_TRACKING_URI.strip()

    if MLFLOW_TRACKING_USERNAME.strip():
        os.environ["MLFLOW_TRACKING_USERNAME"] = MLFLOW_TRACKING_USERNAME.strip()

    if MLFLOW_TRACKING_PASSWORD.strip():
        os.environ["MLFLOW_TRACKING_PASSWORD"] = MLFLOW_TRACKING_PASSWORD.strip()


export_mlflow_environment()



def as_path(value: str | Path) -> Path:
    """
    Normaliza una ruta editable a `Path`.
    """

    return Path(value)


RAW_DATA_ROOT = as_path(RAW_DATA_ROOT)
MASK_ROOT = as_path(MASK_ROOT)
WORKFLOW_ROOT = as_path(WORKFLOW_ROOT)


def build_variable_sources(raw_root: str | Path) -> dict[str, VariableSource]:
    """
    Plantilla de fuentes raw.

    Ajusta estos ficheros según tu estructura real. Cada variable apunta a un
    fichero concreto para evitar lógica implícita dentro del workflow.
    """

    raw_root = Path(raw_root)
    available_monthly_files = {
        "d2m": "d2m_1982_2022_monthly_0.5deg.nc",
        "lai": "lai_1982_2022_monthly_0.5deg.nc",
        "pev": "pev_1982_2022_monthly_0.5deg.nc",
        "spei01": "spei01_1982_2022_monthly_0.5deg.nc",
        "spei02": "spei02_1982_2022_monthly_0.5deg.nc",
        "spei03": "spei03_1982_2022_monthly_0.5deg.nc",
        "spei06": "spei06_1982_2022_monthly_0.5deg.nc",
        "spei09": "spei09_1982_2022_monthly_0.5deg.nc",
        "spei12": "spei12_1982_2022_monthly_0.5deg.nc",
        "spei24": "spei24_1982_2022_monthly_0.5deg.nc",
        "ssrd": "ssrd_1982_2022_monthly_0.5deg.nc",
        "subswc": "subswc_1982_2022_monthly_0.5deg.nc",
        "swvl1": "swvl1_1982_2022_monthly_0.5deg.nc",
        "swvl2": "swvl2_1982_2022_monthly_0.5deg.nc",
        "swvl3": "swvl3_1982_2022_monthly_0.5deg.nc",
        "t2m": "t2m_1982_2022_monthly_0.5deg.nc",
        "tcc": "tcc_1982_2022_monthly_0.5deg.nc",
        "tp": "tp_1982_2022_monthly_0.5deg.nc",
        "u10": "u10_1982_2022_monthly_0.5deg.nc",
        "v10": "v10_1982_2022_monthly_0.5deg.nc",
        "vpd": "vpd_1982_2022_monthly_0.5deg.nc",
        "wind": "wind_1982_2022_monthly_0.5deg.nc",
    }

    sources = {
        variable_name: VariableSource(
            path=raw_root / file_name,
            data_var=variable_name,
        )
        for variable_name, file_name in available_monthly_files.items()
    }

    # Alias de compatibilidad para poder seguir configurando variables con los
    # nombres usados en notebooks previos.
    sources.update(
        {
            "LAI": sources["lai"],
            "SM1": sources["swvl1"],
            "SM2": sources["swvl2"],
            "SM3": sources["swvl3"],
            "TP": sources["tp"],
            "T2M": sources["t2m"],
            "SSRD": sources["ssrd"],
            "VPD": sources["vpd"],
            "TCC": sources["tcc"],
            "D2M": sources["d2m"],
            "U10": sources["u10"],
            "V10": sources["v10"],
            "WIND": sources["wind"],
            "PEV": sources["pev"],
            "SUBSWC": sources["subswc"],
        }
    )

    return sources


def build_mask_sources(mask_root: str | Path) -> dict[str, MaskSource]:
    """
    Plantilla de máscaras disponibles.

    Puedes activar cualquier subconjunto en `filter_mask_names` y elegir una
    máscara categórica distinta para el split espacial.
    """

    mask_root = Path(mask_root)

    return {
        "land": MaskSource(
            path=mask_root / "land_mask_0p5deg.npy",
            kind="boolean",
            description="Máscara binaria tierra",
        ),
        "ebf": MaskSource(
            path=mask_root / "ebf_mask_0p5deg.npy",
            kind="boolean",
            description="Evergreen broadleaf forest",
        ),
        "bs": MaskSource(
            path=mask_root / "bs_mask_0p5deg.npy",
            kind="boolean",
            description="Bare soil / desert",
        ),
        "climate": MaskSource(
            path=mask_root / "climate_mask_0p5_5classes.npy",
            kind="categorical",
            description="Clases climáticas",
            labels={
                1: "equatorial",
                2: "arid",
                3: "warm_temperate",
                4: "snow",
                5: "polar",
            },
        ),
        "landcover": MaskSource(
            path=mask_root / "landcover_mask_0p5_7classes.npy",
            kind="categorical",
            description="Clases de cobertura del suelo",
            labels={
                10: "cropland",
                20: "forest",
                30: "grassland",
                40: "shrubland",
                70: "tundra",
                90: "barren",
                100: "snow_ice",
            },
        ),
    }


COMMON_VARIABLE_SOURCES = build_variable_sources(RAW_DATA_ROOT)
COMMON_MASK_SOURCES = build_mask_sources(MASK_ROOT)


# Duplica y adapta estos bloques para lanzar N configuraciones distintas
# de datos/modelos una detrás de otra.
EXPERIMENTS: list[ExperimentConfig] = [
    
    ExperimentConfig(
        name="monthly_land_example",
        paths=ExperimentPaths(
            intermediate_dir=WORKFLOW_ROOT / "monthly_land_example" / "intermediate",
            split_dir=WORKFLOW_ROOT / "monthly_land_example" / "split",
            model_dir=WORKFLOW_ROOT / "monthly_land_example" / "models",
            report_dir=WORKFLOW_ROOT / "monthly_land_example" / "reports",
        ),
        raw_variables=COMMON_VARIABLE_SOURCES,
        target_name="LAI",
        predictor_names=["SM1", "SM2", "TP", "T2M", "SSRD", "VPD"],
        temporal_resolution="monthly",
        start_year=1982,
        end_year_inclusive=2022,
        dtype="float32",
        roi=None,
        mask_sources=COMMON_MASK_SOURCES,
        filter_mask_names=["land"],
        split=SpatialSplitConfig(
            category_mask_name="landcover",
            split_name="landcover_stratified_30pct",
            prefix="landcover",
            test_fraction=0.10,
            pixel_fraction=0.30,
            ignore_codes=(0,),
            min_valid_fraction=0.0,
            seed=42,
            subset_seed=42,
            reuse_saved_masks=True,
            reuse_exported_arrays=True,
        ),
        model_runs=[
            ModelRunConfig(
                run_name="rf_baseline",
                model_name="rf",
                scaler_name=None,
                random_state=42,
                params={
                    "n_estimators": 300,
                    "max_depth": 25,
                    "min_samples_leaf": 3,
                    "max_features": "sqrt",
                    "n_jobs": -1,
                    "bootstrap": True,
                },
            ),
            ModelRunConfig(
                run_name="rf_baseline",
                model_name="rf",
                scaler_name=None,
                random_state=42,
                params={
                    "n_estimators": 500,
                    "max_depth": 25,
                    "min_samples_leaf": 29,
                    "max_features": "sqrt",
                    "n_jobs": -1,
                    "bootstrap": True,
                },
            ),
            ModelRunConfig(
                run_name="rf_baseline",
                model_name="rf",
                scaler_name=None,
                random_state=42,
                params={
                    "n_estimators": 750,
                    "max_depth": 25,
                    "min_samples_leaf": 45,
                    "max_features": "sqrt",
                    "n_jobs": -1,
                    "bootstrap": True,
                },
            ),
            ModelRunConfig(
                run_name="hgb_depth8",
                model_name="hgb",
                scaler_name=None,
                random_state=42,
                params={
                    "learning_rate": 0.005,
                    "max_iter": 500,
                    "max_depth": 18,
                    "min_samples_leaf": 25,
                },
            ),
            ModelRunConfig(
                run_name="hgb_depth8",
                model_name="hgb",
                scaler_name=None,
                random_state=42,
                params={
                    "learning_rate": 0.05,
                    "max_iter": 100,
                    "max_depth": 18,
                    "min_samples_leaf": 25,
                },
            ),
        ],
        mlflow=MlflowTrackingConfig(
            enabled=True,
            tracking_uri=os.getenv("MLFLOW_TRACKING_URI"),
            experiment_name="profecia-lai",
            tags={
                "pipeline": "run_workflow_and_train.py",
                "target": "LAI",
            },
        ),
        plots=PlotConfig(
            enabled=True,
            scatter_sample_size=20_000,
            n_example_pixels=6,
            top_feature_count=15,
        ),
        lat_block_size=10,
        mmap_mode="r",
        save_processed_netcdf=True,
        cleanup_intermediate_after_models=False,
    ),
]


def configure_logging(workflow_root: Path) -> Path:
    """
    Configura logging a consola y a fichero.
    """

    workflow_root = Path(workflow_root)
    workflow_root.mkdir(parents=True, exist_ok=True)
    log_path = workflow_root / f"workflow_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.log"

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(stream_handler)
    root_logger.addHandler(file_handler)

    return log_path


def main() -> int:
    """
    Ejecuta todos los experimentos configurados secuencialmente.
    """

    log_path = configure_logging(WORKFLOW_ROOT)
    LOGGER.info("Log general: %s", log_path)
    LOGGER.info("Se han configurado %s experimento(s).", len(EXPERIMENTS))

    for experiment in EXPERIMENTS:
        validate_experiment_config(experiment)

    summaries = []
    failures: list[tuple[str, str]] = []

    for experiment in EXPERIMENTS:
        try:
            summary = run_experiment(experiment)
        except Exception as exc:
            LOGGER.exception("El experimento '%s' ha fallado.", experiment.name)
            failures.append((experiment.name, str(exc)))
            if STOP_ON_ERROR:
                return 1
        else:
            summaries.append(summary)

    summary_path = WORKFLOW_ROOT / "workflow_summary.json"
    summary_payload = {
        "executed_at_utc": datetime.now(UTC).isoformat(),
        "stop_on_error": STOP_ON_ERROR,
        "experiments_requested": [experiment.name for experiment in EXPERIMENTS],
        "experiments_completed": [summary["experiment_name"] for summary in summaries],
        "failures": [
            {"experiment_name": name, "error": error}
            for name, error in failures
        ],
    }
    summary_path.write_text(
        json.dumps(summary_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    LOGGER.info("Resumen global guardado en %s", summary_path)

    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
