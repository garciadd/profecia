from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# MÉTRICAS GLOBALES

def compute_global_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """
    Calcula métricas globales de regresión.

    Parameters
    ----------
    y_true : np.ndarray
        Valores reales.
    y_pred : np.ndarray
        Predicciones del modelo.

    Returns
    -------
    dict
        Diccionario con R2, RMSE y MAE globales.
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true y y_pred deben tener la misma shape. "
            f"Recibido: {y_true.shape} vs {y_pred.shape}"
        )

    if y_true.size == 0:
        raise ValueError("y_true está vacío.")

    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "n_samples": int(y_true.size),
    }


# HELPERS MÉTRICAS SEGURAS

def _safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calcula R2 de forma segura.
    Devuelve NaN si no hay suficientes muestras o si la varianza es cero.
    """
    if len(y_true) < 2:
        return np.nan

    if np.allclose(np.var(y_true), 0.0):
        return np.nan

    return float(r2_score(y_true, y_pred))


def _safe_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    RMSE seguro.
    """
    if len(y_true) == 0:
        return np.nan
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _safe_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    MAE seguro.
    """
    if len(y_true) == 0:
        return np.nan
    return float(mean_absolute_error(y_true, y_pred))


# DATAFRAME DE EVALUACIÓN FILA A FILA

def build_prediction_dataframe(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    pixel_id: np.ndarray,
    lat_idx: np.ndarray | None = None,
    lon_idx: np.ndarray | None = None,
    time_idx: np.ndarray | None = None,
    latitude_values: np.ndarray | None = None,
    longitude_values: np.ndarray | None = None,
    time_values: np.ndarray | pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """
    Construye un DataFrame fila a fila para evaluación.

    Cada fila representa una observación individual (time, pixel).

    Parameters
    ----------
    y_true : np.ndarray
        Valores reales.
    y_pred : np.ndarray
        Predicciones.
    pixel_id : np.ndarray
        Identificador de píxel por fila.
    lat_idx : np.ndarray | None
        Índice latitudinal por fila.
    lon_idx : np.ndarray | None
        Índice longitudinal por fila.
    time_idx : np.ndarray | None
        Índice temporal por fila.
    latitude_values : np.ndarray | None
        Vector 1D de coordenadas latitude del grid completo.
    longitude_values : np.ndarray | None
        Vector 1D de coordenadas longitude del grid completo.
    time_values : np.ndarray | pd.DatetimeIndex | None
        Vector 1D de timestamps del dataset completo.

    Returns
    -------
    pd.DataFrame
        DataFrame con columnas base:
        - y_true
        - y_pred
        - pixel_id

        y opcionalmente:
        - lat_idx
        - lon_idx
        - time_idx
        - latitude
        - longitude
        - time
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    pixel_id = np.asarray(pixel_id).reshape(-1)

    n = len(y_true)

    if y_pred.shape[0] != n or pixel_id.shape[0] != n:
        raise ValueError("y_true, y_pred y pixel_id deben tener la misma longitud.")

    data: dict[str, Any] = {
        "y_true": y_true,
        "y_pred": y_pred,
        "pixel_id": pixel_id.astype(np.int64),
    }

    if lat_idx is not None:
        lat_idx = np.asarray(lat_idx).reshape(-1)
        if len(lat_idx) != n:
            raise ValueError("lat_idx debe tener la misma longitud que y_true.")
        data["lat_idx"] = lat_idx.astype(np.int64)

    if lon_idx is not None:
        lon_idx = np.asarray(lon_idx).reshape(-1)
        if len(lon_idx) != n:
            raise ValueError("lon_idx debe tener la misma longitud que y_true.")
        data["lon_idx"] = lon_idx.astype(np.int64)

    if time_idx is not None:
        time_idx = np.asarray(time_idx).reshape(-1)
        if len(time_idx) != n:
            raise ValueError("time_idx debe tener la misma longitud que y_true.")
        data["time_idx"] = time_idx.astype(np.int64)

    df = pd.DataFrame(data)

    if latitude_values is not None:
        latitude_values = np.asarray(latitude_values).reshape(-1)
        if "lat_idx" not in df.columns:
            raise ValueError("Para añadir latitude necesitas proporcionar lat_idx.")
        if np.any((df["lat_idx"].values < 0) | (df["lat_idx"].values >= len(latitude_values))):
            raise ValueError("lat_idx contiene índices fuera de rango para latitude_values.")
        df["latitude"] = latitude_values[df["lat_idx"].values]

    if longitude_values is not None:
        longitude_values = np.asarray(longitude_values).reshape(-1)
        if "lon_idx" not in df.columns:
            raise ValueError("Para añadir longitude necesitas proporcionar lon_idx.")
        if np.any((df["lon_idx"].values < 0) | (df["lon_idx"].values >= len(longitude_values))):
            raise ValueError("lon_idx contiene índices fuera de rango para longitude_values.")
        df["longitude"] = longitude_values[df["lon_idx"].values]

    if time_values is not None:
        time_values = pd.to_datetime(np.asarray(time_values).reshape(-1))
        if "time_idx" not in df.columns:
            raise ValueError("Para añadir time necesitas proporcionar time_idx.")
        if np.any((df["time_idx"].values < 0) | (df["time_idx"].values >= len(time_values))):
            raise ValueError("time_idx contiene índices fuera de rango para time_values.")
        df["time"] = time_values[df["time_idx"].values]

    return df


# MÉTRICAS POR PÍXEL

def compute_pixel_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    pixel_id: np.ndarray,
    lat_idx: np.ndarray | None = None,
    lon_idx: np.ndarray | None = None,
    latitude: np.ndarray | None = None,
    longitude: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Calcula métricas de regresión por píxel.

    Parameters
    ----------
    y_true : np.ndarray
        Valores reales.
    y_pred : np.ndarray
        Predicciones.
    pixel_id : np.ndarray
        Identificador de píxel para cada fila.
    lat_idx : np.ndarray | None
        Índice latitudinal por fila.
    lon_idx : np.ndarray | None
        Índice longitudinal por fila.
    latitude : np.ndarray | None
        Coordenada latitude por fila.
    longitude : np.ndarray | None
        Coordenada longitude por fila.

    Returns
    -------
    pd.DataFrame
        Tabla con una fila por pixel_id y columnas:
        - pixel_id
        - n_samples
        - r2
        - rmse
        - mae
        - y_true_variance
        - lat_idx (si se proporciona)
        - lon_idx (si se proporciona)
        - latitude (si se proporciona)
        - longitude (si se proporciona)
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    pixel_id = np.asarray(pixel_id).reshape(-1)

    n = len(y_true)

    if y_pred.shape[0] != n or pixel_id.shape[0] != n:
        raise ValueError("y_true, y_pred y pixel_id deben tener la misma longitud.")

    data = {
        "pixel_id": pixel_id,
        "y_true": y_true,
        "y_pred": y_pred,
    }

    if lat_idx is not None:
        lat_idx = np.asarray(lat_idx).reshape(-1)
        if len(lat_idx) != n:
            raise ValueError("lat_idx debe tener la misma longitud que y_true.")
        data["lat_idx"] = lat_idx

    if lon_idx is not None:
        lon_idx = np.asarray(lon_idx).reshape(-1)
        if len(lon_idx) != n:
            raise ValueError("lon_idx debe tener la misma longitud que y_true.")
        data["lon_idx"] = lon_idx

    if latitude is not None:
        latitude = np.asarray(latitude).reshape(-1)
        if len(latitude) != n:
            raise ValueError("latitude debe tener la misma longitud que y_true.")
        data["latitude"] = latitude

    if longitude is not None:
        longitude = np.asarray(longitude).reshape(-1)
        if len(longitude) != n:
            raise ValueError("longitude debe tener la misma longitud que y_true.")
        data["longitude"] = longitude

    df = pd.DataFrame(data)

    rows = []

    for pid, sub in df.groupby("pixel_id", sort=True):
        yt = sub["y_true"].values
        yp = sub["y_pred"].values

        row = {
            "pixel_id": int(pid),
            "n_samples": int(len(sub)),
            "r2": _safe_r2(yt, yp),
            "rmse": _safe_rmse(yt, yp),
            "mae": _safe_mae(yt, yp),
            "y_true_variance": float(np.var(yt)) if len(yt) > 0 else np.nan,
        }

        if "lat_idx" in sub.columns:
            row["lat_idx"] = int(sub["lat_idx"].iloc[0])

        if "lon_idx" in sub.columns:
            row["lon_idx"] = int(sub["lon_idx"].iloc[0])

        if "latitude" in sub.columns:
            row["latitude"] = float(sub["latitude"].iloc[0])

        if "longitude" in sub.columns:
            row["longitude"] = float(sub["longitude"].iloc[0])

        rows.append(row)

    out = pd.DataFrame(rows).sort_values("pixel_id").reset_index(drop=True)
    return out


# RESUMEN DE MÉTRICAS POR PÍXEL

def summarize_pixel_metrics(
    pixel_metrics_df: pd.DataFrame,
) -> dict[str, Any]:
    """
    Resume la distribución de métricas por píxel.
    """
    if pixel_metrics_df.empty:
        raise ValueError("pixel_metrics_df está vacío.")

    valid_r2 = pixel_metrics_df["r2"].notna()

    summary = {
        "n_pixels": int(len(pixel_metrics_df)),
        "n_pixels_valid_r2": int(valid_r2.sum()),
        "n_pixels_nan_r2": int((~valid_r2).sum()),
        "r2_mean": float(pixel_metrics_df["r2"].mean(skipna=True)),
        "r2_median": float(pixel_metrics_df["r2"].median(skipna=True)),
        "r2_p05": float(pixel_metrics_df["r2"].quantile(0.05)),
        "r2_p95": float(pixel_metrics_df["r2"].quantile(0.95)),
        "rmse_mean": float(pixel_metrics_df["rmse"].mean(skipna=True)),
        "rmse_median": float(pixel_metrics_df["rmse"].median(skipna=True)),
        "mae_mean": float(pixel_metrics_df["mae"].mean(skipna=True)),
        "mae_median": float(pixel_metrics_df["mae"].median(skipna=True)),
    }

    if "y_true_variance" in pixel_metrics_df.columns:
        summary["y_true_variance_mean"] = float(pixel_metrics_df["y_true_variance"].mean(skipna=True))
        summary["y_true_variance_median"] = float(pixel_metrics_df["y_true_variance"].median(skipna=True))

    return summary


# RANKING DE PÍXELES

def rank_best_and_worst_pixels(
    pixel_metrics_df: pd.DataFrame,
    metric: str = "r2",
    top_k: int = 20,
    ascending: bool | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Devuelve los mejores y peores píxeles según una métrica.

    Parameters
    ----------
    pixel_metrics_df : pd.DataFrame
        Tabla de métricas por píxel.
    metric : str
        Métrica para ordenar ("r2", "rmse", "mae").
    top_k : int
        Número de filas a devolver en mejores/peores.
    ascending : bool | None
        Si es None:
        - r2 -> mejores = mayor, peores = menor
        - rmse/mae -> mejores = menor, peores = mayor

    Returns
    -------
    dict
        {
            "best": DataFrame,
            "worst": DataFrame,
        }
    """
    if metric not in pixel_metrics_df.columns:
        raise ValueError(f"La métrica '{metric}' no existe en pixel_metrics_df.")

    df = pixel_metrics_df.copy()
    df = df[np.isfinite(df[metric])]

    if df.empty:
        return {
            "best": pd.DataFrame(columns=pixel_metrics_df.columns),
            "worst": pd.DataFrame(columns=pixel_metrics_df.columns),
        }

    if ascending is None:
        if metric == "r2":
            best = df.sort_values(metric, ascending=False).head(top_k)
            worst = df.sort_values(metric, ascending=True).head(top_k)
        else:
            best = df.sort_values(metric, ascending=True).head(top_k)
            worst = df.sort_values(metric, ascending=False).head(top_k)
    else:
        best = df.sort_values(metric, ascending=ascending).head(top_k)
        worst = df.sort_values(metric, ascending=not ascending).head(top_k)

    return {
        "best": best.reset_index(drop=True),
        "worst": worst.reset_index(drop=True),
    }


# SERIES TEMPORALES POR PÍXEL

def build_pixel_timeseries_dataframe(
    prediction_df: pd.DataFrame,
    pixel_id: int,
    sort_by_time: bool = True,
) -> pd.DataFrame:
    """
    Extrae la serie temporal real/predicha de un píxel concreto
    a partir del DataFrame de predicciones fila a fila.

    Requiere que prediction_df contenga al menos:
    - pixel_id
    - y_true
    - y_pred

    Y opcionalmente:
    - time_idx
    - time
    - lat_idx / lon_idx / latitude / longitude
    """
    required_cols = {"pixel_id", "y_true", "y_pred"}
    missing = required_cols - set(prediction_df.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas en prediction_df: {sorted(missing)}")

    sub = prediction_df[prediction_df["pixel_id"] == pixel_id].copy()

    if sub.empty:
        return sub

    preferred_cols = [
        "pixel_id",
        "time_idx",
        "time",
        "y_true",
        "y_pred",
        "lat_idx",
        "lon_idx",
        "latitude",
        "longitude",
    ]
    existing_cols = [c for c in preferred_cols if c in sub.columns]
    remaining_cols = [c for c in sub.columns if c not in existing_cols]
    sub = sub[existing_cols + remaining_cols]

    if sort_by_time:
        if "time" in sub.columns:
            sub = sub.sort_values("time")
        elif "time_idx" in sub.columns:
            sub = sub.sort_values("time_idx")

    return sub.reset_index(drop=True)


def build_multiple_pixel_timeseries(
    prediction_df: pd.DataFrame,
    pixel_ids: list[int] | np.ndarray,
    sort_by_time: bool = True,
) -> dict[int, pd.DataFrame]:
    """
    Devuelve un diccionario pixel_id -> DataFrame de serie temporal.
    """
    out: dict[int, pd.DataFrame] = {}
    for pid in pixel_ids:
        out[int(pid)] = build_pixel_timeseries_dataframe(
            prediction_df=prediction_df,
            pixel_id=int(pid),
            sort_by_time=sort_by_time,
        )
    return out