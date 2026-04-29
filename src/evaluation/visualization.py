from __future__ import annotations

from pathlib import Path
import re
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import pandas as pd

from src.evaluation.regression import (
    build_pixel_timeseries_dataframe,
    compute_global_regression_metrics,
    compute_pixel_regression_metrics,
)


DEFAULT_FIGSIZE = (20, 12)
DEFAULT_DPI = 300


def _clean_slug(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug or "figure"


def _as_valid_array(values) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def add_hist_stats(
    ax,
    values,
    title: str,
    xlabel: str,
    color: str = "tab:blue",
    bins: int = 40,
) -> None:
    values = _as_valid_array(values)
    if values.size == 0:
        ax.set_title(f"{title} - sin datos validos")
        ax.set_xlabel(xlabel)
        return

    mean_val = float(np.mean(values))
    median_val = float(np.median(values))
    p05 = float(np.percentile(values, 5))
    p95 = float(np.percentile(values, 95))

    ax.hist(values, bins=bins, color=color, alpha=0.8, edgecolor="black")
    ax.axvline(mean_val, color="black", linestyle="--", linewidth=1.3, label=f"mean={mean_val:.4f}")
    ax.axvline(median_val, color="black", linestyle="-", linewidth=1.3, label=f"median={median_val:.4f}")
    ax.axvline(p05, color="grey", linestyle=":", linewidth=1.1, label=f"P5={p05:.4f}")
    ax.axvline(p95, color="grey", linestyle=":", linewidth=1.1, label=f"P95={p95:.4f}")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.legend(fontsize=8, frameon=True)


def build_metrics_text(metrics: dict[str, float]) -> str:
    lines = []
    for key in ["r2", "rmse", "mae", "bias", "slope", "intercept", "n_samples"]:
        if key not in metrics:
            continue
        value = metrics[key]
        if key == "n_samples":
            lines.append(f"n={int(value)}")
        else:
            lines.append(f"{key.capitalize()}={value:.4f}")
    return "\n".join(lines)


def _safe_corr(y_true: np.ndarray, y_pred: np.ndarray, min_valid: int = 12) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[valid]
    y_pred = y_pred[valid]
    if y_true.size < min_valid:
        return np.nan
    if np.std(y_true) <= 0:
        return np.nan
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def _global_fit_metrics(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[valid]
    y_pred = y_pred[valid]
    if y_true.size == 0:
        raise ValueError("No hay datos válidos para calcular métricas globales.")

    metrics = compute_global_regression_metrics(y_true, y_pred)
    residual = y_pred - y_true
    metrics["bias"] = float(np.mean(residual))

    if y_true.size >= 2:
        slope, intercept = np.polyfit(y_true, y_pred, 1)
    else:
        slope, intercept = np.nan, np.nan
    metrics["slope"] = float(slope)
    metrics["intercept"] = float(intercept)
    return metrics


def _compute_spatial_pixel_metrics(row_df: pd.DataFrame, corr_min_valid: int = 12) -> pd.DataFrame:
    rows = []
    for pixel_id, sub in row_df.groupby("pixel_id", sort=True):
        y_true = sub["y_true"].to_numpy(dtype=float)
        y_pred = sub["y_pred"].to_numpy(dtype=float)
        residual = y_pred - y_true
        row = {
            "pixel_id": int(pixel_id),
            "n_samples": int(len(sub)),
            "rmse": float(np.sqrt(np.mean((residual) ** 2))) if len(sub) else np.nan,
            "mae": float(np.mean(np.abs(residual))) if len(sub) else np.nan,
            "bias": float(np.mean(residual)) if len(sub) else np.nan,
            "corr": _safe_corr(y_true, y_pred, min_valid=corr_min_valid),
            "r2": compute_global_regression_metrics(y_true, y_pred)["r2"] if len(sub) >= 2 and np.var(y_true) > 0 else np.nan,
        }
        for col in ["lat_idx", "lon_idx", "latitude", "longitude"]:
            if col in sub.columns:
                row[col] = sub[col].iloc[0]
        rows.append(row)

    return pd.DataFrame(rows).sort_values("pixel_id").reset_index(drop=True)


def _build_metric_map(
    metrics_df: pd.DataFrame,
    value_col: str,
    shape: tuple[int, int],
    base_mask=None,
) -> np.ndarray:
    if base_mask is not None:
        mask_values = np.asarray(base_mask.values if hasattr(base_mask, "values") else base_mask, dtype=bool)
        if mask_values.shape != shape:
            raise ValueError(f"base_mask tiene shape {mask_values.shape}, esperado {shape}.")
        out = np.where(mask_values, 0.0, np.nan).astype(float)
    else:
        out = np.full(shape, np.nan, dtype=float)
    if metrics_df.empty:
        return out
    lat_idx = metrics_df["lat_idx"].to_numpy(dtype=int)
    lon_idx = metrics_df["lon_idx"].to_numpy(dtype=int)
    values = metrics_df[value_col].to_numpy(dtype=float)
    out[lat_idx, lon_idx] = values
    return out


def _plot_density_panel(ax, y_true, y_pred, metrics: dict[str, float], title: str) -> None:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[valid]
    y_pred = y_pred[valid]
    if y_true.size == 0:
        ax.set_title(f"{title} - sin datos validos")
        return

    robust_max = float(np.percentile(np.concatenate([y_true, y_pred]), 99.5))
    lim_max = robust_max * 1.15 if robust_max > 0 else 1.0
    hb = ax.hexbin(y_true, y_pred, gridsize=55, mincnt=1, bins="log", cmap="viridis", linewidths=0)
    lims = [0.0, lim_max]
    ax.plot(lims, lims, linestyle="--", color="black", linewidth=1.2)
    if np.isfinite(metrics.get("slope", np.nan)) and np.isfinite(metrics.get("intercept", np.nan)):
        x_fit = np.array(lims)
        y_fit = metrics["slope"] * x_fit + metrics["intercept"]
        ax.plot(x_fit, y_fit, color="red", linewidth=1.4)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("LAI real")
    ax.set_ylabel("LAI predicho")
    ax.set_title(title)
    ax.margins(x=0.04, y=0.04)
    ax.text(
        0.03,
        0.97,
        build_metrics_text(metrics),
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
        fontsize=9,
    )
    plt.colorbar(hb, ax=ax, label="log10(count)")


def _plot_residuals_vs_true(ax, y_true, y_pred, title: str) -> None:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[valid]
    residual = y_pred[valid] - y_true
    if y_true.size == 0:
        ax.set_title(f"{title} - sin datos validos")
        return

    hb = ax.hexbin(y_true, residual, gridsize=55, mincnt=1, bins="log", cmap="viridis", linewidths=0)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.1)

    bins = np.linspace(float(np.min(y_true)), float(np.percentile(y_true, 99.5)), 20)
    if np.unique(bins).size >= 3:
        bin_ids = np.digitize(y_true, bins)
        centers = []
        means = []
        for idx in range(1, len(bins)):
            mask = bin_ids == idx
            if mask.sum() < 10:
                continue
            centers.append(float(np.mean(y_true[mask])))
            means.append(float(np.mean(residual[mask])))
        if centers:
            ax.plot(centers, means, color="red", linewidth=1.5)

    rmse = float(np.sqrt(np.mean(residual ** 2)))
    mae = float(np.mean(np.abs(residual)))
    bias = float(np.mean(residual))
    ax.text(
        0.03,
        0.97,
        f"Bias={bias:.4f}\nRMSE={rmse:.4f}\nMAE={mae:.4f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
        fontsize=9,
    )
    ax.set_xlabel("LAI real")
    ax.set_ylabel("pred - real")
    ax.set_title(title)
    x_max = float(np.percentile(y_true, 99.5)) * 1.15 if y_true.size else 1.0
    y_lim = float(np.percentile(np.abs(residual), 99.0)) if residual.size else 1.0
    ax.set_xlim(0.0, x_max if x_max > 0 else 1.0)
    if np.isfinite(y_lim) and y_lim > 0:
        ax.set_ylim(-y_lim * 1.15, y_lim * 1.15)
    ax.margins(x=0.04, y=0.08)
    plt.colorbar(hb, ax=ax, label="log10(count)")


def _plot_residual_hist(ax, residual, title: str) -> None:
    residual = _as_valid_array(residual)
    if residual.size == 0:
        ax.set_title(f"{title} - sin datos validos")
        return
    mean_val = float(np.mean(residual))
    std_val = float(np.std(residual))
    p05 = float(np.percentile(residual, 5))
    p95 = float(np.percentile(residual, 95))
    ax.hist(residual, bins=50, color="#7b3f98", alpha=0.82, edgecolor="black")
    ax.axvline(mean_val, color="black", linestyle="--", linewidth=1.2)
    ax.axvline(p05, color="grey", linestyle=":", linewidth=1.1)
    ax.axvline(p95, color="grey", linestyle=":", linewidth=1.1)
    ax.axvspan(p05, p95, color="#7b3f98", alpha=0.12)
    ax.text(
        0.03,
        0.97,
        f"mean={mean_val:.4f}\nstd={std_val:.4f}\nP5={p05:.4f}\nP95={p95:.4f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
        fontsize=9,
    )
    ax.set_xlabel("pred - real")
    ax.set_title(title)
    ax.margins(x=0.03)


def _plot_metric_hist(ax, values, title: str, xlabel: str, color: str) -> None:
    values = _as_valid_array(values)
    if values.size == 0:
        ax.set_title(f"{title} - sin datos validos")
        ax.set_xlabel(xlabel)
        return
    mean_val = float(np.mean(values))
    median_val = float(np.median(values))
    p05 = float(np.percentile(values, 5))
    p95 = float(np.percentile(values, 95))
    ax.hist(values, bins=40, color=color, alpha=0.82, edgecolor="black")
    ax.axvline(median_val, color="black", linestyle="-", linewidth=1.2)
    ax.axvline(mean_val, color="black", linestyle="--", linewidth=1.2)
    ax.axvline(p05, color="grey", linestyle=":", linewidth=1.0)
    ax.axvline(p95, color="grey", linestyle=":", linewidth=1.0)
    ax.text(
        0.03,
        0.97,
        f"mean={mean_val:.4f}\nmedian={median_val:.4f}\nP5={p05:.4f}\nP95={p95:.4f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
        fontsize=9,
    )
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.margins(x=0.03)


def _plot_percentage_error_hist(ax, y_true, y_pred, title: str) -> None:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(y_pred) & (np.abs(y_true) > 1e-6)
    if not np.any(valid):
        ax.set_title(f"{title} - sin datos validos")
        ax.set_xlabel("% error absoluto")
        return

    ape = np.abs((y_pred[valid] - y_true[valid]) / y_true[valid]) * 100.0
    ape = ape[np.isfinite(ape)]
    if ape.size == 0:
        ax.set_title(f"{title} - sin datos validos")
        ax.set_xlabel("% error absoluto")
        return

    clip_max = float(np.percentile(ape, 99.0))
    ape = ape[ape <= clip_max]
    mean_val = float(np.mean(ape))
    median_val = float(np.median(ape))
    p05 = float(np.percentile(ape, 5))
    p95 = float(np.percentile(ape, 95))

    ax.hist(ape, bins=40, color="steelblue", alpha=0.82, edgecolor="black")
    ax.axvline(median_val, color="black", linestyle="-", linewidth=1.2)
    ax.axvline(mean_val, color="black", linestyle="--", linewidth=1.2)
    ax.axvline(p05, color="grey", linestyle=":", linewidth=1.0)
    ax.axvline(p95, color="grey", linestyle=":", linewidth=1.0)
    ax.text(
        0.03,
        0.97,
        f"mean={mean_val:.2f}%\nmedian={median_val:.2f}%\nP5={p05:.2f}%\nP95={p95:.2f}%",
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
        fontsize=9,
    )
    ax.set_xlabel("% error absoluto")
    ax.set_title(title)
    ax.margins(x=0.03)


def _plot_metric_map_with_background(
    ax,
    data_2d: np.ndarray,
    title: str,
    cmap,
    base_mask=None,
    vmin=None,
    vmax=None,
    norm=None,
) -> None:
    if base_mask is not None:
        base = np.asarray(base_mask.values if hasattr(base_mask, "values") else base_mask, dtype=bool)
        background = np.where(base, 0.0, np.nan)
        bg_cmap = colors.ListedColormap(["#efe8d8"])
        bg_cmap.set_bad(color="#d9d9d9")
        ax.imshow(background, origin="lower", cmap=bg_cmap, aspect="auto", interpolation="none")

    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(color=(1, 1, 1, 0))
    plotted = np.asarray(data_2d, dtype=float).copy()
    if base_mask is not None:
        valid_land = np.asarray(base_mask.values if hasattr(base_mask, "values") else base_mask, dtype=bool)
        plotted[(valid_land) & (plotted == 0)] = np.nan
    im = ax.imshow(
        plotted,
        origin="lower",
        cmap=cmap_obj,
        vmin=vmin,
        vmax=vmax,
        norm=norm,
        aspect="auto",
        interpolation="none",
    )
    ax.set_title(title)
    ax.set_xlabel("Longitude index")
    ax.set_ylabel("Latitude index")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_facecolor("#d9d9d9")


def _plot_metric_scatter_map(
    ax,
    metrics_df: pd.DataFrame,
    value_col: str,
    title: str,
    cmap: str,
    base_mask=None,
    vmin=None,
    vmax=None,
    norm=None,
    cbar_label: str | None = None,
) -> None:
    if base_mask is not None:
        base_mask.astype(int).plot(ax=ax, cmap="Greys", add_colorbar=False, alpha=0.18)

    work = metrics_df[np.isfinite(metrics_df[value_col])].copy()
    if work.empty:
        ax.set_title(f"{title} - sin pixeles")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        return

    sc = ax.scatter(
        work["longitude"],
        work["latitude"],
        c=work[value_col],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        norm=norm,
        s=10,
        edgecolors="none",
    )
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label=cbar_label)


def _save_figure(fig, output_dir: str | Path | None, stem: str) -> dict[str, str]:
    if output_dir is None:
        return {}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{stem}.png"
    fig.savefig(png_path, dpi=DEFAULT_DPI, bbox_inches="tight")
    return {"png": str(png_path)}


def compute_subset_pixel_metrics(row_df: pd.DataFrame) -> pd.DataFrame:
    return compute_pixel_regression_metrics(
        y_true=row_df["y_true"].values,
        y_pred=row_df["y_pred"].values,
        pixel_id=row_df["pixel_id"].values,
        lat_idx=row_df["lat_idx"].values,
        lon_idx=row_df["lon_idx"].values,
        latitude=row_df["latitude"].values,
        longitude=row_df["longitude"].values,
    )


def plot_spatial_bundle(
    title: str,
    row_df: pd.DataFrame,
    metrics_df: pd.DataFrame | None = None,
    map_base_mask=None,
    output_dir: str | Path | None = None,
    filename_slug: str | None = None,
) -> dict[str, float]:
    plot_df = row_df.copy()
    plot_df["residual"] = plot_df["y_pred"] - plot_df["y_true"]
    plot_df["abs_error"] = np.abs(plot_df["residual"])

    global_metrics = _global_fit_metrics(plot_df["y_true"], plot_df["y_pred"])
    spatial_metrics_df = _compute_spatial_pixel_metrics(plot_df)

    if metrics_df is not None and not metrics_df.empty:
        for col in ["climate_code", "climate_label", "landcover_code", "landcover_label", "lai_var", "lai_std"]:
            if col in metrics_df.columns and col not in spatial_metrics_df.columns:
                spatial_metrics_df = spatial_metrics_df.merge(
                    metrics_df[["pixel_id", col]].drop_duplicates(),
                    on="pixel_id",
                    how="left",
                )

    rmse_values = _as_valid_array(spatial_metrics_df["rmse"])
    bias_values = _as_valid_array(spatial_metrics_df["bias"])
    r2_values = _as_valid_array(spatial_metrics_df["r2"])
    rmse_vmax = float(np.percentile(rmse_values, 95)) if rmse_values.size else None
    bias_lim = float(np.percentile(np.abs(bias_values), 95)) if bias_values.size else None
    r2_vmin = 0.0
    r2_vmax = 1.0

    plt.style.use("default")
    fig = plt.figure(figsize=DEFAULT_FIGSIZE, constrained_layout=True)
    gs = fig.add_gridspec(3, 3)

    ax11 = fig.add_subplot(gs[0, 0])
    ax12 = fig.add_subplot(gs[0, 1])
    ax13 = fig.add_subplot(gs[0, 2])
    ax21 = fig.add_subplot(gs[1, 0])
    ax22 = fig.add_subplot(gs[1, 1])
    ax23 = fig.add_subplot(gs[1, 2])
    ax31 = fig.add_subplot(gs[2, 0])
    ax32 = fig.add_subplot(gs[2, 1])
    ax33 = fig.add_subplot(gs[2, 2])

    for ax in [ax11, ax12, ax13, ax21, ax22, ax23, ax31, ax32, ax33]:
        ax.grid(True, alpha=0.22)

    _plot_density_panel(ax11, plot_df["y_true"], plot_df["y_pred"], global_metrics, "1. Densidad: LAI predicho vs LAI real")
    _plot_residuals_vs_true(ax12, plot_df["y_true"], plot_df["y_pred"], "2. Residuos vs LAI real")
    _plot_residual_hist(ax13, plot_df["residual"], "3. Histograma de residuos (pred - real)")
    _plot_metric_hist(ax21, spatial_metrics_df["rmse"], "4. Distribución de RMSE por píxel", "RMSE", color="darkorange")
    _plot_metric_hist(ax22, spatial_metrics_df["mae"], "5. Distribución de MAE por píxel", "MAE", color="forestgreen")
    _plot_percentage_error_hist(ax23, plot_df["y_true"], plot_df["y_pred"], "6. Distribución de % error absoluto")

    rmse_vmin = float(np.percentile(rmse_values, 5)) if rmse_values.size else 0.0
    _plot_metric_scatter_map(
        ax31,
        spatial_metrics_df,
        "rmse",
        "7. Mapa global de RMSE por píxel",
        cmap="RdYlGn",
        base_mask=map_base_mask,
        vmin=rmse_vmin,
        vmax=rmse_vmax,
        cbar_label="RMSE por pixel",
    )
    if bias_lim is not None and np.isfinite(bias_lim) and bias_lim > 0:
        norm = colors.TwoSlopeNorm(vmin=-bias_lim, vcenter=0.0, vmax=bias_lim)
        _plot_metric_scatter_map(
            ax32,
            spatial_metrics_df,
            "bias",
            "8. Mapa global de bias (pred - real)",
            cmap="RdYlGn",
            base_mask=map_base_mask,
            norm=norm,
            cbar_label="Bias por pixel",
        )
    else:
        _plot_metric_scatter_map(
            ax32,
            spatial_metrics_df,
            "bias",
            "8. Mapa global de bias (pred - real)",
            cmap="RdYlGn",
            base_mask=map_base_mask,
            cbar_label="Bias por pixel",
        )
    _plot_metric_scatter_map(
        ax33,
        spatial_metrics_df,
        "r2",
        "9. Mapa global de R2 por píxel",
        cmap="RdYlGn",
        base_mask=map_base_mask,
        vmin=r2_vmin,
        vmax=r2_vmax,
        cbar_label="R2 por pixel",
    )

    slug = filename_slug or _clean_slug(title)
    saved_paths = _save_figure(fig, output_dir=output_dir, stem=slug)
    plt.show()

    global_metrics["figure_png"] = saved_paths.get("png")
    global_metrics["bias"] = float(global_metrics["bias"])
    global_metrics["slope"] = float(global_metrics["slope"])
    global_metrics["intercept"] = float(global_metrics["intercept"])
    return global_metrics


def run_spatial_category_analysis(
    row_df: pd.DataFrame,
    code_col: str,
    labels_map: dict[int, str],
    section_title: str,
    map_base_mask=None,
    ignore_codes: Iterable[int] = (0,),
    output_dir: str | Path | None = None,
    filename_prefix: str | None = None,
) -> pd.DataFrame:
    records = []
    valid_codes = [code for code in sorted(labels_map) if code not in set(ignore_codes)]

    for code in valid_codes:
        subset_rows = row_df[row_df[code_col] == code].copy()
        if subset_rows.empty:
            continue

        subset_pixel_metrics = _compute_spatial_pixel_metrics(subset_rows)
        label = labels_map.get(int(code), f"class_{int(code)}")
        metrics = plot_spatial_bundle(
            title=f"{section_title} - {label}",
            row_df=subset_rows,
            metrics_df=subset_pixel_metrics,
            map_base_mask=map_base_mask,
            output_dir=output_dir,
            filename_slug=f"{filename_prefix or _clean_slug(section_title)}_{_clean_slug(label)}",
        )
        metrics.update(
            {
                "category_code": int(code),
                "category_label": label,
                "n_pixels": int(subset_pixel_metrics.shape[0]),
                "corr_mean": float(subset_pixel_metrics["corr"].mean(skipna=True)) if not subset_pixel_metrics.empty else np.nan,
                "bias_mean": float(subset_pixel_metrics["bias"].mean(skipna=True)) if not subset_pixel_metrics.empty else np.nan,
            }
        )
        records.append(metrics)

    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records).sort_values("r2", ascending=False).reset_index(drop=True)


def plot_selected_pixel_group(
    selected_df: pd.DataFrame,
    prediction_df: pd.DataFrame,
    group_name: str,
) -> None:
    from IPython.display import display

    if selected_df.empty:
        print(f"{group_name}: sin pixeles")
        return

    display(
        selected_df[
            [
                "pixel_id",
                "n_samples",
                "r2",
                "rmse",
                "mae",
                "lai_var",
                "lai_std",
                "climate_label",
                "landcover_label",
                "latitude",
                "longitude",
            ]
        ]
    )

    fig, axes = plt.subplots(2, len(selected_df), figsize=(4.5 * len(selected_df), 9))
    axes = np.atleast_2d(axes)

    for ax, (_, row) in zip(axes[0], selected_df.iterrows()):
        ts_df = build_pixel_timeseries_dataframe(prediction_df, pixel_id=int(row["pixel_id"]))
        _plot_density_panel(
            ax,
            ts_df["y_true"],
            ts_df["y_pred"],
            _global_fit_metrics(ts_df["y_true"], ts_df["y_pred"]),
            f"{group_name} | pid={int(row['pixel_id'])}",
        )

    for ax, (_, row) in zip(axes[1], selected_df.iterrows()):
        ts_df = build_pixel_timeseries_dataframe(prediction_df, pixel_id=int(row["pixel_id"]))
        ax.plot(ts_df["time"], ts_df["y_true"], label="real")
        ax.plot(ts_df["time"], ts_df["y_pred"], label="predicha")
        ax.set_title(f"{group_name} | pid={int(row['pixel_id'])} | R2={row['r2']:.3f}")
        ax.tick_params(axis="x", rotation=45)
        ax.set_xlabel("Time")
        ax.set_ylabel("LAI")
        ax.legend()

    plt.tight_layout()
    plt.show()

    pixel_ids = selected_df["pixel_id"].astype(int).tolist()
    row_subset = prediction_df[prediction_df["pixel_id"].isin(pixel_ids)].copy()
    row_subset["residual"] = row_subset["y_pred"] - row_subset["y_true"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
    _plot_residual_hist(axes[0], row_subset["residual"], f"{group_name} - residuos")
    _plot_metric_hist(axes[1], selected_df["rmse"], f"{group_name} - RMSE por pixel", "RMSE", color="darkorange")
    _plot_metric_hist(axes[2], selected_df["mae"], f"{group_name} - MAE por pixel", "MAE", color="forestgreen")
    plt.tight_layout()
    plt.show()


def binned_r2_summary(df: pd.DataFrame, value_col: str, n_bins: int = 8) -> pd.DataFrame:
    work = df[[value_col, "r2"]].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if work.empty:
        return pd.DataFrame()

    work["bin"] = pd.qcut(work[value_col], q=min(n_bins, work[value_col].nunique()), duplicates="drop")
    return (
        work.groupby("bin", observed=False)
        .agg(
            n_pixels=("r2", "size"),
            value_mean=(value_col, "mean"),
            r2_mean=("r2", "mean"),
            r2_std=("r2", "std"),
            r2_median=("r2", "median"),
        )
        .reset_index()
    )


def metrics_by_time(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for time_value, sub in df.groupby("time", sort=True):
        if len(sub) < 2:
            continue
        metrics = compute_global_regression_metrics(sub["y_true"], sub["y_pred"])
        metrics["time"] = time_value
        rows.append(metrics)
    return pd.DataFrame(rows).sort_values("time").reset_index(drop=True)


def pixel_observation_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["pixel_id", "lat_idx", "lon_idx", "latitude", "longitude"], as_index=False)
        .agg(
            n_test_obs=("y_true", "size"),
            mae_mean=("abs_error", "mean"),
            rmse_mean=("sq_error", lambda x: float(np.sqrt(np.mean(x)))),
            residual_mean=("residual", "mean"),
        )
    )


def _observation_metrics_by_time(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for time_value, sub in df.groupby("time", sort=True):
        if sub.empty:
            continue
        residual = sub["y_pred"].to_numpy(dtype=float) - sub["y_true"].to_numpy(dtype=float)
        rows.append(
            {
                "time": time_value,
                "n_obs": int(len(sub)),
                "rmse": float(np.sqrt(np.mean(residual ** 2))),
                "mae": float(np.mean(np.abs(residual))),
                "bias": float(np.mean(residual)),
            }
        )
    return pd.DataFrame(rows).sort_values("time").reset_index(drop=True)


def _lai_error_bins(df: pd.DataFrame, n_bins: int = 12) -> pd.DataFrame:
    work = df[["y_true", "abs_error"]].replace([np.inf, -np.inf], np.nan).dropna().copy()
    work = work[work["y_true"] > 1e-6]
    if work.empty:
        return pd.DataFrame()

    n_quantiles = min(n_bins, int(work["y_true"].nunique()))
    if n_quantiles < 2:
        return pd.DataFrame()

    work["lai_bin"] = pd.qcut(work["y_true"], q=n_quantiles, duplicates="drop")
    out = (
        work.groupby("lai_bin", observed=False)
        .agg(
            lai_mean=("y_true", "mean"),
            mae_mean=("abs_error", "mean"),
            ape_mean=("abs_error", lambda x: float(np.mean(x / work.loc[x.index, "y_true"]) * 100.0)),
            n=("abs_error", "size"),
        )
        .reset_index()
    )
    return out


def _plot_error_by_lai_bins(ax, df: pd.DataFrame, title: str) -> None:
    bins_df = _lai_error_bins(df)
    if bins_df.empty:
        ax.set_title(f"{title} - sin datos validos")
        ax.set_xlabel("LAI real")
        return

    ax.plot(bins_df["lai_mean"], bins_df["mae_mean"], marker="o", color="darkorange", linewidth=1.5, label="MAE")
    ax2 = ax.twinx()
    ax2.plot(bins_df["lai_mean"], bins_df["ape_mean"], marker="s", color="steelblue", linewidth=1.4, label="% error")
    ax.set_xlabel("LAI real medio del bin")
    ax.set_ylabel("MAE")
    ax2.set_ylabel("% error absoluto")
    ax.set_title(title)
    ax.grid(True, alpha=0.22)

    lines = ax.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in lines]
    ax.legend(lines, labels, loc="upper left", fontsize=8, frameon=True)


def plot_observation_bundle(
    title: str,
    df: pd.DataFrame,
    output_dir: str | Path | None = None,
    filename_slug: str | None = None,
) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    plot_df = df.copy()
    if "residual" not in plot_df.columns:
        plot_df["residual"] = plot_df["y_pred"] - plot_df["y_true"]
    if "abs_error" not in plot_df.columns:
        plot_df["abs_error"] = np.abs(plot_df["residual"])
    if "sq_error" not in plot_df.columns:
        plot_df["sq_error"] = plot_df["residual"] ** 2

    metrics = _global_fit_metrics(plot_df["y_true"], plot_df["y_pred"])
    time_df = _observation_metrics_by_time(plot_df)
    pixel_df = pixel_observation_summary(df)
    fig = plt.figure(figsize=DEFAULT_FIGSIZE, constrained_layout=True)
    gs = fig.add_gridspec(3, 3)

    ax11 = fig.add_subplot(gs[0, 0])
    ax12 = fig.add_subplot(gs[0, 1])
    ax13 = fig.add_subplot(gs[0, 2])
    ax21 = fig.add_subplot(gs[1, 0])
    ax22 = fig.add_subplot(gs[1, 1])
    ax23 = fig.add_subplot(gs[1, 2])
    ax31 = fig.add_subplot(gs[2, 0])
    ax32 = fig.add_subplot(gs[2, 1])
    ax33 = fig.add_subplot(gs[2, 2])

    for ax in [ax11, ax12, ax13, ax21, ax22, ax23, ax31, ax32, ax33]:
        ax.grid(True, alpha=0.22)

    _plot_density_panel(ax11, plot_df["y_true"], plot_df["y_pred"], metrics, "1. Densidad: LAI predicho vs LAI real")
    _plot_residuals_vs_true(ax12, plot_df["y_true"], plot_df["y_pred"], "2. Residuos vs LAI real")
    _plot_residual_hist(ax13, plot_df["residual"], "3. Histograma de residuos (pred - real)")
    _plot_metric_hist(ax21, time_df["rmse"] if not time_df.empty else [], "4. Distribución de RMSE por timestep", "RMSE", color="darkorange")
    _plot_metric_hist(ax22, time_df["mae"] if not time_df.empty else [], "5. Distribución de MAE por timestep", "MAE", color="forestgreen")
    _plot_error_by_lai_bins(ax23, plot_df, "6. Error por bins de LAI real")

    _plot_metric_scatter_map(
        ax31,
        pixel_df.rename(columns={"mae_mean": "mae_plot"}),
        "mae_plot",
        "7. Mapa global de MAE",
        cmap="RdYlGn",
        cbar_label="MAE por pixel",
    )
    _plot_metric_scatter_map(
        ax32,
        pixel_df.rename(columns={"residual_mean": "bias_plot"}),
        "bias_plot",
        "8. Mapa global de bias",
        cmap="RdYlGn",
        cbar_label="Bias por pixel",
    )
    _plot_metric_scatter_map(
        ax33,
        pixel_df.rename(columns={"n_test_obs": "n_obs_plot"}),
        "n_obs_plot",
        "9. Mapa global de n observaciones test",
        cmap="RdYlGn",
        cbar_label="n obs por pixel",
    )

    slug = filename_slug or _clean_slug(title)
    saved_paths = _save_figure(fig, output_dir=output_dir, stem=slug)
    plt.show()

    metrics["figure_png"] = saved_paths.get("png")
    return metrics, time_df, pixel_df


def run_observation_category_analysis(
    df: pd.DataFrame,
    code_col: str,
    labels_map: dict[int, str],
    section_title: str,
    ignore_codes: Iterable[int] = (0,),
    output_dir: str | Path | None = None,
    filename_prefix: str | None = None,
) -> pd.DataFrame:
    rows = []
    for code in sorted(labels_map):
        if code in set(ignore_codes):
            continue
        sub = df[df[code_col] == code].copy()
        if sub.empty:
            continue
        metrics, _, pixel_df = plot_observation_bundle(
            f"{section_title} - {labels_map[code]}",
            sub,
            output_dir=output_dir,
            filename_slug=f"{filename_prefix or _clean_slug(section_title)}_{_clean_slug(labels_map[code])}",
        )
        metrics.update(
            {
                "category_code": int(code),
                "category_label": labels_map[code],
                "n_unique_pixels": int(sub["pixel_id"].nunique()),
                "n_timesteps": int(sub["time"].nunique()),
                "mean_test_obs_per_pixel": float(pixel_df["n_test_obs"].mean()) if not pixel_df.empty else np.nan,
            }
        )
        rows.append(metrics)
    return pd.DataFrame(rows).sort_values("r2", ascending=False).reset_index(drop=True)
