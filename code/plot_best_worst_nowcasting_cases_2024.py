#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Genera figures diagnòstiques del nowcasting local UB v9 durant el 2024:

1) Temperatura: una sola figura amb dos panells:
   - esquerra: millor cas segons MAE mitjana a 3 h;
   - dreta: pitjor cas segons MAE mitjana a 3 h.

2) Precipitació: una sola figura amb dos panells:
   - esquerra: millor cas amb pluja intensa ben representada;
   - dreta: pitjor cas amb pluja observada.

Les figures no tenen títol intern i tenen una sola llegenda compartida.

El codi NO recalcula el model. Llegeix les memòries cau .npz generades pels scripts:
- animate_temperature_nowcasting_v9_2024_v4_clean_window_fixed.py
- animate_precipitation_nowcasting_v9_2024.py

Exemple d'ús:
-------------
python plot_best_worst_nowcasting_cases_2024_combined.py --dataset CR3000_Tabla1_2011_2024_dataset_nowcasting_no_rs.zip --temperature-cache outputs/animation_temperature_2024/temperature_nowcasts_2024-01-01_00_00_00_2025-01-01_00_00_00_fstride1_mall.npz --rain-cache outputs/animation_precipitation_2024/rain_nowcasts_2024-01-01_00_00_00_2025-01-01_00_00_00_fstride1_mall.npz --output-dir outputs/best_worst_cases_2024_combined --display-history-hours 6 --rain-min-accum-mm 1.0 --rain-best-min-step-mm 3.0 --rain-best-model-min-step-mm 2.0 --rain-best-min-peak-fraction 0.45 --rain-best-peak-time-tol-steps 3 --formats png,pdf --dpi 260
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# Utilitats generals
# =============================================================================

def log(msg: str) -> None:
    print(msg, flush=True)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except Exception:
        return None
    return v if np.isfinite(v) else None


def json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return json_safe(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return v if np.isfinite(v) else None
    if isinstance(obj, pd.Timestamp):
        return str(obj)
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


def write_json(obj: Any, path: Path) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(json_safe(obj), f, indent=2, ensure_ascii=False)


def find_single_csv_in_zip(zip_path: Path) -> str:
    with zipfile.ZipFile(zip_path, "r") as zf:
        csv_members = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not csv_members:
            raise FileNotFoundError(f"El ZIP no conté cap CSV: {zip_path}")

        preferred = [n for n in csv_members if "dataset_nowcasting" in Path(n).name]
        return preferred[0] if preferred else csv_members[0]


def resolve_dataset_csv(dataset_arg: str, workdir: Path) -> Path:
    p = Path(dataset_arg).expanduser().resolve()

    if p.is_file() and p.suffix.lower() == ".csv":
        return p

    if p.is_file() and p.suffix.lower() == ".zip":
        ensure_dir(workdir)
        member = find_single_csv_in_zip(p)
        out = workdir / Path(member).name

        if not out.exists():
            log(f"[INFO] Extraient dataset: {Path(member).name}")
            with zipfile.ZipFile(p, "r") as zf:
                with zf.open(member, "r") as src, open(out, "wb") as dst:
                    shutil.copyfileobj(src, dst)

        return out

    raise FileNotFoundError(f"No trobo el dataset CSV/ZIP: {p}")


def read_observations(csv_path: Path, temp_col: str, rain_col: str) -> pd.DataFrame:
    usecols = ["TIMESTAMP", temp_col, rain_col]

    log(f"[INFO] Llegint observacions: {csv_path.name}")
    df = pd.read_csv(csv_path, usecols=usecols, low_memory=False)

    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"], errors="coerce")
    df = df.dropna(subset=["TIMESTAMP"]).sort_values("TIMESTAMP")

    df[temp_col] = pd.to_numeric(df[temp_col], errors="coerce")
    df[rain_col] = pd.to_numeric(df[rain_col], errors="coerce").clip(lower=0.0)

    df = df.drop_duplicates(subset=["TIMESTAMP"], keep="last").set_index("TIMESTAMP")
    return df


def load_npz(path: str | Path) -> Dict[str, np.ndarray]:
    p = Path(path).expanduser().resolve()

    if not p.exists():
        raise FileNotFoundError(f"No trobo la memòria cau: {p}")

    log(f"[INFO] Carregant memòria cau: {p}")
    with np.load(p, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def infer_dt_from_issue_times(issue_times: pd.DatetimeIndex, fallback: str = "10min") -> pd.Timedelta:
    if len(issue_times) >= 2:
        diffs = pd.Series(issue_times).diff().dropna()
        if not diffs.empty:
            return pd.Timedelta(diffs.mode().iloc[0])

    return pd.Timedelta(fallback)


def future_times_for_case(
    t0: pd.Timestamp,
    horizon_steps: int,
    dt: pd.Timedelta,
) -> pd.DatetimeIndex:
    return pd.DatetimeIndex([t0 + (i + 1) * dt for i in range(horizon_steps)])


# =============================================================================
# Mètriques
# =============================================================================

def finite_case_mae(forecast: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> np.ndarray:
    valid = (mask > 0.5) & np.isfinite(forecast) & np.isfinite(truth)
    err = np.abs(forecast - truth)
    err = np.where(valid, err, np.nan)

    with np.errstate(all="ignore"):
        return np.nanmean(err, axis=1)


def rain_accumulations(pred: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model = pred["ensemble_mean"].astype(float)
    baseline = pred["baseline"].astype(float)
    obs = pred["observations"].astype(float)
    mask = pred["mask"].astype(float) > 0.5

    full_valid = (
        mask.all(axis=1)
        & np.isfinite(obs).all(axis=1)
        & np.isfinite(model).all(axis=1)
    )

    obs_acc = np.where(full_valid, np.nansum(obs, axis=1), np.nan)
    mod_acc = np.where(full_valid, np.nansum(model, axis=1), np.nan)
    bas_acc = np.where(full_valid, np.nansum(baseline, axis=1), np.nan)

    return obs_acc, mod_acc, bas_acc, full_valid


@dataclass(frozen=True)
class SelectedCase:
    variable: str
    kind: str
    index: int
    issue_time: pd.Timestamp
    score: float
    model_score: Optional[float]
    baseline_score: Optional[float]

    observed_accum: Optional[float] = None
    model_accum: Optional[float] = None
    baseline_accum: Optional[float] = None

    observed_peak_10min: Optional[float] = None
    model_peak_10min: Optional[float] = None
    baseline_peak_10min: Optional[float] = None

    observed_peak_lead_step: Optional[int] = None
    model_peak_lead_step: Optional[int] = None
    peak_time_error_steps: Optional[int] = None

    model_at_observed_peak_10min: Optional[float] = None
    baseline_at_observed_peak_10min: Optional[float] = None


# =============================================================================
# Selecció de casos
# =============================================================================

def select_temperature_cases(pred: Dict[str, np.ndarray]) -> Tuple[SelectedCase, SelectedCase]:
    issue_times = pd.to_datetime(pred["issue_times"])

    mae_model = finite_case_mae(
        pred["ensemble_mean"].astype(float),
        pred["observations"].astype(float),
        pred["mask"].astype(float),
    )
    mae_base = finite_case_mae(
        pred["baseline"].astype(float),
        pred["observations"].astype(float),
        pred["mask"].astype(float),
    )

    valid = np.isfinite(mae_model)
    if not valid.any():
        raise RuntimeError("No hi ha cap cas vàlid de temperatura.")

    valid_idx = np.where(valid)[0]
    best_idx = int(valid_idx[np.nanargmin(mae_model[valid_idx])])
    worst_idx = int(valid_idx[np.nanargmax(mae_model[valid_idx])])

    best = SelectedCase(
        variable="temperature",
        kind="millor",
        index=best_idx,
        issue_time=pd.Timestamp(issue_times[best_idx]),
        score=float(mae_model[best_idx]),
        model_score=float(mae_model[best_idx]),
        baseline_score=safe_float(mae_base[best_idx]),
    )

    worst = SelectedCase(
        variable="temperature",
        kind="pitjor",
        index=worst_idx,
        issue_time=pd.Timestamp(issue_times[worst_idx]),
        score=float(mae_model[worst_idx]),
        model_score=float(mae_model[worst_idx]),
        baseline_score=safe_float(mae_base[worst_idx]),
    )

    return best, worst


def _rain_peak_arrays(pred: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    obs_all = np.asarray(pred["observations"], dtype=float)
    mod_all = np.asarray(pred["ensemble_mean"], dtype=float)
    bas_all = np.asarray(pred["baseline"], dtype=float)
    mask_all = np.asarray(pred["mask"], dtype=float) > 0.5

    n_cases = obs_all.shape[0]

    obs_peak = np.full(n_cases, np.nan, dtype=float)
    mod_peak = np.full(n_cases, np.nan, dtype=float)
    bas_peak = np.full(n_cases, np.nan, dtype=float)

    obs_peak_idx = np.full(n_cases, -1, dtype=int)
    mod_peak_idx = np.full(n_cases, -1, dtype=int)

    model_at_obs_peak = np.full(n_cases, np.nan, dtype=float)
    baseline_at_obs_peak = np.full(n_cases, np.nan, dtype=float)

    for i in range(n_cases):
        valid_i = mask_all[i] & np.isfinite(obs_all[i]) & np.isfinite(mod_all[i])

        if not valid_i.any():
            continue

        obs_i = np.where(valid_i, obs_all[i], np.nan)
        mod_i = np.where(valid_i, mod_all[i], np.nan)
        bas_i = np.where(valid_i & np.isfinite(bas_all[i]), bas_all[i], np.nan)

        j_obs = int(np.nanargmax(obs_i))
        j_mod = int(np.nanargmax(mod_i))

        obs_peak[i] = float(obs_i[j_obs])
        mod_peak[i] = float(mod_i[j_mod])

        if np.isfinite(bas_i).any():
            bas_peak[i] = float(np.nanmax(bas_i))

        if np.isfinite(mod_i[j_obs]):
            model_at_obs_peak[i] = float(mod_i[j_obs])

        if np.isfinite(bas_i[j_obs]):
            baseline_at_obs_peak[i] = float(bas_i[j_obs])

        obs_peak_idx[i] = j_obs
        mod_peak_idx[i] = j_mod

    return {
        "obs_peak": obs_peak,
        "mod_peak": mod_peak,
        "bas_peak": bas_peak,
        "obs_peak_idx": obs_peak_idx,
        "mod_peak_idx": mod_peak_idx,
        "model_at_obs_peak": model_at_obs_peak,
        "baseline_at_obs_peak": baseline_at_obs_peak,
    }


def select_rain_cases(
    pred: Dict[str, np.ndarray],
    rain_min_accum_mm: float,
    rain_best_min_step_mm: float,
    rain_best_model_min_step_mm: float,
    rain_best_peak_time_tol_steps: int,
    rain_best_min_peak_fraction: float,
) -> Tuple[SelectedCase, SelectedCase]:
    issue_times = pd.to_datetime(pred["issue_times"])

    obs_acc, mod_acc, bas_acc, valid = rain_accumulations(pred)
    abs_acc_error = np.abs(mod_acc - obs_acc)

    peaks = _rain_peak_arrays(pred)
    obs_peak = peaks["obs_peak"]
    mod_peak = peaks["mod_peak"]
    bas_peak = peaks["bas_peak"]
    obs_peak_idx = peaks["obs_peak_idx"]
    mod_peak_idx = peaks["mod_peak_idx"]
    model_at_obs_peak = peaks["model_at_obs_peak"]
    baseline_at_obs_peak = peaks["baseline_at_obs_peak"]

    rainy = valid & np.isfinite(obs_acc) & (obs_acc >= float(rain_min_accum_mm))

    if not rainy.any():
        raise RuntimeError(
            f"No hi ha cap cas de precipitació amb acumulació observada >= {rain_min_accum_mm} mm. "
            "Prova un llindar més baix amb --rain-min-accum-mm."
        )

    intense_obs = (
        rainy
        & np.isfinite(obs_peak)
        & (obs_peak >= float(rain_best_min_step_mm))
    )

    intense_model = (
        np.isfinite(mod_peak)
        & (mod_peak >= float(rain_best_model_min_step_mm))
        & (mod_peak >= float(rain_best_min_peak_fraction) * obs_peak)
    )

    peak_time_error = np.abs(mod_peak_idx - obs_peak_idx)
    peak_time_ok = (
        (obs_peak_idx >= 0)
        & (mod_peak_idx >= 0)
        & (peak_time_error <= int(rain_best_peak_time_tol_steps))
    )

    best_candidates = intense_obs & intense_model & peak_time_ok

    if not best_candidates.any():
        n_intense_obs = int(np.sum(intense_obs))
        n_intense_model = int(np.sum(intense_obs & intense_model))
        n_time_ok = int(np.sum(intense_obs & intense_model & peak_time_ok))

        raise RuntimeError(
            "No hi ha cap cas prou bo de precipitació intensa amb els criteris actuals.\n"
            f"Candidats amb pic observat >= {rain_best_min_step_mm} mm/10 min: {n_intense_obs}\n"
            f"Candidats on el model també fa un pic prou fort: {n_intense_model}\n"
            f"Candidats on el pic també està dins la tolerància temporal: {n_time_ok}\n\n"
            "Prova, per ordre:\n"
            "  1) relaxar --rain-best-model-min-step-mm, per exemple 1.5;\n"
            "  2) relaxar --rain-best-min-peak-fraction, per exemple 0.4;\n"
            "  3) relaxar --rain-best-peak-time-tol-steps, per exemple 4 o 5."
        )

    eps = 1e-6

    peak_rel_err = np.abs(mod_peak - obs_peak) / np.maximum(obs_peak, eps)
    accum_rel_err = np.abs(mod_acc - obs_acc) / np.maximum(obs_acc, eps)
    peak_time_err_scaled = peak_time_error.astype(float) / max(1, int(rain_best_peak_time_tol_steps))

    best_score = (
        0.50 * peak_rel_err
        + 0.30 * accum_rel_err
        + 0.20 * peak_time_err_scaled
    )

    candidate_idx = np.where(best_candidates)[0]
    best_idx = int(candidate_idx[np.nanargmin(best_score[candidate_idx])])

    worst_candidate_idx = np.where(rainy)[0]
    worst_idx = int(worst_candidate_idx[np.nanargmax(abs_acc_error[worst_candidate_idx])])

    def make_case(idx: int, kind: str, score_value: float) -> SelectedCase:
        return SelectedCase(
            variable="rain",
            kind=kind,
            index=idx,
            issue_time=pd.Timestamp(issue_times[idx]),
            score=float(score_value),
            model_score=safe_float(abs_acc_error[idx]),
            baseline_score=safe_float(abs(bas_acc[idx] - obs_acc[idx])),
            observed_accum=safe_float(obs_acc[idx]),
            model_accum=safe_float(mod_acc[idx]),
            baseline_accum=safe_float(bas_acc[idx]),
            observed_peak_10min=safe_float(obs_peak[idx]),
            model_peak_10min=safe_float(mod_peak[idx]),
            baseline_peak_10min=safe_float(bas_peak[idx]),
            observed_peak_lead_step=int(obs_peak_idx[idx]) + 1 if obs_peak_idx[idx] >= 0 else None,
            model_peak_lead_step=int(mod_peak_idx[idx]) + 1 if mod_peak_idx[idx] >= 0 else None,
            peak_time_error_steps=int(peak_time_error[idx]) if obs_peak_idx[idx] >= 0 and mod_peak_idx[idx] >= 0 else None,
            model_at_observed_peak_10min=safe_float(model_at_obs_peak[idx]),
            baseline_at_observed_peak_10min=safe_float(baseline_at_obs_peak[idx]),
        )

    best = make_case(best_idx, "millor", best_score[best_idx])
    worst = make_case(worst_idx, "pitjor", abs_acc_error[worst_idx])

    return best, worst


# =============================================================================
# Figures
# =============================================================================

def local_ylim(
    values: Iterable[np.ndarray],
    pad_frac: float,
    min_pad: float,
    floor_zero: bool = False,
) -> Tuple[float, float]:
    arrs = [np.asarray(v, dtype=float).ravel() for v in values]

    if not arrs:
        return (0.0, 1.0)

    joined = np.concatenate(arrs)
    joined = joined[np.isfinite(joined)]

    if joined.size == 0:
        return (0.0, 1.0)

    vmin, vmax = float(joined.min()), float(joined.max())

    if floor_zero:
        vmin = 0.0

    spread = max(vmax - vmin, min_pad)
    pad = max(min_pad, spread * pad_frac)

    lo = vmin if floor_zero else vmin - pad
    hi = vmax + pad

    if floor_zero:
        lo = 0.0

    return lo, hi


def setup_matplotlib() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10.5,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#d1d5db",
        "axes.linewidth": 1.0,
        "savefig.bbox": "tight",
    })

    return plt, mdates, Polygon


def save_figure(fig: Any, output_base: Path, dpi: int, formats: Tuple[str, ...]) -> None:
    ensure_dir(output_base.parent)

    for fmt in formats:
        out = output_base.with_suffix(f".{fmt}")
        fig.savefig(out, dpi=dpi if fmt.lower() in {"png", "jpg", "jpeg"} else None)
        log(f"[OK] Figura: {out}")


def _shared_legend(fig: Any, ax: Any) -> None:
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))

    order = [
        "Ensemble",
        "Mitjana ensemble",
        "Línia base",
        "Observacions",
        "Observacions futures",
    ]

    ordered_handles = [by_label[k] for k in order if k in by_label]
    ordered_labels = [k for k in order if k in by_label]

    fig.legend(
        ordered_handles,
        ordered_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.96),
        ncol=5,
        frameon=False,
        fontsize=10,
        handlelength=2.6,
        columnspacing=1.4,
    )


def _draw_temperature_case_axis(
    ax: Any,
    case: SelectedCase,
    pred: Dict[str, np.ndarray],
    obs_df: pd.DataFrame,
    display_history_hours: float,
    temp_col: str,
    show_ylabel: bool = True,
) -> Dict[str, Any]:
    plt, mdates, Polygon = setup_matplotlib()

    idx = case.index
    issue_times = pd.to_datetime(pred["issue_times"])
    t0 = pd.Timestamp(issue_times[idx])

    horizon_steps = int(pred["ensemble_mean"].shape[1])
    dt = infer_dt_from_issue_times(issue_times)
    future_times = future_times_for_case(t0, horizon_steps, dt)

    x_future = mdates.date2num(future_times.to_pydatetime())
    x_left = t0 - pd.Timedelta(hours=float(display_history_hours))
    x_right = future_times[-1]

    past = obs_df.loc[(obs_df.index >= x_left) & (obs_df.index <= t0), temp_col]

    future_obs = pred["observations"][idx].astype(float)
    future_mask = pred["mask"][idx].astype(float) > 0.5

    members = pred["member_forecasts"][:, idx, :].astype(float)
    mean = pred["ensemble_mean"][idx].astype(float)
    q10 = pred["ensemble_q10"][idx].astype(float)
    q90 = pred["ensemble_q90"][idx].astype(float)
    baseline = pred["baseline"][idx].astype(float)

    y_limits = local_ylim(
        [past.to_numpy(), future_obs, members, mean, baseline],
        pad_frac=0.12,
        min_pad=0.8,
    )

    ax.set_facecolor("#ffffff")
    ax.axvspan(t0, x_right, color="#e0f2fe", alpha=0.42, zorder=1)

    poly = np.vstack([
        np.column_stack([x_future, q10]),
        np.column_stack([x_future[::-1], q90[::-1]]),
    ])
    ax.add_patch(
        Polygon(
            poly,
            closed=True,
            facecolor="#38bdf8",
            edgecolor="none",
            alpha=0.15,
            zorder=2,
        )
    )

    for m, arr in enumerate(members):
        ax.plot(
            future_times,
            arr,
            color="#7dd3fc",
            linewidth=1.0,
            alpha=0.28,
            label="Ensemble" if m == 0 else None,
            zorder=3,
        )

    ax.plot(
        past.index,
        past.values,
        color="#0f172a",
        linewidth=2.6,
        label="Observacions",
        zorder=6,
    )
    ax.plot(
        future_times[future_mask],
        future_obs[future_mask],
        color="#334155",
        linewidth=2.2,
        linestyle=(0, (3, 2)),
        label="Observacions futures",
        zorder=7,
    )
    ax.plot(
        future_times,
        mean,
        color="#0f766e",
        linewidth=3.2,
        label="Mitjana ensemble",
        zorder=8,
    )
    ax.plot(
        future_times,
        baseline,
        color="#f59e0b",
        linewidth=2.4,
        linestyle="--",
        label="Línia base",
        zorder=8,
    )

    ax.axvline(t0, color="#64748b", linewidth=1.4, linestyle=":", zorder=4)

    ax.set_xlim(x_left, x_right)
    ax.set_ylim(*y_limits)

    ax.set_xlabel("Temps")
    ax.set_ylabel("Temperatura (°C)" if show_ylabel else "")

    ax.grid(axis="y", color="#e2e8f0", linewidth=0.9)
    ax.grid(axis="x", color="#f1f5f9", linewidth=0.75)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M\n%d/%m"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=8))

    return {
        "emissio": str(t0),
        "ylim": y_limits,
        "mae_model_C": case.model_score,
        "mae_baseline_C": case.baseline_score,
    }


def plot_temperature_cases_combined(
    best_case: SelectedCase,
    worst_case: SelectedCase,
    pred: Dict[str, np.ndarray],
    obs_df: pd.DataFrame,
    output_base: Path,
    display_history_hours: float,
    dpi: int,
    formats: Tuple[str, ...],
    temp_col: str,
) -> Dict[str, Any]:
    plt, _, _ = setup_matplotlib()

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(15.0, 6.2),
        sharey=False,
        constrained_layout=False,
    )

    fig.patch.set_facecolor("#f8fafc")
    fig.subplots_adjust(top=0.84, bottom=0.16, left=0.07, right=0.985, wspace=0.12)

    best_info = _draw_temperature_case_axis(
        axes[0],
        best_case,
        pred,
        obs_df,
        display_history_hours=display_history_hours,
        temp_col=temp_col,
        show_ylabel=True,
    )

    worst_info = _draw_temperature_case_axis(
        axes[1],
        worst_case,
        pred,
        obs_df,
        display_history_hours=display_history_hours,
        temp_col=temp_col,
        show_ylabel=True,
    )

    _shared_legend(fig, axes[0])

    save_figure(fig, output_base, dpi=dpi, formats=formats)
    plt.close(fig)

    return {
        "variable": "temperatura",
        "cas": "millor_i_pitjor_en_una_figura",
        "criteri": (
            "Panell esquerre: millor cas segons MAE mitjana a 3 h. "
            "Panell dret: pitjor cas segons MAE mitjana a 3 h."
        ),
        "millor_cas": best_info,
        "pitjor_cas": worst_info,
        "output_base": str(output_base),
    }


def _draw_rain_case_axis(
    ax: Any,
    case: SelectedCase,
    pred: Dict[str, np.ndarray],
    obs_df: pd.DataFrame,
    display_history_hours: float,
    rain_col: str,
    show_ylabel: bool = True,
) -> Dict[str, Any]:
    plt, mdates, Polygon = setup_matplotlib()

    idx = case.index
    issue_times = pd.to_datetime(pred["issue_times"])
    t0 = pd.Timestamp(issue_times[idx])

    horizon_steps = int(pred["ensemble_mean"].shape[1])
    dt = infer_dt_from_issue_times(issue_times)
    future_times = future_times_for_case(t0, horizon_steps, dt)

    x_future = mdates.date2num(future_times.to_pydatetime())
    x_left = t0 - pd.Timedelta(hours=float(display_history_hours))
    x_right = future_times[-1]

    width_days = (dt.total_seconds() / 86400.0) * 0.82

    past = obs_df.loc[(obs_df.index >= x_left) & (obs_df.index <= t0), rain_col].clip(lower=0.0)

    future_obs = np.clip(pred["observations"][idx].astype(float), 0.0, None)
    future_mask = pred["mask"][idx].astype(float) > 0.5

    members = np.clip(pred["member_forecasts"][:, idx, :].astype(float), 0.0, None)
    mean = np.clip(pred["ensemble_mean"][idx].astype(float), 0.0, None)
    q10 = np.clip(pred["ensemble_q10"][idx].astype(float), 0.0, None)
    q90 = np.clip(pred["ensemble_q90"][idx].astype(float), 0.0, None)
    baseline = np.clip(pred["baseline"][idx].astype(float), 0.0, None)

    y_limits = local_ylim(
        [past.to_numpy(), future_obs, members, mean, baseline],
        pad_frac=0.18,
        min_pad=0.15,
        floor_zero=True,
    )

    if y_limits[1] < 1.0:
        y_limits = (0.0, 1.0)

    ax.set_facecolor("#ffffff")
    ax.axvspan(t0, x_right, color="#ecfeff", alpha=0.50, zorder=1)

    ax.bar(
        past.index,
        past.values,
        width=width_days,
        color="#0f172a",
        alpha=0.34,
        label="Observacions",
        zorder=4,
    )
    ax.bar(
        future_times[future_mask],
        future_obs[future_mask],
        width=width_days,
        color="#334155",
        alpha=0.52,
        label="Observacions futures",
        zorder=5,
    )

    poly = np.vstack([
        np.column_stack([x_future, q10]),
        np.column_stack([x_future[::-1], q90[::-1]]),
    ])
    ax.add_patch(
        Polygon(
            poly,
            closed=True,
            facecolor="#38bdf8",
            edgecolor="none",
            alpha=0.15,
            zorder=2,
        )
    )

    for m, arr in enumerate(members):
        ax.plot(
            future_times,
            arr,
            color="#7dd3fc",
            linewidth=1.0,
            alpha=0.30,
            label="Ensemble" if m == 0 else None,
            zorder=3,
        )

    ax.plot(
        future_times,
        mean,
        color="#0f766e",
        linewidth=3.0,
        marker="o",
        markersize=3.4,
        label="Mitjana ensemble",
        zorder=8,
    )
    ax.plot(
        future_times,
        baseline,
        color="#f59e0b",
        linewidth=2.3,
        linestyle="--",
        marker=".",
        label="Línia base",
        zorder=8,
    )

    ax.axvline(t0, color="#64748b", linewidth=1.4, linestyle=":", zorder=4)

    ax.set_xlim(x_left, x_right)
    ax.set_ylim(*y_limits)

    ax.set_xlabel("Temps")
    ax.set_ylabel("Precipitació (mm / 10 min)" if show_ylabel else "")

    ax.grid(axis="y", color="#e2e8f0", linewidth=0.9)
    ax.grid(axis="x", color="#f1f5f9", linewidth=0.75)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M\n%d/%m"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=8))

    return {
        "emissio": str(t0),
        "ylim": y_limits,
        "precipitacio_observada_3h_mm": case.observed_accum,
        "precipitacio_model_3h_mm": case.model_accum,
        "precipitacio_baseline_3h_mm": case.baseline_accum,
        "error_abs_model_3h_mm": case.model_score,
        "error_abs_baseline_3h_mm": case.baseline_score,
        "pic_observat_mm_10min": case.observed_peak_10min,
        "pic_model_mm_10min": case.model_peak_10min,
        "pic_baseline_mm_10min": case.baseline_peak_10min,
        "lead_pic_observat": case.observed_peak_lead_step,
        "lead_pic_model": case.model_peak_lead_step,
        "error_temporal_pic_passos_10min": case.peak_time_error_steps,
        "model_al_pic_observat_mm_10min": case.model_at_observed_peak_10min,
        "baseline_al_pic_observat_mm_10min": case.baseline_at_observed_peak_10min,
    }


def plot_rain_cases_combined(
    best_case: SelectedCase,
    worst_case: SelectedCase,
    pred: Dict[str, np.ndarray],
    obs_df: pd.DataFrame,
    output_base: Path,
    display_history_hours: float,
    dpi: int,
    formats: Tuple[str, ...],
    rain_col: str,
) -> Dict[str, Any]:
    plt, _, _ = setup_matplotlib()

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(15.0, 6.2),
        sharey=True,
        constrained_layout=False,
    )

    fig.patch.set_facecolor("#f8fafc")
    fig.subplots_adjust(top=0.84, bottom=0.16, left=0.07, right=0.985, wspace=0.08)

    best_info = _draw_rain_case_axis(
        axes[0],
        best_case,
        pred,
        obs_df,
        display_history_hours=display_history_hours,
        rain_col=rain_col,
        show_ylabel=True,
    )

    worst_info = _draw_rain_case_axis(
        axes[1],
        worst_case,
        pred,
        obs_df,
        display_history_hours=display_history_hours,
        rain_col=rain_col,
        show_ylabel=False,
    )

    ymax = max(float(best_info["ylim"][1]), float(worst_info["ylim"][1]))
    for ax in axes:
        ax.set_ylim(0.0, ymax)

    _shared_legend(fig, axes[0])

    save_figure(fig, output_base, dpi=dpi, formats=formats)
    plt.close(fig)

    return {
        "variable": "precipitacio",
        "cas": "millor_i_pitjor_en_una_figura",
        "criteri": (
            "Panell esquerre: millor cas de pluja intensa ben representada. "
            "Panell dret: pitjor cas amb pluja observada."
        ),
        "millor_cas": best_info,
        "pitjor_cas": worst_info,
        "output_base": str(output_base),
    }


# =============================================================================
# Paths i arguments
# =============================================================================

def default_cache_path(
    output_dir: Path,
    prefix: str,
    start: str,
    end: str,
    forecast_stride_steps: int,
    max_members: Optional[int],
) -> Path:
    def slug(s: str) -> str:
        return str(pd.Timestamp(s)).replace(":", "_").replace("-", "-").replace(" ", "_")

    member_tag = f"m{max_members}" if max_members is not None else "mall"
    return output_dir / f"{prefix}_nowcasts_{slug(start)}_{slug(end)}_fstride{forecast_stride_steps}_{member_tag}.npz"


def parse_formats(s: str) -> Tuple[str, ...]:
    out = tuple(x.strip().lower().lstrip(".") for x in s.split(",") if x.strip())

    if not out:
        raise ValueError("Cal indicar com a mínim un format de sortida.")

    allowed = {"png", "pdf", "svg", "jpg", "jpeg"}
    bad = [x for x in out if x not in allowed]

    if bad:
        raise ValueError(f"Formats no admesos: {bad}. Formats admesos: {sorted(allowed)}")

    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Genera una figura combinada per temperatura i una figura combinada "
            "per precipitació del nowcasting v9 2024."
        )
    )

    parser.add_argument(
        "--dataset",
        required=True,
        help="CSV o ZIP del dataset nowcasting no-RS.",
    )
    parser.add_argument(
        "--temperature-cache",
        default=None,
        help=".npz de temperatura generat per l'script d'animació.",
    )
    parser.add_argument(
        "--rain-cache",
        default=None,
        help=".npz de precipitació generat per l'script d'animació.",
    )
    parser.add_argument(
        "--temperature-output-dir",
        default="outputs/animation_temperature_2024",
        help="Directori on buscar la memòria cau de temperatura si no s'indica --temperature-cache.",
    )
    parser.add_argument(
        "--rain-output-dir",
        default="outputs/animation_precipitation_2024",
        help="Directori on buscar la memòria cau de precipitació si no s'indica --rain-cache.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/best_worst_cases_2024_combined",
        help="Directori de sortida de les figures.",
    )

    parser.add_argument("--start", default="2024-01-01 00:00:00")
    parser.add_argument("--end", default="2025-01-01 00:00:00")
    parser.add_argument("--forecast-stride-steps", type=int, default=1)

    parser.add_argument(
        "--max-members",
        type=int,
        default=None,
        help="Ha de coincidir amb la memòria cau. Per defecte assumeix tots els membres: mall.",
    )

    parser.add_argument("--temp-col", default="Temp_1_Avg")
    parser.add_argument("--rain-col", default="Lluvia_Tot")
    parser.add_argument("--display-history-hours", type=float, default=6.0)

    parser.add_argument(
        "--rain-min-accum-mm",
        type=float,
        default=1.0,
        help="Llindar mínim d'acumulació observada a 3 h per seleccionar casos de pluja.",
    )
    parser.add_argument(
        "--rain-best-min-step-mm",
        type=float,
        default=3.0,
        help=(
            "Llindar mínim del pic observat dins les 3 h futures per seleccionar "
            "el millor cas de precipitació intensa, en mm/10 min."
        ),
    )
    parser.add_argument(
        "--rain-best-model-min-step-mm",
        type=float,
        default=2.0,
        help=(
            "Pic mínim que també ha de predir el model per considerar un bon cas "
            "de precipitació intensa, en mm/10 min."
        ),
    )
    parser.add_argument(
        "--rain-best-min-peak-fraction",
        type=float,
        default=0.45,
        help=(
            "Fracció mínima del pic observat que ha d'assolir el pic predit pel model. "
            "Exemple: 0.45 vol dir que si s'observen 5 mm/10 min, el model ha de predir "
            "com a mínim 2.25 mm/10 min, a més del llindar absolut."
        ),
    )
    parser.add_argument(
        "--rain-best-peak-time-tol-steps",
        type=int,
        default=3,
        help=(
            "Tolerància màxima, en passos de 10 minuts, entre el pic observat i el pic "
            "predit del millor cas."
        ),
    )

    parser.add_argument("--dpi", type=int, default=260)
    parser.add_argument("--formats", default="png,pdf", help="Formats separats per coma: png,pdf,svg,jpg.")

    return parser


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    outdir = Path(args.output_dir).expanduser().resolve()
    ensure_dir(outdir)

    workdir = outdir / "_work"

    temp_cache = (
        Path(args.temperature_cache).expanduser().resolve()
        if args.temperature_cache
        else default_cache_path(
            Path(args.temperature_output_dir).expanduser().resolve(),
            prefix="temperature",
            start=args.start,
            end=args.end,
            forecast_stride_steps=args.forecast_stride_steps,
            max_members=args.max_members,
        )
    )

    rain_cache = (
        Path(args.rain_cache).expanduser().resolve()
        if args.rain_cache
        else default_cache_path(
            Path(args.rain_output_dir).expanduser().resolve(),
            prefix="rain",
            start=args.start,
            end=args.end,
            forecast_stride_steps=args.forecast_stride_steps,
            max_members=args.max_members,
        )
    )

    csv_path = resolve_dataset_csv(args.dataset, workdir)
    obs_df = read_observations(csv_path, temp_col=args.temp_col, rain_col=args.rain_col)

    temp_pred = load_npz(temp_cache)
    rain_pred = load_npz(rain_cache)

    temp_best, temp_worst = select_temperature_cases(temp_pred)

    rain_best, rain_worst = select_rain_cases(
        rain_pred,
        rain_min_accum_mm=args.rain_min_accum_mm,
        rain_best_min_step_mm=args.rain_best_min_step_mm,
        rain_best_model_min_step_mm=args.rain_best_model_min_step_mm,
        rain_best_peak_time_tol_steps=args.rain_best_peak_time_tol_steps,
        rain_best_min_peak_fraction=args.rain_best_min_peak_fraction,
    )

    formats = parse_formats(args.formats)

    summaries = []

    summaries.append(
        plot_temperature_cases_combined(
            temp_best,
            temp_worst,
            temp_pred,
            obs_df,
            output_base=outdir / "temperatura_casos_2024",
            display_history_hours=args.display_history_hours,
            dpi=args.dpi,
            formats=formats,
            temp_col=args.temp_col,
        )
    )

    summaries.append(
        plot_rain_cases_combined(
            rain_best,
            rain_worst,
            rain_pred,
            obs_df,
            output_base=outdir / "precipitacio_casos_intensos_2024",
            display_history_hours=args.display_history_hours,
            dpi=args.dpi,
            formats=formats,
            rain_col=args.rain_col,
        )
    )

    summary_df = pd.DataFrame(summaries)

    summary_csv = outdir / "resum_millors_pitjors_casos_2024.csv"
    summary_json = outdir / "resum_millors_pitjors_casos_2024.json"

    summary_df.to_csv(summary_csv, index=False)

    write_json(
        {
            "cases": summaries,
            "rain_min_accum_mm": args.rain_min_accum_mm,
            "rain_best_min_step_mm": args.rain_best_min_step_mm,
            "rain_best_model_min_step_mm": args.rain_best_model_min_step_mm,
            "rain_best_min_peak_fraction": args.rain_best_min_peak_fraction,
            "rain_best_peak_time_tol_steps": args.rain_best_peak_time_tol_steps,
            "notes": (
                "Les figures combinades no inclouen títol intern. "
                "La interpretació dels panells s'ha de posar al peu de figura de l'informe."
            ),
        },
        summary_json,
    )

    log(f"[OK] Resum CSV: {summary_csv}")
    log(f"[OK] Resum JSON: {summary_json}")
    log("[RESULTAT] " + json.dumps(json_safe({"cases": summaries}), ensure_ascii=False))


if __name__ == "__main__":
    main()