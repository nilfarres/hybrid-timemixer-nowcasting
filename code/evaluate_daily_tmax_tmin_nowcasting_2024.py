#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Avalua si el model Hybrid-TimeMixer v9 millora la predicció de la temperatura
màxima i mínima diària durant el 2024.

El codi parteix de la memòria cau .npz generada per:
    animate_temperature_nowcasting_v9_2024_v4_clean_window_fixed.py

No recalcula la xarxa neuronal. Fa tres comparacions:

1) Model v9:
   - mitjana de l'ensemble de temperatura.

2) Baseline interna del model:
   - baseline físic-estadística guardada dins de la mateixa memòria cau.

3) Model físic-estadístic local:
   - relaxació de l'anomalia tèrmica cap a una climatologia diürna train-only.
   - fórmula conceptual:
       T_pred(t+h) = T_clim(t+h) + [T(t)-T_clim(t)] exp(-h/tau) + tendència amortida
   - només usa observacions passades/presents de la mateixa estació i calendari.

Anàlisis que genera:
--------------------
A) Extrems diaris reconstruïts amb lead fix:
   - per exemple, Tmax/Tmin obtingudes amb prediccions a 1 h, 2 h i 3 h.
   - serveix per veure si el model preserva bé els extrems diaris quan cada punt
     del dia s'estima amb prediccions del mateix rang temporal.

B) Estimació intradia de l'extrem final del dia:
   - a cada emissió: extrem observat fins aquell moment + forecast de les 3 h següents.
   - serveix per veure si el model ajuda a anticipar millor la Tmax/Tmin final a mesura
     que avança el dia.

Sortides:
---------
- CSVs de prediccions diàries i resums.
- Figures PNG/PDF amb MAE/RMSE i skill.
- JSON de resum.

Exemple d'ús Windows cmd:
python evaluate_daily_tmax_tmin_nowcasting_2024.py ^
  --dataset CR3000_Tabla1_2011_2024_dataset_nowcasting_no_rs.zip ^
  --temperature-cache outputs/animation_temperature_2024/temperature_nowcasts_2024-01-01_00_00_00_2025-01-01_00_00_00_fstride1_mall.npz ^
  --output-dir outputs/daily_tmax_tmin_analysis_2024 ^
  --start "2024-01-01 00:00:00" ^
  --end "2025-01-01 00:00:00" ^
  --train-end "2023-01-01" ^
  --lead-steps 6,12,18 ^
  --formats png,pdf
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# Configuració
# =============================================================================

@dataclass
class Config:
    dataset: str
    temperature_cache: str
    output_dir: str

    start: str = "2024-01-01 00:00:00"
    end: str = "2025-01-01 00:00:00"
    train_end: str = "2023-01-01"

    timestamp_col: str = "TIMESTAMP"
    temp_col: str = "Temp_1_Avg"
    freq: str = "10min"

    lead_steps: str = "6,12,18"
    min_daily_coverage: float = 0.90

    # Model físic-estadístic local
    physical_tau_hours: float = 6.0
    physical_trend_hours: float = 2.0
    physical_trend_weight: float = 0.35
    physical_max_trend_delta_C: float = 2.0

    # Estimació intradia
    intraday_min_hour: int = 0
    intraday_max_hour: int = 23

    # Sortida
    formats: str = "png,pdf"
    dpi: int = 260
    title_prefix: str = "Nowcasting local de temperatura màxima i mínima (2024)"


# =============================================================================
# Utilitats
# =============================================================================

def log(msg: str) -> None:
    print(msg, flush=True)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_args() -> Config:
    p = argparse.ArgumentParser(
        description="Avalua Tmax/Tmin diàries a partir dels nowcasts de temperatura v9."
    )
    p.add_argument("--dataset", required=True, help="CSV o ZIP del dataset nowcasting.")
    p.add_argument("--temperature-cache", required=True, help="NPZ de temperatura generat per l'animació.")
    p.add_argument("--output-dir", required=True)

    p.add_argument("--start", default="2024-01-01 00:00:00")
    p.add_argument("--end", default="2025-01-01 00:00:00")
    p.add_argument("--train-end", default="2023-01-01")

    p.add_argument("--timestamp-col", default="TIMESTAMP")
    p.add_argument("--temp-col", default="Temp_1_Avg")
    p.add_argument("--freq", default="10min")

    p.add_argument(
        "--lead-steps",
        default="6,12,18",
        help="Leads en passos de 10 min per reconstruir extrems diaris. Ex.: 6,12,18 = 1,2,3 h.",
    )
    p.add_argument("--min-daily-coverage", type=float, default=0.90)

    p.add_argument("--physical-tau-hours", type=float, default=6.0)
    p.add_argument("--physical-trend-hours", type=float, default=2.0)
    p.add_argument("--physical-trend-weight", type=float, default=0.35)
    p.add_argument("--physical-max-trend-delta-C", type=float, default=2.0)

    p.add_argument("--intraday-min-hour", type=int, default=0)
    p.add_argument("--intraday-max-hour", type=int, default=23)

    p.add_argument("--formats", default="png,pdf", help="Formats de figures separats per coma.")
    p.add_argument("--dpi", type=int, default=260)
    p.add_argument("--title-prefix", default="Nowcasting local de temperatura màxima i mínima (2024)")

    return Config(**vars(p.parse_args()))


def resolve_dataset_path(dataset: Path, output_dir: Path) -> Path:
    """Admet dataset CSV o ZIP amb un CSV a dins."""
    dataset = dataset.expanduser().resolve()
    if dataset.is_file() and dataset.suffix.lower() == ".csv":
        return dataset
    if not (dataset.is_file() and dataset.suffix.lower() == ".zip"):
        raise FileNotFoundError(f"No trobo un CSV o ZIP vàlid: {dataset}")

    extract_dir = output_dir / "_dataset_extracted"
    ensure_dir(extract_dir)
    with zipfile.ZipFile(dataset, "r") as zf:
        csv_members = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not csv_members:
            raise FileNotFoundError(f"El ZIP no conté cap CSV: {dataset}")
        preferred = [n for n in csv_members if "dataset_nowcasting" in Path(n).name]
        member = preferred[0] if preferred else csv_members[0]
        out = extract_dir / Path(member).name
        if not out.exists():
            log(f"[INFO] Extraient dataset: {Path(member).name}")
            with zf.open(member, "r") as src, open(out, "wb") as dst:
                shutil.copyfileobj(src, dst)
        return out


def read_temperature_series(csv_path: Path, cfg: Config) -> pd.Series:
    usecols = [cfg.timestamp_col, cfg.temp_col]
    df = pd.read_csv(csv_path, usecols=usecols, low_memory=False)
    if cfg.timestamp_col not in df or cfg.temp_col not in df:
        raise ValueError(f"El CSV ha de contenir {cfg.timestamp_col} i {cfg.temp_col}.")

    df[cfg.timestamp_col] = pd.to_datetime(df[cfg.timestamp_col], errors="coerce")
    df[cfg.temp_col] = pd.to_numeric(df[cfg.temp_col], errors="coerce")
    df = df.dropna(subset=[cfg.timestamp_col]).sort_values(cfg.timestamp_col)
    df = df.drop_duplicates(subset=[cfg.timestamp_col], keep="last")

    s = df.set_index(cfg.timestamp_col)[cfg.temp_col].astype(float)
    # No reimputem. Només regularitzem l'índex per facilitar seleccions temporals.
    full_index = pd.date_range(s.index.min(), s.index.max(), freq=cfg.freq)
    s = s.reindex(full_index)
    s.index.name = cfg.timestamp_col
    return s


def load_temperature_cache(path: Path) -> Dict[str, np.ndarray]:
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"No existeix la memòria cau de temperatura: {path}")
    log(f"[INFO] Llegint memòria cau de temperatura: {path}")
    with np.load(path, allow_pickle=False) as data:
        pred = {k: data[k] for k in data.files}

    required = ["issue_times", "ensemble_mean", "baseline", "observations", "mask"]
    missing = [k for k in required if k not in pred]
    if missing:
        raise KeyError(f"Falten claus al NPZ: {missing}")

    pred["issue_times"] = pd.to_datetime(pred["issue_times"])
    return pred


def parse_lead_steps(text: str, horizon_steps: int) -> List[int]:
    leads: List[int] = []
    for token in str(text).split(","):
        token = token.strip()
        if not token:
            continue
        v = int(token)
        if v < 1 or v > horizon_steps:
            raise ValueError(f"Lead step fora de rang: {v}. Ha d'estar entre 1 i {horizon_steps}.")
        leads.append(v)
    if not leads:
        raise ValueError("No s'ha indicat cap lead vàlid.")
    return sorted(set(leads))


def finite_scores(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    if ok.sum() == 0:
        return {"n": 0, "mae": math.nan, "rmse": math.nan, "bias": math.nan}
    err = y_pred[ok] - y_true[ok]
    return {
        "n": int(ok.sum()),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "bias": float(np.mean(err)),
    }


def skill(reference_rmse: float, model_rmse: float) -> float:
    if not np.isfinite(reference_rmse) or reference_rmse <= 0 or not np.isfinite(model_rmse):
        return math.nan
    return float(1.0 - model_rmse / reference_rmse)


def safe_date(x: Any) -> str:
    return str(pd.Timestamp(x).date())


# =============================================================================
# Observació diària
# =============================================================================

def observed_daily_extremes(
    temp: pd.Series,
    start: pd.Timestamp,
    end: pd.Timestamp,
    cfg: Config,
) -> pd.DataFrame:
    """Calcula Tmax/Tmin observades amb cobertura mínima."""
    dt = pd.Timedelta(cfg.freq)
    expected_per_day = int(round(pd.Timedelta(days=1) / dt))

    s = temp.loc[(temp.index >= start) & (temp.index < end)]
    rows = []
    for day, group in s.groupby(s.index.floor("D")):
        n_valid = int(group.notna().sum())
        coverage = n_valid / expected_per_day if expected_per_day > 0 else math.nan
        if coverage < cfg.min_daily_coverage:
            continue
        rows.append({
            "date": pd.Timestamp(day).date(),
            "n_obs": n_valid,
            "coverage": float(coverage),
            "tmax_obs": float(group.max(skipna=True)),
            "tmin_obs": float(group.min(skipna=True)),
            "tmean_obs": float(group.mean(skipna=True)),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("No hi ha cap dia amb cobertura suficient per calcular extrems observats.")
    out["date"] = pd.to_datetime(out["date"])
    return out


# =============================================================================
# Model físic-estadístic local
# =============================================================================

def slot_of_day(index: pd.DatetimeIndex, freq: str) -> np.ndarray:
    dt_seconds = int(round(pd.Timedelta(freq).total_seconds()))
    seconds = (
        index.hour.to_numpy(dtype=np.int64) * 3600
        + index.minute.to_numpy(dtype=np.int64) * 60
        + index.second.to_numpy(dtype=np.int64)
    )
    return np.floor_divide(seconds, dt_seconds).astype(int)


class PhysicalDiurnalRelaxation:
    """
    Model físic-estadístic local:
    - climatologia diürna train-only T_clim(dia de l'any, hora)
    - persistència de l'anomalia respecte a T_clim amb relaxació exponencial
    - tendència recent amortida i limitada
    """

    def __init__(
        self,
        temp: pd.Series,
        train_end: pd.Timestamp,
        freq: str,
        tau_hours: float,
        trend_hours: float,
        trend_weight: float,
        max_trend_delta_C: float,
    ) -> None:
        self.temp = temp.astype(float)
        self.freq = freq
        self.tau_hours = float(tau_hours)
        self.trend_hours = float(trend_hours)
        self.trend_weight = float(trend_weight)
        self.max_trend_delta_C = float(max_trend_delta_C)

        train = self.temp.loc[self.temp.index < train_end].dropna()
        if train.empty:
            raise RuntimeError("No hi ha dades de train per construir el model físic-estadístic.")

        idx = train.index
        df = pd.DataFrame({
            "temp": train.to_numpy(dtype=float),
            "doy": idx.dayofyear.astype(int),
            "slot": slot_of_day(idx, freq),
        })
        clim = df.groupby(["doy", "slot"])["temp"].median()
        slot_clim = df.groupby("slot")["temp"].median()
        global_clim = float(train.median())

        self.clim = clim
        self.slot_clim = slot_clim
        self.global_clim = global_clim

    def climatology(self, times: pd.DatetimeIndex) -> np.ndarray:
        doys = times.dayofyear.astype(int)
        slots = slot_of_day(times, self.freq)
        out = np.empty(len(times), dtype=float)
        for i, (doy, slot) in enumerate(zip(doys, slots)):
            v = self.clim.get((int(doy), int(slot)), np.nan)
            if not np.isfinite(v):
                v = self.slot_clim.get(int(slot), np.nan)
            if not np.isfinite(v):
                v = self.global_clim
            out[i] = float(v)
        return out

    def predict(self, issue_times: pd.DatetimeIndex, target_times: pd.DatetimeIndex) -> np.ndarray:
        issue_times = pd.DatetimeIndex(issue_times)
        target_times = pd.DatetimeIndex(target_times)

        T0 = self.temp.reindex(issue_times).to_numpy(dtype=float)
        eq0 = self.climatology(issue_times)
        eqf = self.climatology(target_times)

        lead_hours = (target_times - issue_times).total_seconds().to_numpy(dtype=float) / 3600.0
        lead_hours = np.maximum(lead_hours, 0.0)

        trend_delta = np.zeros(len(issue_times), dtype=float)
        if self.trend_hours > 0:
            past_times = issue_times - pd.Timedelta(hours=self.trend_hours)
            Tprev = self.temp.reindex(past_times).to_numpy(dtype=float)
            slope = (T0 - Tprev) / self.trend_hours
            slope = np.where(np.isfinite(slope), slope, 0.0)
            raw_delta = slope * self.tau_hours * (1.0 - np.exp(-lead_hours / max(self.tau_hours, 1e-6)))
            raw_delta = np.clip(raw_delta, -abs(self.max_trend_delta_C), abs(self.max_trend_delta_C))
            trend_delta = self.trend_weight * raw_delta

        anomaly = T0 - eq0
        pred = eqf + anomaly * np.exp(-lead_hours / max(self.tau_hours, 1e-6)) + trend_delta
        pred[~np.isfinite(T0)] = np.nan
        return pred.astype(float)


# =============================================================================
# Reconstrucció d'extrems diaris amb lead fix
# =============================================================================

def fixed_lead_daily_extremes(
    pred: Dict[str, np.ndarray],
    temp_series: pd.Series,
    obs_daily: pd.DataFrame,
    physical: PhysicalDiurnalRelaxation,
    leads: Sequence[int],
    start: pd.Timestamp,
    end: pd.Timestamp,
    cfg: Config,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Per cada lead step k, reconstrueix la sèrie de temperatura prevista del dia:
      target_time = issue_time + k*dt

    Aleshores calcula Tmax/Tmin per dia.
    """
    issue_times = pd.DatetimeIndex(pred["issue_times"])
    dt = pd.Timedelta(cfg.freq)

    model = pred["ensemble_mean"].astype(float)
    baseline = pred["baseline"].astype(float)
    truth = pred["observations"].astype(float)
    mask = pred["mask"].astype(float) > 0.5

    obs_daily_idx = obs_daily.set_index("date")
    rows: List[Dict[str, Any]] = []
    point_rows: List[Dict[str, Any]] = []

    for k in leads:
        lead_hours = k * dt.total_seconds() / 3600.0
        target_times = issue_times + k * dt
        in_period = (target_times >= start) & (target_times < end) & mask[:, k - 1]

        t_issue = issue_times[in_period]
        t_target = target_times[in_period]
        if len(t_target) == 0:
            continue

        phys_pred = physical.predict(t_issue, t_target)

        tmp = pd.DataFrame({
            "issue_time": t_issue,
            "target_time": t_target,
            "date": t_target.floor("D"),
            "lead_steps": int(k),
            "lead_hours": float(lead_hours),
            "obs": truth[in_period, k - 1],
            "model": model[in_period, k - 1],
            "baseline": baseline[in_period, k - 1],
            "physical": phys_pred,
        })
        tmp = tmp.replace([np.inf, -np.inf], np.nan).dropna(subset=["obs"])
        point_rows.extend(tmp.to_dict("records"))

        for (day, group) in tmp.groupby("date"):
            if day not in obs_daily_idx.index:
                continue
            obs_row = obs_daily_idx.loc[day]
            # Cal que hi hagi prou punts previstos dins del dia.
            expected = int(round(pd.Timedelta(days=1) / dt))
            if len(group) < cfg.min_daily_coverage * expected:
                continue

            row = {
                "date": pd.Timestamp(day),
                "lead_steps": int(k),
                "lead_hours": float(lead_hours),
                "n_points": int(len(group)),
                "tmax_obs": float(obs_row["tmax_obs"]),
                "tmin_obs": float(obs_row["tmin_obs"]),
            }
            for name in ["model", "baseline", "physical"]:
                row[f"tmax_{name}"] = float(group[name].max(skipna=True))
                row[f"tmin_{name}"] = float(group[name].min(skipna=True))
            rows.append(row)

    daily = pd.DataFrame(rows)
    points = pd.DataFrame(point_rows)
    if daily.empty:
        raise RuntimeError("No s'ha pogut reconstruir cap extrem diari amb els leads indicats.")
    return daily, points


def summarize_fixed_lead(daily: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for lead_hours, g_lead in daily.groupby("lead_hours"):
        for variable in ["tmax", "tmin"]:
            for model_name in ["model", "baseline", "physical"]:
                scores = finite_scores(g_lead[f"{variable}_obs"].to_numpy(), g_lead[f"{variable}_{model_name}"].to_numpy())
                row = {
                    "analysis": "fixed_lead_daily_extreme",
                    "variable": variable,
                    "model": model_name,
                    "lead_hours": float(lead_hours),
                    **scores,
                }
                rows.append(row)

            # Skills del model v9 respecte als dos referents.
            s_model = finite_scores(g_lead[f"{variable}_obs"].to_numpy(), g_lead[f"{variable}_model"].to_numpy())
            s_base = finite_scores(g_lead[f"{variable}_obs"].to_numpy(), g_lead[f"{variable}_baseline"].to_numpy())
            s_phys = finite_scores(g_lead[f"{variable}_obs"].to_numpy(), g_lead[f"{variable}_physical"].to_numpy())

            rows.append({
                "analysis": "fixed_lead_daily_extreme",
                "variable": variable,
                "model": "skill_model_vs_baseline",
                "lead_hours": float(lead_hours),
                "n": s_model["n"],
                "mae": math.nan,
                "rmse": skill(s_base["rmse"], s_model["rmse"]),
                "bias": math.nan,
            })
            rows.append({
                "analysis": "fixed_lead_daily_extreme",
                "variable": variable,
                "model": "skill_model_vs_physical",
                "lead_hours": float(lead_hours),
                "n": s_model["n"],
                "mae": math.nan,
                "rmse": skill(s_phys["rmse"], s_model["rmse"]),
                "bias": math.nan,
            })

    return pd.DataFrame(rows)


# =============================================================================
# Estimació intradia de Tmax/Tmin final del dia
# =============================================================================

def intraday_daily_extreme_updates(
    pred: Dict[str, np.ndarray],
    temp_series: pd.Series,
    obs_daily: pd.DataFrame,
    physical: PhysicalDiurnalRelaxation,
    start: pd.Timestamp,
    end: pd.Timestamp,
    cfg: Config,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    A cada emissió t0:
      estimate_Tmax_today = max(observació del dia fins a t0, forecast t0+10min..t0+3h dins el mateix dia)
      estimate_Tmin_today = min(...)
    """
    issue_times = pd.DatetimeIndex(pred["issue_times"])
    model = pred["ensemble_mean"].astype(float)
    baseline = pred["baseline"].astype(float)
    mask = pred["mask"].astype(float) > 0.5

    dt = pd.Timedelta(cfg.freq)
    horizon_steps = model.shape[1]
    future_offsets = np.arange(1, horizon_steps + 1) * dt

    obs_daily_idx = obs_daily.set_index("date")
    rows: List[Dict[str, Any]] = []

    # Precalculem extrems observats acumulats del dia per eficiència.
    s_period = temp_series.loc[(temp_series.index >= start - pd.Timedelta(days=1)) & (temp_series.index < end)]
    cummax_by_time = s_period.groupby(s_period.index.floor("D")).cummax()
    cummin_by_time = s_period.groupby(s_period.index.floor("D")).cummin()

    for i, t0 in enumerate(issue_times):
        if t0 < start or t0 >= end:
            continue
        if t0.hour < cfg.intraday_min_hour or t0.hour > cfg.intraday_max_hour:
            continue

        day = pd.Timestamp(t0.floor("D"))
        if day not in obs_daily_idx.index:
            continue

        obs_so_far_max = cummax_by_time.reindex([t0]).iloc[0]
        obs_so_far_min = cummin_by_time.reindex([t0]).iloc[0]
        if not np.isfinite(obs_so_far_max) or not np.isfinite(obs_so_far_min):
            continue

        future_times = pd.DatetimeIndex([t0 + off for off in future_offsets])
        same_day = future_times.floor("D") == day
        valid_future = same_day & mask[i, :]
        if not valid_future.any():
            continue

        phys = physical.predict(pd.DatetimeIndex([t0] * horizon_steps), future_times)

        candidates: Dict[str, np.ndarray] = {
            "model": model[i, valid_future],
            "baseline": baseline[i, valid_future],
            "physical": phys[valid_future],
        }

        row: Dict[str, Any] = {
            "issue_time": t0,
            "date": day,
            "issue_hour": int(t0.hour),
            "n_future_points": int(valid_future.sum()),
            "tmax_obs": float(obs_daily_idx.loc[day, "tmax_obs"]),
            "tmin_obs": float(obs_daily_idx.loc[day, "tmin_obs"]),
            "tmax_observed_so_far": float(obs_so_far_max),
            "tmin_observed_so_far": float(obs_so_far_min),
        }

        for name, arr in candidates.items():
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                row[f"tmax_{name}"] = float(obs_so_far_max)
                row[f"tmin_{name}"] = float(obs_so_far_min)
            else:
                row[f"tmax_{name}"] = float(max(obs_so_far_max, np.nanmax(arr)))
                row[f"tmin_{name}"] = float(min(obs_so_far_min, np.nanmin(arr)))

        # Referència simple: no fer nowcast, només usar l'extrem observat fins ara.
        row["tmax_so_far"] = float(obs_so_far_max)
        row["tmin_so_far"] = float(obs_so_far_min)
        rows.append(row)

    updates = pd.DataFrame(rows)
    if updates.empty:
        raise RuntimeError("No s'ha pogut calcular cap actualització intradia.")
    summary = summarize_intraday(updates)
    return updates, summary


def summarize_intraday(updates: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    model_names = ["model", "baseline", "physical", "so_far"]
    for hour, g_hour in updates.groupby("issue_hour"):
        for variable in ["tmax", "tmin"]:
            for model_name in model_names:
                scores = finite_scores(g_hour[f"{variable}_obs"].to_numpy(), g_hour[f"{variable}_{model_name}"].to_numpy())
                rows.append({
                    "analysis": "intraday_daily_extreme_update",
                    "variable": variable,
                    "model": model_name,
                    "issue_hour": int(hour),
                    **scores,
                })
    return pd.DataFrame(rows)


# =============================================================================
# Gràfics
# =============================================================================

def setup_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#d1d5db",
        "axes.linewidth": 1.0,
        "axes.titleweight": "bold",
        "figure.facecolor": "#f8fafc",
        "axes.facecolor": "white",
        "grid.color": "#e2e8f0",
        "grid.linewidth": 0.8,
    })


MODEL_LABELS = {
    "model": "Hybrid-TimeMixer v9",
    "baseline": "Baseline interna",
    "physical": "Model físic-estadístic",
    "so_far": "Observat fins ara",
}

MODEL_COLORS = {
    "model": "#0f766e",
    "baseline": "#f59e0b",
    "physical": "#6366f1",
    "so_far": "#64748b",
}


def save_figure(fig: plt.Figure, out_base: Path, formats: Sequence[str], dpi: int) -> None:
    ensure_dir(out_base.parent)
    for fmt in formats:
        fmt = fmt.strip().lower().lstrip(".")
        if not fmt:
            continue
        path = out_base.with_suffix(f".{fmt}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        log(f"[OK] Figura: {path}")
    plt.close(fig)


def plot_fixed_lead_mae(summary: pd.DataFrame, variable: str, out_dir: Path, cfg: Config, formats: Sequence[str]) -> None:
    setup_style()
    fig, ax = plt.subplots(figsize=(9.2, 5.3))
    sub = summary[
        (summary["analysis"] == "fixed_lead_daily_extreme")
        & (summary["variable"] == variable)
        & (summary["model"].isin(["model", "baseline", "physical"]))
    ].copy()
    for model_name, g in sub.groupby("model"):
        g = g.sort_values("lead_hours")
        ax.plot(
            g["lead_hours"],
            g["mae"],
            marker="o",
            linewidth=2.5,
            label=MODEL_LABELS.get(model_name, model_name),
            color=MODEL_COLORS.get(model_name),
        )
    ax.set_title(f"{cfg.title_prefix}\nMAE de {'Tmax' if variable == 'tmax' else 'Tmin'} segons el lead fix")
    ax.set_xlabel("Lead utilitzat per reconstruir el dia (h)")
    ax.set_ylabel("MAE (°C)")
    ax.grid(True, axis="y")
    ax.legend(frameon=False)
    save_figure(fig, out_dir / f"mae_{variable}_lead_fix", formats, cfg.dpi)


def plot_fixed_lead_skill(summary: pd.DataFrame, variable: str, out_dir: Path, cfg: Config, formats: Sequence[str]) -> None:
    setup_style()
    fig, ax = plt.subplots(figsize=(9.2, 5.3))
    sub = summary[
        (summary["analysis"] == "fixed_lead_daily_extreme")
        & (summary["variable"] == variable)
        & (summary["model"].isin(["skill_model_vs_baseline", "skill_model_vs_physical"]))
    ].copy()
    labels = {
        "skill_model_vs_baseline": "Skill vs baseline interna",
        "skill_model_vs_physical": "Skill vs model físic-estadístic",
    }
    colors = {
        "skill_model_vs_baseline": "#0ea5e9",
        "skill_model_vs_physical": "#8b5cf6",
    }
    for model_name, g in sub.groupby("model"):
        g = g.sort_values("lead_hours")
        ax.plot(
            g["lead_hours"],
            100.0 * g["rmse"],
            marker="o",
            linewidth=2.5,
            label=labels.get(model_name, model_name),
            color=colors.get(model_name),
        )
    ax.axhline(0, color="#94a3b8", linewidth=1.2, linestyle="--")
    ax.set_title(f"{cfg.title_prefix}\nSkill de RMSE de {'Tmax' if variable == 'tmax' else 'Tmin'}")
    ax.set_xlabel("Lead utilitzat per reconstruir el dia (h)")
    ax.set_ylabel("Skill de RMSE (%)")
    ax.grid(True, axis="y")
    ax.legend(frameon=False)
    save_figure(fig, out_dir / f"skill_{variable}_lead_fix", formats, cfg.dpi)


def plot_intraday_mae(summary: pd.DataFrame, variable: str, out_dir: Path, cfg: Config, formats: Sequence[str]) -> None:
    setup_style()
    fig, ax = plt.subplots(figsize=(10.2, 5.5))
    sub = summary[
        (summary["analysis"] == "intraday_daily_extreme_update")
        & (summary["variable"] == variable)
        & (summary["model"].isin(["model", "baseline", "physical", "so_far"]))
    ].copy()
    for model_name, g in sub.groupby("model"):
        g = g.sort_values("issue_hour")
        ax.plot(
            g["issue_hour"],
            g["mae"],
            marker="o",
            linewidth=2.2,
            markersize=4,
            label=MODEL_LABELS.get(model_name, model_name),
            color=MODEL_COLORS.get(model_name),
        )
    ax.set_title(f"{cfg.title_prefix}\nActualització intradia de {'Tmax' if variable == 'tmax' else 'Tmin'}")
    ax.set_xlabel("Hora d'emissió")
    ax.set_ylabel("MAE respecte a l'extrem final del dia (°C)")
    ax.set_xticks(range(0, 24, 2))
    ax.grid(True, axis="y")
    ax.legend(frameon=False, ncol=2)
    save_figure(fig, out_dir / f"mae_intradia_{variable}", formats, cfg.dpi)


def plot_scatter_best_lead(daily: pd.DataFrame, summary: pd.DataFrame, variable: str, out_dir: Path, cfg: Config, formats: Sequence[str]) -> None:
    setup_style()
    sub = summary[
        (summary["analysis"] == "fixed_lead_daily_extreme")
        & (summary["variable"] == variable)
        & (summary["model"] == "model")
    ].sort_values("mae")
    if sub.empty:
        return
    best_lead = float(sub.iloc[0]["lead_hours"])
    d = daily[daily["lead_hours"] == best_lead].copy()
    obs_col = f"{variable}_obs"
    pred_col = f"{variable}_model"
    ok = np.isfinite(d[obs_col]) & np.isfinite(d[pred_col])
    d = d.loc[ok]
    if d.empty:
        return

    vmin = float(min(d[obs_col].min(), d[pred_col].min()))
    vmax = float(max(d[obs_col].max(), d[pred_col].max()))
    pad = max(0.5, 0.05 * (vmax - vmin))

    fig, ax = plt.subplots(figsize=(6.4, 6.1))
    ax.scatter(d[obs_col], d[pred_col], s=24, alpha=0.72, color="#0f766e", edgecolor="none")
    ax.plot([vmin - pad, vmax + pad], [vmin - pad, vmax + pad], color="#94a3b8", linestyle="--", linewidth=1.4)
    ax.set_xlim(vmin - pad, vmax + pad)
    ax.set_ylim(vmin - pad, vmax + pad)
    ax.set_title(f"{'Tmax' if variable == 'tmax' else 'Tmin'} diària: observada vs model\nLead seleccionat: {best_lead:.1f} h")
    ax.set_xlabel("Observació (°C)")
    ax.set_ylabel("Predicció Hybrid-TimeMixer v9 (°C)")
    ax.grid(True)
    save_figure(fig, out_dir / f"scatter_{variable}_millor_lead", formats, cfg.dpi)


# =============================================================================
# Programa principal
# =============================================================================

def main() -> None:
    cfg = parse_args()
    out_dir = Path(cfg.output_dir).expanduser().resolve()
    ensure_dir(out_dir)

    start = pd.Timestamp(cfg.start)
    end = pd.Timestamp(cfg.end)
    train_end = pd.Timestamp(cfg.train_end)
    dt = pd.Timedelta(cfg.freq)

    dataset_csv = resolve_dataset_path(Path(cfg.dataset), out_dir)
    temp = read_temperature_series(dataset_csv, cfg)

    pred = load_temperature_cache(Path(cfg.temperature_cache))
    horizon_steps = int(pred["ensemble_mean"].shape[1])
    leads = parse_lead_steps(cfg.lead_steps, horizon_steps)

    obs_daily = observed_daily_extremes(temp, start, end, cfg)
    log(f"[INFO] Dies observats amb cobertura suficient: {len(obs_daily)}")

    physical = PhysicalDiurnalRelaxation(
        temp=temp,
        train_end=train_end,
        freq=cfg.freq,
        tau_hours=cfg.physical_tau_hours,
        trend_hours=cfg.physical_trend_hours,
        trend_weight=cfg.physical_trend_weight,
        max_trend_delta_C=cfg.physical_max_trend_delta_C,
    )

    daily_fixed, point_fixed = fixed_lead_daily_extremes(
        pred=pred,
        temp_series=temp,
        obs_daily=obs_daily,
        physical=physical,
        leads=leads,
        start=start,
        end=end,
        cfg=cfg,
    )
    summary_fixed = summarize_fixed_lead(daily_fixed)

    updates, summary_intraday = intraday_daily_extreme_updates(
        pred=pred,
        temp_series=temp,
        obs_daily=obs_daily,
        physical=physical,
        start=start,
        end=end,
        cfg=cfg,
    )

    # Export CSV
    daily_fixed_path = out_dir / "daily_extremes_fixed_lead_2024.csv"
    point_fixed_path = out_dir / "point_forecasts_fixed_lead_2024.csv"
    summary_fixed_path = out_dir / "summary_fixed_lead_daily_extremes_2024.csv"
    updates_path = out_dir / "intraday_extreme_updates_2024.csv"
    summary_intraday_path = out_dir / "summary_intraday_extreme_updates_2024.csv"

    daily_fixed.to_csv(daily_fixed_path, index=False)
    point_fixed.to_csv(point_fixed_path, index=False)
    summary_fixed.to_csv(summary_fixed_path, index=False)
    updates.to_csv(updates_path, index=False)
    summary_intraday.to_csv(summary_intraday_path, index=False)

    log(f"[OK] CSV: {daily_fixed_path}")
    log(f"[OK] CSV: {summary_fixed_path}")
    log(f"[OK] CSV: {updates_path}")
    log(f"[OK] CSV: {summary_intraday_path}")

    # Figures
    formats = [f.strip().lower() for f in cfg.formats.split(",") if f.strip()]
    for variable in ["tmax", "tmin"]:
        plot_fixed_lead_mae(summary_fixed, variable, out_dir, cfg, formats)
        plot_fixed_lead_skill(summary_fixed, variable, out_dir, cfg, formats)
        plot_intraday_mae(summary_intraday, variable, out_dir, cfg, formats)
        plot_scatter_best_lead(daily_fixed, summary_fixed, variable, out_dir, cfg, formats)

    # Resum JSON + resposta compacta per terminal
    best_rows: List[Dict[str, Any]] = []
    for variable in ["tmax", "tmin"]:
        sub = summary_fixed[
            (summary_fixed["analysis"] == "fixed_lead_daily_extreme")
            & (summary_fixed["variable"] == variable)
            & (summary_fixed["model"].isin(["model", "baseline", "physical"]))
        ].copy()
        if sub.empty:
            continue
        best_model_row = sub[sub["model"] == "model"].sort_values("mae").iloc[0]
        lead = float(best_model_row["lead_hours"])
        at_lead = sub[sub["lead_hours"] == lead].set_index("model")
        out = {
            "variable": variable,
            "best_model_lead_hours": lead,
            "model_mae_C": float(at_lead.loc["model", "mae"]) if "model" in at_lead.index else math.nan,
            "baseline_mae_C": float(at_lead.loc["baseline", "mae"]) if "baseline" in at_lead.index else math.nan,
            "physical_mae_C": float(at_lead.loc["physical", "mae"]) if "physical" in at_lead.index else math.nan,
            "model_rmse_C": float(at_lead.loc["model", "rmse"]) if "model" in at_lead.index else math.nan,
            "baseline_rmse_C": float(at_lead.loc["baseline", "rmse"]) if "baseline" in at_lead.index else math.nan,
            "physical_rmse_C": float(at_lead.loc["physical", "rmse"]) if "physical" in at_lead.index else math.nan,
        }
        out["rmse_skill_vs_baseline"] = skill(out["baseline_rmse_C"], out["model_rmse_C"])
        out["rmse_skill_vs_physical"] = skill(out["physical_rmse_C"], out["model_rmse_C"])
        best_rows.append(out)

    summary_json = {
        "config": asdict(cfg),
        "n_observed_days": int(len(obs_daily)),
        "lead_steps": leads,
        "lead_hours": [float(k * dt.total_seconds() / 3600.0) for k in leads],
        "physical_model": {
            "name": "diurnal_anomaly_relaxation",
            "tau_hours": cfg.physical_tau_hours,
            "trend_hours": cfg.physical_trend_hours,
            "trend_weight": cfg.physical_trend_weight,
            "max_trend_delta_C": cfg.physical_max_trend_delta_C,
            "train_end": cfg.train_end,
        },
        "best_fixed_lead_summary": best_rows,
    }
    json_path = out_dir / "summary_daily_tmax_tmin_analysis_2024.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2, ensure_ascii=False)
    log(f"[OK] JSON: {json_path}")

    print("\n[RESULTAT]")
    print(json.dumps(summary_json["best_fixed_lead_summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
