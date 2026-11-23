#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comparació temporalment homogènia horària de Tmax/Tmin diàries:
Hybrid-TimeMixer v9 vs models NWP operacionals globals/regionals.

Aquest script és una versió metodològicament més estricta de la comparació
amb GFS/ECMWF/ICON/AROME:

- Observacions UB: només valors a hores exactes.
- Hybrid-TimeMixer v9: només prediccions amb target_time a hores exactes.
- NWP: valors horaris retornats per Open-Meteo Historical Forecast API.
- Tmax/Tmin: calculades totes sobre la mateixa graella horària.

A més, els skill scores del model local respecte a cada NWP es calculen SEMPRE
sobre els dies comuns on tots dos tenen valor vàlid.

Exemple Windows cmd:
--------------------
python compare_tmax_tmin_operational_nwp_hourly_homogeneous_2024.py ^
  --dataset CR3000_Tabla1_2011_2024_dataset_nowcasting_no_rs.zip ^
  --temperature-cache outputs/animation_temperature_2024/temperature_nowcasts_2024-01-01_00_00_00_2025-01-01_00_00_00_fstride1_mall.npz ^
  --output-dir outputs/tmax_tmin_operational_nwp_hourly_2024 ^
  --nwp-cache-dir outputs/tmax_tmin_operational_nwp_arome_2024/nwp_cache ^
  --start "2024-01-01" ^
  --end "2025-01-01" ^
  --latitude 41.3864 ^
  --longitude 2.1177 ^
  --timezone "Europe/Madrid" ^
  --nwp-models arome_france_hd,arome_france,ncep_gfs013,icon_eu,arpege_europe,ecmwf_ifs025 ^
  --lead-step 18 ^
  --formats png,pdf

Notes:
------
- --lead-step 18 equival a +3 h perquè el model local és cada 10 min.
- Si --nwp-cache-dir conté els CSVs Open-Meteo ja descarregats, no es torna a
  descarregar res. Si falta algun model, el script prova de descarregar-lo.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import requests

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

    start: str = "2024-01-01"
    end: str = "2025-01-01"

    latitude: float = 41.3864
    longitude: float = 2.1177
    timezone: str = "Europe/Madrid"

    timestamp_col: str = "TIMESTAMP"
    temp_col: str = "Temp_1_Avg"
    freq: str = "10min"

    # 18 passos * 10 min = 3 h.
    lead_step: int = 18

    nwp_models: str = "arome_france_hd,arome_france,ncep_gfs013,icon_eu,arpege_europe,ecmwf_ifs025"
    openmeteo_url: str = "https://historical-forecast-api.open-meteo.com/v1/forecast"
    openmeteo_timeout: int = 60
    chunk_days: int = 31
    request_sleep_s: float = 0.25
    force_download: bool = False
    nwp_cache_dir: Optional[str] = None

    min_obs_daily_coverage: float = 0.90
    min_local_daily_coverage: float = 0.90
    min_nwp_daily_coverage: float = 0.80

    formats: str = "png,pdf"
    dpi: int = 260
    title_prefix: str = "Comparació horària homogènia de Tmax/Tmin (2024)"


MODEL_LABELS = {
    "hybrid_timemixer_v9": "Hybrid-TimeMixer v9",
    "baseline_interna": "Baseline interna",
    "ncep_gfs013": "GFS 0.11°",
    "ncep_gfs025": "GFS 0.25°",
    "ecmwf_ifs025": "ECMWF IFS 0.25°",
    "ecmwf_ifs04": "ECMWF IFS 0.4°",
    "ecmwf_ifs": "ECMWF IFS",
    "icon_eu": "ICON-EU ~7 km",
    "arpege_europe": "ARPEGE Europe ~11 km",
    "arome_france": "AROME France ~2.5 km",
    "arome_france_hd": "AROME France HD ~1.5 km",
}

MODEL_COLORS = {
    "hybrid_timemixer_v9": "#0f766e",
    "baseline_interna": "#f59e0b",
    "ncep_gfs013": "#2563eb",
    "ncep_gfs025": "#1d4ed8",
    "ecmwf_ifs025": "#7c3aed",
    "ecmwf_ifs04": "#8b5cf6",
    "ecmwf_ifs": "#6d28d9",
    "icon_eu": "#0891b2",
    "arpege_europe": "#dc2626",
    "arome_france": "#9333ea",
    "arome_france_hd": "#581c87",
}


# =============================================================================
# Utilitats
# =============================================================================

def log(msg: str) -> None:
    print(msg, flush=True)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_args() -> Config:
    p = argparse.ArgumentParser(
        description="Comparació horària homogènia de Tmax/Tmin: HTM v9 vs NWP operacional."
    )
    p.add_argument("--dataset", required=True)
    p.add_argument("--temperature-cache", required=True)
    p.add_argument("--output-dir", required=True)

    p.add_argument("--start", default="2024-01-01")
    p.add_argument("--end", default="2025-01-01")
    p.add_argument("--latitude", type=float, default=41.3864)
    p.add_argument("--longitude", type=float, default=2.1177)
    p.add_argument("--timezone", default="Europe/Madrid")

    p.add_argument("--timestamp-col", default="TIMESTAMP")
    p.add_argument("--temp-col", default="Temp_1_Avg")
    p.add_argument("--freq", default="10min")
    p.add_argument("--lead-step", type=int, default=18)

    p.add_argument("--nwp-models", default="arome_france_hd,arome_france,ncep_gfs013,icon_eu,arpege_europe,ecmwf_ifs025")
    p.add_argument("--openmeteo-url", default="https://historical-forecast-api.open-meteo.com/v1/forecast")
    p.add_argument("--openmeteo-timeout", type=int, default=60)
    p.add_argument("--chunk-days", type=int, default=31)
    p.add_argument("--request-sleep-s", type=float, default=0.25)
    p.add_argument("--force-download", action="store_true")
    p.add_argument(
        "--nwp-cache-dir",
        default=None,
        help="Directori amb CSVs Open-Meteo ja descarregats. Si no s'indica, s'usa <output-dir>/nwp_cache.",
    )

    p.add_argument("--min-obs-daily-coverage", type=float, default=0.90)
    p.add_argument("--min-local-daily-coverage", type=float, default=0.90)
    p.add_argument("--min-nwp-daily-coverage", type=float, default=0.80)

    p.add_argument("--formats", default="png,pdf")
    p.add_argument("--dpi", type=int, default=260)
    p.add_argument("--title-prefix", default="Comparació horària homogènia de Tmax/Tmin (2024)")

    return Config(**vars(p.parse_args()))


def split_models(models: str) -> List[str]:
    return [m.strip() for m in str(models).split(",") if m.strip()]


def formats_list(cfg: Config) -> List[str]:
    return [f.strip().lower().lstrip(".") for f in cfg.formats.split(",") if f.strip()]


def is_exact_hour(index: pd.DatetimeIndex) -> np.ndarray:
    return (
        (index.minute.to_numpy() == 0)
        & (index.second.to_numpy() == 0)
        & (index.microsecond.to_numpy() == 0)
    )


# =============================================================================
# Dataset UB i memòria cau del model local
# =============================================================================

def resolve_dataset_path(dataset: Path, output_dir: Path) -> Path:
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
    df = pd.read_csv(csv_path, usecols=[cfg.timestamp_col, cfg.temp_col], low_memory=False)
    df[cfg.timestamp_col] = pd.to_datetime(df[cfg.timestamp_col], errors="coerce")
    df[cfg.temp_col] = pd.to_numeric(df[cfg.temp_col], errors="coerce")
    df = df.dropna(subset=[cfg.timestamp_col]).sort_values(cfg.timestamp_col)
    df = df.drop_duplicates(subset=[cfg.timestamp_col], keep="last")

    s = df.set_index(cfg.timestamp_col)[cfg.temp_col].astype(float)
    full_index = pd.date_range(s.index.min(), s.index.max(), freq=cfg.freq)
    s = s.reindex(full_index)
    s.index.name = cfg.timestamp_col
    return s


def observed_daily_extremes_hourly(temp: pd.Series, cfg: Config) -> pd.DataFrame:
    """Tmax/Tmin observades només amb valors a hores exactes."""
    start = pd.Timestamp(cfg.start)
    end = pd.Timestamp(cfg.end)
    idx = pd.DatetimeIndex(temp.index)
    s = temp.loc[(idx >= start) & (idx < end) & is_exact_hour(idx)]

    rows: List[Dict[str, Any]] = []
    for day, group in s.groupby(s.index.floor("D")):
        n_valid = int(group.notna().sum())
        coverage = n_valid / 24.0
        if coverage < cfg.min_obs_daily_coverage:
            continue
        rows.append({
            "date": pd.Timestamp(day),
            "tmax_obs": float(group.max(skipna=True)),
            "tmin_obs": float(group.min(skipna=True)),
            "tmean_obs": float(group.mean(skipna=True)),
            "n_obs_hourly": n_valid,
            "obs_hourly_coverage": float(coverage),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("No hi ha dies observats horaris amb cobertura suficient.")
    return out


def load_temperature_cache(path: Path) -> Dict[str, np.ndarray]:
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"No existeix la memòria cau de temperatura: {path}")

    log(f"[INFO] Llegint memòria cau del model local: {path}")
    with np.load(path, allow_pickle=False) as data:
        pred = {k: data[k] for k in data.files}
    for k in ["issue_times", "ensemble_mean", "baseline", "observations", "mask"]:
        if k not in pred:
            raise KeyError(f"Falta la clau {k} dins del NPZ.")
    pred["issue_times"] = pd.to_datetime(pred["issue_times"])
    return pred


def local_model_daily_extremes_hourly(pred: Dict[str, np.ndarray], cfg: Config) -> pd.DataFrame:
    """
    Reconstrueix Tmax/Tmin del model local només en target_times horaris.
    Això fa que la comparació amb NWP horari sigui temporalment homogènia.
    """
    issue_times = pd.DatetimeIndex(pred["issue_times"])
    model = pred["ensemble_mean"].astype(float)
    baseline = pred["baseline"].astype(float)
    mask = pred["mask"].astype(float) > 0.5

    k = int(cfg.lead_step)
    horizon_steps = model.shape[1]
    if k < 1 or k > horizon_steps:
        raise ValueError(f"--lead-step={k} fora de rang. Horizon steps={horizon_steps}.")

    dt = pd.Timedelta(cfg.freq)
    target_times = issue_times + k * dt
    start = pd.Timestamp(cfg.start)
    end = pd.Timestamp(cfg.end)

    valid = (
        (target_times >= start)
        & (target_times < end)
        & is_exact_hour(pd.DatetimeIndex(target_times))
        & mask[:, k - 1]
    )

    df = pd.DataFrame({
        "target_time": target_times[valid],
        "date": target_times[valid].floor("D"),
        "hybrid_timemixer_v9": model[valid, k - 1],
        "baseline_interna": baseline[valid, k - 1],
    }).replace([np.inf, -np.inf], np.nan)

    rows: List[Dict[str, Any]] = []
    for day, group in df.groupby("date"):
        n_valid = int(group["hybrid_timemixer_v9"].notna().sum())
        coverage = n_valid / 24.0
        if coverage < cfg.min_local_daily_coverage:
            continue
        row: Dict[str, Any] = {
            "date": pd.Timestamp(day),
            "lead_step": int(k),
            "lead_hours": float(k * dt.total_seconds() / 3600.0),
            "n_local_model_hourly": n_valid,
            "local_model_hourly_coverage": float(coverage),
        }
        for source in ["hybrid_timemixer_v9", "baseline_interna"]:
            row[f"tmax_{source}"] = float(group[source].max(skipna=True))
            row[f"tmin_{source}"] = float(group[source].min(skipna=True))
        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("No s'han pogut reconstruir extrems horaris del model local.")
    return out


# =============================================================================
# Open-Meteo Historical Forecast API i cache NWP
# =============================================================================

def nwp_cache_dir(cfg: Config, out_dir: Path) -> Path:
    if cfg.nwp_cache_dir:
        p = Path(cfg.nwp_cache_dir).expanduser().resolve()
    else:
        p = out_dir / "nwp_cache"
    ensure_dir(p)
    return p


def cache_file_for_model(cfg: Config, cache_dir: Path, model: str) -> Path:
    return cache_dir / f"openmeteo_{model}_{cfg.start}_{cfg.end}_{cfg.latitude:.4f}_{cfg.longitude:.4f}.csv"


def find_existing_cache(cfg: Config, cache_dir: Path, model: str) -> Optional[Path]:
    expected = cache_file_for_model(cfg, cache_dir, model)
    if expected.exists():
        return expected

    # Tolerància per si es va crear amb el mateix nom però en un altre directori o
    # amb variacions menors de decimals.
    pattern = f"openmeteo_{model}_{cfg.start}_{cfg.end}_*.csv"
    candidates = sorted(cache_dir.glob(pattern))
    if candidates:
        return candidates[0]
    return None


def monthlike_chunks(start_date: str, end_exclusive: str, chunk_days: int) -> Iterable[Tuple[pd.Timestamp, pd.Timestamp]]:
    start = pd.Timestamp(start_date).normalize()
    end_excl = pd.Timestamp(end_exclusive).normalize()
    cursor = start
    step = pd.Timedelta(days=max(1, int(chunk_days)))
    while cursor < end_excl:
        nxt_excl = min(cursor + step, end_excl)
        api_end = nxt_excl - pd.Timedelta(days=1)
        yield cursor, api_end
        cursor = nxt_excl


def openmeteo_temperature_key(hourly: Dict[str, Any], model: str) -> str:
    keys = [k for k in hourly.keys() if k.startswith("temperature_2m")]
    if "temperature_2m" in keys:
        return "temperature_2m"
    matching = [k for k in keys if model in k]
    if len(matching) == 1:
        return matching[0]
    if len(keys) == 1:
        return keys[0]
    raise KeyError(f"No puc identificar la clau temperature_2m per {model}. Claus: {keys}")


def fetch_openmeteo_chunk(cfg: Config, model: str, start_day: pd.Timestamp, end_day: pd.Timestamp) -> pd.DataFrame:
    params = {
        "latitude": f"{cfg.latitude:.6f}",
        "longitude": f"{cfg.longitude:.6f}",
        "start_date": start_day.strftime("%Y-%m-%d"),
        "end_date": end_day.strftime("%Y-%m-%d"),
        "hourly": "temperature_2m",
        "temperature_unit": "celsius",
        "timezone": cfg.timezone,
        "models": model,
    }
    url = cfg.openmeteo_url + "?" + urlencode(params)

    last_exc: Optional[Exception] = None
    for attempt in range(1, 4):
        try:
            r = requests.get(url, timeout=int(cfg.openmeteo_timeout))
            if r.status_code >= 400:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:500]}")
            data = r.json()
            if data.get("error"):
                raise RuntimeError(f"Open-Meteo error: {data}")
            hourly = data.get("hourly", {})
            if "time" not in hourly:
                raise RuntimeError(f"Resposta sense hourly.time: {data}")
            key = openmeteo_temperature_key(hourly, model)
            out = pd.DataFrame({
                "time": pd.to_datetime(hourly["time"], errors="coerce"),
                "temperature_2m_C": pd.to_numeric(pd.Series(hourly[key]), errors="coerce"),
            })
            out["model"] = model
            out = out.dropna(subset=["time"]).sort_values("time")
            return out
        except Exception as exc:
            last_exc = exc
            wait = 2.0 * attempt
            log(f"[AVÍS] Error descarregant {model} {start_day.date()}..{end_day.date()} "
                f"(intent {attempt}/3): {exc}")
            time.sleep(wait)

    raise RuntimeError(f"No s'ha pogut descarregar {model} {start_day.date()}..{end_day.date()}: {last_exc}")


def download_or_read_nwp_model(cfg: Config, model: str, out_dir: Path) -> pd.DataFrame:
    cdir = nwp_cache_dir(cfg, out_dir)
    existing = find_existing_cache(cfg, cdir, model)

    if existing is not None and not cfg.force_download:
        log(f"[INFO] Reutilitzant NWP cache: {existing}")
        df = pd.read_csv(existing)
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        return df.dropna(subset=["time"]).sort_values("time")

    frames: List[pd.DataFrame] = []
    log(f"[INFO] Descarregant {model} via Open-Meteo Historical Forecast API...")
    for start_day, end_day in monthlike_chunks(cfg.start, cfg.end, cfg.chunk_days):
        chunk = fetch_openmeteo_chunk(cfg, model, start_day, end_day)
        frames.append(chunk)
        time.sleep(float(cfg.request_sleep_s))

    if not frames:
        raise RuntimeError(f"No s'ha descarregat cap dada per {model}")

    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["time"], keep="last").sort_values("time")
    cache_csv = cache_file_for_model(cfg, cdir, model)
    df.to_csv(cache_csv, index=False)
    log(f"[OK] NWP cache desada: {cache_csv}")
    return df


def nwp_daily_extremes_hourly(df: pd.DataFrame, cfg: Config, model: str) -> pd.DataFrame:
    start = pd.Timestamp(cfg.start)
    end = pd.Timestamp(cfg.end)

    df = df.copy()
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df["temperature_2m_C"] = pd.to_numeric(df["temperature_2m_C"], errors="coerce")
    df = df.dropna(subset=["time"])
    df = df[(df["time"] >= start) & (df["time"] < end)].copy()
    if df.empty:
        raise RuntimeError(f"{model}: no hi ha dades dins el període demanat.")

    rows: List[Dict[str, Any]] = []
    for day, group in df.groupby(df["time"].dt.floor("D")):
        n_valid = int(group["temperature_2m_C"].notna().sum())
        coverage = n_valid / 24.0
        if coverage < cfg.min_nwp_daily_coverage:
            continue
        rows.append({
            "date": pd.Timestamp(day),
            f"tmax_{model}": float(group["temperature_2m_C"].max(skipna=True)),
            f"tmin_{model}": float(group["temperature_2m_C"].min(skipna=True)),
            f"n_hours_{model}": n_valid,
            f"coverage_{model}": float(coverage),
        })

    return pd.DataFrame(rows)


# =============================================================================
# Mètriques i skills corregits
# =============================================================================

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


def rmse_skill(ref_rmse: float, model_rmse: float) -> float:
    if not np.isfinite(ref_rmse) or ref_rmse <= 0 or not np.isfinite(model_rmse):
        return math.nan
    return float(1.0 - model_rmse / ref_rmse)


def mae_reduction(ref_mae: float, model_mae: float) -> float:
    if not np.isfinite(ref_mae) or ref_mae <= 0 or not np.isfinite(model_mae):
        return math.nan
    return float(1.0 - model_mae / ref_mae)


def summarize_sources_common_days(daily: pd.DataFrame, sources: Sequence[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    summary:
      mètriques de cada font sobre els dies on aquella font i l'observació són vàlids.

    skill_summary:
      comparació HTM vs cada referència sobre els dies comuns HTM+referència+obs.
    """
    rows: List[Dict[str, Any]] = []
    skill_rows: List[Dict[str, Any]] = []

    for variable in ["tmax", "tmin"]:
        obs_col = f"{variable}_obs"
        local_col = f"{variable}_hybrid_timemixer_v9"

        for source in sources:
            pred_col = f"{variable}_{source}"
            if pred_col not in daily.columns:
                continue
            scores = finite_scores(daily[obs_col].to_numpy(), daily[pred_col].to_numpy())
            rows.append({
                "variable": variable,
                "source": source,
                "label": MODEL_LABELS.get(source, source),
                "days_basis": "available_days_for_source",
                **scores,
            })

        for ref in sources:
            if ref == "hybrid_timemixer_v9":
                continue
            ref_col = f"{variable}_{ref}"
            if local_col not in daily.columns or ref_col not in daily.columns:
                continue

            ok = (
                np.isfinite(daily[obs_col].to_numpy(dtype=float))
                & np.isfinite(daily[local_col].to_numpy(dtype=float))
                & np.isfinite(daily[ref_col].to_numpy(dtype=float))
            )
            if ok.sum() == 0:
                continue

            local_scores = finite_scores(daily.loc[ok, obs_col].to_numpy(), daily.loc[ok, local_col].to_numpy())
            ref_scores = finite_scores(daily.loc[ok, obs_col].to_numpy(), daily.loc[ok, ref_col].to_numpy())

            skill_rows.append({
                "variable": variable,
                "comparison": f"hybrid_timemixer_v9_vs_{ref}",
                "reference": ref,
                "reference_label": MODEL_LABELS.get(ref, ref),
                "n_common_days": int(ok.sum()),
                "hybrid_mae": local_scores["mae"],
                "reference_mae": ref_scores["mae"],
                "hybrid_rmse": local_scores["rmse"],
                "reference_rmse": ref_scores["rmse"],
                "hybrid_bias": local_scores["bias"],
                "reference_bias": ref_scores["bias"],
                "mae_reduction_fraction": mae_reduction(ref_scores["mae"], local_scores["mae"]),
                "rmse_skill_fraction": rmse_skill(ref_scores["rmse"], local_scores["rmse"]),
                "mae_reduction_percent": 100.0 * mae_reduction(ref_scores["mae"], local_scores["mae"]),
                "rmse_skill_percent": 100.0 * rmse_skill(ref_scores["rmse"], local_scores["rmse"]),
            })

    return pd.DataFrame(rows), pd.DataFrame(skill_rows)


# =============================================================================
# Figures
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


def save_figure(fig: plt.Figure, out_base: Path, formats: Sequence[str], dpi: int) -> None:
    ensure_dir(out_base.parent)
    for fmt in formats:
        path = out_base.with_suffix(f".{fmt}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        log(f"[OK] Figura: {path}")
    plt.close(fig)


def plot_mae_bars(summary: pd.DataFrame, sources: Sequence[str], out_dir: Path, cfg: Config) -> None:
    setup_style()
    formats = formats_list(cfg)

    fig, ax = plt.subplots(figsize=(11.0, 5.9))
    x = np.arange(2)
    width = 0.82 / max(1, len(sources))

    for i, source in enumerate(sources):
        sub = summary[summary["source"] == source].set_index("variable")
        vals = [
            float(sub.loc["tmax", "mae"]) if "tmax" in sub.index else np.nan,
            float(sub.loc["tmin", "mae"]) if "tmin" in sub.index else np.nan,
        ]
        offset = (i - (len(sources) - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            vals,
            width=width,
            label=MODEL_LABELS.get(source, source),
            color=MODEL_COLORS.get(source, None),
            alpha=0.92,
        )
        ax.bar_label(bars, fmt="%.2f", fontsize=7.6, padding=2)

    ax.set_title(f"{cfg.title_prefix}\nMAE calculat amb observacions i prediccions horàries")
    ax.set_xticks(x)
    ax.set_xticklabels(["Tmax", "Tmin"])
    ax.set_ylabel("MAE (°C)")
    ax.grid(True, axis="y")
    ax.legend(frameon=False, ncol=2)
    save_figure(fig, out_dir / "mae_tmax_tmin_horari_model_local_vs_nwp", formats, cfg.dpi)


def plot_skill_bars(skill_summary: pd.DataFrame, out_dir: Path, cfg: Config) -> None:
    if skill_summary.empty:
        return
    setup_style()
    formats = formats_list(cfg)

    comparisons = skill_summary["reference"].unique().tolist()
    fig, ax = plt.subplots(figsize=(11.0, 5.9))
    x = np.arange(2)
    width = 0.82 / max(1, len(comparisons))

    for i, ref in enumerate(comparisons):
        sub = skill_summary[skill_summary["reference"] == ref].set_index("variable")
        vals = [
            float(sub.loc["tmax", "rmse_skill_percent"]) if "tmax" in sub.index else np.nan,
            float(sub.loc["tmin", "rmse_skill_percent"]) if "tmin" in sub.index else np.nan,
        ]
        offset = (i - (len(comparisons) - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            vals,
            width=width,
            label=f"vs {MODEL_LABELS.get(ref, ref)}",
            alpha=0.92,
        )
        ax.bar_label(bars, fmt="%.1f", fontsize=7.4, padding=2)

    ax.axhline(0, color="#94a3b8", linewidth=1.2, linestyle="--")
    ax.set_title(f"{cfg.title_prefix}\nSkill de RMSE de l'Hybrid-TimeMixer v9 sobre dies comuns")
    ax.set_xticks(x)
    ax.set_xticklabels(["Tmax", "Tmin"])
    ax.set_ylabel("Skill de RMSE (%)")
    ax.grid(True, axis="y")
    ax.legend(frameon=False, ncol=2)
    save_figure(fig, out_dir / "skill_rmse_horari_hybrid_vs_nwp_dies_comuns", formats, cfg.dpi)


def plot_timeseries(daily: pd.DataFrame, sources: Sequence[str], variable: str, out_dir: Path, cfg: Config) -> None:
    setup_style()
    formats = formats_list(cfg)

    fig, ax = plt.subplots(figsize=(13.2, 5.5))
    obs_col = f"{variable}_obs"
    ax.plot(daily["date"], daily[obs_col], color="#111827", linewidth=1.7, label="Observació horària", zorder=8)

    for source in sources:
        col = f"{variable}_{source}"
        if col not in daily.columns:
            continue
        ax.plot(
            daily["date"],
            daily[col],
            linewidth=1.25 if source != "hybrid_timemixer_v9" else 1.8,
            alpha=0.86,
            label=MODEL_LABELS.get(source, source),
            color=MODEL_COLORS.get(source, None),
        )

    label = "Tmax" if variable == "tmax" else "Tmin"
    ax.set_title(f"{cfg.title_prefix}\nSèrie diària horària de {label}")
    ax.set_xlabel("Data")
    ax.set_ylabel("Temperatura (°C)")
    ax.grid(True, axis="y")
    ax.legend(frameon=False, ncol=3)
    fig.autofmt_xdate()
    save_figure(fig, out_dir / f"serie_diaria_horaria_{variable}_model_local_vs_nwp", formats, cfg.dpi)


def plot_scatter(daily: pd.DataFrame, sources: Sequence[str], variable: str, out_dir: Path, cfg: Config) -> None:
    setup_style()
    formats = formats_list(cfg)

    preferred = ["hybrid_timemixer_v9"] + [s for s in sources if s not in ("hybrid_timemixer_v9", "baseline_interna")]
    preferred = preferred[:5]

    obs_col = f"{variable}_obs"
    fig, ax = plt.subplots(figsize=(6.9, 6.5))

    vals_list = [daily[obs_col].to_numpy(dtype=float)]
    for source in preferred:
        col = f"{variable}_{source}"
        if col in daily.columns:
            vals_list.append(daily[col].to_numpy(dtype=float))
    vals = np.concatenate([v[np.isfinite(v)] for v in vals_list if np.asarray(v).size])
    if vals.size == 0:
        plt.close(fig)
        return

    vmin, vmax = float(vals.min()), float(vals.max())
    pad = max(0.7, 0.06 * (vmax - vmin))

    for source in preferred:
        col = f"{variable}_{source}"
        if col not in daily.columns:
            continue
        ok = np.isfinite(daily[obs_col]) & np.isfinite(daily[col])
        ax.scatter(
            daily.loc[ok, obs_col],
            daily.loc[ok, col],
            s=22,
            alpha=0.62,
            edgecolor="none",
            label=MODEL_LABELS.get(source, source),
            color=MODEL_COLORS.get(source, None),
        )

    ax.plot([vmin - pad, vmax + pad], [vmin - pad, vmax + pad], color="#94a3b8", linestyle="--", linewidth=1.3)
    ax.set_xlim(vmin - pad, vmax + pad)
    ax.set_ylim(vmin - pad, vmax + pad)
    label = "Tmax" if variable == "tmax" else "Tmin"
    ax.set_title(f"{label}: predicció horària vs observació horària")
    ax.set_xlabel("Observació horària (°C)")
    ax.set_ylabel("Predicció horària (°C)")
    ax.grid(True)
    ax.legend(frameon=False)
    save_figure(fig, out_dir / f"scatter_horari_{variable}_model_local_vs_nwp", formats, cfg.dpi)


# =============================================================================
# Programa principal
# =============================================================================

def main() -> None:
    cfg = parse_args()
    out_dir = Path(cfg.output_dir).expanduser().resolve()
    ensure_dir(out_dir)

    log("[INFO] Preparant observacions UB horàries...")
    csv_path = resolve_dataset_path(Path(cfg.dataset), out_dir)
    temp_series = read_temperature_series(csv_path, cfg)
    obs_daily = observed_daily_extremes_hourly(temp_series, cfg)

    log("[INFO] Preparant extrems diaris horaris del model local...")
    pred = load_temperature_cache(Path(cfg.temperature_cache))
    local_daily = local_model_daily_extremes_hourly(pred, cfg)

    daily = obs_daily.merge(local_daily, on="date", how="inner")
    sources = ["hybrid_timemixer_v9", "baseline_interna"]

    log("[INFO] Preparant NWP horaris...")
    for model in split_models(cfg.nwp_models):
        try:
            nwp_hourly = download_or_read_nwp_model(cfg, model, out_dir)
            nwp_daily = nwp_daily_extremes_hourly(nwp_hourly, cfg, model)
            if nwp_daily.empty:
                log(f"[AVÍS] {model}: sense extrems diaris vàlids.")
                continue
            daily = daily.merge(nwp_daily, on="date", how="left")
            sources.append(model)
        except Exception as exc:
            log(f"[AVÍS] S'ignora {model} perquè ha fallat: {exc}")

    summary, skill_summary = summarize_sources_common_days(daily, sources)

    # Exportació
    daily_path = out_dir / "daily_tmax_tmin_hourly_homogeneous_model_local_vs_nwp_2024.csv"
    summary_path = out_dir / "summary_tmax_tmin_hourly_homogeneous_model_local_vs_nwp_2024.csv"
    skill_path = out_dir / "skill_tmax_tmin_hourly_homogeneous_common_days_2024.csv"
    json_path = out_dir / "summary_tmax_tmin_hourly_homogeneous_model_local_vs_nwp_2024.json"

    daily.to_csv(daily_path, index=False)
    summary.to_csv(summary_path, index=False)
    skill_summary.to_csv(skill_path, index=False)

    meta = {
        "config": asdict(cfg),
        "evaluation_frequency": "1H",
        "methodological_note": (
            "Observacions UB, Hybrid-TimeMixer i NWP es comparen només en hores exactes. "
            "Tmax/Tmin es calculen sobre la mateixa graella horària. Els skills es calculen "
            "sobre dies comuns entre HTM i cada referència."
        ),
        "coordinates_used": {
            "latitude": cfg.latitude,
            "longitude": cfg.longitude,
            "timezone": cfg.timezone,
        },
        "sources": sources,
        "labels": {s: MODEL_LABELS.get(s, s) for s in sources},
        "n_days_after_obs_and_local_join": int(len(daily)),
        "lead_step": int(cfg.lead_step),
        "lead_hours": float(int(cfg.lead_step) * pd.Timedelta(cfg.freq).total_seconds() / 3600.0),
        "summary_records": summary.replace([np.inf, -np.inf], np.nan).where(pd.notna(summary), None).to_dict("records"),
        "skill_records_common_days": skill_summary.replace([np.inf, -np.inf], np.nan).where(pd.notna(skill_summary), None).to_dict("records"),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    log(f"[OK] CSV diari: {daily_path}")
    log(f"[OK] CSV resum: {summary_path}")
    log(f"[OK] CSV skills comuns: {skill_path}")
    log(f"[OK] JSON: {json_path}")

    # Figures
    plot_mae_bars(summary, sources, out_dir, cfg)
    plot_skill_bars(skill_summary, out_dir, cfg)
    for variable in ["tmax", "tmin"]:
        plot_timeseries(daily, sources, variable, out_dir, cfg)
        plot_scatter(daily, sources, variable, out_dir, cfg)

    print("\n[RESULTAT - MÈTRIQUES]")
    print(summary.to_string(index=False))
    print("\n[RESULTAT - SKILLS SOBRE DIES COMUNS]")
    print(skill_summary.to_string(index=False))


if __name__ == "__main__":
    main()
