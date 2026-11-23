#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Construeix el conjunt de dades final per entrenar el model de nowcasting meteorològic (1 estació, horitzó fins a 3h), a partir del csv de features generat pel codi corr_var.py.

IMPORTANT:
----------
Aquest codi s'ha d'executar després dels codis "imputa_NaN.py" i "corr_var.py", en aquest ordre.

Idees clau:
-----------
1) Regularització temporal:
   - Es reindexa a una graella regular (10 min). Això pot reintroduir timestamps sense dades que no existien al csv imputat.
   - Aquests timestamps solen tenir NaNs a totes les features i flags.
   - Per coherència amb el codi "imputa_NaN.py", s'eliminen aquestes files després del reindex.

2) Segmentació:
   - Per defecte es trenca segment per gaps temporals reals (differences > gap_factor * dt).
   - Opcionalment es pot trencar també per missing de target (configurable), però no ve per defecte.

3) Targets:
   - No s'imputen.
   - Es conserven NaNs i es creen flags: Temp_1_Avg__miss i Lluvia_Tot__miss.

4) QC:
   - S'afegeixen flags de qualitat.
   - La pressió fa servir llindars locals + bounds train-only per evitar leakage.

5) Redundància / feature sets:
   - "compact" elimina els predictors redundants recomanats per maximitzar generalització: VP_Tot, Vel_Viento_WVc(3) i el paquet radiatiu colinear/derivat.

Exemple d'ús:
python dataset_nowcasting.py --input "Dataset_final/CR3000_Tabla1_2011_2024_features_nowcasting.csv" --output "Dataset_final/CR3000_Tabla1_2011_2024_dataset_nowcasting.csv" --report "Dataset_final/dataset_nowcasting_2011_2024_report.json" --freq "10min" --history-hours 48 --horizon-hours 3 --train-end "2023-01-01" --feature-set compact
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


# =============================================================================
# Configuració
# =============================================================================

@dataclass
class Config:
    """
    Configuració del pipeline.
    Aquest dataclass només conté paràmetres. La lògica està tota en funcions.
    """
    input_csv: str
    output_csv: str
    report_json: str

    timestamp_col: str = "TIMESTAMP"
    freq: str = "10min"

    temp_col: str = "Temp_1_Avg"
    rain_col: str = "Lluvia_Tot"

    history_hours: float = 48.0
    horizon_hours: float = 3.0

    # Segmentació: trencament per gap temporal (> gap_factor * dt)
    gap_factor: float = 1.5

    # Opcional: trencar segment per missing de target ("none" | "any" | "temp" | "rain")
    segment_break_on_missing_target: str = "none"

    add_engineered: bool = True
    add_qc_flags: bool = True

    # IMPORTANT: elimina files offline reintroduïdes pel reindex
    drop_offline_rows: bool = True

    # Tall temporal inferior del dataset final (per coherència amb el radiosondatge, si s'utilitza)
    data_start: Optional[str] = "2012-01-01"

    # Per QC train-only (sense leakage) i per report
    train_end: str = "2023-01-01"

    # Redundància / neteja
    feature_set: str = "compact"  # "full" o "compact"; per defecte usem el preset més robust
    drop_aux_cols: Tuple[str, ...] = ("rain_occ", "rain_log1p")

    # Rolling windows (minutes)
    rain_roll_minutes: Tuple[int, ...] = (30, 60, 180)
    temp_roll_minutes: Tuple[int, ...] = (60, 180)
    pres_roll_minutes: Tuple[int, ...] = (60,)

    # Tendències (diff) sobre variables base
    trend_cols: Tuple[str, ...] = (
        # Només variables base que realment volem conservar al model final.
        "Temp_1_Avg",
        "Hum_Rel_1_Avg",
        "Pres_Atm_Avg",
        "VPD_Tot",
        "Rad_Global_Avg",
        "Vel_Viento_WVc(1)",
    )

    # QC thresholds
    qc_temp_min_C: float = -50.0
    qc_temp_max_C: float = 60.0
    qc_rh_min_pct: float = 0.0
    qc_rh_max_pct: float = 100.0

    # Pressió: llindars locals (hPa) + opcional train-only bounds (per flags)
    qc_pres_min_hPa: float = 930.0
    qc_pres_max_hPa: float = 1060.0
    qc_pres_use_train_bounds: bool = True
    qc_pres_train_method: str = "iqr"  # "quantile" o "iqr"
    qc_pres_q_low: float = 0.001
    qc_pres_q_high: float = 0.999
    qc_pres_iqr_k: float = 8.0
    qc_pres_min_span_hPa: float = 30.0

    # Vent
    qc_wind_max_ms: float = 120.0

    # Radiosondatge
    radiosonde_dir: Optional[str] = None
    radiosonde_pattern: str = "*.txt"
    radiosonde_max_age_hours: float = 18.0
    radiosonde_levels_hPa: Tuple[int, ...] = (1000, 925, 850, 700, 500)

    # Neteja final de predictors no informatius
    drop_qc_predictors: bool = True
    drop_constant_predictors: bool = True


# =============================================================================
# Helpers generals
# =============================================================================

def _safe_to_numeric(df: pd.DataFrame, exclude: List[str]) -> pd.DataFrame:
    """
    Converteix columnes 'object' a numèric (quan toca) per evitar sorpreses.

    - Les columnes a 'exclude' no es toquen (p. ex. TIMESTAMP).
    - Si la conversió falla, es posa NaN (errors='coerce').
    """
    out = df.copy()
    for c in out.columns:
        if c in exclude:
            continue
        if out[c].dtype == object:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _infer_dt_minutes(freq: str) -> float:
    """Retorna el pas temporal (freq) en minuts."""
    return pd.Timedelta(freq).total_seconds() / 60.0


def _unique_columns(df: pd.DataFrame) -> None:
    """
    Assegura que no hi hagi columnes duplicades.

    És una validació útil perquè, si hi ha duplicitats, qualsevol selecció de columnes pot tenir comportaments inesperats.
    """
    if df.columns.duplicated().any():
        dups = df.columns[df.columns.duplicated()].tolist()
        raise ValueError(f"Hi ha columnes duplicades: {dups}")


def _drop_cols_with_miss(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    Elimina columnes i els seus flags associats __miss, si existeixen.

    Exemple:
    - Si cols inclou 'rain_occ', també s'intenta eliminar 'rain_occ__miss'.
    """
    to_drop: List[str] = []
    for c in cols:
        if c in df.columns:
            to_drop.append(c)
        mc = f"{c}__miss"
        if mc in df.columns:
            to_drop.append(mc)
    return df.drop(columns=to_drop, errors="ignore")


def _prune_predictors(df: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    Elimina predictors no informatius o potencialment problemàtics abans de l'export final.

    Criteris:
    - Es retiren les metadades QC del conjunt de predictors, perquè descriuen qualitat
      de la dada i no pas estat meteorològic.
    - S'eliminen columnes constants exactes entre files, ja que no aporten cap
      informació a l'aprenentatge ni a la generalització.

    Es preserven sempre els targets i la metainformació estructural del dataset
    (segmentació i valid_t_end), encara que puntualment fossin constants.
    """
    out = df.copy()
    dropped_qc: List[str] = []
    dropped_constant: List[str] = []

    if cfg.drop_qc_predictors:
        qc_cols = [c for c in ("qc_n_violations", "qc_any") if c in out.columns]
        if qc_cols:
            out = out.drop(columns=qc_cols)
            dropped_qc = qc_cols

    if cfg.drop_constant_predictors:
        protected = {
            cfg.temp_col,
            cfg.rain_col,
            f"{cfg.temp_col}__miss",
            f"{cfg.rain_col}__miss",
            "segment_id",
            "segment_pos",
            "segment_len",
            "valid_t_end",
        }
        candidate_cols = [c for c in out.columns if c not in protected]
        nunique = out[candidate_cols].nunique(dropna=False) if candidate_cols else pd.Series(dtype=int)
        dropped_constant = [c for c in candidate_cols if int(nunique.get(c, 0)) <= 1]
        if dropped_constant:
            out = out.drop(columns=dropped_constant)

    info = {
        "dropped_qc_predictors": dropped_qc,
        "dropped_constant_predictors": dropped_constant,
        "n_dropped_qc_predictors": int(len(dropped_qc)),
        "n_dropped_constant_predictors": int(len(dropped_constant)),
    }
    return out, info



def _apply_data_start_filter(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    Limita el dataset a partir d'una data mínima comuna.
    """
    if not cfg.data_start:
        return df

    out = df.copy()
    out[cfg.timestamp_col] = pd.to_datetime(out[cfg.timestamp_col], errors="coerce")
    t0 = pd.Timestamp(cfg.data_start)
    out = out.loc[out[cfg.timestamp_col] >= t0].copy()
    if out.empty:
        raise ValueError(f"No queden files després d'aplicar data_start={cfg.data_start}.")
    return out


# =============================================================================
# Radiosondatge: lectura robusta, resum físic i fusió causal
# =============================================================================

_RADIOSONDE_RENAME = {
    "ftr_alt": "alt_m",
    "ftr_pres": "pres_hPa",
    "ftr_temp": "temp_C",
    "ftr_hum": "rh_pct",
    "ftr_DP": "dewpoint_C",
    "ftr_VEF": "u_ms",
    "ftr_VNF": "v_ms",
    "ftr_WF": "wind_speed_ms",
    "ftr_WD": "wind_dir_deg",
    "ftr_LAT": "lat",
    "ftr_LON": "lon",
}


def _radiosonde_filename_to_timestamp(path: Path) -> Optional[pd.Timestamp]:
    """
    Extreu el timestamp de llançament a partir del nom del fitxer.

    S'espera el format habitual YYMMDDHH.txt, per exemple:
      - 24010100.txt  -> 2024-01-01 00:00
      - 24010112.txt  -> 2024-01-01 12:00
    """
    stem = path.stem.strip()
    if not re.fullmatch(r"\d{8}", stem):
        return None
    try:
        return pd.Timestamp(pd.to_datetime(stem, format="%y%m%d%H"))
    except Exception:
        return None


def _read_one_radiosonde_file(path: Path) -> pd.DataFrame:
    """
    Llegeix un radiosondatge en format txt/tabular i en fa una neteja mínima.

    Regles:
    - només es conserven variables físiques útils;
    - s'eliminen files sense pressió/altitud vàlides;
    - la columna vertical de referència és la pressió;
    - es força un ordre monotònic de pressió i es col·lapsen duplicats.
    """
    try:
        df = pd.read_csv(path, sep=r"\s+", engine="python")
    except Exception:
        try:
            df = pd.read_csv(path, sep="\t")
        except Exception:
            return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    # Elimina columnes residuals buides o sense nom.
    df = df.loc[:, [c for c in df.columns if not str(c).startswith("Unnamed")]].copy()

    keep = [c for c in _RADIOSONDE_RENAME.keys() if c in df.columns]
    if not {"ftr_pres", "ftr_alt", "ftr_temp", "ftr_hum", "ftr_VEF", "ftr_VNF"}.issubset(set(keep)):
        return pd.DataFrame()

    df = df[keep].rename(columns=_RADIOSONDE_RENAME).copy()
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Control físic bàsic.
    if "pres_hPa" in df.columns:
        df = df[df["pres_hPa"].between(20.0, 1100.0, inclusive="both")]
    if "alt_m" in df.columns:
        df = df[df["alt_m"].between(-500.0, 50000.0, inclusive="both")]
    if "temp_C" in df.columns:
        df.loc[~df["temp_C"].between(-120.0, 60.0, inclusive="both"), "temp_C"] = np.nan
    if "rh_pct" in df.columns:
        df.loc[~df["rh_pct"].between(0.0, 100.0, inclusive="both"), "rh_pct"] = np.nan
    if "wind_speed_ms" in df.columns:
        df.loc[~df["wind_speed_ms"].between(0.0, 200.0, inclusive="both"), "wind_speed_ms"] = np.nan
    for c in ("u_ms", "v_ms"):
        if c in df.columns:
            df.loc[~df[c].between(-200.0, 200.0, inclusive="both"), c] = np.nan

    df = df.dropna(subset=["pres_hPa", "alt_m"])
    if len(df) < 10:
        return pd.DataFrame()

    df = df.sort_values(["pres_hPa", "alt_m"], ascending=[False, True])

    # Col·lapsa nivells de pressió duplicats fent-ne la mediana.
    agg = {c: "median" for c in df.columns if c != "pres_hPa"}
    df = df.groupby("pres_hPa", as_index=False).agg(agg)
    df = df.sort_values("pres_hPa", ascending=False).reset_index(drop=True)

    if len(df) < 10:
        return pd.DataFrame()

    return df


def _max_interp_gap_hpa(level_hpa: float) -> float:
    """
    Llindar màxim de buit vertical admissible per interpolar a un nivell estàndard.

    És més estricte a la baixa troposfera, on petits detalls termodinàmics són
    molt importants per al nowcasting.
    """
    p = float(level_hpa)
    if p >= 900.0:
        return 35.0
    if p >= 800.0:
        return 40.0
    if p >= 650.0:
        return 55.0
    return 80.0


def _interp_profile_value(
    pres_desc: np.ndarray,
    values_desc: np.ndarray,
    level_hpa: float,
) -> float:
    """
    Interpola una variable a un nivell de pressió estàndard en espai log(p).

    Només s'interpola si:
    - el nivell cau dins del rang observat;
    - els punts bracketing tenen valors finits;
    - el gap vertical no supera un llindar físicament raonable.
    """
    mask = np.isfinite(pres_desc) & np.isfinite(values_desc)
    if int(mask.sum()) < 2:
        return float("nan")

    p = pres_desc[mask].astype(float)
    v = values_desc[mask].astype(float)

    order = np.argsort(p)  # ascendent per facilitar searchsorted
    p = p[order]
    v = v[order]

    lvl = float(level_hpa)
    if lvl < float(p[0]) or lvl > float(p[-1]):
        return float("nan")

    # Match exacte.
    exact = np.where(np.isclose(p, lvl, atol=1e-6))[0]
    if exact.size > 0:
        return float(v[exact[0]])

    j = int(np.searchsorted(p, lvl, side="left"))
    if j <= 0 or j >= len(p):
        return float("nan")

    p_lo, p_hi = float(p[j - 1]), float(p[j])
    v_lo, v_hi = float(v[j - 1]), float(v[j])
    if not (np.isfinite(v_lo) and np.isfinite(v_hi)):
        return float("nan")

    gap = float(p_hi - p_lo)
    if gap > _max_interp_gap_hpa(lvl):
        return float("nan")

    x = np.log(np.array([p_lo, p_hi], dtype=float))
    y = np.array([v_lo, v_hi], dtype=float)
    return float(np.interp(np.log(lvl), x, y))


def _layer_lapse_rate(t_low: float, z_low: float, t_high: float, z_high: float) -> float:
    """Retorna lapse rate en °C/km (positiu si la temperatura decreix amb l'altura)."""
    if not (np.isfinite(t_low) and np.isfinite(z_low) and np.isfinite(t_high) and np.isfinite(z_high)):
        return float("nan")
    dz = float(z_high - z_low)
    if abs(dz) < 1.0:
        return float("nan")
    return float((t_low - t_high) / dz * 1000.0)


def _layer_shear(u_low: float, v_low: float, u_high: float, v_high: float) -> float:
    """Mòdul del vector de cisallament entre dos nivells."""
    if not all(np.isfinite(x) for x in (u_low, v_low, u_high, v_high)):
        return float("nan")
    return float(np.hypot(u_high - u_low, v_high - v_low))


def _build_radiosonde_feature_row(path: Path, cfg: Config) -> Optional[Dict[str, object]]:
    """
    Resumeix un radiosondatge en un vector compacte i robust.

    Estratègia:
    - s'interpola només a nivells de pressió estàndard;
    - es deriven poques variables però molt informatives;
    - s'eviten features clarament redundants amb la superfície.
    """
    ts = _radiosonde_filename_to_timestamp(path)
    if ts is None:
        return None

    prof = _read_one_radiosonde_file(path)

    row: Dict[str, object] = {
        "rs_launch_time": ts,
        "rs_available": 0.0,
        "rs_std_avail_frac": np.nan,
    }

    if prof.empty:
        return row

    row["rs_available"] = 1.0
    row["rs_top_pres_hPa"] = float(prof["pres_hPa"].min())

    pres = prof["pres_hPa"].to_numpy(dtype=float)
    gaps = np.diff(pres * -1.0)  # pressió descendent -> gaps positius
    if len(gaps) > 0:
        row["rs_max_pressure_gap_hPa"] = float(np.nanmax(gaps))
    else:
        row["rs_max_pressure_gap_hPa"] = 0.0

    levels = [int(x) for x in cfg.radiosonde_levels_hPa]
    interp: Dict[str, Dict[int, float]] = {name: {} for name in ("temp", "rh", "u", "v", "z")}
    for lev in levels:
        interp["temp"][lev] = _interp_profile_value(pres, prof["temp_C"].to_numpy(dtype=float), lev)
        interp["rh"][lev] = _interp_profile_value(pres, prof["rh_pct"].to_numpy(dtype=float), lev)
        interp["u"][lev] = _interp_profile_value(pres, prof["u_ms"].to_numpy(dtype=float), lev)
        interp["v"][lev] = _interp_profile_value(pres, prof["v_ms"].to_numpy(dtype=float), lev)
        interp["z"][lev] = _interp_profile_value(pres, prof["alt_m"].to_numpy(dtype=float), lev)

    # Fracció de nivells estàndard útils.
    core_ok = []
    for lev in levels:
        core_ok.append(
            np.isfinite(interp["temp"][lev])
            and np.isfinite(interp["rh"][lev])
            and np.isfinite(interp["u"][lev])
            and np.isfinite(interp["v"][lev])
        )
    if core_ok:
        row["rs_std_avail_frac"] = float(np.mean(core_ok))

    # Directes compactes: T/RH/U/V a 925/850/700/500.
    for lev in (925, 850, 700, 500):
        if lev in levels:
            row[f"rs_temp_{lev}hPa_C"] = interp["temp"][lev]
            row[f"rs_rh_{lev}hPa_pct"] = interp["rh"][lev]
            row[f"rs_u_{lev}hPa_ms"] = interp["u"][lev]
            row[f"rs_v_{lev}hPa_ms"] = interp["v"][lev]

    # Derivades robustes: lapse rates, gruix geopotencial i cisallament.
    if all(lev in levels for lev in (925, 850)):
        row["rs_lapse_925_850_C_km"] = _layer_lapse_rate(
            interp["temp"][925], interp["z"][925], interp["temp"][850], interp["z"][850]
        )
    if all(lev in levels for lev in (850, 700)):
        row["rs_lapse_850_700_C_km"] = _layer_lapse_rate(
            interp["temp"][850], interp["z"][850], interp["temp"][700], interp["z"][700]
        )
    if all(lev in levels for lev in (700, 500)):
        row["rs_lapse_700_500_C_km"] = _layer_lapse_rate(
            interp["temp"][700], interp["z"][700], interp["temp"][500], interp["z"][500]
        )
    if all(lev in levels for lev in (1000, 500)):
        if np.isfinite(interp["z"][1000]) and np.isfinite(interp["z"][500]):
            row["rs_thickness_1000_500_m"] = float(interp["z"][500] - interp["z"][1000])
        row["rs_shear_1000_500_ms"] = _layer_shear(
            interp["u"][1000], interp["v"][1000], interp["u"][500], interp["v"][500]
        )
    if all(lev in levels for lev in (1000, 700)):
        row["rs_shear_1000_700_ms"] = _layer_shear(
            interp["u"][1000], interp["v"][1000], interp["u"][700], interp["v"][700]
        )

    return row


def _load_radiosonde_features_table(cfg: Config) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    Llegeix tots els fitxers de radiosondatge d'un directori i en construeix una taula resum.

    La taula resultant queda indexada per l'hora de llançament i es farà servir
    més tard amb una fusió causal de tipus merge_asof.
    """
    if not cfg.radiosonde_dir:
        return pd.DataFrame(), {"enabled": False}

    rs_dir = Path(cfg.radiosonde_dir).expanduser().resolve()
    if not rs_dir.exists() or not rs_dir.is_dir():
        raise FileNotFoundError(f"Directori de radiosondatge inexistent o invàlid: {rs_dir}")

    rows: List[Dict[str, object]] = []
    n_seen = 0
    n_parsed = 0

    files = [p for p in sorted(rs_dir.glob(cfg.radiosonde_pattern)) if p.is_file()]
    iterator = files
    if tqdm is not None:
        iterator = tqdm(files, desc="Llegint radiosondatges", unit="fitxer")

    for path in iterator:
        n_seen += 1
        row = _build_radiosonde_feature_row(path, cfg)
        if row is None:
            continue
        n_parsed += 1
        rows.append(row)

    if not rows:
        return pd.DataFrame(), {
            "enabled": True,
            "directory": str(rs_dir),
            "pattern": cfg.radiosonde_pattern,
            "n_files_seen": n_seen,
            "n_profiles_loaded": 0,
        }

    rs_df = pd.DataFrame(rows)
    rs_df = rs_df.sort_values("rs_launch_time").drop_duplicates(subset=["rs_launch_time"], keep="last").reset_index(drop=True)

    info = {
        "enabled": True,
        "directory": str(rs_dir),
        "pattern": cfg.radiosonde_pattern,
        "n_files_seen": int(n_seen),
        "n_profiles_loaded": int(n_parsed),
        "time_start": str(rs_df["rs_launch_time"].min()),
        "time_end": str(rs_df["rs_launch_time"].max()),
        "n_rows": int(len(rs_df)),
    }
    return rs_df, info


def _merge_radiosonde_asof(
    df_idx: pd.DataFrame,
    rs_df: pd.DataFrame,
    cfg: Config,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    Fusiona de manera causal el darrer radiosondatge disponible amb cada timestamp del dataset.

    Regles:
    - només s'utilitza el sondatge més recent anterior o igual a l'instant.
    - si l'edat supera radiosonde_max_age_hours, es considera no disponible.
    - es conserva rs_available com a flag operatiu.
    """
    if rs_df.empty:
        return df_idx.copy(), {
            "n_rows_with_rs_launch": 0,
            "n_rows_rs_available": 0,
            "n_rows_rs_too_old": 0,
        }

    left = df_idx.copy()
    left.index.name = cfg.timestamp_col
    left = left.reset_index().sort_values(cfg.timestamp_col)
    right = rs_df.sort_values("rs_launch_time").copy()

    merged = pd.merge_asof(
        left,
        right,
        left_on=cfg.timestamp_col,
        right_on="rs_launch_time",
        direction="backward",
    )

    merged["rs_age_hours"] = (
        (merged[cfg.timestamp_col] - merged["rs_launch_time"]).dt.total_seconds() / 3600.0
    )

    rs_cols = [c for c in right.columns if c != "rs_launch_time"]
    rows_with_launch = merged["rs_launch_time"].notna()

    too_old = pd.Series(False, index=merged.index)
    if cfg.radiosonde_max_age_hours is not None and float(cfg.radiosonde_max_age_hours) > 0.0:
        too_old = rows_with_launch & (merged["rs_age_hours"] > float(cfg.radiosonde_max_age_hours))

    # Quan no hi ha perfil usable, tots els descriptors continus passen a NaN i els flags binaris a 0.
    binary_cols = {"rs_available"}
    invalid = (~rows_with_launch) | too_old | (merged.get("rs_available", 0.0).fillna(0.0) <= 0.5)
    for c in rs_cols:
        if c in binary_cols:
            merged.loc[invalid, c] = 0.0
        else:
            merged.loc[invalid, c] = np.nan

    merged.loc[invalid, "rs_available"] = 0.0
    merged.loc[invalid, "rs_age_hours"] = np.nan

    out = merged.set_index(cfg.timestamp_col)
    out.index.name = cfg.timestamp_col
    out = out.drop(columns=["rs_launch_time"], errors="ignore")

    info = {
        "n_rows_with_rs_launch": int(rows_with_launch.sum()),
        "n_rows_rs_available": int((out.get("rs_available", pd.Series(index=out.index, dtype=float)).fillna(0.0) > 0.5).sum()),
        "n_rows_rs_too_old": int(too_old.sum()),
    }
    return out, info


def _add_and_impute_radiosonde_features(
    df: pd.DataFrame,
    cfg: Config,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    Afegeix radiosondatge al dataset final i en resol els missing de manera robusta.

    - es creen flags __miss per a descriptors continus del radiosondatge.
    - la imputació es fa amb mediana de TRAIN per evitar leakage.
    - rs_available queda com a flag explícit d'existència de perfil usable.
    """
    rs_df, load_info = _load_radiosonde_features_table(cfg)
    if rs_df.empty:
        return df.copy(), {"enabled": bool(cfg.radiosonde_dir), **load_info}

    merged, merge_info = _merge_radiosonde_asof(df, rs_df, cfg)
    out = merged.copy()

    rs_all_cols = [c for c in out.columns if c.startswith("rs_")]
    binary_cols = {"rs_available"}
    continuous_cols = [c for c in rs_all_cols if c not in binary_cols]

    # Els binaris es defineixen explícitament a 0 quan no hi ha radiosondatge usable.
    for c in binary_cols:
        if c in out.columns:
            out[c] = out[c].fillna(0.0).astype(np.float32)

    # Salvaguarda final: si rs_available=0, cap descriptor continu del radiosondatge
    # no s'ha de considerar observat. Això evita inconsistències residuals en casos
    # de perfil invàlid, fitxer buit o sondatge massa antic.
    invalid_rs = out.get("rs_available", pd.Series(index=out.index, dtype=float)).fillna(0.0) <= 0.5
    if continuous_cols:
        out.loc[invalid_rs, continuous_cols] = np.nan

    train_mask = out.index < pd.Timestamp(cfg.train_end)
    fill_values: Dict[str, float] = {}
    cont_iter = continuous_cols
    if tqdm is not None and continuous_cols:
        cont_iter = tqdm(continuous_cols, desc="Imputant variables de radiosondatge", unit="variable")
    for c in cont_iter:
        miss = out[c].isna().astype(np.int8)
        out[f"{c}__miss"] = miss

        train_vals = pd.to_numeric(out.loc[train_mask, c], errors="coerce")
        train_vals = train_vals[np.isfinite(train_vals)]
        if len(train_vals) > 0:
            fill = float(np.nanmedian(train_vals))
        else:
            fill = 0.0
        fill_values[c] = fill
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(fill).astype(np.float32)

    info = {
        "enabled": True,
        **load_info,
        **merge_info,
        "n_continuous_features": int(len(continuous_cols)),
        "n_binary_features": int(sum(c in out.columns for c in binary_cols)),
        "n_rows_rs_invalid_forced_nan": int(invalid_rs.sum()),
        "fill_values_train_median": fill_values,
    }
    return out, info

# =============================================================================
# Regularització a graella i eliminació de files sense dades (offline)
# =============================================================================

def _regularize_to_grid(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    Regularitza la sèrie temporal a una graella uniforme de pas cfg.freq.

    Passos:
    - Parseja TIMESTAMP, elimina NaT, ordena i elimina duplicats.
    - Reindexa a date_range complet [min, max] amb freq.
    - Retorna DF indexat per TIMESTAMP.

    Nota:
    - Aquest reindex és el que pot crear files "offline" noves (tot NaN).
    """
    df = df.copy()
    df[cfg.timestamp_col] = pd.to_datetime(df[cfg.timestamp_col], errors="coerce")
    df = df.dropna(subset=[cfg.timestamp_col]).sort_values(cfg.timestamp_col)
    df = df.drop_duplicates(subset=[cfg.timestamp_col], keep="last")

    start, end = df[cfg.timestamp_col].min(), df[cfg.timestamp_col].max()
    full_index = pd.date_range(start=start, end=end, freq=cfg.freq)

    out = df.set_index(cfg.timestamp_col).reindex(full_index)
    out.index.name = cfg.timestamp_col
    return out


def _drop_offline_rows_after_reindex(df_idx: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, int]:
    """
    Elimina files creades pel reindex on totes les columnes no-target numèriques són NaN.

    Això replica la filosofia del codi "imputa_NaN.py":
    - Si el reindex crea timestamps d'aturada, acostumen a tenir NaN massius.
    - Aquests NaNs no són "missing" real de sensors, sinó absència de dades.
    - Es treuen per evitar que apareguin NaNs a features/flags a la sortida final.

    Retorna:
    - (df_filtrat, n_offline_eliminades)
    """
    core_cols = [c for c in df_idx.columns if c not in {cfg.temp_col, cfg.rain_col}]
    num_core = [c for c in core_cols if pd.api.types.is_numeric_dtype(df_idx[c])]
    if not num_core:
        return df_idx, 0

    offline = df_idx[num_core].isna().all(axis=1)
    n_off = int(offline.sum())
    if n_off:
        df_idx = df_idx.loc[~offline].copy()
    return df_idx, n_off


# =============================================================================
# Segmentació
# =============================================================================

def _build_segments(df_idx: pd.DataFrame, cfg: Config) -> pd.Series:
    """
    Crea segment_id segons salts temporals (i opcionalment segons missing de target).

    Segmentació per defecte:
    - boundary = gap_break (differences > gap_factor * dt)

    Opcional:
    - segment_break_on_missing_target:
        - "any": trenca si falta temp o pluja
        - "temp": trenca si falta temperatura
        - "rain": trenca si falta pluja
        - "none": no ho fa (default)

    Retorna:
    - Series segment_id alineada amb df_idx.index
    """
    dt = pd.Timedelta(cfg.freq)
    thresh = cfg.gap_factor * dt

    diffs = df_idx.index.to_series().diff()
    gap_break = (diffs > thresh).fillna(False)

    boundary = gap_break.copy()

    mode = (cfg.segment_break_on_missing_target or "none").lower().strip()
    if mode != "none":
        miss_temp = df_idx[cfg.temp_col].isna() if cfg.temp_col in df_idx.columns else False
        miss_rain = df_idx[cfg.rain_col].isna() if cfg.rain_col in df_idx.columns else False

        if mode == "any":
            miss = miss_temp | miss_rain
        elif mode == "temp":
            miss = miss_temp
        elif mode == "rain":
            miss = miss_rain
        else:
            raise ValueError("segment_break_on_missing_target: none|any|temp|rain")

        # Trenca a l'instant del missing i també a l'instant anterior (per no barrejar límits)
        miss_break = miss | miss.shift(1, fill_value=False)
        boundary = boundary | miss_break

    seg = boundary.cumsum().astype(np.int64)
    seg.name = "segment_id"
    return seg


def _add_segment_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """
    Afegeix metadades de segment:
    - segment_pos: posició 0..(len-1) dins del segment.
    - segment_len: longitud total del segment.

    Aquestes metadades són útils per decidir quins instants poden ser t_end vàlids (valid_t_end).
    """
    out = df.copy()
    out["segment_pos"] = out.groupby("segment_id").cumcount()
    out["segment_len"] = out.groupby("segment_id")["segment_id"].transform("size").astype(np.int64)
    return out


def _compute_valid_t_end(df: pd.DataFrame, cfg: Config) -> pd.Series:
    """
    Marca quins instants poden ser "t_end" vàlids per construir finestres causals.

    Definicions:
    - history_hours: quantes hores de passat s'usen per la finestra d'entrada.
    - horizon_hours: quantes hores al futur es vol preveure.

    Condició geomètrica dins del segment:
    - pos >= (H - 1)
    - pos <= (segment_len - O - 1)

    Retorna:
    - Series valid_t_end (int8: 0/1)
    """
    dt_min = _infer_dt_minutes(cfg.freq)
    steps_per_hour = int(round(60.0 / dt_min))

    H = int(round(cfg.history_hours * steps_per_hour))
    O = int(round(cfg.horizon_hours * steps_per_hour))
    H = max(H, 1)
    O = max(O, 1)

    pos = df["segment_pos"].to_numpy()
    seglen = df["segment_len"].to_numpy()

    valid = (pos >= (H - 1)) & (pos <= (seglen - O - 1))
    return pd.Series(valid.astype(np.int8), index=df.index, name="valid_t_end")


# =============================================================================
# Feature engineering causal (per segment)
# =============================================================================

def _rolling_sum_by_minutes(s: pd.Series, minutes: int, cfg: Config) -> pd.Series:
    """
    Suma mòbil causal de finestra 'minutes' (en minuts) sobre una sèrie.

    - Omple NaN amb 0.0 (interpretació: absència de pluja = 0; o bé feature derivada robusta).
    - rolling amb min_periods=1 per evitar NaNs al principi.
    """
    dt_min = _infer_dt_minutes(cfg.freq)
    w = max(int(round(minutes / dt_min)), 1)
    return s.fillna(0.0).rolling(window=w, min_periods=1).sum()


def _rolling_stats_by_minutes(s: pd.Series, minutes: int, cfg: Config) -> Tuple[pd.Series, pd.Series]:
    """
    Mitjana i desviació estàndard mòbils (causals) amb finestra en minuts.

    - std amb ddof=0 (poblacional) i NaN inicials omplerts amb 0.0 per estabilitat.
    """
    dt_min = _infer_dt_minutes(cfg.freq)
    w = max(int(round(minutes / dt_min)), 1)
    roll = s.rolling(window=w, min_periods=1)
    mean = roll.mean()
    std = roll.std(ddof=0).fillna(0.0)
    return mean, std


def _add_engineered_features(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    Afegeix features derivades (engineering) de manera causal i per segment.

    Regla general anti-leakage:
    - Tot el que es calcula usa només valors actuals i passats dins del segment.
    - No es fa cap ús d'informació futura.

    Inclou:
    - Pluja: sumes recents + temps des de l'última pluja.
    - Tendències (diff) a 10/30/60 min de diverses variables base.
    - Estadístiques rolling (mean/std) per T, pressió i altres trend_cols.
    - Radiació: fracció difusa.
    - Vent: components u/v si hi ha velocitat i sin/cos de direcció.

    Important:
    - Qualsevol nova derivada que encara tingui NaNs s'omple amb 0.0 per garantir
      que no quedin NaNs en no-targets al final del pipeline.
    """
    out = df.copy()
    base_cols = set(out.columns)
    g = out.groupby("segment_id", group_keys=False)

    # --- Pluja: sumes recents + temps des de l'última pluja (>0)
    if cfg.rain_col in out.columns:
        for m in cfg.rain_roll_minutes:
            out[f"rain_sum_{m}min"] = g[cfg.rain_col].apply(lambda s: _rolling_sum_by_minutes(s, m, cfg))

        dt_min = _infer_dt_minutes(cfg.freq)

        def _tsr_series(rain_seg: pd.Series) -> pd.Series:
            rr = rain_seg.to_numpy(dtype=float)
            rr = np.where(np.isfinite(rr), rr, np.nan)
            out_arr = np.zeros_like(rr, dtype=float)
            last: Optional[float] = np.nan
            for i in range(len(rr)):
                if np.isfinite(rr[i]) and rr[i] > 0.0:
                    last = 0.0
                else:
                    last = (0.0 if np.isnan(last) else last + dt_min)
                out_arr[i] = last
            return pd.Series(out_arr, index=rain_seg.index)

        out["rain_since_last_min"] = (
            out.groupby("segment_id")[cfg.rain_col]
            .apply(_tsr_series)
            .reset_index(level=0, drop=True)
        )

    # --- Tendències (diff) a 10/30/60 minuts
    dt_min = _infer_dt_minutes(cfg.freq)
    steps_10 = max(int(round(10 / dt_min)), 1)
    steps_30 = max(int(round(30 / dt_min)), 1)
    steps_60 = max(int(round(60 / dt_min)), 1)
    lag_defs = [("d10min", steps_10), ("d30min", steps_30), ("d60min", steps_60)]

    for col in cfg.trend_cols:
        if col not in out.columns:
            continue
        for label, k in lag_defs:
            out[f"{col}_{label}"] = g[col].apply(lambda s, kk=k: (s - s.shift(kk)).fillna(0.0))

    # --- Rolling mean/std de temperatura
    if cfg.temp_col in out.columns:
        for m in cfg.temp_roll_minutes:
            mean_s = g[cfg.temp_col].apply(lambda s: _rolling_stats_by_minutes(s, m, cfg)[0])
            std_s = g[cfg.temp_col].apply(lambda s: _rolling_stats_by_minutes(s, m, cfg)[1])
            out[f"{cfg.temp_col}_mean_{m}min"] = mean_s
            out[f"{cfg.temp_col}_std_{m}min"] = std_s

    # --- Rolling mean/std de pressió
    if "Pres_Atm_Avg" in out.columns:
        for m in cfg.pres_roll_minutes:
            mean_s = g["Pres_Atm_Avg"].apply(lambda s: _rolling_stats_by_minutes(s, m, cfg)[0])
            std_s = g["Pres_Atm_Avg"].apply(lambda s: _rolling_stats_by_minutes(s, m, cfg)[1])
            out[f"Pres_Atm_Avg_mean_{m}min"] = mean_s
            out[f"Pres_Atm_Avg_std_{m}min"] = std_s

    # --- Rolling mean/std (60 min) per altres variables base
    for col in cfg.trend_cols:
        if col not in out.columns:
            continue
        if col in (cfg.temp_col, "Pres_Atm_Avg"):
            continue
        mean_s = g[col].apply(lambda s: _rolling_stats_by_minutes(s, 60, cfg)[0])
        std_s = g[col].apply(lambda s: _rolling_stats_by_minutes(s, 60, cfg)[1])
        out[f"{col}_mean_60min"] = mean_s
        out[f"{col}_std_60min"] = std_s

    # --- Radiació: fracció difusa (evita divisions per zero i valors absurds)
    if "Rad_Global_Avg" in out.columns and "Rad_Difusa_Avg" in out.columns:
        glob = out["Rad_Global_Avg"].to_numpy(dtype=float)
        difu = out["Rad_Difusa_Avg"].to_numpy(dtype=float)
        frac = np.full_like(glob, 0.0, dtype=float)
        mask = np.isfinite(glob) & (glob > 20.0) & np.isfinite(difu) & (difu >= 0.0)
        frac[mask] = difu[mask] / (glob[mask] + 1e-6)
        out["rad_diffuse_frac"] = np.clip(frac, 0.0, 2.0)

    # --- No afegim components u/v de superfície perquè serien una transformació
    # exacta de Vel_Viento_WVc(1) + wind_dir_sin/cos i, per tant, redundants.

    # --- Evita NaNs a derivades noves (no-targets)
    new_cols = [c for c in out.columns if c not in base_cols]
    for c in new_cols:
        if out[c].isna().any():
            out[c] = out[c].fillna(0.0)

    return out


# =============================================================================
# QC flags
# =============================================================================

def _compute_train_pressure_bounds(p_train: pd.Series, cfg: Config) -> Tuple[float, float]:
    """
    Estima bounds de pressió a train (sense leakage) per a flags QC.

    Mètodes:
    - quantile: [q_low, q_high]
    - iqr: mediana +/- k*IQR

    També s'aplica una amplada mínima (qc_pres_min_span_hPa) i finalment es clipeja dins els llindars locals durs (qc_pres_min_hPa/max).
    """
    x = pd.to_numeric(p_train, errors="coerce").dropna().astype(float)
    x = x[(x >= cfg.qc_pres_min_hPa) & (x <= cfg.qc_pres_max_hPa)]
    if len(x) < 1000:
        return float(cfg.qc_pres_min_hPa), float(cfg.qc_pres_max_hPa)

    m = cfg.qc_pres_train_method.lower().strip()
    if m == "quantile":
        lo = float(x.quantile(cfg.qc_pres_q_low))
        hi = float(x.quantile(cfg.qc_pres_q_high))
    elif m == "iqr":
        q25, q75 = x.quantile([0.25, 0.75]).to_numpy()
        iqr = float(q75 - q25)
        med = float(x.median())
        lo = med - cfg.qc_pres_iqr_k * iqr
        hi = med + cfg.qc_pres_iqr_k * iqr
    else:
        raise ValueError("qc_pres_train_method: quantile|iqr")

    if (hi - lo) < cfg.qc_pres_min_span_hPa:
        med = float(x.median())
        half = cfg.qc_pres_min_span_hPa / 2.0
        lo, hi = med - half, med + half

    lo = max(lo, float(cfg.qc_pres_min_hPa))
    hi = min(hi, float(cfg.qc_pres_max_hPa))
    return float(lo), float(hi)


def _add_qc_flags(
    df: pd.DataFrame,
    cfg: Config,
    pres_bounds: Optional[Tuple[float, float]] = None
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    """
    Afegeix flags QC sense modificar les dades originals.

    Retorna:
    - df amb:
        - qc_n_violations: nombre de checks fallits a cada fila
        - qc_any: 1 si hi ha alguna violació, 0 si no
    - qc_summary: diccionari amb comptatges i fraccions per cada flag

    Nota:
    - pres_bounds pot venir calculat amb train-only; si és None, s'usen llindars locals.
    """
    out = df.copy()
    qc: Dict[str, Dict[str, float]] = {}
    n = len(out)

    def _stat(name: str, flag: np.ndarray) -> None:
        qc[name] = {"count": int(flag.sum()), "frac": float(flag.mean()) if n > 0 else 0.0}

    # Temperatura
    T = out[cfg.temp_col].to_numpy(dtype=float)
    fT = (T < cfg.qc_temp_min_C) | (T > cfg.qc_temp_max_C) | ~np.isfinite(T)
    _stat("qc_temp_out_of_range", fT)

    # Pluja negativa (si n'hi hagués)
    R = out[cfg.rain_col].to_numpy(dtype=float)
    fR = np.isfinite(R) & (R < 0.0)
    _stat("qc_rain_negative", fR)

    # Humitat relativa (si existeix)
    if "Hum_Rel_1_Avg" in out.columns:
        RH = out["Hum_Rel_1_Avg"].to_numpy(dtype=float)
        fRH = (RH < cfg.qc_rh_min_pct) | (RH > cfg.qc_rh_max_pct) | ~np.isfinite(RH)
        _stat("qc_rh_out_of_range", fRH)
    else:
        fRH = np.zeros(n, dtype=bool)

    # Pressió (si existeix)
    if "Pres_Atm_Avg" in out.columns:
        P = out["Pres_Atm_Avg"].to_numpy(dtype=float)
        lo, hi = pres_bounds if pres_bounds is not None else (cfg.qc_pres_min_hPa, cfg.qc_pres_max_hPa)
        fP = (P < lo) | (P > hi) | ~np.isfinite(P)
        _stat("qc_pres_out_of_range", fP)
        qc["qc_pres_bounds_hPa"] = {"lo": float(lo), "hi": float(hi)}
    else:
        fP = np.zeros(n, dtype=bool)

    # Vent (si existeix)
    for wcol in ("Vel_Viento_WVc(1)",):
        if wcol in out.columns:
            W = out[wcol].to_numpy(dtype=float)
            fW = (W < -1e-6) | (W > cfg.qc_wind_max_ms) | ~np.isfinite(W)
            _stat(f"qc_{wcol}_out_of_range", fW)

    # Resum global de violacions
    flags = np.vstack([fT, fR, fRH, fP]).T
    out["qc_n_violations"] = flags.sum(axis=1).astype(np.int16)
    out["qc_any"] = (out["qc_n_violations"] > 0).astype(np.int8)

    return out, qc


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> Config:
    """
    Parseja arguments de línia d'ordres i construeix Config.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", dest="input_csv", required=True)
    ap.add_argument("--output", dest="output_csv", required=True)
    ap.add_argument("--report", dest="report_json", required=True)

    ap.add_argument("--freq", default="10min")
    ap.add_argument("--history-hours", type=float, default=48.0)
    ap.add_argument("--horizon-hours", type=float, default=3.0)
    ap.add_argument("--gap-factor", type=float, default=1.5)

    ap.add_argument("--segment-break-on-missing-target", default="none", choices=["none", "any", "temp", "rain"])

    ap.add_argument("--data-start", default="2012-01-01", help="Data inicial del dataset final (recomanat: 2012-01-01).")
    ap.add_argument("--train-end", default="2023-01-01")

    ap.add_argument("--feature-set", default="compact", choices=["full", "compact"])
    ap.add_argument("--keep-aux-cols", action="store_true")

    ap.add_argument("--no-engineered", action="store_true")
    ap.add_argument("--no-qc", action="store_true")

    # Offline rows
    ap.add_argument("--keep-offline-rows", action="store_true", help="No eliminar files offline reintroduïdes pel reindex.")

    # QC pressió
    ap.add_argument("--qc-pres-min-hpa", type=float, default=930.0)
    ap.add_argument("--qc-pres-max-hpa", type=float, default=1060.0)
    ap.add_argument("--no-qc-pres-train-bounds", action="store_true")
    ap.add_argument("--qc-pres-train-method", default="iqr", choices=["quantile", "iqr"])
    ap.add_argument("--qc-pres-q-low", type=float, default=0.001)
    ap.add_argument("--qc-pres-q-high", type=float, default=0.999)
    ap.add_argument("--qc-pres-iqr-k", type=float, default=8.0)
    ap.add_argument("--qc-pres-min-span-hpa", type=float, default=30.0)

    ap.add_argument("--qc-wind-max-ms", type=float, default=120.0)

    # Radiosondatge
    ap.add_argument("--radiosonde-dir", default=None, help="Directori amb fitxers de radiosondatge YYMMDDHH.txt.")
    ap.add_argument("--radiosonde-pattern", default="*.txt", help="Patró glob per trobar fitxers de radiosondatge.")
    ap.add_argument("--radiosonde-max-age-hours", type=float, default=18.0, help="Edat màxima admissible del darrer sondatge per considerar-lo usable.")

    ap.add_argument("--keep-qc-predictors", action="store_true", help="Conserva qc_n_violations i qc_any al dataset final.")
    ap.add_argument("--keep-constant-predictors", action="store_true", help="No eliminar columnes constants exactes del dataset final.")

    args = ap.parse_args()

    return Config(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        report_json=args.report_json,
        freq=args.freq,
        history_hours=args.history_hours,
        horizon_hours=args.horizon_hours,
        gap_factor=args.gap_factor,
        segment_break_on_missing_target=args.segment_break_on_missing_target,
        data_start=args.data_start,
        train_end=args.train_end,
        feature_set=args.feature_set,
        add_engineered=not args.no_engineered,
        add_qc_flags=not args.no_qc,
        drop_offline_rows=not args.keep_offline_rows,
        qc_pres_min_hPa=args.qc_pres_min_hpa,
        qc_pres_max_hPa=args.qc_pres_max_hpa,
        qc_pres_use_train_bounds=not args.no_qc_pres_train_bounds,
        qc_pres_train_method=args.qc_pres_train_method,
        qc_pres_q_low=args.qc_pres_q_low,
        qc_pres_q_high=args.qc_pres_q_high,
        qc_pres_iqr_k=args.qc_pres_iqr_k,
        qc_pres_min_span_hPa=args.qc_pres_min_span_hpa,
        qc_wind_max_ms=args.qc_wind_max_ms,
        radiosonde_dir=args.radiosonde_dir,
        radiosonde_pattern=args.radiosonde_pattern,
        radiosonde_max_age_hours=args.radiosonde_max_age_hours,
        drop_qc_predictors=not args.keep_qc_predictors,
        drop_constant_predictors=not args.keep_constant_predictors,
        drop_aux_cols=tuple([]) if args.keep_aux_cols else Config.drop_aux_cols,
    )


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    """
    Pipeline principal.

    Ordre d'operacions:
    1) Lectura i validacions bàsiques.
    2) Neteja de metadades velles i auxiliars (evita duplicats/leakage).
    3) Conversió a numèric i reindex a graella.
    4) Eliminació de files offline reintroduïdes pel reindex (si activat).
    5) Flags de missing de targets (sense imputació).
    6) Segmentació + metadades de segment + valid_t_end.
    7) Feature engineering (opcional).
    8) Integració causal del radiosondatge (opcional) + imputació train-only.
    9) QC flags (opcional, i pressió amb bounds train-only si es pot).
    10) Neteja infinits i verificació final: NaNs només als targets.
    11) Export csv i report json.
    """
    t0_all = time.perf_counter()
    cfg = parse_args()

    print("[INFO] Llegint CSV d'entrada...")
    t0_csv = time.perf_counter()
    df = pd.read_csv(cfg.input_csv)
    print(f"[INFO] CSV llegit en {time.perf_counter() - t0_csv:.1f} s")

    # --- Validacions mínimes
    if cfg.timestamp_col not in df.columns:
        raise ValueError(f"No existeix {cfg.timestamp_col} al csv d'entrada.")
    for req in (cfg.temp_col, cfg.rain_col):
        if req not in df.columns:
            raise ValueError(f"Falta la columna requerida {req} al csv d'entrada.")

    # --- Limita el període per coherència temporal amb el radiosondatge
    df = _apply_data_start_filter(df, cfg)

    # --- Evita duplicats de metadades si venien d'execucions prèvies
    df = df.drop(columns=["segment_id", "segment_pos", "segment_len", "valid_t_end"], errors="ignore")

    # --- Elimina auxiliars del target (evita leakage accidental)
    if cfg.drop_aux_cols:
        df = _drop_cols_with_miss(df, list(cfg.drop_aux_cols))

    # --- Redundància radiativa: preset "compact"
    if cfg.feature_set == "compact":
        # Preset compacte reforçat: elimina totes les variables base considerades
        # redundants o poc informatives per a la generalització del model.
        df = _drop_cols_with_miss(
            df,
            [
                "VP_Tot",
                "Vel_Viento_WVc(3)",
                "Rad_Difusa_Avg",
                "NetRs_Avg",
                "NetRl_Avg",
                "Albedo_Avg",
                "UVA_Avg",
                "UVB_Avg",
            ],
        )

    # --- Converteix a numèric (excepte TIMESTAMP)
    df = _safe_to_numeric(df, exclude=[cfg.timestamp_col])

    # --- Regularitza a graella
    df_idx = _regularize_to_grid(df, cfg)

    # IMPORTANT: elimina files offline reintroduïdes pel reindex
    if cfg.drop_offline_rows:
        df_idx, n_off = _drop_offline_rows_after_reindex(df_idx, cfg)
        if n_off:
            print(f"[INFO] Eliminades {n_off} files offline (reindex)")

    # --- Flags de missing dels targets a la graella (no imputem targets)
    df_idx[f"{cfg.temp_col}__miss"] = df_idx[cfg.temp_col].isna().astype(np.int8)
    df_idx[f"{cfg.rain_col}__miss"] = df_idx[cfg.rain_col].isna().astype(np.int8)

    # --- Segmentació (per defecte: només gaps reals)
    df_idx["segment_id"] = _build_segments(df_idx, cfg)

    # --- Check de columnes duplicades
    _unique_columns(df_idx)

    # --- Metadades de segment
    df2 = _add_segment_metadata(df_idx)

    # --- Valid t_end (geometria de finestres)
    df2["valid_t_end"] = _compute_valid_t_end(df2, cfg)

    # --- Feature engineering causal (per segment)
    if cfg.add_engineered:
        df2 = _add_engineered_features(df2, cfg)

    # --- Radiosondatge (fusió causal + imputació train-only)
    radiosonde_info: Dict[str, object] = {"enabled": False}
    if cfg.radiosonde_dir:
        print("[INFO] Integrant radiosondatge...")
        t0_rs = time.perf_counter()
        df2, radiosonde_info = _add_and_impute_radiosonde_features(df2, cfg)
        print(f"[INFO] Radiosondatge integrat en {time.perf_counter() - t0_rs:.1f} s")

    # --- QC (flags, sense tocar dades)
    qc_summary: Dict[str, Dict[str, float]] = {}
    pres_bounds = None
    if cfg.add_qc_flags:
        if cfg.qc_pres_use_train_bounds and "Pres_Atm_Avg" in df2.columns:
            train_end = pd.Timestamp(cfg.train_end)
            train_mask = df2.index < train_end
            if int(train_mask.sum()) >= 10000:
                pres_bounds = _compute_train_pressure_bounds(df2.loc[train_mask, "Pres_Atm_Avg"], cfg)
        df2, qc_summary = _add_qc_flags(df2, cfg, pres_bounds=pres_bounds)

    # --- Neteja infinits
    df2 = df2.replace([np.inf, -np.inf], np.nan)

    # --- Poda final de predictors no informatius
    df2, pruning_info = _prune_predictors(df2, cfg)
    if pruning_info["n_dropped_qc_predictors"] > 0:
        print(f"[INFO] Eliminats predictors QC del dataset final: {pruning_info['dropped_qc_predictors']}")
    if pruning_info["n_dropped_constant_predictors"] > 0:
        print(f"[INFO] Eliminades {pruning_info['n_dropped_constant_predictors']} columnes constants del dataset final")

    # Permetem NaNs només als targets (per loss masking)
    non_target_cols = [c for c in df2.columns if c not in {cfg.temp_col, cfg.rain_col}]
    if df2[non_target_cols].isna().any().any():
        bad = df2[non_target_cols].isna().mean().sort_values(ascending=False).head(30)
        raise RuntimeError(f"Encara queden NaNs a no-targets. Pitjors:\n{bad}")

    # --- Export csv
    out_df = df2.reset_index().rename(columns={cfg.timestamp_col: "TIMESTAMP"})
    out_df.to_csv(cfg.output_csv, index=False)

    # --- Report json
    seg_sizes = df2.groupby("segment_id").size()
    report = {
        "config": asdict(cfg),
        "qc": qc_summary,
        "radiosonde": radiosonde_info,
        "pruning": pruning_info,
        "summary": {
            "time_start": str(df2.index.min()),
            "data_start_applied": str(pd.Timestamp(cfg.data_start)) if cfg.data_start else None,
            "time_end": str(df2.index.max()),
            "n_rows_out": int(len(df2)),
            "n_cols_out": int(df2.shape[1]),
            "n_rows_valid_t_end": int(df2["valid_t_end"].sum()) if "valid_t_end" in df2.columns else 0,
            "n_temp_missing": int(df2[f"{cfg.temp_col}__miss"].sum()),
            "n_rain_missing": int(df2[f"{cfg.rain_col}__miss"].sum()),
            "n_rows_rs_available": int((df2["rs_available"] > 0.5).sum()) if "rs_available" in df2.columns else 0,
        },
        "segments": {
            "n_segments": int(seg_sizes.shape[0]),
            "rows_per_segment_min": int(seg_sizes.min()) if len(seg_sizes) else 0,
            "rows_per_segment_median": float(seg_sizes.median()) if len(seg_sizes) else 0.0,
            "rows_per_segment_max": int(seg_sizes.max()) if len(seg_sizes) else 0,
        },
        "columns": {"output_columns": ["TIMESTAMP"] + [c for c in df2.columns]},
    }

    with open(cfg.report_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"[OK] Output: {cfg.output_csv}")
    print(f"[OK] Report: {cfg.report_json}")
    print(f"[INFO] Rows out: {len(out_df)}, Cols out: {out_df.shape[1]}")
    print(f"[INFO] Temps total: {time.perf_counter() - t0_all:.1f} s")


if __name__ == "__main__":
    main()