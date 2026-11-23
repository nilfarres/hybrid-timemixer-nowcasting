#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Imputació ML-safe (sense leakage) per a la sèrie de l'estació meteorològica de la UB (cada 10 min).

Objectiu principal:
------------------
Preparar "features" per a un model de nowcasting assegurant que:
- No hi hagi "data leakage": qualsevol estadística utilitzada per imputar es calcula només amb train.
- No s'utilitzi informació del futur: el farciment local és un forward-fill limitat (sense interpolació bidireccional).
- No s'inventin etiquetes: els targets (temperatura i pluja) no s'imputen.
- Es descartin variables massa incompletes segons un llindar de fracció de NaN al train (després d'uns filtres físics).
- No es propagui informació entre segments: el forward-fill es fa per segment_id.

L'script genera "features" completes (sense NaNs) i, per a cada feature, un flag __miss indicant si el valor original era NaN abans d'imputar.
Els targets poden contenir NaN (si a l'original hi ha buits). En l'entrenament cal filtrar finestres amb target disponible.

Sortides:
--------
- CR3000_Tabla1_2011_2024_imputed.csv
- nan_fraction_train_after_filters.csv
- imputation_report.json

Exemple d'ús:
------------
python imputa_NaN.py --input "Dataset_csv/CR3000_Tabla1_2011_2024_clean.csv" --output-dir "Dataset_csv" --train-end "2023-01-01" --max-nan-frac 0.40 --freq "10min" --timestamp-col "TIMESTAMP" --temp-target "Temp_1_Avg" --rain-target "Lluvia_Tot" --wind-dir-col "Vel_Viento_WVc(2)" --pressure-train-method iqr --pressure-iqr-k 10
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# Configuració
# =============================================================================

@dataclass
class Config:
    """
    Contenidor de paràmetres del pipeline d'imputació.

    Les opcions estan pensades per garantir un flux ML-safe: tot el que sigui "model" o "climatologia" s'estima només amb train.
    """
    input_csv: str
    output_dir: str

    # Eix temporal
    timestamp_col: str = "TIMESTAMP"
    freq: str = "10min"

    # Targets (no s'imputen)
    temp_target: str = "Temp_1_Avg"
    rain_target: str = "Lluvia_Tot"

    # Tall train per evitar leakage (train = índex < train_end)
    train_end: str = "2023-01-01"

    # Elimina features massa incompletes (fracció de NaN a train després de filtres físics)
    max_nan_frac: float = 0.40

    # Gaps màxims (en minuts) per al forward-fill
    max_gap_thermo_min: int = 360
    max_gap_wind_min: int = 60
    max_gap_rad_min: int = 60
    max_gap_default_min: int = 180

    # Filtres físics bàsics
    temp_min_C: float = -50.0
    temp_max_C: float = 50.0

    # Pressió: llindars durs (hPa) + filtre robust train-only
    pressure_min_hPa: float = 930.0
    pressure_max_hPa: float = 1060.0

    pressure_train_filter: bool = True
    pressure_train_method: str = "quantile"  # "quantile" o "iqr"
    pressure_q_low: float = 0.001
    pressure_q_high: float = 0.999
    pressure_iqr_k: float = 8.0
    pressure_min_span_hPa: float = 30.0  # evita bounds massa estrets en casos rars

    # Humitat relativa (sensor 1)
    rh_min_pct: float = 0.0
    rh_max_pct: float = 100.0

    # Vent
    wind_speed_max_ms: float = 120.0
    encode_wind_dir: bool = True
    wind_dir_col: str = "Vel_Viento_WVc(2)"  # al CSV és 0..360

    # Downtime: elimina files creades pel reindex que són quasi tot NaN
    drop_offline_rows: bool = True

    # Columnes a descartar explícitament (per defecte: sensor 2 i VP/VPD 2)
    drop_cols_explicit: Tuple[str, ...] = (
        "Temp_2_Min", "Temp_2_Max", "Temp_2_Avg",
        "Hum_Rel_2_Min", "Hum_Rel_2_Max", "Hum_Rel_2_Avg",
        "VP_2_Tot", "VPD_2_Tot",
    )


# =============================================================================
# I/O i regularització temporal
# =============================================================================

def ensure_dir(path: str) -> None:
    """Crea el directori si no existeix."""
    os.makedirs(path, exist_ok=True)


def read_and_regularize(cfg: Config) -> pd.DataFrame:
    """
    Llegeix el CSV i força una malla temporal regular a "cfg.freq".

    Passos:
    1) Llegeix el CSV i parseja la columna temporal.
    2) Ordena, elimina timestamps invàlids i duplicats.
    3) Reindexa a un "date_range" complet (això pot crear files noves amb NaNs).
    4) (Opcional) Elimina downtime: files on tots els canals numèrics (excepte RECORD) són NaN.
    5) Defineix "segment_id" per tallar el forward-fill en salts temporals grans.

    Retorna:
    - DataFrame indexat per temps, amb "segment_id" com a columna.
    """
    df_raw = pd.read_csv(cfg.input_csv)
    if cfg.timestamp_col not in df_raw.columns:
        raise ValueError(f"No trobo {cfg.timestamp_col} al CSV.")

    # Parse temporal (errors='coerce' converteix valors invàlids a NaT)
    df_raw[cfg.timestamp_col] = pd.to_datetime(df_raw[cfg.timestamp_col], errors="coerce")
    df_raw = df_raw.dropna(subset=[cfg.timestamp_col]).sort_values(cfg.timestamp_col)

    # Duplicats: ens quedem amb l'últim valor
    df_raw = df_raw.drop_duplicates(subset=[cfg.timestamp_col], keep="last").reset_index(drop=True)

    # Reindex a malla temporal regular
    start, end = df_raw[cfg.timestamp_col].min(), df_raw[cfg.timestamp_col].max()
    full_index = pd.date_range(start=start, end=end, freq=cfg.freq)

    df = df_raw.set_index(cfg.timestamp_col).reindex(full_index)
    df.index.name = cfg.timestamp_col

    # Elimina downtime: files que el reindex ha creat però que són buides (totes NaN)
    if cfg.drop_offline_rows:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        core = [c for c in num_cols if c != "RECORD"]
        if core:
            offline = df[core].isna().all(axis=1)
            df = df.loc[~offline].copy()

    # Segmentació: evita creuar salts grans (p. ex. buits llargs)
    dt = pd.Timedelta(cfg.freq)
    diffs = df.index.to_series().diff()
    df["segment_id"] = (diffs > dt * 1.5).cumsum().fillna(0).astype(int).to_numpy()

    return df


# =============================================================================
# Filtres físics
# =============================================================================

def apply_physical_filters(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    Aplica filtres físics per convertir valors clarament invàlids a NaN:

    Els filtres només eliminen valors físicament erronis. Això facilita que la imputació posterior sigui més estable i coherent.
    La pluja (target) només es transforma a no-negativa. No s'imputa.
    L'albedo del fitxer es força a NaN perquè es recomputa després de la imputació.
    """
    out = df.copy()

    # Descarta explícitament certes columnes (per defecte: sensor 2, que és molt incomplet, i VP/VPD 2 que no fem servir)
    for c in cfg.drop_cols_explicit:
        if c in out.columns:
            out[c] = np.nan

    # UVA (p. ex. -7999): qualsevol valor molt negatiu és un codi d'error
    for c in ["Rad_UVA_Avg", "Rad_UVA_Tot"]:
        if c in out.columns:
            out[c] = out[c].where(out[c] > -1000.0, np.nan)

    # Temperatures en ºC
    temp_like = [
        "Temp_1_Min", "Temp_1_Max", "Temp_1_Avg",
        "T_datalogger", "CNR1TC_Avg",
    ]
    for c in temp_like:
        if c in out.columns:
            out[c] = out[c].where((out[c] > cfg.temp_min_C) & (out[c] < cfg.temp_max_C), np.nan)

    # CNR1TK (Kelvin): rang físic raonable per temperatura de sensor
    if "CNR1TK_Avg" in out.columns:
        out["CNR1TK_Avg"] = out["CNR1TK_Avg"].where(
            (out["CNR1TK_Avg"] >= 200.0) & (out["CNR1TK_Avg"] <= 340.0),
            np.nan
        )

    # RH 0-100 (sensor 1)
    for c in ["Hum_Rel_1_Min", "Hum_Rel_1_Max", "Hum_Rel_1_Avg"]:
        if c in out.columns:
            out[c] = out[c].where((out[c] >= cfg.rh_min_pct) & (out[c] <= cfg.rh_max_pct), np.nan)

    # Pressió (llindar local, hPa)
    if "Pres_Atm_Avg" in out.columns:
        out["Pres_Atm_Avg"] = out["Pres_Atm_Avg"].where(
            (out["Pres_Atm_Avg"] >= cfg.pressure_min_hPa) & (out["Pres_Atm_Avg"] <= cfg.pressure_max_hPa),
            np.nan
        )

    # Vent:
    # - Vel_Viento_Max: velocitat (m/s), no pot ser negativa i té un màxim físic alt
    if "Vel_Viento_Max" in out.columns:
        out["Vel_Viento_Max"] = out["Vel_Viento_Max"].where(
            (out["Vel_Viento_Max"] >= 0.0) & (out["Vel_Viento_Max"] <= cfg.wind_speed_max_ms),
            np.nan
        )

    # - Direcció (0..360)
    if cfg.wind_dir_col in out.columns:
        out[cfg.wind_dir_col] = out[cfg.wind_dir_col].where(
            (out[cfg.wind_dir_col] >= 0.0) & (out[cfg.wind_dir_col] <= 360.0),
            np.nan
        )

    # - Altres canals de vent: filtres suaus (0..200, segons el fitxer original)
    for c in [col for col in out.columns if col.startswith("Vel_Viento") and col not in {"Vel_Viento_Max", cfg.wind_dir_col}]:
        out[c] = out[c].where((out[c] >= 0.0) & (out[c] <= 200.0), np.nan)

    # Pluja no-negativa (target): només coherència física, NO imputació
    if cfg.rain_target in out.columns:
        out[cfg.rain_target] = out[cfg.rain_target].clip(lower=0.0)

    # Radiació: elimina outliers poc coherents (rang ampli per evitar eliminar la variabilitat real)
    rad_cols = [c for c in out.columns if any(k in c for k in ["Rad_", "CM3", "CG3", "UVA_Avg", "UVB_Avg"])]
    for c in rad_cols:
        if c in out.columns and pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c].where((out[c] > -5000.0) & (out[c] < 5000.0), np.nan)

    # Albedo del fitxer: no el fem servir (el recomputem després)
    if "Albedo_Avg" in out.columns:
        out["Albedo_Avg"] = np.nan

    return out


# =============================================================================
# Pressió: bounds train-only
# =============================================================================

def compute_train_pressure_bounds(p_train: pd.Series, cfg: Config) -> Tuple[float, float]:
    """
    Estimació de bounds de pressió a TRAIN (sense leakage).

    - Converteix a numèric i filtra dins els llindars establerts.
    - Si train és massa petit, retorna directament els llindars.
    - Estima bounds amb:
       -> "quantile": quantils baixos/alts (cfg.pressure_q_low / cfg.pressure_q_high)
       -> "iqr": mediana +/- k*IQR (cfg.pressure_iqr_k)
    - Força una amplada mínima (cfg.pressure_min_span_hPa) per evitar bounds massa estrets.
    - Retalla finalment dins els llindars.

    Retorna:
    - (lo, hi) en hPa.
    """
    x = pd.to_numeric(p_train, errors="coerce").dropna().astype(float)
    x = x[(x >= cfg.pressure_min_hPa) & (x <= cfg.pressure_max_hPa)]

    # Si train és massa petit o la columna és gairebé buida, fem fallback als llindars
    if len(x) < 1000:
        return float(cfg.pressure_min_hPa), float(cfg.pressure_max_hPa)

    m = cfg.pressure_train_method.lower().strip()
    if m == "quantile":
        lo = float(x.quantile(cfg.pressure_q_low))
        hi = float(x.quantile(cfg.pressure_q_high))
    elif m == "iqr":
        q25, q75 = x.quantile([0.25, 0.75]).to_numpy()
        iqr = float(q75 - q25)
        med = float(x.median())
        lo = med - cfg.pressure_iqr_k * iqr
        hi = med + cfg.pressure_iqr_k * iqr
    else:
        raise ValueError(f"pressure_train_method no suportat: {cfg.pressure_train_method}")

    # Evita bounds massa estrets (protecció contra casos rars)
    if (hi - lo) < cfg.pressure_min_span_hPa:
        med = float(x.median())
        half = cfg.pressure_min_span_hPa / 2.0
        lo, hi = med - half, med + half

    # Talla dins els límits establerts
    lo = max(lo, float(cfg.pressure_min_hPa))
    hi = min(hi, float(cfg.pressure_max_hPa))

    return float(lo), float(hi)


# =============================================================================
# VP/VPD: recomputació (Pa) amb Bolton
# =============================================================================

def es_bolton_hPa(Tc: np.ndarray) -> np.ndarray:
    """Pressió de vapor de saturació (hPa) amb la formulació de Bolton, donada T en ºC."""
    return 6.112 * np.exp(17.67 * Tc / (Tc + 243.5))


def recompute_vp_vpd(df: pd.DataFrame, T_col: str, RH_col: str, vp_col: str, vpd_col: str) -> pd.DataFrame:
    """
    Recomputa VP i VPD en Pa a partir de T (ºC) i RH (%).

    Notes:
    - vp = e (Pa), vpd = es - e (Pa)
    - Es talla a no-negatiu.
    - Només recalcula si existeixen les columnes d'entrada.
    """
    out = df.copy()
    if not (T_col in out.columns and RH_col in out.columns):
        return out

    T = out[T_col].to_numpy(dtype=float)
    RH = out[RH_col].to_numpy(dtype=float)
    mask = np.isfinite(T) & np.isfinite(RH)

    vp = np.full_like(T, np.nan, dtype=float)
    vpd = np.full_like(T, np.nan, dtype=float)

    if mask.any():
        es = es_bolton_hPa(T[mask])
        e = es * (RH[mask] / 100.0)
        v = es - e
        vp[mask] = np.clip(e * 100.0, 0.0, None)     # Pa
        vpd[mask] = np.clip(v * 100.0, 0.0, None)    # Pa

    if vp_col in out.columns:
        out[vp_col] = vp
    if vpd_col in out.columns:
        out[vpd_col] = vpd

    return out


# =============================================================================
# Derivades radiatives
# =============================================================================

def recompute_radiation_derivatives(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recomputa derivades radiatives per assegurar coherència física entre fluxos.

    Exemples:
    - NetRs_Avg = CM3Up_Avg - CM3Dn_Avg
    - NetRl_Avg = CG3Up_Avg - CG3Dn_Avg
    - UpTot_Avg = CM3Up_Avg + CG3Up_Avg
    - DnTot_Avg = CM3Dn_Avg + CG3Dn_Avg
    - NetTot_Avg = UpTot_Avg - DnTot_Avg

    Albedo:
    - Es calcula com dn/up en condicions diürnes (up > 20), i a la nit es posa 0.0.
    - Es clipeja a [0, 1.5] per evitar valors que no siguin coherents.
    """
    out = df.copy()

    if all(c in out.columns for c in ["CM3Up_Avg", "CM3Dn_Avg"]):
        out["NetRs_Avg"] = out["CM3Up_Avg"] - out["CM3Dn_Avg"]

        # Albedo físic: dia = dn/up, nit = 0.0 (sense NaNs)
        up = out["CM3Up_Avg"].to_numpy(dtype=float)
        dn = out["CM3Dn_Avg"].to_numpy(dtype=float)
        alb = np.zeros_like(up, dtype=float)
        m = (up > 20.0) & np.isfinite(up) & np.isfinite(dn) & (dn >= 0.0)
        alb[m] = dn[m] / up[m]
        out["Albedo_Avg"] = np.clip(alb, 0.0, 1.5)

    if all(c in out.columns for c in ["CG3Up_Avg", "CG3Dn_Avg"]):
        out["NetRl_Avg"] = out["CG3Up_Avg"] - out["CG3Dn_Avg"]

    if all(c in out.columns for c in ["CM3Up_Avg", "CG3Up_Avg"]):
        out["UpTot_Avg"] = out["CM3Up_Avg"] + out["CG3Up_Avg"]
    if all(c in out.columns for c in ["CM3Dn_Avg", "CG3Dn_Avg"]):
        out["DnTot_Avg"] = out["CM3Dn_Avg"] + out["CG3Dn_Avg"]
    if all(c in out.columns for c in ["UpTot_Avg", "DnTot_Avg"]):
        out["NetTot_Avg"] = out["UpTot_Avg"] - out["DnTot_Avg"]

    return out


# =============================================================================
# Fluxos totals: factors estimats a train (train-only)
# =============================================================================

def estimate_flux_factor(avg: pd.Series, tot: pd.Series, min_abs_avg: float = 10.0) -> Optional[float]:
    """
    Estima un factor per reconstruir totals a partir d'una mitjana.

    Si "tot" i "avg" estan relacionats aproximadament com "tot ≈ avg * fac", aleshores "fac" es pot estimar com la mediana de "tot/avg".

    Retorna:
    - fac (float) o None si no hi ha prou mostres.
    """
    m = avg.notna() & tot.notna() & (avg.abs() > min_abs_avg) & (tot > 0)
    if m.sum() < 100:
        return None
    return float(np.nanmedian((tot[m] / avg[m]).to_numpy(dtype=float)))


def recompute_flux_totals(df: pd.DataFrame, factors: Dict[str, Tuple[str, float]]) -> pd.DataFrame:
    """
    Recomputa columnes *_Tot a partir de columnes *_Avg i factors.

    Es tracten els NaNs d'avg com a 0.0 per evitar NaNs finals.
    Es clipeja a no-negatiu.
    """
    out = df.copy()
    for tot_col, (avg_col, fac) in factors.items():
        if tot_col not in out.columns or avg_col not in out.columns:
            continue
        avg = out[avg_col].to_numpy(dtype=float)
        avg = np.where(np.isfinite(avg), avg, 0.0)
        avg = np.clip(avg, 0.0, None)
        out[tot_col] = np.clip(avg * fac, 0.0, None)
    return out


# =============================================================================
# Climatologies train-only
# =============================================================================

def build_train_climatologies(train_df: pd.DataFrame, cols: List[str]) -> Dict[str, object]:
    """
    Construeix climatologies només amb TRAIN per imputar sense leakage.

    Es creen tres nivells de farciment:
    1) clim_doyhm: mediana per (dia de l'any, hora, minut) -> patró estacional + diürn.
    2) clim_monthh: mediana per (mes, hora) -> fallback més agregat.
    3) global_med: mediana global de train -> últim recurs.

    Retorna:
    - Diccionari amb tres objectes (DataFrame / Series).
    """
    idx = train_df.index
    key1 = pd.MultiIndex.from_arrays(
        [idx.dayofyear.to_numpy(), idx.hour.to_numpy(), idx.minute.to_numpy()],
        names=["doy", "hour", "minute"]
    )
    key2 = pd.MultiIndex.from_arrays(
        [idx.month.to_numpy(), idx.hour.to_numpy()],
        names=["month", "hour"]
    )
    sub = train_df[cols]
    clim1 = sub.groupby(key1).median(numeric_only=True)
    clim2 = sub.groupby(key2).median(numeric_only=True)
    gmed = sub.median(numeric_only=True)
    return {"clim_doyhm": clim1, "clim_monthh": clim2, "global_med": gmed}


def impute_features_ml_safe(
    df: pd.DataFrame,
    df_train: pd.DataFrame,
    cols: List[str],
    cfg: Config,
    thermo_cols: Set[str],
    wind_cols: Set[str],
    rad_cols: Set[str],
    dt: pd.Timedelta
) -> pd.DataFrame:
    """
    Imputa "features" (no targets) de manera ML-safe.

    Estratègia per cada columna "col":
    1) Crea un flag "{col}__miss" amb 1 si el valor original és NaN, 0 altrament.
    2) Farciment local sense futur: forward-fill limitat per "segment_id". El límit (en passos) depèn del tipus de variable (termòmetres, vent, radiació, etc.).
    3) Si encara hi ha NaNs: imputació amb climatologia train-only:
       3.1) per (doy, hora, minut)
       3.2) per (mes, hora)
       3.3) mediana global de TRAIN
       3.4) si tot falla: 0.0 (protecció final per garantir "features" completes)

    Retorna:
    - DataFrame amb totes les "cols" imputades i flags __miss.
    """
    out = df.copy()

    # Climatologies calculades només amb TRAIN
    clim = build_train_climatologies(df_train, cols)
    clim1: pd.DataFrame = clim["clim_doyhm"]
    clim2: pd.DataFrame = clim["clim_monthh"]
    gmed: pd.Series = clim["global_med"]

    idx_all = out.index
    key1_all = pd.MultiIndex.from_arrays(
        [idx_all.dayofyear.to_numpy(), idx_all.hour.to_numpy(), idx_all.minute.to_numpy()],
        names=["doy", "hour", "minute"]
    )
    key2_all = pd.MultiIndex.from_arrays(
        [idx_all.month.to_numpy(), idx_all.hour.to_numpy()],
        names=["month", "hour"]
    )

    seg = out["segment_id"]

    for col in cols:
        # Flag de missing (abans d'imputar)
        out[f"{col}__miss"] = out[col].isna().astype(np.int8)

        # Selecció del gap màxim segons el tipus de variable
        if col in thermo_cols:
            mg = cfg.max_gap_thermo_min
        elif col in wind_cols:
            mg = cfg.max_gap_wind_min
        elif col in rad_cols:
            mg = cfg.max_gap_rad_min
        else:
            mg = cfg.max_gap_default_min

        # Converteix minuts a nombre de passos de "dt" (mínim 1)
        steps = max(1, int(round(pd.Timedelta(minutes=mg) / dt)))

        # Forward-fill per segment: evita barrejar blocs separats per salts temporals
        out[col] = out.groupby(seg)[col].ffill(limit=steps)

        # Fallback 1: climatologia (doy, hora, minut)
        mask = out[col].isna()
        if mask.any():
            aligned1 = clim1[col].reindex(key1_all).to_numpy()
            vals = out[col].to_numpy()
            m = mask.to_numpy()
            repl = m & ~np.isnan(aligned1)
            vals[repl] = aligned1[repl]
            out[col] = pd.Series(vals, index=out.index, name=col)

        # Fallback 2: climatologia (mes, hora)
        mask = out[col].isna()
        if mask.any():
            aligned2 = clim2[col].reindex(key2_all).to_numpy()
            vals = out[col].to_numpy()
            m = mask.to_numpy()
            repl = m & ~np.isnan(aligned2)
            vals[repl] = aligned2[repl]
            out[col] = pd.Series(vals, index=out.index, name=col)

        # Fallback 3: mediana global de train
        mask = out[col].isna()
        if mask.any():
            fill = gmed.get(col, np.nan)
            if not np.isfinite(fill):
                fill = 0.0
            out[col] = out[col].fillna(float(fill))

    return out


# =============================================================================
# Direcció del vent: codificació circular (sin/cos)
# =============================================================================

def add_wind_dir_sincos(df: pd.DataFrame, dir_col: str) -> pd.DataFrame:
    """
    Codifica la direcció del vent en sin/cos per evitar discontinuïtats (0 i 360).

    També propaga el flag de missing de dir_col, si existeix, a:
    - wind_dir_sin__miss
    - wind_dir_cos__miss
    """
    out = df.copy()
    if dir_col not in out.columns:
        return out

    ang = np.deg2rad(out[dir_col].to_numpy(dtype=float))
    out["wind_dir_sin"] = np.sin(ang)
    out["wind_dir_cos"] = np.cos(ang)

    if f"{dir_col}__miss" in out.columns:
        out["wind_dir_sin__miss"] = out[f"{dir_col}__miss"]
        out["wind_dir_cos__miss"] = out[f"{dir_col}__miss"]

    return out


# =============================================================================
# Pipeline principal
# =============================================================================

def run_pipeline(cfg: Config) -> None:
    """
    Executa el pipeline complet d'imputació i escriu sortides a disc.

    Resum d'etapes:
    1) Lectura + regularització temporal + segmentació.
    2) Filtres físics + recomputació VP/VPD.
    3) Separació TRAIN/ALL segons train_end.
    4) Filtre robust train-only de pressió.
    5) Selecció de features per incompletesa a train.
    6) Imputació ML-safe de features + recomputacions radiatives + fluxos totals.
    7) Codificació sin/cos de direcció del vent.
    8) Verificació final: cap NaN a features + flags.
    9) Exporta a CSV + informes.
    """
    ensure_dir(cfg.output_dir)

    # 1) Lectura i regularització
    df = read_and_regularize(cfg)
    dt = pd.Timedelta(cfg.freq)

    # 2) Filtres físics i recomputacions coherents
    df = apply_physical_filters(df, cfg)
    df = recompute_vp_vpd(df, "Temp_1_Avg", "Hum_Rel_1_Avg", "VP_Tot", "VPD_Tot")

    # 3) Partició train (per evitar leakage)
    train_end = pd.Timestamp(cfg.train_end)
    train_mask = df.index < train_end
    if int(train_mask.sum()) < 10000:
        raise ValueError("Train massa petit; revisa train_end.")
    df_train = df.loc[train_mask].copy()

    # 4) Pressió: segon pas train-only (sense leakage) per eliminar outliers locals
    pressure_bounds = None
    if cfg.pressure_train_filter and "Pres_Atm_Avg" in df.columns:
        lo_p, hi_p = compute_train_pressure_bounds(df_train["Pres_Atm_Avg"], cfg)

        # Marquem com a NaN (a tot el període) els punts fora dels bounds estimats a train
        bad_all = (df["Pres_Atm_Avg"] < lo_p) | (df["Pres_Atm_Avg"] > hi_p)
        n_bad = int(bad_all.fillna(False).sum())
        df.loc[bad_all, "Pres_Atm_Avg"] = np.nan

        # I també ho reflectim al dataframe train (coherent amb l'estimació)
        bad_tr = (df_train["Pres_Atm_Avg"] < lo_p) | (df_train["Pres_Atm_Avg"] > hi_p)
        df_train.loc[bad_tr, "Pres_Atm_Avg"] = np.nan

        pressure_bounds = {
            "method": cfg.pressure_train_method,
            "lo_hPa": float(lo_p),
            "hi_hPa": float(hi_p),
            "q_low": float(cfg.pressure_q_low),
            "q_high": float(cfg.pressure_q_high),
            "iqr_k": float(cfg.pressure_iqr_k),
            "min_span_hPa": float(cfg.pressure_min_span_hPa),
            "n_outliers_set_nan": n_bad,
        }

    # 5) Selecció de features candidates (numèriques) excloent metadades i targets
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude = {"RECORD", "segment_id", cfg.temp_target, cfg.rain_target}
    candidate = [c for c in numeric_cols if c not in exclude]

    # Variables derivades que es recomputen: es mantenen encara que tinguin NaNs a train
    derived_recompute = {
        "Albedo_Avg", "NetRs_Avg", "NetRl_Avg", "UpTot_Avg", "DnTot_Avg", "NetTot_Avg",
        "VP_Tot", "VPD_Tot"
    }

    # Fracció de NaNs a train (després de filtres físics)
    nan_frac_train = df_train[candidate].isna().mean().sort_values(ascending=False)

    # Es queden:
    # - Les derivades recomputables
    # - La resta si fracció NaN a train <= max_nan_frac
    kept = [c for c in candidate if (c in derived_recompute) or (float(nan_frac_train.get(c, 0.0)) <= cfg.max_nan_frac)]
    dropped = sorted(set(candidate) - set(kept))

    # Les derivades radiatives les recomputarem després, per tant aquí són "base_features"
    base_features = [c for c in kept if c not in {"Albedo_Avg", "NetRs_Avg", "NetRl_Avg", "UpTot_Avg", "DnTot_Avg", "NetTot_Avg"}]

    # 6) Classificació per tipus (només per decidir gaps de forward-fill)
    thermo = {
        "Temp_1_Min", "Temp_1_Max", "T_datalogger", "CNR1TC_Avg", "CNR1TK_Avg",
        "Hum_Rel_1_Min", "Hum_Rel_1_Max", "Hum_Rel_1_Avg", "Pres_Atm_Avg", "VP_Tot", "VPD_Tot"
    }
    wind = {c for c in base_features if c.startswith("Vel_Viento")}
    rad = {c for c in base_features if any(k in c for k in ["Rad_", "CM3", "CG3", "UVA_Avg", "UVB_Avg"])}

    # 7) Imputació ML-safe de features
    df_imp = impute_features_ml_safe(
        df=df,
        df_train=df_train,
        cols=base_features,
        cfg=cfg,
        thermo_cols=thermo,
        wind_cols=wind,
        rad_cols=rad,
        dt=dt
    )

    # 8) Recomputa derivades radiatives (coherència)
    df_imp = recompute_radiation_derivatives(df_imp)

    # 9) Fluxos totals: estima factors a train i recomputa totals
    flux_pairs = {
        "Rad_Global_acum_Tot": "Rad_Global_Avg",
        "Rad_Difusa_acum_Tot": "Rad_Difusa_Avg",
    }
    factors: Dict[str, Tuple[str, float]] = {}
    for tot, avg in flux_pairs.items():
        if tot in df_imp.columns and avg in df_imp.columns:
            fac = estimate_flux_factor(df_train[avg], df_train[tot])
            if fac is not None:
                factors[tot] = (avg, fac)
    if factors:
        df_imp = recompute_flux_totals(df_imp, factors)

    # 10) Direcció del vent: sin/cos
    if cfg.encode_wind_dir and cfg.wind_dir_col in df_imp.columns and cfg.wind_dir_col in base_features:
        df_imp = add_wind_dir_sincos(df_imp, cfg.wind_dir_col)

    # 11) Defineix llista final de features (base + derivades recomputades)
    derived_features = [
        c for c in ["Albedo_Avg", "NetRs_Avg", "NetRl_Avg", "UpTot_Avg", "DnTot_Avg", "NetTot_Avg"]
        if c in df_imp.columns
    ]
    final_features = base_features + derived_features

    # Si hem creat sin/cos, traiem la direcció original i afegim les dues components
    if cfg.encode_wind_dir and cfg.wind_dir_col in final_features:
        final_features = [c for c in final_features if c != cfg.wind_dir_col]
        final_features += ["wind_dir_sin", "wind_dir_cos"]

    # Flags de missing (un per feature imputada)
    miss_cols = [c for c in df_imp.columns if c.endswith("__miss")]

    # Columnes base (metadades i targets, si existeixen)
    base_cols = [c for c in ["RECORD", "segment_id", cfg.temp_target, cfg.rain_target] if c in df_imp.columns]

    # 12) Construcció de la sortida: evitem duplicats mantenint l'ordre
    out_cols = list(dict.fromkeys(base_cols + final_features + miss_cols))
    out = df_imp[out_cols].copy().reset_index()

    # 13) Sanity check: cap NaN a features ni flags (els targets poden tenir NaN)
    check_cols = final_features + miss_cols + [c for c in base_cols if c not in {cfg.temp_target, cfg.rain_target}]
    if out[check_cols].isna().any().any():
        bad = out[check_cols].isna().mean().sort_values(ascending=False).head(25)
        raise RuntimeError(f"Encara hi ha NaNs a features. Pitjors:\n{bad}")

    # 14) Escriu sortides
    out_path = os.path.join(cfg.output_dir, "CR3000_Tabla1_2011_2024_imputed.csv")
    out.to_csv(out_path, index=False)

    nan_path = os.path.join(cfg.output_dir, "nan_fraction_train_after_filters.csv")
    nan_frac_train.to_csv(nan_path)

    report = {
        "input_csv": cfg.input_csv,
        "freq": cfg.freq,
        "train_end": cfg.train_end,
        "max_nan_frac": cfg.max_nan_frac,
        "encode_wind_dir": cfg.encode_wind_dir,
        "wind_dir_col": cfg.wind_dir_col,
        "drop_cols_explicit": list(cfg.drop_cols_explicit),
        "pressure_hard_bounds_hPa": [float(cfg.pressure_min_hPa), float(cfg.pressure_max_hPa)],
        "pressure_bounds_train": pressure_bounds,
        "dropped_by_missingness_train": dropped,
        "final_features": final_features,
        "missing_flag_cols": miss_cols,
        "n_rows": int(len(out)),
    }
    rep_path = os.path.join(cfg.output_dir, "imputation_report.json")
    with open(rep_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"[OK] Output: {out_path}")
    print(f"[OK] Report: {rep_path}")
    print(f"[OK] NaN rates train: {nan_path}")
    print(f"[INFO] Dropped: {dropped}")


# =============================================================================
# CLI (interfície de línia d'ordres)
# =============================================================================

def parse_args() -> Config:
    """
    Parseja els arguments de la CLI i construeix un Config.
    """
    p = argparse.ArgumentParser(description="Imputació ML-safe (sense leakage) per dades UB.")
    p.add_argument("--input", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--timestamp-col", default="TIMESTAMP")
    p.add_argument("--freq", default="10min")

    p.add_argument("--temp-target", default="Temp_1_Avg")
    p.add_argument("--rain-target", default="Lluvia_Tot")

    p.add_argument("--train-end", default="2023-01-01")
    p.add_argument("--max-nan-frac", type=float, default=0.40)

    p.add_argument("--max-gap-thermo-min", type=int, default=360)
    p.add_argument("--max-gap-wind-min", type=int, default=60)
    p.add_argument("--max-gap-rad-min", type=int, default=60)
    p.add_argument("--max-gap-default-min", type=int, default=180)

    # Pressió (llindars durs + filtre robust train-only)
    p.add_argument("--pressure-min-hpa", type=float, default=930.0)
    p.add_argument("--pressure-max-hpa", type=float, default=1060.0)
    p.add_argument("--no-pressure-train-filter", action="store_true",
                   help="Desactiva el filtre robust train-only de pressió.")
    p.add_argument("--pressure-train-method", default="quantile", choices=["quantile", "iqr"])
    p.add_argument("--pressure-q-low", type=float, default=0.001)
    p.add_argument("--pressure-q-high", type=float, default=0.999)
    p.add_argument("--pressure-iqr-k", type=float, default=8.0)
    p.add_argument("--pressure-min-span-hpa", type=float, default=30.0)

    p.add_argument("--no-winddir-encode", action="store_true",
                   help="No crear sin/cos per a la direcció del vent.")
    p.add_argument("--wind-dir-col", default="Vel_Viento_WVc(2)")

    p.add_argument("--keep-offline-rows", action="store_true")

    a = p.parse_args()

    return Config(
        input_csv=a.input,
        output_dir=a.output_dir,
        timestamp_col=a.timestamp_col,
        freq=a.freq,
        temp_target=a.temp_target,
        rain_target=a.rain_target,
        train_end=a.train_end,
        max_nan_frac=a.max_nan_frac,
        max_gap_thermo_min=a.max_gap_thermo_min,
        max_gap_wind_min=a.max_gap_wind_min,
        max_gap_rad_min=a.max_gap_rad_min,
        max_gap_default_min=a.max_gap_default_min,
        pressure_min_hPa=a.pressure_min_hpa,
        pressure_max_hPa=a.pressure_max_hpa,
        pressure_train_filter=not a.no_pressure_train_filter,
        pressure_train_method=a.pressure_train_method,
        pressure_q_low=a.pressure_q_low,
        pressure_q_high=a.pressure_q_high,
        pressure_iqr_k=a.pressure_iqr_k,
        pressure_min_span_hPa=a.pressure_min_span_hpa,
        encode_wind_dir=not a.no_winddir_encode,
        wind_dir_col=a.wind_dir_col,
        drop_offline_rows=not a.keep_offline_rows,
    )


def main() -> None:
    """Punt d'entrada del script."""
    cfg = parse_args()
    ensure_dir(cfg.output_dir)
    run_pipeline(cfg)


if __name__ == "__main__":
    main()