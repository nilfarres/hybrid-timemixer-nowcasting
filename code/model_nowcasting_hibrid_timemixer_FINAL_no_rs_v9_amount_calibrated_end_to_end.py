#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model híbrid físic-IA amb backbone Hybrid-TimeMixer per al nowcasting meteorològic
local de temperatura i pluja amb una sola estació meteorològica.

Versió v9 end-to-end amount-calibrated:
- és un codi únic d'entrenament: pot entrenar la base CC-compact i després fer fine-tuning només de pluja dins la mateixa execució;
- no requereix tenir prèviament cap carpeta v7 entrenada quan s'usa amb --v9-end-to-end;
- incorpora una baseline de temperatura amb tendència recent amortida i modulació solar;
- afegeix features causals de règim meteorològic, dèficit radiatiu, saturació, depressió del punt de rosada, Clausius-Clapeyron, tendències de pressió, vent i temperatura;
- usa una versió compacta de la termodinàmica: evita duplicar VPD_Tot i conserva només predictors físics no lineals o agregats útils;
- afegeix caps auxiliars de règim de precipitació: onset, continuïtat i final/decay a 1/2/3 h;
- afegeix caps auxiliars d'acumulació directa a 1/2/3 h;
- reforça l'aprenentatge de precipitació intensa amb pèrdues asimètriques sobre quantitat i acumulació;
- manté el deadband operatiu de precipitació (<0,1 mm -> 0) només a la sortida predictiva.\n- permet fine-tuning exclusiu de la branca de precipitació carregant pesos previs i congelant la part de temperatura; també permet entrenament v8 end-to-end en dues fases automàtiques.

Idees clau del disseny
---------------------
1) El model NO intenta substituir la física a gran escala. Amb una sola estació i
   sense radar, satèl·lit ni NWP, això no seria científicament defensable.
2) El model sí que aprofita una física/meteorologia local interpretable:
   - persistència de molt curt termini,
   - cicle diürn i climatologia condicionada al calendari,
   - persistència del dia anterior a la mateixa hora,
   - continuïtat humit/sec de la pluja mitjançant una cadena de Markov d'ordre 1.
3) La xarxa neuronal no prediu directament des de zero, sinó que aprèn el residu
   sobre aquestes línies base. Això la converteix en una arquitectura híbrida
   físic-estadística + IA.
4) S'afegeixen regularitzacions de coherència:
   - suavitat temporal de la temperatura prevista,
   - coherència d'acumulació de pluja per a horitzons agregats.

Aquest fitxer està pensat com a versió professional i autocontinguda del model
híbrid. Manté l'esperit del pipeline original del TFM, però simplifica algunes
parts per fer explícita la hibridació física i facilitar-ne la defensa científica.

Versió revisada no-RS:
- preset compact de predictors per reduir col·linearitat i millorar generalització,
- selecció robusta de model encara que alguna mètrica sigui NaN,
- NLL d'ensemble Zero-Inflated Gamma amb log-sum-exp,
- correcció del mode predict i reconstrucció exacta dels hiperparàmetres arquitectònics,
- ús real de focal/pos-weight en la pèrdua auxiliar d'ocurrència de pluja,
- verificació probabilista reforçada: CRPS de pluja per lead time, diagnòstics d'acumulació,
  calibratge i mètriques condicionades per règim meteorològic.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


# =============================================================================
# Utilitats generals
# =============================================================================


def seed_everything(seed: int = 42) -> None:
    """Fixa totes les llavors rellevants per afavorir la reproductibilitat."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def ensure_dir(path: str) -> None:
    """Crea un directori si no existeix."""
    os.makedirs(path, exist_ok=True)



def to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    """Mou tots els tensors d'un batch al dispositiu indicat."""
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}



def infer_steps(freq: str, history_hours: float, horizon_hours: float) -> Tuple[int, int, int]:
    """Converteix freqüències i horitzons en passos discrets."""
    dt_seconds = pd.Timedelta(freq).total_seconds()
    steps_per_hour = max(1, int(round(3600.0 / dt_seconds)))
    history_steps = max(4, int(round(history_hours * steps_per_hour)))
    horizon_steps = max(1, int(round(horizon_hours * steps_per_hour)))
    return steps_per_hour, history_steps, horizon_steps



def time_slot_of_day(index: pd.DatetimeIndex, freq: str) -> np.ndarray:
    """Retorna el slot intradiari discret segons la resolució temporal."""
    dt_seconds = int(round(pd.Timedelta(freq).total_seconds()))
    steps_per_day = max(1, int(round(86400 / dt_seconds)))

    seconds = (
        index.hour.to_numpy(dtype=np.int64) * 3600
        + index.minute.to_numpy(dtype=np.int64) * 60
        + index.second.to_numpy(dtype=np.int64)
    )
    slot = np.floor_divide(seconds, dt_seconds).astype(np.int64)
    return np.clip(slot, 0, steps_per_day - 1)




def _ensure_naive_utc(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """
    Converteix un índex temporal a UTC naïf per fer càlculs astronòmics de manera
    robusta, independentment de si originalment tenia zona horària o no.

    Nota:
    - Si l'índex és naïf, s'assumeix que representa el temps local del dataset.
    - Si és timezone-aware, es passa a UTC i se'n treu la timezone.
    """
    if index.tz is None:
        return index
    return index.tz_convert("UTC").tz_localize(None)


def solar_geometry_features(
    index: pd.DatetimeIndex,
    latitude_deg: float,
    longitude_deg: float,
) -> Dict[str, np.ndarray]:
    """
    Calcula una geometria solar explícita i causal per a cada timestamp.

    Les fórmules segueixen aproximacions astronòmiques estàndard de tipus NOAA /
    Spencer, suficientment precises per a la finalitat de nowcasting local:
      - declinació solar,
      - correcció de l'equació del temps,
      - hora solar local,
      - angle horari,
      - cosinus de l'angle zenital,
      - elevació solar,
      - irradiància extraterrestre projectada sobre pla horitzontal.

    Retorna arrays de longitud T amb:
      - cos_zenith
      - sin_elevation
      - hour_angle_sin / hour_angle_cos
      - declination_sin / declination_cos
      - eccentricity_correction
      - toa_sw : irradiància extraterrestre horitzontal [W m-2]
    """
    idx = _ensure_naive_utc(index)

    doy = idx.dayofyear.to_numpy(dtype=np.float64)
    minutes = (
        idx.hour.to_numpy(dtype=np.float64) * 60.0
        + idx.minute.to_numpy(dtype=np.float64)
        + idx.second.to_numpy(dtype=np.float64) / 60.0
    )

    lat = np.deg2rad(float(latitude_deg))
    lon = float(longitude_deg)

    gamma = 2.0 * np.pi * ((doy - 1.0) + (minutes / 1440.0)) / 365.0

    eccentricity = (
        1.000110
        + 0.034221 * np.cos(gamma)
        + 0.001280 * np.sin(gamma)
        + 0.000719 * np.cos(2.0 * gamma)
        + 0.000077 * np.sin(2.0 * gamma)
    )

    decl = (
        0.006918
        - 0.399912 * np.cos(gamma)
        + 0.070257 * np.sin(gamma)
        - 0.006758 * np.cos(2.0 * gamma)
        + 0.000907 * np.sin(2.0 * gamma)
        - 0.002697 * np.cos(3.0 * gamma)
        + 0.001480 * np.sin(3.0 * gamma)
    )

    eqtime = 229.18 * (
        0.000075
        + 0.001868 * np.cos(gamma)
        - 0.032077 * np.sin(gamma)
        - 0.014615 * np.cos(2.0 * gamma)
        - 0.040849 * np.sin(2.0 * gamma)
    )

    true_solar_time = (minutes + eqtime + 4.0 * lon) % 1440.0
    hour_angle_deg = true_solar_time / 4.0 - 180.0
    hour_angle = np.deg2rad(hour_angle_deg)

    cos_zenith = (
        np.sin(lat) * np.sin(decl)
        + np.cos(lat) * np.cos(decl) * np.cos(hour_angle)
    )
    cos_zenith = np.clip(cos_zenith, -1.0, 1.0)
    sin_elevation = np.clip(cos_zenith, 0.0, 1.0)

    solar_constant = 1361.0
    toa_sw = solar_constant * eccentricity * sin_elevation
    toa_sw = np.clip(toa_sw, 0.0, None)

    return {
        "cos_zenith": cos_zenith.astype(np.float32),
        "sin_elevation": sin_elevation.astype(np.float32),
        "hour_angle_sin": np.sin(hour_angle).astype(np.float32),
        "hour_angle_cos": np.cos(hour_angle).astype(np.float32),
        "declination_sin": np.sin(decl).astype(np.float32),
        "declination_cos": np.cos(decl).astype(np.float32),
        "eccentricity_correction": eccentricity.astype(np.float32),
        "toa_sw": toa_sw.astype(np.float32),
    }


def get_time_feature_columns(use_solar_geometry: bool = True) -> List[str]:
    """
    Retorna l'ordre canònic de les features temporals/futures conegudes.
    """
    cols = ["sin_hod", "cos_hod", "sin_doy", "cos_doy", "sin_dow", "cos_dow"]
    if use_solar_geometry:
        cols.extend(
            [
                "solar_cos_zenith",
                "solar_sin_elevation",
                "solar_hour_angle_sin",
                "solar_hour_angle_cos",
                "solar_declination_sin",
                "solar_declination_cos",
                "solar_eccentricity",
                "solar_toa_sw",
            ]
        )
    return cols


def make_time_features(
    index: pd.DatetimeIndex,
    latitude_deg: Optional[float] = None,
    longitude_deg: Optional[float] = None,
    use_solar_geometry: bool = True,
) -> np.ndarray:
    """
    Variables temporals deterministes conegudes també al futur.

    Sempre retorna els cicles:
      - hora del dia,
      - dia de l'any,
      - dia de la setmana.

    Opcionalment hi afegeix geometria solar explícita:
      - cosinus zenital,
      - elevació solar,
      - angle horari,
      - declinació,
      - correcció d'excentricitat orbital,
      - irradiància extraterrestre horitzontal.

    Aquestes variables són completament conegudes al futur i són especialment
    útils per a la temperatura i per a la part convectiva/diürna de la pluja.
    """
    hod = index.hour.to_numpy() + index.minute.to_numpy() / 60.0
    doy = index.dayofyear.to_numpy()
    dow = index.dayofweek.to_numpy()

    sin_hod = np.sin(2.0 * np.pi * hod / 24.0)
    cos_hod = np.cos(2.0 * np.pi * hod / 24.0)
    sin_doy = np.sin(2.0 * np.pi * doy / 365.25)
    cos_doy = np.cos(2.0 * np.pi * doy / 365.25)
    sin_dow = np.sin(2.0 * np.pi * dow / 7.0)
    cos_dow = np.cos(2.0 * np.pi * dow / 7.0)

    feats = [sin_hod, cos_hod, sin_doy, cos_doy, sin_dow, cos_dow]

    if use_solar_geometry:
        if latitude_deg is None or longitude_deg is None:
            raise ValueError(
                "Per activar la geometria solar cal indicar latitude_deg i longitude_deg."
            )
        solar = solar_geometry_features(index, latitude_deg=latitude_deg, longitude_deg=longitude_deg)
        feats.extend(
            [
                solar["cos_zenith"],
                solar["sin_elevation"],
                solar["hour_angle_sin"],
                solar["hour_angle_cos"],
                solar["declination_sin"],
                solar["declination_cos"],
                solar["eccentricity_correction"],
                solar["toa_sw"] / 1361.0,
            ]
        )

    return np.stack(feats, axis=1).astype(np.float32)



# =============================================================================
# Escalador robust
# =============================================================================


class RobustScaler:
    """
    Escalador basat en mediana i IQR.

    IMPORTANT:
    S'ha d'ajustar exclusivament amb TRAIN per evitar leakage temporal.
    """

    def __init__(self) -> None:
        self.median_: Optional[np.ndarray] = None
        self.iqr_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "RobustScaler":
        med = np.nanmedian(X, axis=0)
        q75 = np.nanpercentile(X, 75, axis=0)
        q25 = np.nanpercentile(X, 25, axis=0)
        iqr = q75 - q25
        iqr = np.where(iqr < 1e-6, 1.0, iqr)
        self.median_ = med.astype(np.float32)
        self.iqr_ = iqr.astype(np.float32)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self.median_ is not None and self.iqr_ is not None
        X_scaled = ((X - self.median_) / self.iqr_).astype(np.float32)
        return np.clip(X_scaled, -20.0, 20.0).astype(np.float32)

    def state_dict(self) -> Dict[str, list]:
        assert self.median_ is not None and self.iqr_ is not None
        return {"median": self.median_.tolist(), "iqr": self.iqr_.tolist()}

    @classmethod
    def from_state_dict(cls, d: Dict[str, list]) -> "RobustScaler":
        obj = cls()
        obj.median_ = np.array(d["median"], dtype=np.float32)
        obj.iqr_ = np.array(d["iqr"], dtype=np.float32)
        return obj


# =============================================================================
# Configuració
# =============================================================================


@dataclass
class TrainConfig:
    # Fitxers i columnes
    csv: str
    datetime_col: str
    temp_col: str
    rain_col: str
    segment_col: str = "segment_id"

    # Selecció de predictors: "compact" és el preset recomanat per al dataset sense radiosondatge.
    # Redueix col·linearitat termodinàmica/radiativa i força el model a aprendre senyals físics més robustos.
    feature_set: str = "compact"

    # Temps i splits
    timezone: str = "naive"
    freq: str = "10min"
    history_hours: float = 48.0
    horizon_hours: float = 3.0
    accumulation_horizons_hours: Tuple[float, ...] = (1.0, 2.0, 3.0)
    train_end: str = "2023-01-01"
    val_end: str = "2024-01-01"
    test_end: str = "2025-01-01"

    # Optimització
    seed: int = 42
    batch_size: int = 128
    num_workers: int = 0
    max_epochs: int = 30
    patience: int = 6
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    eps: float = 1e-6

    # Arquitectura
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 4
    dropout: float = 0.15
    patch_len_hours: float = 1.0
    patch_stride_frac: float = 0.5
    recent_window_hours: float = 6.0
    recent_patch_len_hours: float = 0.5
    recent_patch_stride_frac: float = 0.5
    timemixer_scales: Tuple[int, ...] = (1, 2, 4, 8)
    timemixer_ma_kernel: int = 5
    residual_temp_max_sigma: float = 3.0
    residual_occ_logit_max: float = 4.0
    residual_log_wet_mean_max: float = 3.0

    # Pèrdues
    rain_loss_weight: float = 1.0
    n_ensembles: int = 1
    ensemble_seed_stride: int = 1
    rain_pos_weight: float = 0.0
    rain_intensity_alpha: float = 0.0
    rain_occurrence_bce_weight: float = 0.0
    rain_occ_pos_weight: float = 0.0
    rain_occ_focal_gamma: float = 0.0
    physics_temp_smooth_weight: float = 0.01
    physics_temp_anchor_weight: float = 0.02
    physics_rain_accum_weight: float = 0.20
    use_solar_geometry: bool = True
    station_latitude_deg: Optional[float] = None
    station_longitude_deg: Optional[float] = None
    solar_temp_baseline_weight: float = 0.25

    # Avaluació probabilista
    val_temp_crps_weight: float = 0.20
    val_rain_mae_weight: float = 0.55
    val_rain_brier_weight: float = 3.50
    val_rain_nll_weight: float = 0.25
    val_rain_sedi_weight: float = 0.20
    # Pes moderat de la CRPS de pluja en la selecció del checkpoint.
    # En nowcasting local no ha de dominar Brier/SEDI/MAE, però sí informar la qualitat probabilista.
    val_rain_crps_weight: float = 0.10

    # Verificació probabilista i calibratge
    compute_crps: bool = True
    reliability_bins: int = 10
    crps_rain_n_samples: int = 30
    # Mostreig CRPS de pluja: el valor històric crps_rain_max_points es conserva
    # per compatibilitat i s'interpreta com a mínim per lead time en l'avaluació final.
    crps_rain_max_points: int = 2000
    crps_rain_max_points_per_lead: int = 2000
    crps_rain_min_samples: int = 100

    # Calibratge post-hoc de probabilitat de pluja. Es calibra només amb VALIDACIÓ
    # i s'aplica a TEST, sense tocar els pesos del model ni fer leakage temporal.
    calibrate_rain_probability: bool = True
    rain_calibration_bins: int = 20
    rain_calibration_smoothing: float = 25.0
    rain_calibration_max_points: int = 600000
    extreme_temp_q: float = 0.99
    extreme_rain_q: float = 0.99
    rain_event_threshold: float = 0.1

    # Postprocessament operatiu de precipitació.
    # Els valors previstos per sota del llindar instrumental/operatiu es mantenen
    # com a diagnòstic cru, però es posen a zero a la sortida operativa.
    rain_amount_deadband: float = 0.1

    # V9: calibratge condicional de quantitat de pluja.
    # S'ajusta només amb validació i només s'aplica en règim humit probable o amb pluja recent.
    calibrate_rain_amount: bool = True
    rain_amount_calibration_max_points: int = 600000
    rain_amount_calibration_smoothing: float = 120.0
    rain_amount_calibration_min_cell_count: int = 80
    rain_amount_calibration_min_points: int = 200
    rain_amount_calibration_min_factor: float = 1.0
    rain_amount_calibration_max_factor: float = 3.0
    rain_amount_apply_p_cal_threshold: float = 0.20
    rain_amount_apply_p_raw_threshold: float = 0.35
    rain_amount_apply_mu_threshold: float = 0.03

    # V7: features causals i aprenentatge per règims.
    # Aquestes variables es calculen només amb informació present/passada del dataset.
    use_v7_regime_features: bool = True
    radiation_column: str = "Rad_Global_Avg"
    humidity_column: str = "Hum_Rel_1_Avg"
    pressure_column: str = "Pres_Atm_Avg"
    wind_speed_column: str = "Vel_Viento_WVc(1)"
    wind_dir_column: str = "Dir_Viento_Avg"

    # V7: caps auxiliars de règim i acumulació directa. Són pèrdues auxiliars:
    # milloren la representació, però la sortida principal continua sent ZI-Gamma.
    rain_regime_aux_weight: float = 0.06
    rain_onset_pos_weight: float = 4.0
    rain_continue_pos_weight: float = 1.5
    rain_decay_pos_weight: float = 2.0
    rain_accum_direct_weight: float = 0.08
    rain_accum_direct_under_weight: float = 1.35

    # V7: ponderació suau per finestres difícils.
    # Pre-onset i decay són els dos règims on una estació local té més dificultat.
    rain_preonset_sample_weight: float = 0.20
    rain_decay_sample_weight: float = 0.15

    # V7: refredament tèrmic indirecte durant règims humits.
    # No és un postprocessament: és una pèrdua suau que ajuda a evitar prediccions
    # massa càlides quan el model ja preveu pluja significativa.
    physics_temp_rain_cooling_weight: float = 0.003

    # Baseline tèrmica v6 conservadora: tendència recent causal, amortida i feble, modulada per geometria solar.
    use_temp_trend_baseline: bool = True
    temp_trend_baseline_weight: float = 0.15  # v6: prior feble; evita rigiditzar massa la temperatura
    temp_trend_short_hours: float = 1.0
    temp_trend_mid_hours: float = 3.0
    temp_trend_long_hours: float = 6.0
    temp_trend_tau_hours: float = 1.75
    temp_trend_max_abs_slope_C_per_hour: float = 2.0
    temp_trend_max_delta_C: float = 1.25
    physics_temp_trend_weight: float = 0.005
    physics_temp_diurnal_trend_weight: float = 0.0

    # Postprocessament de temperatura: per defecte desactivat; la coherència s'aprèn entrenant.
    apply_temperature_trend_postprocess: bool = False
    temp_trend_window_hours: float = 3.0
    temp_trend_max_blend_weight: float = 0.0

    # Precipitació intensa v6 conservadora: penalitza la subestimació només de manera suau i més selectiva.
    rain_heavy_quantile: float = 0.97  # v6: més selectiu que v5; només episodis realment intensos
    rain_heavy_step_weight: float = 0.08
    rain_heavy_accum_weight: float = 0.06
    rain_heavy_under_weight: float = 1.50
    rain_heavy_intensity_power: float = 0.35

    rain_cap_quantile: float = 0.999

    # GPU
    use_tf32: bool = True
    compile: bool = False
    compile_mode: str = "max-autotune"

    # V8: fine-tuning selectiu de precipitació.
    # Si init_model_dir apunta a una execució prèvia compatible, cada membre
    # carrega el checkpoint amb el mateix nom i es pot congelar tot excepte la
    # branca de pluja. Això conserva la predicció de temperatura de la v7 CC.
    init_model_dir: Optional[str] = None
    rain_finetune_only: bool = False

    # V9 end-to-end: dues fases dins del mateix script.
    # Fase 1: entrenament complet CC-compact.
    # Fase 2: fine-tuning només de la branca de pluja inicialitzant des de la fase 1.
    v9_end_to_end: bool = False
    v9_stage1_outdir: Optional[str] = None
    v9_stage1_max_epochs: int = 30
    v9_stage1_patience: int = 6
    v9_stage1_lr: Optional[float] = None
    v9_stage2_max_epochs: int = 12
    v9_stage2_patience: int = 4
    v9_stage2_lr: Optional[float] = None
    v9_skip_stage1_if_complete: bool = False



# =============================================================================
# Features causals v7 de règim, radiació efectiva i persistència humida
# =============================================================================


def _series_or_none(df: pd.DataFrame, col: str) -> Optional[pd.Series]:
    """Retorna una sèrie numèrica si la columna existeix; altrament None."""
    if col and col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return None


def _rolling_mean_causal(s: pd.Series, n: int) -> pd.Series:
    """Mitjana mòbil causal, incloent l'instant present i només passat."""
    n = max(1, int(n))
    return s.rolling(window=n, min_periods=1).mean()


def _rolling_sum_causal(s: pd.Series, n: int) -> pd.Series:
    """Suma mòbil causal, incloent l'instant present i només passat."""
    n = max(1, int(n))
    return s.rolling(window=n, min_periods=1).sum()


def saturation_vapor_pressure_magnus_hpa(temp_c: pd.Series) -> pd.Series:
    """
    Pressió de vapor de saturació sobre aigua [hPa] amb la fórmula de Magnus-Bolton.

    e_s(T) = 6.112 * exp(17.67 T / (T + 243.5))

    És una aproximació estàndard i estable per a temperatures meteorològiques
    superficials. Es limita T a un rang ampli però físicament raonable per evitar
    valors absurds en cas d'outliers instrumentals.
    """
    t = pd.to_numeric(temp_c, errors="coerce").astype(float).clip(lower=-80.0, upper=60.0)
    return 6.112 * np.exp((17.67 * t) / (t + 243.5))


def dewpoint_from_temperature_rh_magnus_c(temp_c: pd.Series, rh_percent: pd.Series) -> pd.Series:
    """
    Punt de rosada [°C] a partir de temperatura i humitat relativa.

    S'utilitza la inversa de Magnus-Bolton. El resultat és causal perquè només
    depèn de T i HR observades en l'instant present.
    """
    t = pd.to_numeric(temp_c, errors="coerce").astype(float).clip(lower=-80.0, upper=60.0)
    rh = pd.to_numeric(rh_percent, errors="coerce").astype(float).clip(lower=1e-3, upper=100.0)
    gamma = np.log(rh / 100.0) + (17.67 * t) / (t + 243.5)
    td = (243.5 * gamma) / (17.67 - gamma)
    return pd.Series(td, index=temp_c.index).clip(lower=-100.0, upper=60.0)


def clausius_clapeyron_slope_hpa_per_k(temp_c: pd.Series, esat_hpa: pd.Series) -> pd.Series:
    """
    Derivada local d e_s / dT [hPa K-1] a partir de Clausius-Clapeyron.

    Aproximació física:
        d ln(e_s) / dT = L_v / (R_v T_K^2)
        d e_s / dT = e_s * L_v / (R_v T_K^2)

    Aquesta variable informa el model de com de ràpid canvia la capacitat de
    contenir vapor d'aigua amb la temperatura. Al voltant de 20 °C equival
    aproximadament a un 6-7% per kelvin.
    """
    t_k = pd.to_numeric(temp_c, errors="coerce").astype(float).clip(lower=-80.0, upper=60.0) + 273.15
    es = pd.to_numeric(esat_hpa, errors="coerce").astype(float).clip(lower=0.0)
    latent_heat_vaporization = 2.5e6  # J kg-1
    gas_constant_water_vapor = 461.5  # J kg-1 K-1
    dlnes_dtk = latent_heat_vaporization / (gas_constant_water_vapor * np.square(t_k))
    return pd.Series(es * dlnes_dtk, index=temp_c.index)


def add_v7_regime_features(df: pd.DataFrame, cfg: TrainConfig) -> pd.DataFrame:
    """
    Afegeix predictors causals específics de la v7.

    Objectiu físic:
      - representar l'estat radiatiu real respecte al potencial solar;
      - capturar saturació, baixa VPD, cua humida i tendències locals;
      - donar senyal al model per distingir pre-onset, continuïtat i decay.

    Totes les variables es calculen amb informació present/passada. No hi ha cap
    acumulació futura ni cap target futur, per tant no introdueix leakage temporal.
    """
    if not bool(getattr(cfg, "use_v7_regime_features", True)):
        return df

    out = df.copy()
    steps_per_hour, _, _ = infer_steps(cfg.freq, cfg.history_hours, cfg.horizon_hours)
    n30 = max(1, int(round(0.5 * steps_per_hour)))
    n60 = max(1, int(round(1.0 * steps_per_hour)))
    n180 = max(1, int(round(3.0 * steps_per_hour)))

    # Radiació efectiva: compara radiació global observada amb el potencial solar TOA.
    rad = _series_or_none(out, getattr(cfg, "radiation_column", "Rad_Global_Avg"))
    if rad is None:
        rad = pd.Series(0.0, index=out.index)
    toa_norm = _series_or_none(out, "solar_toa_sw")
    if toa_norm is None:
        toa_wm2 = pd.Series(0.0, index=out.index)
    else:
        toa_wm2 = 1361.0 * toa_norm.clip(lower=0.0)

    day_mask = toa_wm2 > 25.0
    clear_sky_ratio = pd.Series(0.0, index=out.index, dtype="float64")
    clear_sky_ratio.loc[day_mask] = (rad.loc[day_mask] / toa_wm2.loc[day_mask].clip(lower=25.0)).astype(float)
    clear_sky_ratio = clear_sky_ratio.clip(lower=0.0, upper=1.5)
    radiation_deficit = (1.0 - clear_sky_ratio).clip(lower=0.0, upper=1.5)

    out["v7_clear_sky_ratio"] = clear_sky_ratio.astype(np.float32)
    out["v7_radiation_deficit"] = radiation_deficit.astype(np.float32)
    out["v7_radiation_deficit_roll_30min"] = _rolling_mean_causal(radiation_deficit, n30).astype(np.float32)
    out["v7_radiation_deficit_roll_60min"] = _rolling_mean_causal(radiation_deficit, n60).astype(np.float32)
    out["v7_radiation_deficit_roll_180min"] = _rolling_mean_causal(radiation_deficit, n180).astype(np.float32)
    out["v7_radiation_deficit_diff_60min"] = (radiation_deficit - radiation_deficit.shift(n60)).fillna(0.0).astype(np.float32)

    # Humitat, punt de rosada i termodinàmica de Clausius-Clapeyron.
    rh = _series_or_none(out, getattr(cfg, "humidity_column", "Hum_Rel_1_Avg"))
    if rh is None:
        rh = pd.Series(0.0, index=out.index)
    rh = rh.clip(lower=0.0, upper=100.0)
    sat = (rh >= 98.0).astype(float)
    out["v7_rh_saturation_fraction_30min"] = _rolling_mean_causal(sat, n30).astype(np.float32)
    out["v7_rh_saturation_fraction_60min"] = _rolling_mean_causal(sat, n60).astype(np.float32)
    out["v7_rh_saturation_fraction_180min"] = _rolling_mean_causal(sat, n180).astype(np.float32)

    temp_for_thermo = pd.to_numeric(out[cfg.temp_col], errors="coerce")
    esat_hpa = saturation_vapor_pressure_magnus_hpa(temp_for_thermo)
    eact_hpa = (rh / 100.0) * esat_hpa
    vpd_cc_hpa = (esat_hpa - eact_hpa).clip(lower=0.0)
    dewpoint_c = dewpoint_from_temperature_rh_magnus_c(temp_for_thermo, rh)
    dewpoint_depression_c = (temp_for_thermo - dewpoint_c).clip(lower=0.0, upper=80.0)

    cc_slope_hpa_per_k = clausius_clapeyron_slope_hpa_per_k(temp_for_thermo, esat_hpa)
    with np.errstate(divide="ignore", invalid="ignore"):
        cc_frac_per_k = (cc_slope_hpa_per_k / esat_hpa.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
    cc_frac_per_k = cc_frac_per_k.fillna(0.0).clip(lower=0.0, upper=0.15)

    # Si la temperatura baixa, la pressió de saturació també baixa: amb aire ja humit,
    # això aproxima un potencial local de condensació. No és una prova que plourà,
    # però és un senyal físic causal compatible amb Clausius-Clapeyron.
    esat_drop_1h_hpa = (esat_hpa.shift(n60) - esat_hpa).fillna(0.0).clip(lower=0.0)
    esat_drop_3h_hpa = (esat_hpa.shift(n180) - esat_hpa).fillna(0.0).clip(lower=0.0)
    rh_norm = (rh / 100.0).clip(0.0, 1.0)
    cc_condensation_potential_1h = (esat_drop_1h_hpa * rh_norm).clip(lower=0.0, upper=20.0)
    cc_condensation_potential_3h = (esat_drop_3h_hpa * rh_norm).clip(lower=0.0, upper=30.0)

    # Conjunt termodinàmic compacte: evitem afegir variables que són simples
    # duplicats de T i HR, com el punt de rosada brut, e_s i e_a. Conservem només
    # variables amb lectura física directa per precipitació local: proximitat a
    # saturació, VPD coherent en hPa, sensibilitat Clausius-Clapeyron i potencial
    # de condensació per refredament recent.
    out["v7_dewpoint_depression_C"] = dewpoint_depression_c.astype(np.float32)
    out["v7_dewpoint_depression_roll_30min"] = _rolling_mean_causal(dewpoint_depression_c, n30).astype(np.float32)
    out["v7_dewpoint_depression_roll_60min"] = _rolling_mean_causal(dewpoint_depression_c, n60).astype(np.float32)
    out["v7_dewpoint_depression_roll_180min"] = _rolling_mean_causal(dewpoint_depression_c, n180).astype(np.float32)
    out["v7_dewpoint_depression_drop_60min"] = (dewpoint_depression_c.shift(n60) - dewpoint_depression_c).fillna(0.0).astype(np.float32)
    out["v7_dewpoint_rise_60min"] = (dewpoint_c - dewpoint_c.shift(n60)).fillna(0.0).astype(np.float32)

    out["v7_vpd_cc_hpa"] = vpd_cc_hpa.astype(np.float32)
    out["v7_vpd_cc_roll_30min"] = _rolling_mean_causal(vpd_cc_hpa, n30).astype(np.float32)
    out["v7_vpd_cc_roll_60min"] = _rolling_mean_causal(vpd_cc_hpa, n60).astype(np.float32)
    out["v7_vpd_cc_roll_180min"] = _rolling_mean_causal(vpd_cc_hpa, n180).astype(np.float32)

    out["v7_cc_esat_sensitivity_pct_per_K"] = (100.0 * cc_frac_per_k).astype(np.float32)
    out["v7_cc_esat_drop_1h_hpa"] = esat_drop_1h_hpa.astype(np.float32)
    out["v7_cc_esat_drop_3h_hpa"] = esat_drop_3h_hpa.astype(np.float32)
    out["v7_cc_condensation_potential_1h"] = cc_condensation_potential_1h.astype(np.float32)
    out["v7_cc_condensation_potential_3h"] = cc_condensation_potential_3h.astype(np.float32)

    # VPD baix només amb la formulació termodinàmica coherent. No barregem aquí
    # el VPD_Tot original del dataset per evitar dues famílies redundants amb
    # unitats potencialment diferents.
    low_vpd = (vpd_cc_hpa <= 1.0).astype(float)
    out["v7_vpd_low_fraction_30min"] = _rolling_mean_causal(low_vpd, n30).astype(np.float32)
    out["v7_vpd_low_fraction_60min"] = _rolling_mean_causal(low_vpd, n60).astype(np.float32)
    out["v7_vpd_low_fraction_180min"] = _rolling_mean_causal(low_vpd, n180).astype(np.float32)

    # Persistència i cua humida observada.
    rain = pd.to_numeric(out[cfg.rain_col], errors="coerce").fillna(0.0).clip(lower=0.0)
    wet = (rain >= float(cfg.rain_event_threshold)).astype(float)
    out["v7_wet_fraction_30min"] = _rolling_mean_causal(wet, n30).astype(np.float32)
    out["v7_wet_fraction_60min"] = _rolling_mean_causal(wet, n60).astype(np.float32)
    out["v7_wet_fraction_180min"] = _rolling_mean_causal(wet, n180).astype(np.float32)
    out["v7_rain_sum_30min"] = _rolling_sum_causal(rain, n30).astype(np.float32)
    out["v7_rain_sum_60min"] = _rolling_sum_causal(rain, n60).astype(np.float32)
    out["v7_rain_sum_180min"] = _rolling_sum_causal(rain, n180).astype(np.float32)
    out["v7_dry_after_wet_60min"] = (((_rolling_sum_causal(wet, n60) > 0.0) & (wet < 0.5)).astype(float)).astype(np.float32)

    # Tendències locals.
    temp = pd.to_numeric(out[cfg.temp_col], errors="coerce")
    out["v7_temp_drop_30min"] = (temp.shift(n30) - temp).fillna(0.0).astype(np.float32)
    out["v7_temp_drop_60min"] = (temp.shift(n60) - temp).fillna(0.0).astype(np.float32)

    pres = _series_or_none(out, getattr(cfg, "pressure_column", "Pres_Atm_Avg"))
    if pres is None:
        pres = pd.Series(0.0, index=out.index)
    out["v7_pressure_tendency_1h"] = (pres - pres.shift(n60)).fillna(0.0).astype(np.float32)
    out["v7_pressure_tendency_3h"] = (pres - pres.shift(n180)).fillna(0.0).astype(np.float32)

    wspd = _series_or_none(out, getattr(cfg, "wind_speed_column", "Vel_Viento_Avg"))
    if wspd is None:
        wspd = pd.Series(0.0, index=out.index)
    out["v7_wind_speed_diff_1h"] = (wspd - wspd.shift(n60)).fillna(0.0).astype(np.float32)

    wdir = _series_or_none(out, getattr(cfg, "wind_dir_column", "Dir_Viento_Avg"))
    if wdir is not None:
        d = ((wdir - wdir.shift(n60) + 180.0) % 360.0 - 180.0).abs() / 180.0
        out["v7_wind_dir_shift_1h"] = d.fillna(0.0).clip(lower=0.0, upper=1.0).astype(np.float32)
    elif ("wind_dir_sin" in out.columns) and ("wind_dir_cos" in out.columns):
        # Fallback per al dataset compact del TFM: la direcció del vent ja està
        # codificada com sin/cos. Cal calcular el canvi angular sense reconstruir
        # graus explícits ni introduir discontinuïtats a 0/360°.
        sin_now = pd.to_numeric(out["wind_dir_sin"], errors="coerce").astype(float)
        cos_now = pd.to_numeric(out["wind_dir_cos"], errors="coerce").astype(float)
        sin_prev = sin_now.shift(n60)
        cos_prev = cos_now.shift(n60)
        dot = (sin_now * sin_prev + cos_now * cos_prev).clip(lower=-1.0, upper=1.0)
        angular_shift = np.arccos(dot) / np.pi
        out["v7_wind_dir_shift_1h"] = (
            pd.Series(angular_shift, index=out.index)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .clip(lower=0.0, upper=1.0)
            .astype(np.float32)
        )
    else:
        out["v7_wind_dir_shift_1h"] = np.float32(0.0)

    # Índex local de risc pre-onset: no decideix el futur, només resumeix símptomes locals.
    # Pesos suaus, tots causals i físicament interpretables. La part termodinàmica
    # inclou proximitat al punt de rosada i potencial de condensació per refredament
    # segons Clausius-Clapeyron.
    press_fall = (-out["v7_pressure_tendency_3h"].astype(float) / 4.0).clip(0.0, 1.0)
    temp_drop = (out["v7_temp_drop_60min"].astype(float) / 2.0).clip(0.0, 1.0)
    dewpoint_close = (1.0 - out["v7_dewpoint_depression_roll_60min"].astype(float) / 4.0).clip(0.0, 1.0)
    cc_condensation = (out["v7_cc_condensation_potential_1h"].astype(float) / 2.0).clip(0.0, 1.0)
    recent_dry = (1.0 - out["v7_wet_fraction_60min"].astype(float)).clip(0.0, 1.0)
    out["v7_local_preonset_risk"] = (
        0.22 * radiation_deficit
        + 0.20 * rh_norm
        + 0.18 * out["v7_vpd_low_fraction_60min"].astype(float)
        + 0.15 * dewpoint_close
        + 0.12 * press_fall
        + 0.08 * temp_drop
        + 0.05 * cc_condensation
    ).clip(0.0, 1.5).astype(np.float32)
    out["v7_local_preonset_risk_dryweighted"] = (
        out["v7_local_preonset_risk"].astype(float) * recent_dry
    ).clip(0.0, 1.5).astype(np.float32)

    # Neteja final per evitar NaN/inf en predictors.
    v7_cols = [c for c in out.columns if str(c).startswith("v7_")]
    out[v7_cols] = out[v7_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    return out


# =============================================================================
# Càrrega del dataset i splits temporals
# =============================================================================



def load_dataset(cfg: TrainConfig) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    Llegeix el CSV final del pipeline i valida l'estructura mínima necessària.
    """
    if cfg.use_solar_geometry and (cfg.station_latitude_deg is None or cfg.station_longitude_deg is None):
        raise ValueError(
            "Amb --use-solar-geometry cal indicar --station-latitude-deg i --station-longitude-deg."
        )
    df = pd.read_csv(cfg.csv, low_memory=False)

    required_cols = [cfg.datetime_col, cfg.temp_col, cfg.rain_col, cfg.segment_col]
    missing_required = [c for c in required_cols if c not in df.columns]
    if missing_required:
        raise ValueError(
            "Falten columnes obligatòries al CSV: "
            f"{missing_required}. Revisa la sortida de dataset_nowcasting.py."
        )

    df[cfg.datetime_col] = pd.to_datetime(df[cfg.datetime_col], errors="coerce")
    df = df.dropna(subset=[cfg.datetime_col]).sort_values(cfg.datetime_col)
    df = df.drop_duplicates(subset=[cfg.datetime_col], keep="last")
    df = df.set_index(cfg.datetime_col)

    # Conversió numèrica robusta
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Coherència física mínima
    df[cfg.rain_col] = df[cfg.rain_col].clip(lower=0.0)

    # Afegim variables temporals deterministes explícites al DataFrame.
    tf_cols = get_time_feature_columns(cfg.use_solar_geometry)
    tf_df = pd.DataFrame(
        make_time_features(
            df.index,
            latitude_deg=cfg.station_latitude_deg,
            longitude_deg=cfg.station_longitude_deg,
            use_solar_geometry=cfg.use_solar_geometry,
        ),
        index=df.index,
        columns=tf_cols,
    )
    df = pd.concat([df, tf_df], axis=1).copy()
    df = add_v7_regime_features(df, cfg)

    if df[[cfg.temp_col, cfg.rain_col]].isna().any().any():
        bad_cols = df[[cfg.temp_col, cfg.rain_col]].isna().mean().sort_values(ascending=False)
        raise ValueError(f"Encara hi ha NaN als targets. Pitjors columnes:\n{bad_cols}")

    dt_expected = pd.Timedelta(cfg.freq)
    diffs = df.index.to_series().diff().dropna()
    bad = int(np.sum(np.abs((diffs - dt_expected).dt.total_seconds()) > 1.0))

    meta = {
        "rows": int(len(df)),
        "time_start": str(df.index.min()),
        "time_end": str(df.index.max()),
        "dt_expected": str(dt_expected),
        "n_bad_dt_steps": bad,
    }
    return df, meta



def split_by_time(df: pd.DataFrame, cfg: TrainConfig) -> Dict[str, pd.DataFrame]:
    """Crea train/val/test amb talls temporals estrictes."""
    t_train_end = pd.Timestamp(cfg.train_end)
    t_val_end = pd.Timestamp(cfg.val_end)
    t_test_end = pd.Timestamp(cfg.test_end)

    if not (t_train_end < t_val_end < t_test_end):
        raise ValueError(
            "Els talls temporals han de complir train_end < val_end < test_end. "
            f"Rebut: train_end={cfg.train_end}, val_end={cfg.val_end}, test_end={cfg.test_end}."
        )

    train = df.loc[df.index < t_train_end].copy()
    val = df.loc[(df.index >= t_train_end) & (df.index < t_val_end)].copy()
    test = df.loc[(df.index >= t_val_end) & (df.index < t_test_end)].copy()

    sizes = {"train": len(train), "val": len(val), "test": len(test)}
    empty = [k for k, v in sizes.items() if v == 0]
    if empty:
        raise ValueError(f"Almenys un split ha quedat buit. Mides: {sizes}.")

    return {"train": train, "val": val, "test": test}



def build_feature_columns(df: pd.DataFrame, cfg: TrainConfig) -> Tuple[List[str], List[str]]:
    """
    Selecciona predictors meteorològics i columnes temporals.

    El mode ``compact`` és el recomanat per al dataset sense radiosondatge:
    conserva els senyals físics locals principals i elimina famílies molt
    col·linears o derivades que poden fer sobreajust, especialment en radiació,
    vapor d'aigua i VPD. En aquesta versió es prefereix el VPD termodinàmic
    coherent calculat amb Magnus/Clausius-Clapeyron i s'elimina VPD_Tot del
    preset compacte. Els targets es conserven sempre com a predictors causals
    del passat.
    """
    time_cols = get_time_feature_columns(cfg.use_solar_geometry)
    banned = {
        cfg.segment_col,
        "segment_pos",
        "segment_len",
        "valid_t_end",
        f"{cfg.temp_col}__miss",
        f"{cfg.rain_col}__miss",
    }

    cols: List[str] = []
    for c in df.columns:
        if c in time_cols or c in banned:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)

    # Garanteix autoregressió explícita dels dos targets al passat.
    if cfg.temp_col not in cols:
        cols.append(cfg.temp_col)
    if cfg.rain_col not in cols:
        cols.append(cfg.rain_col)

    cols = list(dict.fromkeys(cols))

    # Auxiliars que serien leakage/duplicació de target perquè el model ja
    # construeix internament ocurrència i log-pluja a partir del passat causal.
    banned_exact = {"rain_occ", "rain_log1p"}

    feature_set = str(getattr(cfg, "feature_set", "compact")).lower().strip()
    if feature_set not in {"compact", "full"}:
        raise ValueError("feature_set ha de ser 'compact' o 'full'.")

    if feature_set == "compact":
        # Reducció de redundància per al dataset no-RS. Manté: T, P, HR, pressió,
        # VPD, vent mitjà/vectorial, direcció sin/cos, radiació global i derivades
        # temporals d'aquestes variables. Elimina variables quasi equivalents o
        # fortament derivades que poden degradar la generalització.
        banned_exact.update(
            {
                "qc_n_violations",
                "qc_any",
                "VP_Tot",
                "VP_Tot__miss",
                "VPD_Tot",
                "VPD_Tot__miss",
                "v7_dew_point_C",
                "v7_esat_magnus_hpa",
                "v7_eact_magnus_hpa",
                "v7_cc_slope_hpa_per_K",
                "Vel_Viento_Max",
                "Vel_Viento_Max__miss",
                "Vel_Viento_WVc(3)",
                "Vel_Viento_WVc(3)__miss",
                "Rad_Difusa_Avg",
                "Rad_Difusa_Avg__miss",
                "NetRs_Avg",
                "NetRs_Avg__miss",
                "NetRl_Avg",
                "NetRl_Avg__miss",
                "Albedo_Avg",
                "Albedo_Avg__miss",
                "UVA_Avg",
                "UVA_Avg__miss",
                "UVB_Avg",
                "UVB_Avg__miss",
                "rad_diffuse_frac",
            }
        )
        banned_prefixes = (
            "VP_Tot_",
            "VPD_Tot_",
            "Vel_Viento_Max_",
            "Vel_Viento_WVc(3)_",
            "Rad_Difusa_Avg_",
            "NetRs_Avg_",
            "NetRl_Avg_",
            "Albedo_Avg_",
            "UVA_Avg_",
            "UVB_Avg_",
        )
    else:
        banned_prefixes = tuple()

    cols = [
        c for c in cols
        if c not in banned_exact and not any(str(c).startswith(pref) for pref in banned_prefixes)
    ]

    # Assegurem que els targets causals no s'han eliminat accidentalment.
    for target_col in (cfg.temp_col, cfg.rain_col):
        if target_col not in cols:
            cols.append(target_col)

    return list(dict.fromkeys(cols)), time_cols



def valid_t_end_indices_per_segment(
    df: pd.DataFrame,
    cfg: TrainConfig,
    history_steps: int,
    horizon_steps: int,
) -> np.ndarray:
    """
    Retorna els índexs vàlids de t_end sense travessar segments ni sortir del split.
    """
    seg_ids = df[cfg.segment_col].to_numpy(dtype=np.int64)
    n = len(df)
    idxs = np.arange(n, dtype=np.int64)

    changes = np.where(seg_ids[1:] != seg_ids[:-1])[0] + 1
    starts = np.concatenate(([0], changes))
    ends = np.concatenate((changes, [n]))

    valid_parts: List[np.ndarray] = []
    for s, e in zip(starts, ends):
        seg_len = e - s
        if seg_len < (history_steps + horizon_steps):
            continue
        t_min = s + (history_steps - 1)
        t_max = e - horizon_steps - 1
        if t_max >= t_min:
            valid_parts.append(idxs[t_min:t_max + 1])

    if not valid_parts:
        return np.array([], dtype=np.int64)

    valid_idx = np.concatenate(valid_parts)
    if "valid_t_end" in df.columns:
        valid_flag = df["valid_t_end"].to_numpy(dtype=np.float32) > 0.5
        valid_idx = valid_idx[valid_flag[valid_idx]]

    return valid_idx.astype(np.int64)


# =============================================================================
# Climatologies i baselines train-only
# =============================================================================



def climatology_lookup(
    month_idx: np.ndarray,
    slot_idx: np.ndarray,
    month_slot_values: np.ndarray,
    slot_values: np.ndarray,
    global_value: float,
    clip_min: Optional[float] = None,
    clip_max: Optional[float] = None,
) -> np.ndarray:
    """
    Lookup robust amb fallback: mes-slot -> slot -> global.
    """
    if month_slot_values.size == 0 or slot_values.size == 0:
        out = np.full(month_idx.shape, float(global_value), dtype=np.float64)
    else:
        out = month_slot_values[month_idx - 1, slot_idx]
        out = np.where(np.isfinite(out), out, slot_values[slot_idx])
        out = np.where(np.isfinite(out), out, float(global_value))

    if clip_min is not None:
        out = np.maximum(out, float(clip_min))
    if clip_max is not None:
        out = np.minimum(out, float(clip_max))
    return out.astype(np.float64)



def build_baseline_climatology(
    train_df: pd.DataFrame,
    cfg: TrainConfig,
    steps_per_hour: int,
) -> Dict[str, object]:
    """
    Construeix totes les climatologies i transicions train-only necessàries.

    Inclou:
      - temperatura per mes-slot i slot,
      - ocurrència i quantitat mitjana de pluja,
      - intensitat condicionada a estar humit,
      - transicions Markov sec->humit i humit->humit,
      - acumulacions futures condicionades al fet que ara plou.
    """
    steps_per_day = int(24 * steps_per_hour)
    months = train_df.index.month.to_numpy(dtype=np.int64)
    slots = time_slot_of_day(train_df.index, cfg.freq).astype(np.int64)

    temp = train_df[cfg.temp_col].to_numpy(dtype=np.float32)
    rain = np.clip(train_df[cfg.rain_col].to_numpy(dtype=np.float32), 0.0, None)
    rain_occ = (rain >= float(cfg.rain_event_threshold)).astype(np.float32)
    rain_wet_only = np.where(rain_occ > 0.5, rain, np.nan).astype(np.float64)

    seg = train_df[cfg.segment_col].to_numpy(dtype=np.int64)

    temp_global = float(np.nanmedian(temp))
    rain_occ_global = float(np.nanmean(rain_occ))
    rain_amt_global = float(np.nanmean(rain))
    rain_amt_wet_global = float(np.nanmean(rain_wet_only)) if np.any(np.isfinite(rain_wet_only)) else 0.0

    temp_slot = np.full(steps_per_day, np.nan, dtype=np.float64)
    rain_occ_slot = np.full(steps_per_day, np.nan, dtype=np.float64)
    rain_amt_slot = np.full(steps_per_day, np.nan, dtype=np.float64)
    rain_amt_wet_slot = np.full(steps_per_day, np.nan, dtype=np.float64)

    temp_month_slot = np.full((12, steps_per_day), np.nan, dtype=np.float64)
    rain_occ_month_slot = np.full((12, steps_per_day), np.nan, dtype=np.float64)
    rain_amt_month_slot = np.full((12, steps_per_day), np.nan, dtype=np.float64)
    rain_amt_wet_month_slot = np.full((12, steps_per_day), np.nan, dtype=np.float64)

    for s in range(steps_per_day):
        ms = slots == s
        if np.any(ms):
            temp_slot[s] = float(np.nanmedian(temp[ms]))
            rain_occ_slot[s] = float(np.nanmean(rain_occ[ms]))
            rain_amt_slot[s] = float(np.nanmean(rain[ms]))
            if np.any(rain_occ[ms] > 0.5):
                rain_amt_wet_slot[s] = float(np.nanmean(rain_wet_only[ms]))

    for month in range(1, 13):
        mm = months == month
        if not np.any(mm):
            continue
        for s in range(steps_per_day):
            mms = mm & (slots == s)
            if np.any(mms):
                temp_month_slot[month - 1, s] = float(np.nanmedian(temp[mms]))
                rain_occ_month_slot[month - 1, s] = float(np.nanmean(rain_occ[mms]))
                rain_amt_month_slot[month - 1, s] = float(np.nanmean(rain[mms]))
                if np.any(rain_occ[mms] > 0.5):
                    rain_amt_wet_month_slot[month - 1, s] = float(np.nanmean(rain_wet_only[mms]))

    # Markov(1) per a ocurrència de pluja.
    p01_slot = np.full(steps_per_day, np.nan, dtype=np.float64)
    p11_slot = np.full(steps_per_day, np.nan, dtype=np.float64)
    p01_month_slot = np.full((12, steps_per_day), np.nan, dtype=np.float64)
    p11_month_slot = np.full((12, steps_per_day), np.nan, dtype=np.float64)

    if len(train_df) >= 2:
        same_seg_next = seg[:-1] == seg[1:]
        prev_wet = rain_occ[:-1].astype(np.int64)
        next_wet = rain_occ[1:].astype(np.float64)
        next_month = months[1:]
        next_slot = slots[1:]

        dry_all = same_seg_next & (prev_wet == 0)
        wet_all = same_seg_next & (prev_wet == 1)
        p01_global = float(np.nanmean(next_wet[dry_all])) if np.any(dry_all) else float(rain_occ_global)
        p11_global = float(np.nanmean(next_wet[wet_all])) if np.any(wet_all) else float(rain_occ_global)

        for s in range(steps_per_day):
            ms = same_seg_next & (next_slot == s)
            dry = ms & (prev_wet == 0)
            wetm = ms & (prev_wet == 1)
            if np.any(dry):
                p01_slot[s] = float(np.nanmean(next_wet[dry]))
            if np.any(wetm):
                p11_slot[s] = float(np.nanmean(next_wet[wetm]))

        for month in range(1, 13):
            mm = same_seg_next & (next_month == month)
            if not np.any(mm):
                continue
            for s in range(steps_per_day):
                mms = mm & (next_slot == s)
                dry = mms & (prev_wet == 0)
                wetm = mms & (prev_wet == 1)
                if np.any(dry):
                    p01_month_slot[month - 1, s] = float(np.nanmean(next_wet[dry]))
                if np.any(wetm):
                    p11_month_slot[month - 1, s] = float(np.nanmean(next_wet[wetm]))
    else:
        p01_global = float(rain_occ_global)
        p11_global = float(rain_occ_global)

    # Acumulacions futures de pluja quan l'episodi ja està iniciat.
    event_h_hours = [
        float(h) for h in cfg.accumulation_horizons_hours
        if float(h) > 0.0 and float(h) <= float(cfg.horizon_hours) + 1e-9
    ]
    event_h_steps = [max(1, int(round(h * steps_per_hour))) for h in event_h_hours]
    if len(event_h_steps) == 0:
        event_h_hours = [float(cfg.horizon_hours)]
        event_h_steps = [max(1, int(round(float(cfg.horizon_hours) * steps_per_hour)))]

    event_accum_global = np.full(len(event_h_steps), np.nan, dtype=np.float64)
    event_accum_month = np.full((12, len(event_h_steps)), np.nan, dtype=np.float64)

    rain_fill = np.where(np.isfinite(rain), rain, 0.0).astype(np.float64)
    rain_fin = np.isfinite(rain).astype(np.int64)
    prefix_rain = np.concatenate([[0.0], np.cumsum(rain_fill, dtype=np.float64)])
    prefix_fin = np.concatenate([[0], np.cumsum(rain_fin, dtype=np.int64)])

    for hi, h_steps in enumerate(event_h_steps):
        n = len(rain) - h_steps
        if n <= 0:
            continue
        future_sum = prefix_rain[1 + h_steps:n + 1 + h_steps] - prefix_rain[1:n + 1]
        future_fin = prefix_fin[1 + h_steps:n + 1 + h_steps] - prefix_fin[1:n + 1]
        valid = future_fin == h_steps
        wet_now = rain_occ[:n] > 0.5
        same_seg_h = seg[:n] == seg[h_steps:n + h_steps]
        month_now = months[:n]

        all_mask = valid & wet_now & same_seg_h
        if np.any(all_mask):
            event_accum_global[hi] = float(np.nanmean(future_sum[all_mask]))

        for month in range(1, 13):
            mm = all_mask & (month_now == month)
            if np.any(mm):
                event_accum_month[month - 1, hi] = float(np.nanmean(future_sum[mm]))

    return {
        "steps_per_day": int(steps_per_day),
        "temp_global": temp_global,
        "temp_slot": temp_slot.tolist(),
        "temp_month_slot": temp_month_slot.tolist(),
        "rain_occ_global": rain_occ_global,
        "rain_occ_slot": rain_occ_slot.tolist(),
        "rain_occ_month_slot": rain_occ_month_slot.tolist(),
        "rain_amt_global": rain_amt_global,
        "rain_amt_slot": rain_amt_slot.tolist(),
        "rain_amt_month_slot": rain_amt_month_slot.tolist(),
        "rain_amt_wet_global": rain_amt_wet_global,
        "rain_amt_wet_slot": rain_amt_wet_slot.tolist(),
        "rain_amt_wet_month_slot": rain_amt_wet_month_slot.tolist(),
        "markov_p01_global": float(np.clip(p01_global, 0.0, 1.0)),
        "markov_p01_slot": np.clip(p01_slot, 0.0, 1.0).tolist(),
        "markov_p01_month_slot": np.clip(p01_month_slot, 0.0, 1.0).tolist(),
        "markov_p11_global": float(np.clip(p11_global, 0.0, 1.0)),
        "markov_p11_slot": np.clip(p11_slot, 0.0, 1.0).tolist(),
        "markov_p11_month_slot": np.clip(p11_month_slot, 0.0, 1.0).tolist(),
        "event_accum_horizons_hours": [float(h) for h in event_h_hours],
        "event_accum_horizons_steps": [int(h) for h in event_h_steps],
        "event_accum_global": np.where(np.isfinite(event_accum_global), event_accum_global, np.nan).tolist(),
        "event_accum_month": np.where(np.isfinite(event_accum_month), event_accum_month, np.nan).tolist(),
    }



# =============================================================================
# Tendència tèrmica causal per a la baseline v6
# =============================================================================


def robust_recent_temperature_slope(
    temp_history: np.ndarray,
    steps_per_hour: float,
    short_hours: float = 1.0,
    mid_hours: float = 3.0,
    long_hours: float = 6.0,
    max_abs_slope: float = 2.0,
) -> float:
    """
    Estima una pendent recent de temperatura [°C h-1] només amb informació passada.

    S'usen tres escales temporals per evitar reaccionar massa a soroll de 10 minuts:
    1 h capta canvis ràpids locals, 3 h dona inèrcia mesoescalar i 6 h estabilitza
    el senyal radiatiu/diürn. La mitjana ponderada de pendents és robusta davant
    petites oscil·lacions i manté causalitat estricta.
    """
    arr = np.asarray(temp_history, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size < 3 or steps_per_hour <= 0:
        return 0.0

    last = float(arr[-1])
    candidates: List[float] = []
    weights: List[float] = []
    for h, w in ((short_hours, 0.25), (mid_hours, 0.45), (long_hours, 0.30)):
        n = int(round(float(h) * float(steps_per_hour)))
        if n < 1 or arr.size <= n:
            continue
        prev = float(arr[-1 - n])
        if not np.isfinite(prev):
            continue
        slope = (last - prev) / max(float(h), 1e-6)
        candidates.append(float(slope))
        weights.append(float(w))

    if not candidates:
        return 0.0

    c = np.asarray(candidates, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    slope = float(np.sum(c * w) / max(np.sum(w), 1e-6))
    return float(np.clip(slope, -abs(max_abs_slope), abs(max_abs_slope)))


def diurnal_temperature_coherence(
    slope_c_per_h: float,
    current_toa: float,
    future_toa: np.ndarray,
    min_weight: float = 0.45,
) -> np.ndarray:
    """
    Dona més pes a la tendència recent quan és coherent amb el forçament solar.

    - Durant la nit o amb irradiància decreixent, un refredament recent és coherent.
    - Amb irradiància creixent al matí/migdia, un escalfament recent és coherent.
    - Si el signe recent contradiu el cicle solar, no s'anul·la: pot ser un front,
      núvols, precipitació o advecció; només se'n redueix el pes.
    """
    fut = np.asarray(future_toa, dtype=np.float64)
    if fut.size == 0 or abs(float(slope_c_per_h)) < 1e-6:
        return np.ones_like(fut, dtype=np.float64)

    ref = np.concatenate([[float(current_toa)], fut[:-1]])
    d_toa = fut - ref
    expected = np.where((fut < 30.0) | (d_toa < -5.0), -1.0, np.where(d_toa > 5.0, 1.0, 0.0))
    slope_sign = 1.0 if slope_c_per_h > 0.0 else -1.0
    coherent = (expected == 0.0) | (expected == slope_sign)
    return np.where(coherent, 1.0, float(min_weight)).astype(np.float64)


def damped_temperature_trend_prior(
    last_temp: float,
    slope_c_per_h: float,
    future_toa: np.ndarray,
    current_toa: float,
    steps_per_hour: float,
    trend_weight: float = 0.35,
    tau_hours: float = 1.75,
    max_delta_c: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Retorna una trajectòria de persistència+tendència amortida i el component de tendència.

    La tendència no s'extrapola linealment indefinidament: s'integra amb saturació
    exponencial perquè un descens/ascens recent no domini tot l'horitzó de 3 h.
    """
    O = len(future_toa)
    lead_hours = (np.arange(O, dtype=np.float64) + 1.0) / max(float(steps_per_hour), 1e-6)
    coherence = diurnal_temperature_coherence(slope_c_per_h, current_toa=current_toa, future_toa=future_toa)
    trend_delta = float(slope_c_per_h) * float(tau_hours) * (1.0 - np.exp(-lead_hours / max(float(tau_hours), 1e-6)))
    trend_delta = np.clip(trend_delta, -abs(float(max_delta_c)), abs(float(max_delta_c)))
    trend_delta = float(trend_weight) * coherence * trend_delta
    prior = float(last_temp) + trend_delta
    return prior.astype(np.float32), trend_delta.astype(np.float32)

# =============================================================================
# Dataset de finestres amb baselines híbrids
# =============================================================================


class WindowDataset(torch.utils.data.Dataset):
    """
    Dataset causal history -> horizon.

    A més dels tensors habituals, retorna línies base híbrides per al futur:
      - temp_baseline_raw: baseline físic-estadístic de temperatura,
      - rain_occ_baseline: probabilitat baseline de pluja,
      - rain_mu_baseline: pluja esperada baseline.
    """

    def __init__(
        self,
        X_meteo: np.ndarray,
        X_time: np.ndarray,
        y_temp_scaled: np.ndarray,
        y_temp_raw: np.ndarray,
        y_rain: np.ndarray,
        month_all: np.ndarray,
        slot_all: np.ndarray,
        segment_all: np.ndarray,
        steps_per_day: int,
        history_steps: int,
        horizon_steps: int,
        valid_t_idx: np.ndarray,
        baseline_ctx: Dict[str, object],
        future_cosz_all: np.ndarray,
        future_toa_all: np.ndarray,
        solar_temp_baseline_weight: float,
        use_temp_trend_baseline: bool,
        temp_trend_baseline_weight: float,
        temp_trend_short_hours: float,
        temp_trend_mid_hours: float,
        temp_trend_long_hours: float,
        temp_trend_tau_hours: float,
        temp_trend_max_abs_slope_C_per_hour: float,
        temp_trend_max_delta_C: float,
        temp_mean: float,
        temp_std: float,
        mask_temp_all: Optional[np.ndarray] = None,
        mask_rain_all: Optional[np.ndarray] = None,
        rain_event_threshold: float = 0.1,
    ) -> None:
        self.X = np.concatenate([X_meteo, X_time], axis=1).astype(np.float32)
        self.time = X_time.astype(np.float32)

        self.y_temp = y_temp_scaled.astype(np.float32)
        self.y_temp_raw = y_temp_raw.astype(np.float32)
        self.y_rain = y_rain.astype(np.float32)

        self.month_all = month_all.astype(np.int64)
        self.slot_all = slot_all.astype(np.int64)
        self.segment_all = segment_all.astype(np.int64)
        self.steps_per_day = int(steps_per_day)

        self.H = int(history_steps)
        self.O = int(horizon_steps)
        self.valid = valid_t_idx.astype(np.int64)
        self.future_cosz_all = future_cosz_all.astype(np.float32)
        self.future_toa_all = future_toa_all.astype(np.float32)
        self.solar_temp_baseline_weight = float(solar_temp_baseline_weight)
        self.use_temp_trend_baseline = bool(use_temp_trend_baseline)
        self.temp_trend_baseline_weight = float(temp_trend_baseline_weight)
        self.temp_trend_short_hours = float(temp_trend_short_hours)
        self.temp_trend_mid_hours = float(temp_trend_mid_hours)
        self.temp_trend_long_hours = float(temp_trend_long_hours)
        self.temp_trend_tau_hours = float(temp_trend_tau_hours)
        self.temp_trend_max_abs_slope_C_per_hour = float(temp_trend_max_abs_slope_C_per_hour)
        self.temp_trend_max_delta_C = float(temp_trend_max_delta_C)
        self.steps_per_hour = float(self.steps_per_day) / 24.0
        self.temp_mean = float(temp_mean)
        self.temp_std = float(temp_std)

        self.mask_temp_all = mask_temp_all.astype(np.float32) if mask_temp_all is not None else None
        self.mask_rain_all = mask_rain_all.astype(np.float32) if mask_rain_all is not None else None
        self.rain_event_threshold = float(rain_event_threshold)

        # Baselines train-only en format NumPy.
        self.bc = {
            k: (np.asarray(v, dtype=np.float64) if isinstance(v, list) else v)
            for k, v in baseline_ctx.items()
        }

    def __len__(self) -> int:
        return int(len(self.valid))

    def _temp_baseline(
        self,
        last_temp: float,
        future_month: np.ndarray,
        future_slot: np.ndarray,
        prevday_temp_future: np.ndarray,
        future_cosz: np.ndarray,
        future_toa: np.ndarray,
        temp_history: Optional[np.ndarray] = None,
        current_toa: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Baseline físic-estadístic de temperatura amb tendència recent v6.

        Components:
          - persistència de molt curt termini,
          - tendència recent amortida i causal,
          - persistència del dia anterior a la mateixa hora,
          - climatologia mes-slot train-only,
          - modulació solar de la memòria diürna.

        Retorna:
          - temp_baseline_raw: baseline final usada pel model,
          - temp_trend_prior_raw: trajectòria persistència+tendència usada només com a prior de signe.
        """
        temp_clim = climatology_lookup(
            future_month,
            future_slot,
            self.bc["temp_month_slot"],
            self.bc["temp_slot"],
            float(self.bc["temp_global"]),
        )

        temp_prevday = np.where(np.isfinite(prevday_temp_future), prevday_temp_future, temp_clim)
        lead_hours = (np.arange(self.O, dtype=np.float64) + 1.0) / max(self.steps_per_hour, 1e-6)

        slope = 0.0
        if self.use_temp_trend_baseline and temp_history is not None:
            slope = robust_recent_temperature_slope(
                temp_history,
                steps_per_hour=self.steps_per_hour,
                short_hours=self.temp_trend_short_hours,
                mid_hours=self.temp_trend_mid_hours,
                long_hours=self.temp_trend_long_hours,
                max_abs_slope=self.temp_trend_max_abs_slope_C_per_hour,
            )

        temp_trend_prior, _trend_delta = damped_temperature_trend_prior(
            last_temp=float(last_temp),
            slope_c_per_h=slope,
            future_toa=future_toa,
            current_toa=float(current_toa),
            steps_per_hour=self.steps_per_hour,
            trend_weight=self.temp_trend_baseline_weight if self.use_temp_trend_baseline else 0.0,
            tau_hours=self.temp_trend_tau_hours,
            max_delta_c=self.temp_trend_max_delta_C,
        )
        temp_trend_prior = temp_trend_prior.astype(np.float64)

        # Persistència exponencialment decreixent. Durant el dia, la geometria
        # solar fa guanyar pes a la memòria diürna del dia anterior.
        solar_gate = np.clip(np.asarray(future_toa, dtype=np.float64) / 1000.0, 0.0, 1.0)
        w_persist = np.exp(-lead_hours / 1.5) * (1.0 - 0.15 * solar_gate)
        remainder = np.clip(1.0 - w_persist, 0.0, 1.0)
        solar_w = float(np.clip(self.solar_temp_baseline_weight, 0.0, 1.0))
        base_prevday = 0.45 + solar_w * solar_gate + 0.05 * np.clip(future_cosz, 0.0, 1.0)
        base_prevday = np.clip(base_prevday, 0.20, 0.85)
        w_prevday = remainder * base_prevday
        w_clim = remainder - w_prevday

        # El component de tendència substitueix només la branca de persistència.
        # Així evitem extrapolar massa lluny i preservem el cicle diürn/climatològic.
        baseline = w_persist * temp_trend_prior + w_prevday * temp_prevday + w_clim * temp_clim
        return baseline.astype(np.float32), temp_trend_prior.astype(np.float32)

    def _rain_markov_baseline(
        self,
        last_rain: float,
        future_month: np.ndarray,
        future_slot: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Baseline de pluja amb cadena de Markov(1) per a ocurrència i climatologia
        humida per a la quantitat esperada.
        """
        wet_state = float(last_rain >= self.rain_event_threshold)
        p_prev = wet_state

        occ = np.zeros(self.O, dtype=np.float64)
        rain_mu = np.zeros(self.O, dtype=np.float64)

        for k in range(self.O):
            mk = future_month[k:k + 1]
            sk = future_slot[k:k + 1]
            p01 = climatology_lookup(
                mk,
                sk,
                self.bc["markov_p01_month_slot"],
                self.bc["markov_p01_slot"],
                float(self.bc["markov_p01_global"]),
                clip_min=0.0,
                clip_max=1.0,
            )[0]
            p11 = climatology_lookup(
                mk,
                sk,
                self.bc["markov_p11_month_slot"],
                self.bc["markov_p11_slot"],
                float(self.bc["markov_p11_global"]),
                clip_min=0.0,
                clip_max=1.0,
            )[0]
            p_prev = p_prev * p11 + (1.0 - p_prev) * p01
            occ[k] = p_prev

            wet_amount = climatology_lookup(
                mk,
                sk,
                self.bc["rain_amt_wet_month_slot"],
                self.bc["rain_amt_wet_slot"],
                float(self.bc["rain_amt_wet_global"]),
                clip_min=0.0,
            )[0]
            rain_mu[k] = p_prev * wet_amount

        return occ.astype(np.float32), rain_mu.astype(np.float32)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        t_end = int(self.valid[i])

        x_past = self.X[t_end - self.H + 1:t_end + 1, :]
        t_future = self.time[t_end + 1:t_end + 1 + self.O, :]

        y_temp = self.y_temp[t_end + 1:t_end + 1 + self.O]
        y_temp_raw = self.y_temp_raw[t_end + 1:t_end + 1 + self.O]
        y_rain = self.y_rain[t_end + 1:t_end + 1 + self.O]

        if self.mask_temp_all is None:
            mask_temp = np.ones_like(y_temp, dtype=np.float32)
        else:
            mask_temp = self.mask_temp_all[t_end + 1:t_end + 1 + self.O].astype(np.float32)

        if self.mask_rain_all is None:
            mask_rain = np.ones_like(y_rain, dtype=np.float32)
        else:
            mask_rain = self.mask_rain_all[t_end + 1:t_end + 1 + self.O].astype(np.float32)

        y_occ = ((y_rain >= self.rain_event_threshold).astype(np.float32) * mask_rain).astype(np.float32)

        future_month = self.month_all[t_end + 1:t_end + 1 + self.O].astype(np.int64)
        future_slot = self.slot_all[t_end + 1:t_end + 1 + self.O].astype(np.int64)
        future_cosz = self.future_cosz_all[t_end + 1:t_end + 1 + self.O].astype(np.float32)
        future_toa = self.future_toa_all[t_end + 1:t_end + 1 + self.O].astype(np.float32)
        current_month = np.int64(self.month_all[t_end])

        prevday_temp_future = np.full(self.O, np.nan, dtype=np.float32)
        prevday_idx = np.arange(t_end + 1, t_end + 1 + self.O, dtype=np.int64) - self.steps_per_day
        if self.steps_per_day > 0 and prevday_idx[0] >= 0:
            same_seg_future = np.all(self.segment_all[prevday_idx] == self.segment_all[t_end])
            if same_seg_future:
                prevday_temp_future = self.y_temp_raw[prevday_idx].astype(np.float32)

        last_temp_raw = float(self.y_temp_raw[t_end])
        last_rain_raw = float(self.y_rain[t_end])

        temp_history_raw = self.y_temp_raw[t_end - self.H + 1:t_end + 1].astype(np.float32)
        current_toa = float(self.future_toa_all[t_end]) if 0 <= t_end < len(self.future_toa_all) else 0.0

        temp_baseline_raw, temp_trend_prior_raw = self._temp_baseline(
            last_temp=last_temp_raw,
            future_month=future_month,
            future_slot=future_slot,
            prevday_temp_future=prevday_temp_future,
            future_cosz=future_cosz,
            future_toa=future_toa,
            temp_history=temp_history_raw,
            current_toa=current_toa,
        )
        rain_occ_baseline, rain_mu_baseline = self._rain_markov_baseline(
            last_rain=last_rain_raw,
            future_month=future_month,
            future_slot=future_slot,
        )

        return {
            "x_past": torch.from_numpy(x_past),
            "t_future": torch.from_numpy(t_future),
            "y_temp": torch.from_numpy(y_temp),
            "y_temp_raw": torch.from_numpy(y_temp_raw.astype(np.float32)),
            "y_rain": torch.from_numpy(y_rain),
            "y_rain_occ": torch.from_numpy(y_occ),
            "mask_temp": torch.from_numpy(mask_temp),
            "mask_rain": torch.from_numpy(mask_rain),
            "temp_baseline_raw": torch.from_numpy(temp_baseline_raw),
            "temp_trend_prior_raw": torch.from_numpy(temp_trend_prior_raw),
            "rain_occ_baseline": torch.from_numpy(rain_occ_baseline),
            "rain_mu_baseline": torch.from_numpy(rain_mu_baseline),
            "last_temp_raw": torch.tensor(last_temp_raw, dtype=torch.float32),
            "last_rain_raw": torch.tensor(last_rain_raw, dtype=torch.float32),
            "future_month": torch.from_numpy(future_month),
            "future_slot": torch.from_numpy(future_slot),
            "current_month": torch.tensor(int(current_month), dtype=torch.int64),
            "future_cosz": torch.from_numpy(future_cosz),
            "future_toa": torch.from_numpy(future_toa),
        }



def build_datasets(
    splits: Dict[str, pd.DataFrame],
    cfg: TrainConfig,
) -> Tuple[Dict[str, WindowDataset], Dict[str, object]]:
    """
    Construeix els datasets de train/val/test, ajusta escaladors amb TRAIN i
    genera els artefactes necessaris.
    """
    steps_per_h, H, O = infer_steps(cfg.freq, cfg.history_hours, cfg.horizon_hours)
    feat_cols, time_cols = build_feature_columns(splits["train"], cfg)

    X_train = splits["train"][feat_cols].to_numpy(dtype=np.float32)
    meteo_scaler = RobustScaler().fit(X_train)

    y_temp_train = splits["train"][cfg.temp_col].to_numpy(dtype=np.float32)
    temp_mean = float(np.mean(y_temp_train))
    temp_std = float(np.std(y_temp_train) + 1e-6)

    y_rain_train = np.clip(splits["train"][cfg.rain_col].to_numpy(dtype=np.float32), 0.0, None)
    rain_pos = y_rain_train[y_rain_train >= float(cfg.rain_event_threshold)]
    if rain_pos.size >= 100:
        rain_cap_99p9 = float(np.quantile(rain_pos, float(cfg.rain_cap_quantile)))
    elif rain_pos.size > 0:
        rain_cap_99p9 = float(np.max(rain_pos))
    else:
        rain_cap_99p9 = float(cfg.rain_event_threshold)

    qT = float(cfg.extreme_temp_q)
    temp_p_lo = float(np.quantile(y_temp_train, 1.0 - qT))
    temp_p_hi = float(np.quantile(y_temp_train, qT))

    if rain_pos.size >= 100:
        rain_pos_p = float(np.quantile(rain_pos, float(cfg.extreme_rain_q)))
        rain_heavy_thr = float(np.quantile(rain_pos, float(cfg.rain_heavy_quantile)))
    elif rain_pos.size > 0:
        rain_pos_p = float(np.max(rain_pos))
        rain_heavy_thr = float(np.max(rain_pos))
    else:
        rain_pos_p = float(cfg.rain_event_threshold)
        rain_heavy_thr = float(cfg.rain_event_threshold)
    rain_heavy_thr = max(float(cfg.rain_event_threshold), float(rain_heavy_thr))

    mt_col = f"{cfg.temp_col}__miss"
    mr_col = f"{cfg.rain_col}__miss"

    baseline_ctx = build_baseline_climatology(splits["train"], cfg, steps_per_h)

    artifacts = {
        "feature_cols": feat_cols,
        "time_cols": time_cols,
        "segment_col": cfg.segment_col,
        "temp_feature_index": int(feat_cols.index(cfg.temp_col)),
        "rain_feature_index": int(feat_cols.index(cfg.rain_col)),
        "history_steps": int(H),
        "horizon_steps": int(O),
        "steps_per_hour": int(steps_per_h),
        "meteo_scaler": meteo_scaler.state_dict(),
        "temp_mean": temp_mean,
        "temp_std": temp_std,
        "rain_cap_99p9": rain_cap_99p9,
        "temp_extreme_lo": temp_p_lo,
        "temp_extreme_hi": temp_p_hi,
        "rain_pos_extreme_thr": rain_pos_p,
        "rain_heavy_threshold": rain_heavy_thr,
        "temperature_baseline_version": "trend_v7_regime_aware" if cfg.use_temp_trend_baseline else "legacy_v3",
        "v7_thermodynamic_feature_version": "cc_dewpoint_compact_no_VPD_Tot",
        "rain_event_threshold": float(cfg.rain_event_threshold),
        "freq": cfg.freq,
        "baseline_climatology": baseline_ctx,
        "use_solar_geometry": bool(cfg.use_solar_geometry),
        "station_latitude_deg": cfg.station_latitude_deg,
        "station_longitude_deg": cfg.station_longitude_deg,
    }

    datasets: Dict[str, WindowDataset] = {}
    for name, dfi in splits.items():
        X_meteo = meteo_scaler.transform(dfi[feat_cols].to_numpy(dtype=np.float32))
        X_time = dfi[time_cols].to_numpy(dtype=np.float32)

        y_temp_raw = dfi[cfg.temp_col].to_numpy(dtype=np.float32)
        y_temp_scaled = ((y_temp_raw - temp_mean) / temp_std).astype(np.float32)
        y_rain = np.clip(dfi[cfg.rain_col].to_numpy(dtype=np.float32), 0.0, None).astype(np.float32)

        if mt_col in dfi.columns:
            mask_temp_all = 1.0 - dfi[mt_col].to_numpy(dtype=np.float32)
        else:
            mask_temp_all = np.ones(len(dfi), dtype=np.float32)

        if mr_col in dfi.columns:
            mask_rain_all = 1.0 - dfi[mr_col].to_numpy(dtype=np.float32)
        else:
            mask_rain_all = np.ones(len(dfi), dtype=np.float32)

        valid_idx = valid_t_end_indices_per_segment(dfi, cfg, history_steps=H, horizon_steps=O)
        if len(valid_idx) == 0:
            raise ValueError(f"El split '{name}' no té cap finestra vàlida (H={H}, O={O}).")

        month_all = dfi.index.month.to_numpy(dtype=np.int64)
        slot_all = time_slot_of_day(dfi.index, cfg.freq).astype(np.int64)
        segment_all = dfi[cfg.segment_col].to_numpy(dtype=np.int64)
        if cfg.use_solar_geometry and ("solar_cos_zenith" in dfi.columns):
            future_cosz_all = dfi["solar_cos_zenith"].to_numpy(dtype=np.float32)
            future_toa_all = (1361.0 * np.clip(dfi["solar_toa_sw"].to_numpy(dtype=np.float32), 0.0, None)).astype(np.float32)
        else:
            future_cosz_all = np.zeros(len(dfi), dtype=np.float32)
            future_toa_all = np.zeros(len(dfi), dtype=np.float32)

        datasets[name] = WindowDataset(
            X_meteo=X_meteo,
            X_time=X_time,
            y_temp_scaled=y_temp_scaled,
            y_temp_raw=y_temp_raw,
            y_rain=y_rain,
            month_all=month_all,
            slot_all=slot_all,
            segment_all=segment_all,
            steps_per_day=int(24 * steps_per_h),
            history_steps=H,
            horizon_steps=O,
            valid_t_idx=valid_idx,
            baseline_ctx=baseline_ctx,
            future_cosz_all=future_cosz_all,
            future_toa_all=future_toa_all,
            solar_temp_baseline_weight=cfg.solar_temp_baseline_weight,
            use_temp_trend_baseline=cfg.use_temp_trend_baseline,
            temp_trend_baseline_weight=cfg.temp_trend_baseline_weight,
            temp_trend_short_hours=cfg.temp_trend_short_hours,
            temp_trend_mid_hours=cfg.temp_trend_mid_hours,
            temp_trend_long_hours=cfg.temp_trend_long_hours,
            temp_trend_tau_hours=cfg.temp_trend_tau_hours,
            temp_trend_max_abs_slope_C_per_hour=cfg.temp_trend_max_abs_slope_C_per_hour,
            temp_trend_max_delta_C=cfg.temp_trend_max_delta_C,
            temp_mean=temp_mean,
            temp_std=temp_std,
            mask_temp_all=mask_temp_all,
            mask_rain_all=mask_rain_all,
            rain_event_threshold=cfg.rain_event_threshold,
        )

    return datasets, artifacts



def make_loaders(
    datasets: Dict[str, WindowDataset],
    cfg: TrainConfig,
    use_cuda: bool,
) -> Dict[str, torch.utils.data.DataLoader]:
    """Crea DataLoaders per a train, val i test."""
    loaders: Dict[str, torch.utils.data.DataLoader] = {}

    common_kwargs = dict(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=use_cuda,
    )
    if cfg.num_workers > 0:
        common_kwargs.update({"persistent_workers": True, "prefetch_factor": 4})

    for name, ds in datasets.items():
        loaders[name] = torch.utils.data.DataLoader(
            ds,
            shuffle=(name == "train"),
            drop_last=(name == "train"),
            **common_kwargs,
        )
    return loaders



def configure_gpu_perf(cfg: TrainConfig) -> None:
    """Configura opcions de rendiment quan hi ha GPU disponible."""
    if not torch.cuda.is_available():
        return
    if cfg.use_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


# =============================================================================
# Model híbrid residual amb backbone Hybrid-TimeMixer
# =============================================================================


class MovingAverage1D(nn.Module):
    """Mitjana mòbil senzilla per a la descomposició trend-seasonal."""

    def __init__(self, kernel_size: int = 5) -> None:
        super().__init__()
        self.kernel_size = max(1, int(kernel_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, C]
        if self.kernel_size <= 1 or x.shape[1] <= 2:
            return x
        k = min(self.kernel_size, int(x.shape[1]))
        if k % 2 == 0:
            k = max(1, k - 1)
        if k <= 1:
            return x
        pad = k // 2
        x_t = x.transpose(1, 2)
        x_pad = F.pad(x_t, (pad, pad), mode="replicate")
        trend = F.avg_pool1d(x_pad, kernel_size=k, stride=1)
        return trend.transpose(1, 2)


def _consecutive_tail_fraction(binary_seq: torch.Tensor) -> torch.Tensor:
    """Fracció de passos humits consecutius des de l'instant actual cap enrere."""
    rev = torch.flip(binary_seq, dims=[1])
    tail = torch.cumprod(rev, dim=1)
    return tail.sum(dim=1) / max(1, binary_seq.shape[1])


class ScaleMixerBlock(nn.Module):
    """Bloc de mixing sobre canal i escala, inspirat en la filosofia de TimeMixer."""

    def __init__(self, n_scales: int, d_model: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.channel_mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.scale_mlp = nn.Sequential(
            nn.Linear(n_scales, 2 * n_scales),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * n_scales, n_scales),
            nn.Dropout(dropout),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: [B, S, D]
        z = z + self.channel_mlp(self.norm1(z))
        zt = self.norm2(z).transpose(1, 2)
        z = z + self.scale_mlp(zt).transpose(1, 2)
        return z


class HybridResidualNowcaster(nn.Module):
    """
    Model híbrid residual amb backbone tipus TimeMixer adaptat a una sola estació.

    Idees del disseny:
      - Extracció multiescala del passat mitjançant downsampling temporal.
      - Descomposició simple en component estacional i tendència per a cada escala.
      - Mixing separat estacional/trend en l'eix d'escala.
      - Future mixing condicionat per exògenes futures conegudes i per les baselines.

    No és una rèplica literal del TimeMixer original, sinó una adaptació híbrida,
    causal i orientada a nowcasting local probabilístic.
    """

    def __init__(
        self,
        in_channels: int,
        time_feat_dim: int,
        history_steps: int,
        horizon_steps: int,
        steps_per_hour: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        dropout: float = 0.15,
        patch_len_hours: float = 1.0,
        patch_stride_frac: float = 0.5,
        recent_window_hours: float = 6.0,
        recent_patch_len_hours: float = 0.5,
        recent_patch_stride_frac: float = 0.5,
        timemixer_scales: Tuple[int, ...] = (1, 2, 4, 8),
        timemixer_ma_kernel: int = 5,
        residual_temp_max_sigma: float = 3.0,
        residual_occ_logit_max: float = 4.0,
        residual_log_wet_mean_max: float = 3.0,
        temp_feature_index: int = 0,
        rain_feature_index: int = 1,
    ) -> None:
        super().__init__()
        self.horizon_steps = int(horizon_steps)
        self.history_steps = int(history_steps)
        self.residual_temp_max_sigma = float(residual_temp_max_sigma)
        self.residual_occ_logit_max = float(residual_occ_logit_max)
        self.residual_log_wet_mean_max = float(residual_log_wet_mean_max)
        self.temp_feature_index = int(temp_feature_index)
        self.rain_feature_index = int(rain_feature_index)
        self.in_channels = int(in_channels)

        scales = []
        for s in timemixer_scales:
            s_int = max(1, int(s))
            if s_int <= self.history_steps and s_int not in scales:
                scales.append(s_int)
        if 1 not in scales:
            scales.insert(0, 1)
        self.scale_factors = tuple(scales)
        self.n_scales = len(self.scale_factors)
        self.ma_kernel = max(1, int(timemixer_ma_kernel))
        self.moving_average = MovingAverage1D(self.ma_kernel)

        self.recent_steps = min(self.history_steps, max(6, int(round(float(recent_window_hours) * steps_per_hour))))

        summary_in_dim = 6 * self.in_channels
        self.season_mlps = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(summary_in_dim),
                nn.Linear(summary_in_dim, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model),
            )
            for _ in range(self.n_scales)
        ])
        self.trend_mlps = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(summary_in_dim),
                nn.Linear(summary_in_dim, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model),
            )
            for _ in range(self.n_scales)
        ])

        self.scale_embed = nn.Parameter(torch.zeros(1, self.n_scales, d_model))
        nn.init.trunc_normal_(self.scale_embed, std=0.02)

        self.season_mixers = nn.ModuleList([ScaleMixerBlock(self.n_scales, d_model, dropout) for _ in range(max(1, n_layers))])
        self.trend_mixers = nn.ModuleList([ScaleMixerBlock(self.n_scales, d_model, dropout) for _ in range(max(1, n_layers))])
        self.scale_norm = nn.LayerNorm(d_model)

        self.future_time_mlp = nn.Sequential(
            nn.Linear(time_feat_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.baseline_mlp = nn.Sequential(
            nn.Linear(11, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

        self.temp_recent_mlp = nn.Sequential(
            nn.Linear(2 * self.recent_steps, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.rain_recent_mlp = nn.Sequential(
            nn.Linear(4 * self.recent_steps + 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

        self.lead_embed = nn.Parameter(torch.zeros(1, self.horizon_steps, d_model))
        nn.init.trunc_normal_(self.lead_embed, std=0.02)

        self.temp_query_mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.rain_query_mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

        self.temp_fmm_gate = nn.Sequential(
            nn.LayerNorm(2 * d_model),
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        self.rain_fmm_gate = nn.Sequential(
            nn.LayerNorm(2 * d_model),
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

        future_layers = max(1, min(2, n_layers))
        temp_future_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        rain_future_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.temp_future_encoder = nn.TransformerEncoder(temp_future_layer, num_layers=future_layers)
        self.rain_future_encoder = nn.TransformerEncoder(rain_future_layer, num_layers=future_layers)
        self.temp_out_norm = nn.LayerNorm(d_model)
        self.rain_out_norm = nn.LayerNorm(d_model)

        self.temp_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 2),
        )
        self.rain_occ_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        self.rain_amt_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 2),
        )

        # V7: caps auxiliars a escala d'episodi.
        # No substitueixen la distribució principal ZI-Gamma; actuen com a tasques
        # auxiliars perquè la representació distingeixi inici, continuïtat i final.
        self.rain_regime_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 9),  # onset/continue/decay x 1h/2h/3h
        )
        self.rain_accum_direct_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 3),  # acumulat directe 1h/2h/3h en log1p
        )

        self.rain_occ_residual_gate = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))
        self.rain_amt_residual_gate = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))
        nn.init.constant_(self.rain_occ_residual_gate[-1].bias, -1.0)
        nn.init.constant_(self.rain_amt_residual_gate[-1].bias, -1.0)

    def _downsample(self, x: torch.Tensor, factor: int) -> torch.Tensor:
        if factor <= 1:
            return x
        xt = x.transpose(1, 2)
        L = xt.shape[-1]
        pad = (-L) % factor
        if pad > 0:
            xt = F.pad(xt, (pad, 0), mode="replicate")
        xs = F.avg_pool1d(xt, kernel_size=factor, stride=factor)
        return xs.transpose(1, 2)

    def _summary(self, seq: torch.Tensor) -> torch.Tensor:
        last = seq[:, -1, :]
        mean = seq.mean(dim=1)
        tail_len = max(1, min(seq.shape[1], 6))
        recent = seq[:, -tail_len:, :].mean(dim=1)
        return torch.cat([last, recent, mean], dim=-1)

    def _encode_scales(self, x_past: torch.Tensor) -> torch.Tensor:
        season_tokens = []
        trend_tokens = []
        for i, factor in enumerate(self.scale_factors):
            xs = self._downsample(x_past, factor)
            trend = self.moving_average(xs)
            season = xs - trend
            season_tokens.append(self.season_mlps[i](torch.cat([self._summary(season), self._summary(xs)], dim=-1)))
            trend_tokens.append(self.trend_mlps[i](torch.cat([self._summary(trend), self._summary(xs)], dim=-1)))

        z_season = torch.stack(season_tokens, dim=1)
        z_trend = torch.stack(trend_tokens, dim=1)

        for blk in self.season_mixers:
            z_season = blk(z_season)

        z_trend_rev = torch.flip(z_trend, dims=[1])
        for blk in self.trend_mixers:
            z_trend_rev = blk(z_trend_rev)
        z_trend = torch.flip(z_trend_rev, dims=[1])

        z = self.scale_norm(z_season + z_trend + self.scale_embed[:, : self.n_scales, :])
        return z

    def forward(
        self,
        x_past: torch.Tensor,
        t_future: torch.Tensor,
        temp_baseline_scaled: torch.Tensor,
        rain_occ_baseline: torch.Tensor,
        rain_mu_baseline: torch.Tensor,
        last_temp_scaled: torch.Tensor,
        last_rain_raw: torch.Tensor,
        future_cosz: torch.Tensor,
        future_toa: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        z_scales = self._encode_scales(x_past)
        ctx_scales = z_scales.mean(dim=1)

        x_recent = x_past[:, -self.recent_steps :, :]
        temp_hist = x_recent[:, :, self.temp_feature_index]
        rain_hist_raw = torch.clamp(x_recent[:, :, self.rain_feature_index], min=0.0)
        rain_hist = torch.log1p(rain_hist_raw)
        wet_hist = (rain_hist_raw > 0.0).to(x_past.dtype)

        temp_diff = F.pad(temp_hist[:, 1:] - temp_hist[:, :-1], (1, 0))
        rain_diff = F.pad(rain_hist[:, 1:] - rain_hist[:, :-1], (1, 0))
        recent_wet_frac = wet_hist.mean(dim=1)
        recent_wet_tail = _consecutive_tail_fraction(wet_hist)

        temp_recent = self.temp_recent_mlp(torch.cat([temp_hist, temp_diff], dim=-1))
        rain_recent = self.rain_recent_mlp(torch.cat([rain_hist, wet_hist, rain_diff, temp_hist, recent_wet_frac[:, None], recent_wet_tail[:, None]], dim=-1))

        O = self.horizon_steps
        last_temp_rep = last_temp_scaled.unsqueeze(1).expand(-1, O)
        last_rain_rep = torch.log1p(torch.clamp(last_rain_raw, min=0.0)).unsqueeze(1).expand(-1, O)
        wet_flag_rep = (last_rain_raw > 0.0).to(x_past.dtype).unsqueeze(1).expand(-1, O)
        recent_wet_frac_rep = recent_wet_frac.unsqueeze(1).expand(-1, O)
        recent_wet_tail_rep = recent_wet_tail.unsqueeze(1).expand(-1, O)
        lead_frac = torch.linspace(1.0 / O, 1.0, O, device=x_past.device, dtype=x_past.dtype).unsqueeze(0).expand(x_past.shape[0], -1)

        ft = self.future_time_mlp(t_future)
        base_pack = torch.stack(
            [
                temp_baseline_scaled,
                torch.clamp(rain_occ_baseline, 0.0, 1.0),
                torch.log1p(torch.clamp(rain_mu_baseline, min=0.0)),
                last_temp_rep,
                last_rain_rep,
                wet_flag_rep,
                recent_wet_frac_rep,
                recent_wet_tail_rep,
                torch.clamp(future_cosz, min=-1.0, max=1.0),
                torch.log1p(torch.clamp(future_toa, min=0.0)) / math.log1p(1361.0),
                lead_frac,
            ],
            dim=-1,
        )
        fb = self.baseline_mlp(base_pack)

        shared_q = ft + fb + self.lead_embed[:, :O, :] + ctx_scales.unsqueeze(1)
        temp_q = self.temp_query_mlp(shared_q + temp_recent.unsqueeze(1))
        rain_q = self.rain_query_mlp(shared_q + rain_recent.unsqueeze(1))

        z_exp = z_scales.unsqueeze(1).expand(-1, O, -1, -1)
        temp_q_exp = temp_q.unsqueeze(2).expand(-1, -1, self.n_scales, -1)
        rain_q_exp = rain_q.unsqueeze(2).expand(-1, -1, self.n_scales, -1)

        temp_alpha = torch.softmax(self.temp_fmm_gate(torch.cat([temp_q_exp, z_exp], dim=-1)).squeeze(-1), dim=-1)
        rain_alpha = torch.softmax(self.rain_fmm_gate(torch.cat([rain_q_exp, z_exp], dim=-1)).squeeze(-1), dim=-1)

        temp_mix = (temp_alpha.unsqueeze(-1) * z_exp).sum(dim=2)
        rain_mix = (rain_alpha.unsqueeze(-1) * z_exp).sum(dim=2)

        temp_h = self.temp_future_encoder(temp_q + temp_mix)
        rain_h = self.rain_future_encoder(rain_q + rain_mix)
        temp_h = self.temp_out_norm(temp_h)
        rain_h = self.rain_out_norm(rain_h)

        temp_params = self.temp_head(temp_h)
        delta_temp = self.residual_temp_max_sigma * torch.tanh(temp_params[..., 0])
        temp_mu = temp_baseline_scaled + delta_temp
        log_sigma = temp_params[..., 1].clamp(min=-7.0, max=4.0)
        temp_sigma = torch.exp(log_sigma)

        occ_delta_raw = self.rain_occ_head(rain_h).squeeze(-1)
        amt_params = self.rain_amt_head(rain_h)
        occ_gate = torch.sigmoid(self.rain_occ_residual_gate(rain_h).squeeze(-1))
        amt_gate = torch.sigmoid(self.rain_amt_residual_gate(rain_h).squeeze(-1))

        delta_occ_logit = occ_gate * self.residual_occ_logit_max * torch.tanh(occ_delta_raw)
        delta_log_wet_mu = amt_gate * self.residual_log_wet_mean_max * torch.tanh(amt_params[..., 0])
        k = torch.clamp(F.softplus(amt_params[..., 1]) + 1e-3, min=1e-3, max=20.0)

        p_occ_base = torch.clamp(rain_occ_baseline, min=1e-4, max=1.0 - 1e-4)
        occ_logit = torch.log(p_occ_base / (1.0 - p_occ_base)) + delta_occ_logit
        p_occ = torch.sigmoid(occ_logit)

        wet_mu_base = torch.where(
            rain_occ_baseline > 1e-4,
            rain_mu_baseline / torch.clamp(rain_occ_baseline, min=1e-4),
            torch.full_like(rain_mu_baseline, 0.05),
        )
        wet_mu_base = torch.clamp(wet_mu_base, min=0.05)
        wet_mu = wet_mu_base * torch.exp(delta_log_wet_mu)
        theta = torch.clamp(wet_mu / torch.clamp(k, min=1e-3), min=1e-4, max=100.0)

        rain_logit_p0 = -occ_logit

        # V7: tasques auxiliars d'episodi. S'usen durant l'entrenament i també
        # s'exposen en mode predict com a diagnòstic probabilístic.
        rain_episode_context = rain_h.mean(dim=1)
        rain_regime_logits = self.rain_regime_head(rain_episode_context).view(-1, 3, 3)
        rain_accum_direct_log1p = F.softplus(self.rain_accum_direct_head(rain_episode_context))

        return {
            "temp_mu": temp_mu,
            "temp_sigma": temp_sigma,
            "rain_logit_p0": rain_logit_p0,
            "rain_k": k,
            "rain_theta": theta,
            "p_occ": p_occ,
            "wet_mu": wet_mu,
            "rain_regime_logits": rain_regime_logits,
            "rain_accum_direct_log1p": rain_accum_direct_log1p,
        }


# =============================================================================
# Pèrdues i mètriques
# =============================================================================



def gaussian_nll(y: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    sigma = torch.clamp(sigma, min=eps)
    return 0.5 * ((y - mu) / sigma) ** 2 + torch.log(sigma) + 0.5 * math.log(2.0 * math.pi)



def zigamma_nll(
    y: torch.Tensor,
    logit_p0: torch.Tensor,
    k: torch.Tensor,
    theta: torch.Tensor,
    eps: float = 1e-6,
    cap: Optional[float] = None,
    pos_weight: float = 0.0,
    intensity_alpha: float = 0.0,
) -> torch.Tensor:
    y = torch.clamp(y, min=0.0)
    if cap is not None and cap > 0:
        y = torch.clamp(y, max=float(cap))

    p0 = torch.sigmoid(logit_p0)
    p0 = torch.clamp(p0, min=eps, max=1.0 - eps)

    is_zero = (y <= 0.0).to(y.dtype)
    is_pos = 1.0 - is_zero
    log_y = torch.log(torch.clamp(y, min=eps))

    log_pdf = (k - 1.0) * log_y - (y / theta) - torch.lgamma(k) - k * torch.log(theta)
    log_prob = is_zero * torch.log(p0) + is_pos * (torch.log(1.0 - p0) + log_pdf)
    nll = -log_prob

    if (pos_weight > 0.0) or (intensity_alpha > 0.0):
        w = torch.ones_like(nll)
        if pos_weight > 0.0:
            w = w + float(pos_weight) * is_pos
        if intensity_alpha > 0.0:
            w = w + float(intensity_alpha) * torch.log1p(y) * is_pos
        nll = nll * w

    return nll




def zigamma_mixture_nll(
    y: torch.Tensor,
    p0_stack: torch.Tensor,
    k_stack: torch.Tensor,
    theta_stack: torch.Tensor,
    eps: float = 1e-6,
    cap: Optional[float] = None,
) -> torch.Tensor:
    """
    NLL exacta d'una barreja equiprobable de distribucions Zero-Inflated Gamma.

    Implementació en espai logarítmic amb log-sum-exp. Això evita underflow
    quan les densitats Gamma són molt petites en episodis de pluja intensa.
    """
    y = torch.clamp(y, min=0.0)
    if cap is not None and cap > 0:
        y = torch.clamp(y, max=float(cap))

    p0_stack = torch.clamp(p0_stack, min=eps, max=1.0 - eps)
    k_stack = torch.clamp(k_stack, min=eps)
    theta_stack = torch.clamp(theta_stack, min=eps)

    y_exp = y.unsqueeze(0)
    is_zero = y_exp <= 0.0
    log_y = torch.log(torch.clamp(y_exp, min=eps))
    log_pdf = (
        (k_stack - 1.0) * log_y
        - (y_exp / theta_stack)
        - torch.lgamma(k_stack)
        - k_stack * torch.log(theta_stack)
    )

    log_prob_member = torch.where(
        is_zero,
        torch.log(p0_stack),
        torch.log1p(-p0_stack) + log_pdf,
    )
    log_mix_prob = torch.logsumexp(log_prob_member, dim=0) - math.log(float(p0_stack.shape[0]))
    return -log_mix_prob


def expected_rain_from_params(logit_p0: torch.Tensor, k: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    p0 = torch.sigmoid(logit_p0)
    return (1.0 - p0) * (k * theta)



def rain_occurrence_bce_loss(
    rain_logit_p0: torch.Tensor,
    y_occ: torch.Tensor,
    eps: float = 1e-6,
    focal_gamma: float = 0.0,
    pos_weight: float = 0.0,
) -> torch.Tensor:
    """Pèrdua auxiliar Bernoulli/focal per a l'ocurrència de pluja."""
    p_occ = torch.clamp(1.0 - torch.sigmoid(rain_logit_p0), min=eps, max=1.0 - eps)
    ce = F.binary_cross_entropy(p_occ, y_occ, reduction="none")
    if focal_gamma > 0.0:
        pt = y_occ * p_occ + (1.0 - y_occ) * (1.0 - p_occ)
        ce = ce * torch.pow(torch.clamp(1.0 - pt, min=0.0, max=1.0), float(focal_gamma))
    if pos_weight > 0.0:
        ce = ce * (1.0 + float(pos_weight) * y_occ)
    return ce





def masked_mean(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return (x * mask).sum() / torch.clamp(mask.sum(), min=eps)



def temperature_smoothness_loss(temp_mu_raw: torch.Tensor, mask_temp: torch.Tensor) -> torch.Tensor:
    """
    Penalitza curvatures temporals excessives a la predicció de temperatura.

    Això no imposa una física forta, però evita serrats no plausibles a 10 minuts.
    """
    if temp_mu_raw.shape[1] < 3:
        return temp_mu_raw.new_zeros(())

    d2 = temp_mu_raw[:, 2:] - 2.0 * temp_mu_raw[:, 1:-1] + temp_mu_raw[:, :-2]
    m = mask_temp[:, 2:] * mask_temp[:, 1:-1] * mask_temp[:, :-2]
    return masked_mean(torch.abs(d2), m)



def temperature_anchor_loss(
    temp_mu_scaled: torch.Tensor,
    temp_baseline_scaled: torch.Tensor,
    mask_temp: torch.Tensor,
) -> torch.Tensor:
    """Manté la xarxa prop de la baseline quan la informació no és prou concloent."""
    return masked_mean((temp_mu_scaled - temp_baseline_scaled) ** 2, mask_temp)



def accumulation_consistency_loss(
    rain_mu: torch.Tensor,
    y_rain: torch.Tensor,
    mask_rain: torch.Tensor,
    horizon_steps: List[int],
) -> torch.Tensor:
    """
    Penalitza incoherències en acumulacions futures de pluja.

    El model treballa a 10 minuts, però operativament també importa molt 1 h / 2 h / 3 h.
    """
    losses: List[torch.Tensor] = []
    for h in horizon_steps:
        if h <= 0 or h > rain_mu.shape[1]:
            continue
        pred_acc = rain_mu[:, :h].sum(dim=1)
        true_acc = y_rain[:, :h].sum(dim=1)
        valid = (mask_rain[:, :h].min(dim=1).values > 0.5).to(rain_mu.dtype)
        losses.append(masked_mean(torch.abs(pred_acc - true_acc), valid))
    if not losses:
        return rain_mu.new_zeros(())
    return torch.stack(losses).mean()




def temperature_trend_consistency_loss(
    temp_mu_raw: torch.Tensor,
    temp_trend_prior_raw: torch.Tensor,
    mask_temp: torch.Tensor,
    steps_per_hour: float,
    min_prior_slope_c_per_h: float = 0.08,
    margin_c_per_h: float = 0.02,
) -> torch.Tensor:
    """
    Penalitza prediccions que giren clarament contra una tendència recent robusta.

    No força la magnitud exacta de la tendència; només actua quan el prior té un signe
    prou clar. Això evita que una situació de refredament persistent al vespre/nit es
    transformi artificialment en escalfament, però deixa llibertat en fronts o canvis reals.
    """
    if temp_mu_raw.shape[1] < 2 or temp_trend_prior_raw.shape[1] < 2:
        return temp_mu_raw.new_zeros(())
    pred_slope = (temp_mu_raw[:, 1:] - temp_mu_raw[:, :-1]) * float(steps_per_hour)
    prior_slope = (temp_trend_prior_raw[:, 1:] - temp_trend_prior_raw[:, :-1]) * float(steps_per_hour)
    sign = torch.sign(prior_slope)
    strength = torch.clamp(torch.abs(prior_slope) / 0.5, min=0.0, max=1.0)
    active = (torch.abs(prior_slope) >= float(min_prior_slope_c_per_h)).to(temp_mu_raw.dtype)
    m = mask_temp[:, 1:] * mask_temp[:, :-1] * active
    penalty = F.relu(float(margin_c_per_h) - sign * pred_slope) * strength
    return masked_mean(penalty, m)


def temperature_diurnal_trend_loss(
    temp_mu_raw: torch.Tensor,
    future_toa: torch.Tensor,
    mask_temp: torch.Tensor,
    steps_per_hour: float,
) -> torch.Tensor:
    """
    Regularització molt feble de cicle diürn per dies estables.

    Només penalitza girs clarament contraris al forçament solar quan el senyal TOA és net.
    Té un pes petit per no impedir canvis per núvols, advecció, fronts o precipitació.
    """
    if temp_mu_raw.shape[1] < 2 or future_toa.shape[1] < 2:
        return temp_mu_raw.new_zeros(())
    pred_slope = (temp_mu_raw[:, 1:] - temp_mu_raw[:, :-1]) * float(steps_per_hour)
    d_toa = future_toa[:, 1:] - future_toa[:, :-1]
    expected = torch.where(
        (future_toa[:, 1:] < 30.0) | (d_toa < -5.0),
        torch.full_like(d_toa, -1.0),
        torch.where(d_toa > 5.0, torch.ones_like(d_toa), torch.zeros_like(d_toa)),
    )
    active = (expected != 0.0).to(temp_mu_raw.dtype)
    strength = torch.clamp(torch.abs(d_toa) / 200.0, min=0.0, max=1.0)
    m = mask_temp[:, 1:] * mask_temp[:, :-1] * active
    penalty = F.relu(0.0 - expected * pred_slope) * strength
    return masked_mean(penalty, m)


def rain_heavy_amount_loss(
    rain_mu: torch.Tensor,
    y_rain: torch.Tensor,
    mask_rain: torch.Tensor,
    heavy_threshold: float,
    under_weight: float = 2.5,
    intensity_power: float = 0.5,
) -> torch.Tensor:
    """
    Penalització específica per quantitat de pluja intensa.

    Es calcula en log1p per no deixar que pocs extrems dominin tot l'entrenament, però
    penalitza més la subestimació que la sobreestimació perquè el biaix detectat és cap avall.
    """
    thr = max(float(heavy_threshold), 1e-6)
    heavy = (y_rain >= thr).to(rain_mu.dtype)
    if torch.sum(heavy * mask_rain) <= 0:
        return rain_mu.new_zeros(())
    log_pred = torch.log1p(torch.clamp(rain_mu, min=0.0))
    log_true = torch.log1p(torch.clamp(y_rain, min=0.0))
    abs_err = torch.abs(log_pred - log_true)
    under = F.relu(log_true - log_pred)
    intensity = torch.pow(torch.clamp(y_rain / thr, min=1.0), float(intensity_power))
    intensity = torch.clamp(intensity, min=1.0, max=6.0)
    loss = (abs_err + (float(under_weight) - 1.0) * under) * intensity
    return masked_mean(loss, mask_rain * heavy)


def rain_heavy_accumulation_loss(
    rain_mu: torch.Tensor,
    y_rain: torch.Tensor,
    mask_rain: torch.Tensor,
    horizon_steps: List[int],
    step_heavy_threshold: float,
    under_weight: float = 2.5,
) -> torch.Tensor:
    """
    Penalitza subestimació d'acumulats intensos a 1/2/3 h.

    El llindar d'acumulat s'escala amb sqrt(h) per detectar episodis importants sense exigir
    que tots els passos siguin forts. És una regularització, no una mètrica oficial.
    """
    losses: List[torch.Tensor] = []
    thr_step = max(float(step_heavy_threshold), 1e-6)
    for h in horizon_steps:
        if h <= 0 or h > rain_mu.shape[1]:
            continue
        pred_acc = torch.clamp(rain_mu[:, :h], min=0.0).sum(dim=1)
        true_acc = torch.clamp(y_rain[:, :h], min=0.0).sum(dim=1)
        valid = (mask_rain[:, :h].min(dim=1).values > 0.5).to(rain_mu.dtype)
        thr_acc = thr_step * math.sqrt(float(h))
        heavy = (true_acc >= thr_acc).to(rain_mu.dtype)
        if torch.sum(valid * heavy) <= 0:
            continue
        log_pred = torch.log1p(pred_acc)
        log_true = torch.log1p(true_acc)
        abs_err = torch.abs(log_pred - log_true)
        under = F.relu(log_true - log_pred)
        losses.append(masked_mean(abs_err + (float(under_weight) - 1.0) * under, valid * heavy))
    if not losses:
        return rain_mu.new_zeros(())
    return torch.stack(losses).mean()


def _accumulation_targets_by_horizon(
    y_rain: torch.Tensor,
    mask_rain: torch.Tensor,
    horizon_steps: List[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Retorna acumulats observats i màscares vàlides per a 1/2/3 h."""
    accs: List[torch.Tensor] = []
    masks: List[torch.Tensor] = []
    for h in horizon_steps:
        h = int(h)
        if h <= 0 or h > y_rain.shape[1]:
            continue
        accs.append(torch.clamp(y_rain[:, :h], min=0.0).sum(dim=1))
        masks.append((mask_rain[:, :h].min(dim=1).values > 0.5).to(y_rain.dtype))
    if not accs:
        z = y_rain.new_zeros((y_rain.shape[0], 0))
        return z, z
    return torch.stack(accs, dim=1), torch.stack(masks, dim=1)



def temperature_rain_cooling_consistency_loss(
    temp_mu_raw: torch.Tensor,
    rain_mu: torch.Tensor,
    p_occ: torch.Tensor,
    mask_temp: torch.Tensor,
    mask_rain: torch.Tensor,
    future_toa: torch.Tensor,
    steps_per_hour: float,
    event_threshold: float,
) -> torch.Tensor:
    """
    Regularització v7 molt feble: si la predicció indica règim humit i radiació
    efectiva potencialment limitada, s'eviten escalfaments ràpids no plausibles.

    No força refredament sempre que plou; només penalitza pujades positives fortes
    en passos humits/probables, que era un error recurrent en episodis amb núvols
    i precipitació.
    """
    if temp_mu_raw.shape[1] < 2:
        return temp_mu_raw.new_zeros(())
    slope = (temp_mu_raw[:, 1:] - temp_mu_raw[:, :-1]) * float(steps_per_hour)
    wet_signal = torch.clamp(
        0.5 * (p_occ[:, 1:] + p_occ[:, :-1])
        + 0.5 * ((rain_mu[:, 1:] + rain_mu[:, :-1]) >= float(event_threshold)).to(temp_mu_raw.dtype),
        0.0,
        1.0,
    )
    # Si el TOA és alt no anul·lem la penalització, però la fem més feble.
    toa_gate = torch.where(future_toa[:, 1:] > 300.0, torch.full_like(slope, 0.45), torch.ones_like(slope))
    valid = mask_temp[:, 1:] * mask_temp[:, :-1] * mask_rain[:, 1:] * mask_rain[:, :-1]
    penalty = F.relu(slope - 0.10) * wet_signal * toa_gate
    return masked_mean(penalty, valid)




def rain_regime_auxiliary_loss(
    regime_logits: torch.Tensor,
    y_rain: torch.Tensor,
    mask_rain: torch.Tensor,
    last_rain_raw: torch.Tensor,
    horizon_steps: List[int],
    event_threshold: float,
    onset_pos_weight: float = 4.0,
    continue_pos_weight: float = 1.5,
    decay_pos_weight: float = 2.0,
) -> torch.Tensor:
    """
    Pèrdua auxiliar de règim de precipitació.

    Caps:
      0) onset: estava sec i apareix precipitació dins l'horitzó h;
      1) continue: ja plovia i continua acumulant dins h;
      2) decay/end: ja plovia però l'acumulat futur és petit respecte al pas actual.

    És deliberadament suau: no canvia directament p_rain, sinó que força la
    representació latent a separar pre-onset, maduresa i final de l'episodi.
    """
    if regime_logits.ndim != 3 or regime_logits.shape[1] < 3:
        return y_rain.new_zeros(())

    # Compatibilitat si només tenim 1/2/3 h.
    horizon_steps = [int(h) for h in horizon_steps if int(h) > 0 and int(h) <= y_rain.shape[1]]
    Hn = min(len(horizon_steps), regime_logits.shape[2])
    if Hn <= 0:
        return y_rain.new_zeros(())

    acc, valid_h = _accumulation_targets_by_horizon(y_rain, mask_rain, horizon_steps[:Hn])
    last = torch.clamp(last_rain_raw, min=0.0).unsqueeze(1).expand(-1, Hn)
    last_wet = (last >= float(event_threshold)).to(y_rain.dtype)
    future_wet = (acc >= float(event_threshold)).to(y_rain.dtype)

    onset_t = (1.0 - last_wet) * future_wet
    continue_t = last_wet * future_wet

    # Decay: l'episodi és actiu ara, però el futur h acumula menys que una
    # persistència mínima del pas actual. Això ataca la sobrepersistència.
    h_scale = torch.tensor(
        [max(1.0, float(h) / max(1.0, float(horizon_steps[0]))) for h in horizon_steps[:Hn]],
        dtype=y_rain.dtype,
        device=y_rain.device,
    ).unsqueeze(0)
    persistence_floor = torch.maximum(
        torch.full_like(last, 0.5),
        last * torch.sqrt(h_scale) * 0.65,
    )
    decay_t = last_wet * (acc < persistence_floor).to(y_rain.dtype)

    targets = torch.stack([onset_t, continue_t, decay_t], dim=1)  # [B,3,Hn]
    valid = valid_h.unsqueeze(1).expand(-1, 3, -1)
    logits = regime_logits[:, :, :Hn]

    pos_w = torch.tensor(
        [float(onset_pos_weight), float(continue_pos_weight), float(decay_pos_weight)],
        dtype=y_rain.dtype,
        device=y_rain.device,
    ).view(1, 3, 1)
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    weights = 1.0 + (pos_w - 1.0) * targets
    return masked_mean(bce * weights, valid)


def rain_accum_direct_loss(
    accum_direct_log1p: torch.Tensor,
    rain_mu: torch.Tensor,
    y_rain: torch.Tensor,
    mask_rain: torch.Tensor,
    horizon_steps: List[int],
    under_weight: float = 1.35,
) -> torch.Tensor:
    """
    Pèrdua auxiliar d'acumulació directa a 1/2/3 h.

    El cap directe aprèn l'acumulat agregat, que és meteorològicament més estable
    que la seqüència de 10 minuts. També s'imposa consistència suau amb la suma
    dels passos principals.
    """
    if accum_direct_log1p.ndim != 2:
        return y_rain.new_zeros(())
    horizon_steps = [int(h) for h in horizon_steps if int(h) > 0 and int(h) <= y_rain.shape[1]]
    Hn = min(len(horizon_steps), accum_direct_log1p.shape[1])
    if Hn <= 0:
        return y_rain.new_zeros(())

    true_acc, valid_h = _accumulation_targets_by_horizon(y_rain, mask_rain, horizon_steps[:Hn])
    pred_acc = torch.expm1(torch.clamp(accum_direct_log1p[:, :Hn], min=0.0, max=math.log1p(500.0)))
    step_acc = torch.stack([torch.clamp(rain_mu[:, :h], min=0.0).sum(dim=1) for h in horizon_steps[:Hn]], dim=1)

    log_pred = torch.log1p(torch.clamp(pred_acc, min=0.0))
    log_true = torch.log1p(torch.clamp(true_acc, min=0.0))
    under = F.relu(log_true - log_pred)
    abs_err = torch.abs(log_pred - log_true)
    main = abs_err + (float(under_weight) - 1.0) * under
    consistency = torch.abs(torch.log1p(torch.clamp(pred_acc, min=0.0)) - torch.log1p(torch.clamp(step_acc.detach(), min=0.0)))
    return masked_mean(main + 0.25 * consistency, valid_h)


def rain_regime_weighted_step_loss(
    rain_mu: torch.Tensor,
    y_rain: torch.Tensor,
    mask_rain: torch.Tensor,
    last_rain_raw: torch.Tensor,
    event_threshold: float,
    preonset_weight: float = 0.20,
    decay_weight: float = 0.15,
) -> torch.Tensor:
    """
    Penalització addicional molt suau per finestres difícils:
      - pre-onset: actualment sec però plourà dins l'horitzó;
      - decay: actualment humit però l'acumulat futur és feble.

    Es calcula en log1p per estabilitat i no substitueix la NLL ZI-Gamma.
    """
    if float(preonset_weight) <= 0.0 and float(decay_weight) <= 0.0:
        return y_rain.new_zeros(())

    future_acc = torch.clamp(y_rain, min=0.0).sum(dim=1)
    last_wet = (last_rain_raw >= float(event_threshold)).to(y_rain.dtype)
    future_wet = (future_acc >= float(event_threshold)).to(y_rain.dtype)

    preonset = (1.0 - last_wet) * future_wet
    decay = last_wet * (future_acc < torch.maximum(torch.full_like(last_rain_raw, 0.5), last_rain_raw * 1.25)).to(y_rain.dtype)

    sample_w = 1.0 + float(preonset_weight) * preonset + float(decay_weight) * decay
    err = torch.abs(torch.log1p(torch.clamp(rain_mu, min=0.0)) - torch.log1p(torch.clamp(y_rain, min=0.0)))
    return masked_mean(err * sample_w.unsqueeze(1), mask_rain)



def crps_gaussian(mu: torch.Tensor, sigma: torch.Tensor, y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    sigma = torch.clamp(sigma, min=eps)
    z = (y - mu) / sigma
    sqrt2 = math.sqrt(2.0)
    Phi = 0.5 * (1.0 + torch.erf(z / sqrt2))
    phi = (1.0 / math.sqrt(2.0 * math.pi)) * torch.exp(-0.5 * z ** 2)
    return sigma * (z * (2.0 * Phi - 1.0) + 2.0 * phi - 1.0 / math.sqrt(math.pi))



def fair_crps_from_samples(samples: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    N = samples.shape[0]
    if N < 2:
        return torch.mean(torch.abs(samples - y.unsqueeze(0)), dim=0)
    term1 = torch.mean(torch.abs(samples - y.unsqueeze(0)), dim=0)
    dif = torch.abs(samples.unsqueeze(1) - samples.unsqueeze(0))
    term2 = dif.sum(dim=(0, 1)) / (2.0 * N * (N - 1))
    return term1 - term2



def sample_zigamma(p0: torch.Tensor, k: torch.Tensor, theta: torch.Tensor, n_samples: int) -> torch.Tensor:
    p0 = torch.clamp(p0, min=1e-6, max=1.0 - 1e-6)
    k = torch.clamp(k, min=1e-6)
    theta = torch.clamp(theta, min=1e-6)
    rate = 1.0 / theta
    gamma = torch.distributions.Gamma(concentration=k, rate=rate)
    samp = gamma.sample((n_samples,))
    u = torch.rand_like(samp)
    mask0 = (u < p0.unsqueeze(0)).to(samp.dtype)
    return samp * (1.0 - mask0)



def brier_score(y_true01: np.ndarray, y_prob: np.ndarray) -> float:
    return float(np.mean((y_prob - y_true01) ** 2))



def sedi(tp: int, fn: int, fp: int, tn: int, eps: float = 1e-9) -> float:
    if (tp + fn) == 0 or (fp + tn) == 0:
        return float("nan")
    H = tp / (tp + fn)
    Fp = fp / (fp + tn)
    H = float(np.clip(H, eps, 1.0 - eps))
    Fp = float(np.clip(Fp, eps, 1.0 - eps))
    num = math.log(Fp) - math.log(H) - math.log(1.0 - Fp) + math.log(1.0 - H)
    den = math.log(Fp) + math.log(H) + math.log(1.0 - Fp) + math.log(1.0 - H)
    return float(num / den)



def safe_nanmean(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0 or not np.isfinite(x).any():
        return float("nan")
    return float(np.nanmean(x))


def finite_or(value: object, default: float = 0.0) -> float:
    """Retorna un float finit o un valor per defecte per fer robusta la selecció de model."""
    try:
        v = float(value)
    except Exception:
        return float(default)
    return v if math.isfinite(v) else float(default)



# =============================================================================
# Diagnòstics avançats de verificació probabilista i per règim
# =============================================================================


def init_temp_coverage_accumulators(O: int) -> Dict[str, np.ndarray]:
    """Inicialitza acumuladors de cobertura probabilista per a temperatura."""
    return {
        "n": np.zeros(O, dtype=np.int64),
        "hit80": np.zeros(O, dtype=np.int64),
        "hit90": np.zeros(O, dtype=np.int64),
        "width80": np.zeros(O, dtype=np.float64),
        "width90": np.zeros(O, dtype=np.float64),
        "pit_sum": np.zeros(O, dtype=np.float64),
        "pit_sq_sum": np.zeros(O, dtype=np.float64),
    }


def update_temp_coverage_accumulators(
    acc: Dict[str, np.ndarray],
    mu: np.ndarray,
    sigma: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
) -> None:
    """Actualitza cobertura 80/90% i diagnòstic PIT mitjà per lead time."""
    z80 = 1.2815515655446004
    z90 = 1.6448536269514722
    sigma = np.maximum(np.asarray(sigma, dtype=np.float64), 1e-6)
    mu = np.asarray(mu, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.asarray(mask, dtype=np.float64)
    O = mu.shape[1]
    for k in range(O):
        valid = mask[:, k] > 0.5
        if not np.any(valid):
            continue
        s = sigma[valid, k]
        m = mu[valid, k]
        yy = y[valid, k]
        acc["n"][k] += int(valid.sum())
        acc["hit80"][k] += int(np.sum((yy >= m - z80 * s) & (yy <= m + z80 * s)))
        acc["hit90"][k] += int(np.sum((yy >= m - z90 * s) & (yy <= m + z90 * s)))
        acc["width80"][k] += float(np.sum(2.0 * z80 * s))
        acc["width90"][k] += float(np.sum(2.0 * z90 * s))
        z = (yy - m) / s
        # CDF normal estàndard via erf vectoritzada amb NumPy.
        pit = 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))
        acc["pit_sum"][k] += float(np.sum(pit))
        acc["pit_sq_sum"][k] += float(np.sum(pit ** 2))


def finalize_temp_coverage_accumulators(acc: Dict[str, np.ndarray], steps_per_hour: int) -> Dict[str, object]:
    """Finalitza els diagnòstics de cobertura de temperatura."""
    n = acc["n"].astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        cov80 = np.where(n > 0, acc["hit80"] / n, np.nan)
        cov90 = np.where(n > 0, acc["hit90"] / n, np.nan)
        width80 = np.where(n > 0, acc["width80"] / n, np.nan)
        width90 = np.where(n > 0, acc["width90"] / n, np.nan)
        pit_mean = np.where(n > 0, acc["pit_sum"] / n, np.nan)
        pit_var = np.where(n > 0, acc["pit_sq_sum"] / n - pit_mean ** 2, np.nan)
    return {
        "summary": {
            "coverage80_mean": safe_nanmean(cov80),
            "coverage90_mean": safe_nanmean(cov90),
            "width80_mean_C": safe_nanmean(width80),
            "width90_mean_C": safe_nanmean(width90),
            "pit_mean": safe_nanmean(pit_mean),
            "pit_variance": safe_nanmean(pit_var),
            "target_coverage80": 0.80,
            "target_coverage90": 0.90,
        },
        "by_lead": {
            "lead_hours": ((np.arange(len(n), dtype=np.float64) + 1.0) / float(steps_per_hour)).tolist(),
            "coverage80": cov80.tolist(),
            "coverage90": cov90.tolist(),
            "width80_C": width80.tolist(),
            "width90_C": width90.tolist(),
            "pit_mean": pit_mean.tolist(),
            "pit_variance": pit_var.tolist(),
            "n": acc["n"].tolist(),
        },
    }


def init_reliability_accumulators(O: int, n_bins: int = 10) -> Dict[str, np.ndarray]:
    """Inicialitza acumuladors de calibratge probabilista per a ocurrència de pluja."""
    n_bins = max(2, int(n_bins))
    return {
        "n_bins": np.array([n_bins], dtype=np.int64),
        "n": np.zeros(n_bins, dtype=np.int64),
        "sum_prob": np.zeros(n_bins, dtype=np.float64),
        "sum_obs": np.zeros(n_bins, dtype=np.float64),
        "sum_brier": np.zeros(n_bins, dtype=np.float64),
        "n_by_lead": np.zeros((O, n_bins), dtype=np.int64),
        "sum_prob_by_lead": np.zeros((O, n_bins), dtype=np.float64),
        "sum_obs_by_lead": np.zeros((O, n_bins), dtype=np.float64),
    }


def update_reliability_accumulators(
    acc: Dict[str, np.ndarray],
    prob: np.ndarray,
    obs01: np.ndarray,
    mask: np.ndarray,
) -> None:
    """Actualitza diagrama de fiabilitat global i per lead time."""
    p = np.clip(np.asarray(prob, dtype=np.float64), 0.0, 1.0)
    y = np.asarray(obs01, dtype=np.float64)
    m = np.asarray(mask, dtype=np.float64) > 0.5
    O = p.shape[1]
    n_bins = int(acc["n_bins"][0])
    bin_idx = np.minimum((p * n_bins).astype(np.int64), n_bins - 1)
    for k in range(O):
        valid = m[:, k]
        if not np.any(valid):
            continue
        bk = bin_idx[valid, k]
        pk = p[valid, k]
        yk = y[valid, k]
        for b in range(n_bins):
            mb = bk == b
            if not np.any(mb):
                continue
            nb = int(mb.sum())
            acc["n"][b] += nb
            acc["sum_prob"][b] += float(np.sum(pk[mb]))
            acc["sum_obs"][b] += float(np.sum(yk[mb]))
            acc["sum_brier"][b] += float(np.sum((pk[mb] - yk[mb]) ** 2))
            acc["n_by_lead"][k, b] += nb
            acc["sum_prob_by_lead"][k, b] += float(np.sum(pk[mb]))
            acc["sum_obs_by_lead"][k, b] += float(np.sum(yk[mb]))


def finalize_reliability_accumulators(acc: Dict[str, np.ndarray], steps_per_hour: int) -> Dict[str, object]:
    """Finalitza el calibratge de probabilitat de pluja."""
    n_bins = int(acc["n_bins"][0])
    n = acc["n"].astype(np.float64)
    centers = (np.arange(n_bins, dtype=np.float64) + 0.5) / float(n_bins)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_prob = np.where(n > 0, acc["sum_prob"] / n, np.nan)
        obs_freq = np.where(n > 0, acc["sum_obs"] / n, np.nan)
        brier = np.where(n > 0, acc["sum_brier"] / n, np.nan)
    calib_err = np.abs(mean_prob - obs_freq)

    n_by = acc["n_by_lead"].astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_prob_by = np.where(n_by > 0, acc["sum_prob_by_lead"] / n_by, np.nan)
        obs_freq_by = np.where(n_by > 0, acc["sum_obs_by_lead"] / n_by, np.nan)
    lead_hours = ((np.arange(n_by.shape[0], dtype=np.float64) + 1.0) / float(steps_per_hour)).tolist()
    return {
        "summary": {
            "n_bins": n_bins,
            "expected_calibration_error": safe_nanmean(calib_err),
            "mean_bin_brier": safe_nanmean(brier),
            "total_points": int(acc["n"].sum()),
        },
        "global_bins": {
            "bin_center": centers.tolist(),
            "count": acc["n"].tolist(),
            "mean_forecast_probability": mean_prob.tolist(),
            "observed_frequency": obs_freq.tolist(),
            "brier": brier.tolist(),
        },
        "by_lead": {
            "lead_hours": lead_hours,
            "bin_center": centers.tolist(),
            "count": acc["n_by_lead"].tolist(),
            "mean_forecast_probability": mean_prob_by.tolist(),
            "observed_frequency": obs_freq_by.tolist(),
        },
    }


def init_accumulation_diagnostics(horizons_steps: List[int]) -> Dict[str, object]:
    """Inicialitza mètriques d'acumulació de pluja a 1/2/3 h o horitzons configurats."""
    hs = [int(h) for h in horizons_steps if int(h) > 0]
    return {
        "horizons_steps": hs,
        "n": np.zeros(len(hs), dtype=np.int64),
        "sum_abs": np.zeros(len(hs), dtype=np.float64),
        "sum_sq": np.zeros(len(hs), dtype=np.float64),
        "sum_bias": np.zeros(len(hs), dtype=np.float64),
        "sum_brier_event": np.zeros(len(hs), dtype=np.float64),
        "tp": np.zeros(len(hs), dtype=np.int64),
        "fn": np.zeros(len(hs), dtype=np.int64),
        "fp": np.zeros(len(hs), dtype=np.int64),
        "tn": np.zeros(len(hs), dtype=np.int64),
    }


def update_accumulation_diagnostics(
    acc: Dict[str, object],
    rain_mu: np.ndarray,
    rain_true: np.ndarray,
    p_occ: np.ndarray,
    mask: np.ndarray,
    event_threshold: float,
) -> None:
    """Actualitza mètriques d'acumulació de pluja."""
    rain_mu = np.asarray(rain_mu, dtype=np.float64)
    rain_true = np.asarray(rain_true, dtype=np.float64)
    p_occ = np.asarray(p_occ, dtype=np.float64)
    mask = np.asarray(mask, dtype=np.float64)
    hs = list(acc["horizons_steps"])
    for i, h in enumerate(hs):
        if h <= 0 or h > rain_mu.shape[1]:
            continue
        valid = np.min(mask[:, :h], axis=1) > 0.5
        if not np.any(valid):
            continue
        pred_acc = np.sum(rain_mu[valid, :h], axis=1)
        true_acc = np.sum(rain_true[valid, :h], axis=1)
        # Probabilitat que plogui en algun moment dins l'acumulació, assumint independència condicional aproximada.
        p_event = 1.0 - np.prod(1.0 - np.clip(p_occ[valid, :h], 0.0, 1.0), axis=1)
        y_event = (true_acc >= float(event_threshold)).astype(np.int64)
        err = pred_acc - true_acc
        acc["n"][i] += int(valid.sum())
        acc["sum_abs"][i] += float(np.sum(np.abs(err)))
        acc["sum_sq"][i] += float(np.sum(err ** 2))
        acc["sum_bias"][i] += float(np.sum(err))
        acc["sum_brier_event"][i] += float(np.sum((p_event - y_event) ** 2))
        yhat = (p_event >= 0.5).astype(np.int64)
        acc["tp"][i] += int(np.sum((yhat == 1) & (y_event == 1)))
        acc["fn"][i] += int(np.sum((yhat == 0) & (y_event == 1)))
        acc["fp"][i] += int(np.sum((yhat == 1) & (y_event == 0)))
        acc["tn"][i] += int(np.sum((yhat == 0) & (y_event == 0)))


def finalize_accumulation_diagnostics(acc: Dict[str, object], steps_per_hour: int) -> Dict[str, object]:
    """Finalitza mètriques d'acumulació de pluja."""
    n = acc["n"].astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        mae = np.where(n > 0, acc["sum_abs"] / n, np.nan)
        rmse = np.where(n > 0, np.sqrt(acc["sum_sq"] / n), np.nan)
        bias = np.where(n > 0, acc["sum_bias"] / n, np.nan)
        brier = np.where(n > 0, acc["sum_brier_event"] / n, np.nan)
    sedi_vals = np.array([
        sedi(int(acc["tp"][i]), int(acc["fn"][i]), int(acc["fp"][i]), int(acc["tn"][i])) if acc["n"][i] > 0 else np.nan
        for i in range(len(acc["n"]))
    ], dtype=np.float64)
    h_steps = list(acc["horizons_steps"])
    return {
        "summary": {
            "rain_accum_mae_mean": safe_nanmean(mae),
            "rain_accum_rmse_mean": safe_nanmean(rmse),
            "rain_accum_bias_mean": safe_nanmean(bias),
            "rain_accum_brier_event_mean": safe_nanmean(brier),
            "rain_accum_sedi_event_mean": safe_nanmean(sedi_vals),
        },
        "by_horizon": {
            "horizon_hours": [float(h) / float(steps_per_hour) for h in h_steps],
            "horizon_steps": h_steps,
            "mae": mae.tolist(),
            "rmse": rmse.tolist(),
            "bias": bias.tolist(),
            "brier_event": brier.tolist(),
            "sedi_event": sedi_vals.tolist(),
            "n": acc["n"].tolist(),
        },
    }


def init_regime_accumulators(regime_names: List[str]) -> Dict[str, Dict[str, float]]:
    """Inicialitza mètriques condicionades per règim meteorològic."""
    return {
        name: {
            "n_temp": 0.0,
            "sum_abs_temp": 0.0,
            "sum_sq_temp": 0.0,
            "n_rain": 0.0,
            "sum_abs_rain": 0.0,
            "sum_sq_rain": 0.0,
            "sum_brier": 0.0,
            "tp": 0.0,
            "fn": 0.0,
            "fp": 0.0,
            "tn": 0.0,
        }
        for name in regime_names
    }


def update_regime_accumulators(
    acc: Dict[str, Dict[str, float]],
    regime_masks: Dict[str, np.ndarray],
    temp_mu: np.ndarray,
    temp_true: np.ndarray,
    rain_mu: np.ndarray,
    rain_true: np.ndarray,
    p_occ: np.ndarray,
    occ_true: np.ndarray,
    mask_temp: np.ndarray,
    mask_rain: np.ndarray,
) -> None:
    """Actualitza mètriques agregades segons règim de la finestra."""
    for name, smask in regime_masks.items():
        if name not in acc:
            continue
        smask = np.asarray(smask, dtype=bool)
        if smask.ndim != 1 or not np.any(smask):
            continue
        mt = mask_temp[smask, :] > 0.5
        if np.any(mt):
            err_t = temp_mu[smask, :] - temp_true[smask, :]
            err_t = err_t[mt]
            acc[name]["n_temp"] += float(err_t.size)
            acc[name]["sum_abs_temp"] += float(np.sum(np.abs(err_t)))
            acc[name]["sum_sq_temp"] += float(np.sum(err_t ** 2))
        mr = mask_rain[smask, :] > 0.5
        if np.any(mr):
            err_r = rain_mu[smask, :] - rain_true[smask, :]
            err_r = err_r[mr]
            acc[name]["n_rain"] += float(err_r.size)
            acc[name]["sum_abs_rain"] += float(np.sum(np.abs(err_r)))
            acc[name]["sum_sq_rain"] += float(np.sum(err_r ** 2))
            p = np.clip(p_occ[smask, :][mr], 0.0, 1.0)
            y = occ_true[smask, :][mr].astype(np.int64)
            acc[name]["sum_brier"] += float(np.sum((p - y) ** 2))
            yhat = (p >= 0.5).astype(np.int64)
            acc[name]["tp"] += float(np.sum((yhat == 1) & (y == 1)))
            acc[name]["fn"] += float(np.sum((yhat == 0) & (y == 1)))
            acc[name]["fp"] += float(np.sum((yhat == 1) & (y == 0)))
            acc[name]["tn"] += float(np.sum((yhat == 0) & (y == 0)))


def finalize_regime_accumulators(acc: Dict[str, Dict[str, float]]) -> Dict[str, object]:
    """Finalitza les mètriques condicionades per règim."""
    out: Dict[str, object] = {}
    for name, a in acc.items():
        nt = max(float(a["n_temp"]), 0.0)
        nr = max(float(a["n_rain"]), 0.0)
        temp_mae = a["sum_abs_temp"] / nt if nt > 0 else float("nan")
        temp_rmse = math.sqrt(a["sum_sq_temp"] / nt) if nt > 0 else float("nan")
        rain_mae = a["sum_abs_rain"] / nr if nr > 0 else float("nan")
        rain_rmse = math.sqrt(a["sum_sq_rain"] / nr) if nr > 0 else float("nan")
        brier = a["sum_brier"] / nr if nr > 0 else float("nan")
        sedi_v = sedi(int(a["tp"]), int(a["fn"]), int(a["fp"]), int(a["tn"])) if nr > 0 else float("nan")
        out[name] = {
            "n_temp_points": int(nt),
            "n_rain_points": int(nr),
            "temp_mae": temp_mae,
            "temp_rmse": temp_rmse,
            "rain_mae": rain_mae,
            "rain_rmse": rain_rmse,
            "rain_brier": brier,
            "rain_sedi": sedi_v,
            "tp": int(a["tp"]),
            "fn": int(a["fn"]),
            "fp": int(a["fp"]),
            "tn": int(a["tn"]),
        }
    return out

# =============================================================================
# Prediccions baseline i skill scores
# =============================================================================



def init_baseline_accumulators(O: int) -> Dict[str, np.ndarray]:
    return {
        "n_temp": np.zeros(O, dtype=np.int64),
        "sum_abs_temp": np.zeros(O, dtype=np.float64),
        "sum_sq_temp": np.zeros(O, dtype=np.float64),
        "n_rain": np.zeros(O, dtype=np.int64),
        "sum_abs_rain": np.zeros(O, dtype=np.float64),
        "sum_sq_rain": np.zeros(O, dtype=np.float64),
        "sum_brier": np.zeros(O, dtype=np.float64),
        "tp": np.zeros(O, dtype=np.int64),
        "fn": np.zeros(O, dtype=np.int64),
        "fp": np.zeros(O, dtype=np.int64),
        "tn": np.zeros(O, dtype=np.int64),
    }



def update_baseline_accumulators(
    acc: Dict[str, np.ndarray],
    temp_pred: np.ndarray,
    rain_pred: np.ndarray,
    occ_prob: np.ndarray,
    temp_true: np.ndarray,
    rain_true: np.ndarray,
    occ_true: np.ndarray,
    mask_temp: np.ndarray,
    mask_rain: np.ndarray,
) -> None:
    O = temp_pred.shape[1]
    for k in range(O):
        vt = mask_temp[:, k] > 0.5
        if np.any(vt):
            acc["n_temp"][k] += int(vt.sum())
            err = temp_pred[vt, k] - temp_true[vt, k]
            acc["sum_abs_temp"][k] += float(np.sum(np.abs(err)))
            acc["sum_sq_temp"][k] += float(np.sum(err ** 2))

        vr = mask_rain[:, k] > 0.5
        if np.any(vr):
            acc["n_rain"][k] += int(vr.sum())
            errr = rain_pred[vr, k] - rain_true[vr, k]
            acc["sum_abs_rain"][k] += float(np.sum(np.abs(errr)))
            acc["sum_sq_rain"][k] += float(np.sum(errr ** 2))

            y01 = occ_true[vr, k].astype(int)
            p = np.clip(occ_prob[vr, k], 0.0, 1.0)
            acc["sum_brier"][k] += float(np.sum((p - y01) ** 2))

            yhat = (p >= 0.5).astype(int)
            acc["tp"][k] += int(np.sum((yhat == 1) & (y01 == 1)))
            acc["fn"][k] += int(np.sum((yhat == 0) & (y01 == 1)))
            acc["fp"][k] += int(np.sum((yhat == 1) & (y01 == 0)))
            acc["tn"][k] += int(np.sum((yhat == 0) & (y01 == 0)))



def finalize_baseline_metrics(
    acc: Dict[str, np.ndarray],
    steps_per_hour: int,
    name: str,
) -> Dict[str, object]:
    def safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        out = np.full_like(a, np.nan, dtype=np.float64)
        m = b > 0
        out[m] = a[m] / b[m]
        return out

    temp_mae = safe_div(acc["sum_abs_temp"], acc["n_temp"])
    temp_rmse = np.sqrt(safe_div(acc["sum_sq_temp"], acc["n_temp"]))
    rain_mae = safe_div(acc["sum_abs_rain"], acc["n_rain"])
    rain_rmse = np.sqrt(safe_div(acc["sum_sq_rain"], acc["n_rain"]))
    brier = safe_div(acc["sum_brier"], acc["n_rain"])
    sedi_by_k = np.array(
        [
            sedi(int(acc["tp"][i]), int(acc["fn"][i]), int(acc["fp"][i]), int(acc["tn"][i]))
            if acc["n_rain"][i] > 0 else np.nan
            for i in range(len(acc["n_rain"]))
        ],
        dtype=np.float64,
    )

    return {
        "summary": {
            "baseline_name": name,
            "temp_mae_mean": safe_nanmean(temp_mae),
            "temp_rmse_mean": safe_nanmean(temp_rmse),
            "rain_mae_mean": safe_nanmean(rain_mae),
            "rain_rmse_mean": safe_nanmean(rain_rmse),
            "rain_brier_mean": safe_nanmean(brier),
            "rain_sedi_mean": safe_nanmean(sedi_by_k),
        },
        "by_lead": {
            "lead_hours": ((np.arange(len(temp_mae), dtype=np.float64) + 1.0) / float(steps_per_hour)).tolist(),
            "temp_mae": temp_mae.tolist(),
            "temp_rmse": temp_rmse.tolist(),
            "rain_mae": rain_mae.tolist(),
            "rain_rmse": rain_rmse.tolist(),
            "rain_brier": brier.tolist(),
            "rain_sedi": sedi_by_k.tolist(),
        },
    }



def lower_is_better_skill(model_values: np.ndarray, ref_values: np.ndarray) -> np.ndarray:
    model_values = np.asarray(model_values, dtype=np.float64)
    ref_values = np.asarray(ref_values, dtype=np.float64)
    out = np.full(model_values.shape, np.nan, dtype=np.float64)
    m = np.isfinite(model_values) & np.isfinite(ref_values) & (np.abs(ref_values) > 1e-12)
    out[m] = 1.0 - (model_values[m] / ref_values[m])
    return out



def last_positive_skill_horizon_hours(skill: np.ndarray, steps_per_hour: int) -> float:
    skill = np.asarray(skill, dtype=np.float64)
    good = np.where(np.isfinite(skill) & (skill > 0.0))[0]
    if good.size == 0:
        return float("nan")
    return float((good[-1] + 1) / float(steps_per_hour))



def build_skill_scores(
    model_by_lead: Dict[str, object],
    baselines: Dict[str, Dict[str, object]],
    steps_per_hour: int,
) -> Dict[str, object]:
    temp_mae = np.asarray(model_by_lead["temp_mae"], dtype=np.float64)
    rain_mae = np.asarray(model_by_lead["rain_mae"], dtype=np.float64)
    rain_brier = np.asarray(model_by_lead["rain_brier"], dtype=np.float64)

    p_by = baselines["persistence"]["by_lead"]
    c_by = baselines["climatology"]["by_lead"]
    d_by = baselines["daily_persistence_temp"]["by_lead"]
    m_by = baselines["rain_markov"]["by_lead"]

    temp_skill_vs_persistence = lower_is_better_skill(temp_mae, np.asarray(p_by["temp_mae"], dtype=np.float64))
    temp_skill_vs_climatology = lower_is_better_skill(temp_mae, np.asarray(c_by["temp_mae"], dtype=np.float64))
    temp_skill_vs_daily = lower_is_better_skill(temp_mae, np.asarray(d_by["temp_mae"], dtype=np.float64))

    rain_amount_skill_vs_markov = lower_is_better_skill(rain_mae, np.asarray(m_by["rain_mae"], dtype=np.float64))
    rain_bss_vs_markov = lower_is_better_skill(rain_brier, np.asarray(m_by["rain_brier"], dtype=np.float64))

    return {
        "summary": {
            "temp_mae_skill_vs_persistence_mean": safe_nanmean(temp_skill_vs_persistence),
            "temp_mae_skill_vs_climatology_mean": safe_nanmean(temp_skill_vs_climatology),
            "temp_mae_skill_vs_daily_persistence_mean": safe_nanmean(temp_skill_vs_daily),
            "rain_amount_mae_skill_vs_markov_mean": safe_nanmean(rain_amount_skill_vs_markov),
            "rain_occurrence_bss_vs_markov_mean": safe_nanmean(rain_bss_vs_markov),
            "temp_last_positive_skill_hours_vs_daily_persistence": last_positive_skill_horizon_hours(temp_skill_vs_daily, steps_per_hour),
            "rain_last_positive_bss_hours_vs_markov": last_positive_skill_horizon_hours(rain_bss_vs_markov, steps_per_hour),
        },
        "by_lead": {
            "lead_hours": model_by_lead["lead_hours"],
            "temp_mae_skill_vs_persistence": temp_skill_vs_persistence.tolist(),
            "temp_mae_skill_vs_climatology": temp_skill_vs_climatology.tolist(),
            "temp_mae_skill_vs_daily_persistence": temp_skill_vs_daily.tolist(),
            "rain_amount_mae_skill_vs_markov": rain_amount_skill_vs_markov.tolist(),
            "rain_occurrence_bss_vs_markov": rain_bss_vs_markov.tolist(),
        },
    }


# =============================================================================
# Avaluació
# =============================================================================


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    cfg: TrainConfig,
    temp_mean: float,
    temp_std: float,
    rain_cap: float,
    artifacts: Dict[str, object],
) -> Dict[str, object]:
    """
    Avalua un membre individual amb mètriques adequades per a nowcasting local.

    Inclou mètriques deterministes, probabilistes, skill contra baselines,
    CRPS de pluja mostrejat per lead time, acumulacions 1/2/3 h, calibratge
    probabilista i verificació condicionada per règim.
    """
    model.eval()

    O = int(artifacts["horizon_steps"])
    steps_per_hour = int(artifacts["steps_per_hour"])
    rain_event_threshold = float(cfg.rain_event_threshold)
    heavy_threshold = max(rain_event_threshold, finite_or(artifacts.get("rain_pos_extreme_thr"), rain_event_threshold))
    horizons_steps = [
        max(1, int(round(float(h) * steps_per_hour)))
        for h in cfg.accumulation_horizons_hours
        if float(h) > 0.0 and float(h) <= float(cfg.horizon_hours) + 1e-9
    ]

    n_temp = np.zeros(O, dtype=np.int64)
    sum_abs_temp = np.zeros(O, dtype=np.float64)
    sum_sq_temp = np.zeros(O, dtype=np.float64)
    sum_nll_temp = np.zeros(O, dtype=np.float64)
    sum_crps_temp = np.zeros(O, dtype=np.float64)

    n_rain = np.zeros(O, dtype=np.int64)
    sum_abs_rain = np.zeros(O, dtype=np.float64)
    sum_sq_rain = np.zeros(O, dtype=np.float64)
    sum_nll_rain = np.zeros(O, dtype=np.float64)
    sum_brier = np.zeros(O, dtype=np.float64)
    tp = np.zeros(O, dtype=np.int64)
    fn = np.zeros(O, dtype=np.int64)
    fp = np.zeros(O, dtype=np.int64)
    tn = np.zeros(O, dtype=np.int64)

    # CRPS de pluja per lead time, mostrejat sobre un subconjunt per mantenir cost raonable en validació.
    sum_crps_rain = np.zeros(O, dtype=np.float64)
    n_crps_rain = np.zeros(O, dtype=np.int64)
    crps_rain_points = 0
    losses: List[float] = []

    baseline_accs = {
        "persistence": init_baseline_accumulators(O),
        "climatology": init_baseline_accumulators(O),
        "daily_persistence_temp": init_baseline_accumulators(O),
        "rain_markov": init_baseline_accumulators(O),
    }
    temp_coverage_acc = init_temp_coverage_accumulators(O)
    reliability_acc = init_reliability_accumulators(O, n_bins=int(getattr(cfg, "reliability_bins", 10)))
    accum_acc = init_accumulation_diagnostics(horizons_steps)
    regime_acc = init_regime_accumulators(["all", "dry_window", "already_raining", "rain_onset", "heavy_future"])

    for batch in loader:
        batch = to_device(batch, device)

        temp_baseline_scaled = (batch["temp_baseline_raw"] - temp_mean) / temp_std
        last_temp_scaled = (batch["last_temp_raw"] - temp_mean) / temp_std

        out = model(
            batch["x_past"],
            batch["t_future"],
            temp_baseline_scaled=temp_baseline_scaled,
            rain_occ_baseline=batch["rain_occ_baseline"],
            rain_mu_baseline=batch["rain_mu_baseline"],
            last_temp_scaled=last_temp_scaled,
            last_rain_raw=batch["last_rain_raw"],
            future_cosz=batch["future_cosz"],
            future_toa=batch["future_toa"],
        )

        mT = batch["mask_temp"]
        mR = batch["mask_rain"]

        lt_el = gaussian_nll(batch["y_temp"], out["temp_mu"], out["temp_sigma"], eps=cfg.eps)
        lr_el = zigamma_nll(
            batch["y_rain"],
            out["rain_logit_p0"],
            out["rain_k"],
            out["rain_theta"],
            eps=cfg.eps,
            cap=rain_cap,
            pos_weight=cfg.rain_pos_weight,
            intensity_alpha=cfg.rain_intensity_alpha,
        )

        temp_mu_raw = out["temp_mu"] * temp_std + temp_mean
        temp_sigma_raw = out["temp_sigma"] * temp_std
        rain_mu = expected_rain_from_params(out["rain_logit_p0"], out["rain_k"], out["rain_theta"])
        p_occ = 1.0 - torch.sigmoid(out["rain_logit_p0"])

        loss_temp = masked_mean(lt_el, mT)
        loss_rain = masked_mean(lr_el, mR)
        loss_rain_occ = masked_mean(
            rain_occurrence_bce_loss(
                out["rain_logit_p0"],
                batch["y_rain_occ"],
                eps=cfg.eps,
                focal_gamma=cfg.rain_occ_focal_gamma,
                pos_weight=cfg.rain_occ_pos_weight,
            ),
            mR,
        )
        loss_smooth = temperature_smoothness_loss(temp_mu_raw, mT)
        loss_anchor = temperature_anchor_loss(out["temp_mu"], temp_baseline_scaled, mT)
        loss_accum = accumulation_consistency_loss(
            rain_mu=rain_mu,
            y_rain=batch["y_rain"],
            mask_rain=mR,
            horizon_steps=horizons_steps,
        )

        loss = (
            loss_temp
            + cfg.rain_loss_weight * loss_rain
            + cfg.rain_occurrence_bce_weight * loss_rain_occ
            + cfg.physics_temp_smooth_weight * loss_smooth
            + cfg.physics_temp_anchor_weight * loss_anchor
            + cfg.physics_rain_accum_weight * loss_accum
        )
        losses.append(float(loss.item()))

        temp_true_raw = batch["y_temp_raw"]
        if cfg.compute_crps:
            crpsT = crps_gaussian(temp_mu_raw, temp_sigma_raw, temp_true_raw, eps=cfg.eps)
        else:
            crpsT = None

        tmu = temp_mu_raw.detach().cpu().numpy()
        tsig = temp_sigma_raw.detach().cpu().numpy()
        ttr = temp_true_raw.detach().cpu().numpy()
        rmu = rain_mu.detach().cpu().numpy()
        rtr = batch["y_rain"].detach().cpu().numpy()
        occp = p_occ.detach().cpu().numpy()
        occ_true = batch["y_rain_occ"].detach().cpu().numpy()
        mTn = mT.detach().cpu().numpy()
        mRn = mR.detach().cpu().numpy()
        lt_np = lt_el.detach().cpu().numpy()
        lr_np = lr_el.detach().cpu().numpy()
        crpsT_np = crpsT.detach().cpu().numpy() if crpsT is not None else None

        # Baselines del batch, ja precalculades pel dataset.
        temp_persistence = batch["last_temp_raw"].detach().cpu().numpy()[:, None].repeat(O, axis=1)
        rain_persistence = batch["last_rain_raw"].detach().cpu().numpy()[:, None].repeat(O, axis=1)
        occ_persistence = (rain_persistence >= rain_event_threshold).astype(np.float64)
        temp_climatology = batch["temp_baseline_raw"].detach().cpu().numpy()
        rain_markov_mu = batch["rain_mu_baseline"].detach().cpu().numpy()
        rain_markov_occ = batch["rain_occ_baseline"].detach().cpu().numpy()
        temp_daily = temp_climatology.copy()

        update_baseline_accumulators(baseline_accs["persistence"], temp_persistence, rain_persistence, occ_persistence, ttr, rtr, occ_true, mTn, mRn)
        update_baseline_accumulators(baseline_accs["climatology"], temp_climatology, np.zeros_like(rain_markov_mu), np.zeros_like(rain_markov_occ), ttr, rtr, occ_true, mTn, np.zeros_like(mRn))
        update_baseline_accumulators(baseline_accs["daily_persistence_temp"], temp_daily, np.zeros_like(rain_markov_mu), np.zeros_like(rain_markov_occ), ttr, rtr, occ_true, mTn, np.zeros_like(mRn))
        update_baseline_accumulators(baseline_accs["rain_markov"], np.zeros_like(temp_daily), rain_markov_mu, rain_markov_occ, ttr, rtr, occ_true, np.zeros_like(mTn), mRn)

        update_temp_coverage_accumulators(temp_coverage_acc, tmu, tsig, ttr, mTn)
        update_reliability_accumulators(reliability_acc, occp, occ_true, mRn)
        update_accumulation_diagnostics(accum_acc, rmu, rtr, occp, mRn, event_threshold=rain_event_threshold)

        last_rain_np = batch["last_rain_raw"].detach().cpu().numpy()
        future_sum = np.sum(np.where(mRn > 0.5, rtr, 0.0), axis=1)
        future_max = np.max(np.where(mRn > 0.5, rtr, 0.0), axis=1)
        regime_masks = {
            "all": np.ones_like(last_rain_np, dtype=bool),
            "dry_window": (last_rain_np < rain_event_threshold) & (future_sum < rain_event_threshold),
            "already_raining": last_rain_np >= rain_event_threshold,
            "rain_onset": (last_rain_np < rain_event_threshold) & (future_max >= rain_event_threshold),
            "heavy_future": future_max >= heavy_threshold,
        }
        update_regime_accumulators(regime_acc, regime_masks, tmu, ttr, rmu, rtr, occp, occ_true, mTn, mRn)

        for k in range(O):
            vt = mTn[:, k] > 0.5
            if np.any(vt):
                n_temp[k] += int(vt.sum())
                err = tmu[vt, k] - ttr[vt, k]
                sum_abs_temp[k] += float(np.sum(np.abs(err)))
                sum_sq_temp[k] += float(np.sum(err ** 2))
                sum_nll_temp[k] += float(np.sum(lt_np[vt, k]))
                if crpsT_np is not None:
                    sum_crps_temp[k] += float(np.sum(crpsT_np[vt, k]))

            vr = mRn[:, k] > 0.5
            if np.any(vr):
                n_rain[k] += int(vr.sum())
                errr = rmu[vr, k] - rtr[vr, k]
                sum_abs_rain[k] += float(np.sum(np.abs(errr)))
                sum_sq_rain[k] += float(np.sum(errr ** 2))
                sum_nll_rain[k] += float(np.sum(lr_np[vr, k]))
                y01 = occ_true[vr, k].astype(int)
                p = np.clip(occp[vr, k], 0.0, 1.0)
                sum_brier[k] += float(np.sum((p - y01) ** 2))
                yhat = (p >= 0.5).astype(int)
                tp[k] += int(np.sum((yhat == 1) & (y01 == 1)))
                fn[k] += int(np.sum((yhat == 0) & (y01 == 1)))
                fp[k] += int(np.sum((yhat == 1) & (y01 == 0)))
                tn[k] += int(np.sum((yhat == 0) & (y01 == 0)))

        if cfg.compute_crps and (cfg.crps_rain_max_points > 0) and (crps_rain_points < cfg.crps_rain_max_points):
            r_valid = (mR > 0.5).view(-1)
            if torch.any(r_valid):
                idx_all = torch.where(r_valid)[0]
                remaining = int(cfg.crps_rain_max_points) - crps_rain_points
                take = int(min(remaining, idx_all.numel(), 512))
                if take > 0:
                    perm = torch.randperm(idx_all.numel(), device=idx_all.device)[:take]
                    sel = idx_all[perm]
                    y_sel = batch["y_rain"].view(-1)[sel]
                    p0_sel = torch.sigmoid(out["rain_logit_p0"].view(-1)[sel])
                    k_sel = out["rain_k"].view(-1)[sel]
                    th_sel = out["rain_theta"].view(-1)[sel]
                    samp = sample_zigamma(p0_sel, k_sel, th_sel, n_samples=int(cfg.crps_rain_n_samples))
                    crps = fair_crps_from_samples(samp, y_sel)
                    crps_np = crps.detach().cpu().numpy()
                    lead_np = (sel.detach().cpu().numpy() % O).astype(np.int64)
                    for kk, vv in zip(lead_np, crps_np):
                        sum_crps_rain[kk] += float(vv)
                        n_crps_rain[kk] += 1
                    crps_rain_points += int(crps.numel())

    def safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        out = np.full_like(a, np.nan, dtype=np.float64)
        m = b > 0
        out[m] = a[m] / b[m]
        return out

    temp_mae = safe_div(sum_abs_temp, n_temp)
    temp_rmse = np.sqrt(safe_div(sum_sq_temp, n_temp))
    temp_nll = safe_div(sum_nll_temp, n_temp)
    temp_crps = safe_div(sum_crps_temp, n_temp) if cfg.compute_crps else np.full(O, np.nan)
    rain_mae = safe_div(sum_abs_rain, n_rain)
    rain_rmse = np.sqrt(safe_div(sum_sq_rain, n_rain))
    rain_nll = safe_div(sum_nll_rain, n_rain)
    rain_brier = safe_div(sum_brier, n_rain)
    rain_sedi = np.array([sedi(int(tp[i]), int(fn[i]), int(fp[i]), int(tn[i])) if n_rain[i] > 0 else np.nan for i in range(O)], dtype=np.float64)
    rain_crps = safe_div(sum_crps_rain, n_crps_rain) if cfg.compute_crps else np.full(O, np.nan)

    by_lead = {
        "lead_hours": ((np.arange(O, dtype=np.float64) + 1.0) / float(steps_per_hour)).tolist(),
        "temp_mae": temp_mae.tolist(),
        "temp_rmse": temp_rmse.tolist(),
        "temp_nll": temp_nll.tolist(),
        "temp_crps": temp_crps.tolist(),
        "rain_mae": rain_mae.tolist(),
        "rain_rmse": rain_rmse.tolist(),
        "rain_nll": rain_nll.tolist(),
        "rain_brier": rain_brier.tolist(),
        "rain_sedi": rain_sedi.tolist(),
        "rain_crps": rain_crps.tolist(),
        "rain_crps_n": n_crps_rain.tolist(),
        "n_temp": n_temp.tolist(),
        "n_rain": n_rain.tolist(),
    }

    baselines = {
        "persistence": finalize_baseline_metrics(baseline_accs["persistence"], steps_per_hour, "persistence"),
        "climatology": finalize_baseline_metrics(baseline_accs["climatology"], steps_per_hour, "climatology"),
        "daily_persistence_temp": finalize_baseline_metrics(baseline_accs["daily_persistence_temp"], steps_per_hour, "daily_persistence_temp"),
        "rain_markov": finalize_baseline_metrics(baseline_accs["rain_markov"], steps_per_hour, "rain_markov"),
    }
    skill_scores = build_skill_scores(by_lead, baselines, steps_per_hour)
    diagnostics = {
        "temperature_coverage": finalize_temp_coverage_accumulators(temp_coverage_acc, steps_per_hour),
        "rain_reliability": finalize_reliability_accumulators(reliability_acc, steps_per_hour),
        "rain_accumulation": finalize_accumulation_diagnostics(accum_acc, steps_per_hour),
        "regime_metrics": finalize_regime_accumulators(regime_acc),
    }

    summary = {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "temp_mae_mean": safe_nanmean(temp_mae),
        "temp_rmse_mean": safe_nanmean(temp_rmse),
        "temp_nll_mean": safe_nanmean(temp_nll),
        "temp_crps_mean": safe_nanmean(temp_crps) if cfg.compute_crps else float("nan"),
        "rain_mae_mean": safe_nanmean(rain_mae),
        "rain_rmse_mean": safe_nanmean(rain_rmse),
        "rain_nll_mean": safe_nanmean(rain_nll),
        "rain_brier_mean": safe_nanmean(rain_brier),
        "rain_sedi_mean": safe_nanmean(rain_sedi),
        "rain_crps_mean_sampled": safe_nanmean(rain_crps) if cfg.compute_crps else float("nan"),
        "crps_rain_mean_sampled": safe_nanmean(rain_crps) if cfg.compute_crps else float("nan"),
        "crps_rain_points": int(crps_rain_points),
        "temp_coverage80_mean": diagnostics["temperature_coverage"]["summary"]["coverage80_mean"],
        "temp_coverage90_mean": diagnostics["temperature_coverage"]["summary"]["coverage90_mean"],
        "rain_reliability_ece": diagnostics["rain_reliability"]["summary"]["expected_calibration_error"],
        "rain_accum_mae_mean": diagnostics["rain_accumulation"]["summary"]["rain_accum_mae_mean"],
    }

    return {
        "summary": summary,
        "by_lead": by_lead,
        "baselines": baselines,
        "skill_scores": skill_scores,
        "diagnostics": diagnostics,
    }



def validation_selection_score(metrics: Dict[str, object], cfg: TrainConfig) -> float:
    """
    Score híbrid per seleccionar el millor model de validació.

    Menor és millor. És robust a mètriques no disponibles: si CRPS està
    desactivat o SEDI queda indefinit per manca d'esdeveniments, no bloqueja
    l'early stopping ni impedeix desar el millor checkpoint.
    """
    s = metrics["summary"]
    return float(
        finite_or(s.get("temp_mae_mean"), 1e6)
        + cfg.val_temp_crps_weight * finite_or(s.get("temp_crps_mean"), 0.0)
        + cfg.val_rain_mae_weight * finite_or(s.get("rain_mae_mean"), 1e6)
        + cfg.val_rain_brier_weight * finite_or(s.get("rain_brier_mean"), 1e6)
        + cfg.val_rain_nll_weight * finite_or(s.get("rain_nll_mean"), 1e6)
        + cfg.val_rain_crps_weight * finite_or(s.get("rain_crps_mean_sampled"), 0.0)
        - cfg.val_rain_sedi_weight * finite_or(s.get("rain_sedi_mean"), 0.0)
    )


# =============================================================================
# Entrenament
# =============================================================================



def _resolve_initial_checkpoint(init_model_dir: Optional[str], model_name: str) -> Optional[str]:
    """
    Troba el checkpoint inicial compatible per a un membre de l'ensemble.

    En fine-tuning v8 es parteix normalment de la v7 CC-compact windfix. Si
    l'ensemble té els mateixos seeds, el nom coincideix: best_model_seed42.pt,
    ..., best_model_seed51.pt. Com a fallback s'accepta best_model.pt.
    """
    if not init_model_dir:
        return None
    base = os.path.abspath(os.path.expanduser(str(init_model_dir)))
    candidates = [
        os.path.join(base, model_name),
        os.path.join(base, "best_model.pt"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        "No he trobat cap checkpoint inicial compatible. "
        f"init_model_dir={base}, model_name={model_name}"
    )


def _freeze_model_for_rain_finetune(model: nn.Module) -> Tuple[int, int]:
    """
    Congela tot el model excepte els mòduls específics de precipitació.

    Objectiu: mantenir intacta la predicció de temperatura de la v7 CC i reajustar
    només ocurrència/quantitat/règims/acumulació de pluja. L'encoder compartit,
    les features temporals i la branca tèrmica queden congelats.
    """
    for p in model.parameters():
        p.requires_grad_(False)

    trainable_prefixes = (
        "rain_recent_mlp",
        "rain_query_mlp",
        "rain_fmm_gate",
        "rain_future_encoder",
        "rain_out_norm",
        "rain_occ_head",
        "rain_amt_head",
        "rain_regime_head",
        "rain_accum_direct_head",
        "rain_occ_residual_gate",
        "rain_amt_residual_gate",
    )
    total = 0
    trainable = 0
    for name, p in model.named_parameters():
        total += p.numel()
        if name.startswith(trainable_prefixes):
            p.requires_grad_(True)
            trainable += p.numel()
    if trainable == 0:
        raise RuntimeError("No s'ha activat cap paràmetre entrenable per al fine-tuning de pluja.")
    return int(trainable), int(total)


def _train_single(cfg: TrainConfig, outdir: str, model_name: str) -> Dict[str, object]:
    seed_everything(cfg.seed)

    df, meta = load_dataset(cfg)
    splits = split_by_time(df, cfg)
    datasets, artifacts = build_datasets(splits, cfg)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Device: {device}")

    configure_gpu_perf(cfg)
    loaders = make_loaders(datasets, cfg, use_cuda)

    model_base = HybridResidualNowcaster(
        in_channels=len(artifacts["feature_cols"]) + len(artifacts["time_cols"]),
        time_feat_dim=len(artifacts["time_cols"]),
        history_steps=int(artifacts["history_steps"]),
        horizon_steps=int(artifacts["horizon_steps"]),
        steps_per_hour=int(artifacts["steps_per_hour"]),
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        dropout=cfg.dropout,
        patch_len_hours=cfg.patch_len_hours,
        patch_stride_frac=cfg.patch_stride_frac,
        recent_window_hours=cfg.recent_window_hours,
        recent_patch_len_hours=cfg.recent_patch_len_hours,
        recent_patch_stride_frac=cfg.recent_patch_stride_frac,
        timemixer_scales=cfg.timemixer_scales,
        timemixer_ma_kernel=cfg.timemixer_ma_kernel,
        residual_temp_max_sigma=cfg.residual_temp_max_sigma,
        residual_occ_logit_max=cfg.residual_occ_logit_max,
        residual_log_wet_mean_max=cfg.residual_log_wet_mean_max,
        temp_feature_index=int(artifacts["temp_feature_index"]),
        rain_feature_index=int(artifacts["rain_feature_index"]),
    ).to(device)

    init_ckpt_path = _resolve_initial_checkpoint(getattr(cfg, "init_model_dir", None), model_name)
    if init_ckpt_path is not None:
        print(f"[INFO] Carregant pesos inicials per fine-tuning: {init_ckpt_path}")
        ckpt_init = torch.load(init_ckpt_path, map_location=device)
        state_init = ckpt_init.get("model_state", ckpt_init)
        missing, unexpected = model_base.load_state_dict(state_init, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                "El checkpoint inicial no és estrictament compatible amb l'arquitectura v8. "
                f"missing={missing}, unexpected={unexpected}"
            )

    if bool(getattr(cfg, "rain_finetune_only", False)):
        n_trainable, n_total = _freeze_model_for_rain_finetune(model_base)
        print(
            "[INFO] Mode v8 rain_finetune_only activat: "
            f"{n_trainable:,}/{n_total:,} paràmetres entrenables "
            f"({100.0 * n_trainable / max(n_total, 1):.2f}%)."
        )

    model = model_base
    if cfg.compile:
        try:
            model = torch.compile(model_base, mode=cfg.compile_mode)
            print(f"[INFO] torch.compile activat (mode='{cfg.compile_mode}').")
        except Exception as e:
            print(f"[AVÍS] No s'ha pogut aplicar torch.compile ({type(e).__name__}: {e}).")
            model = model_base

    trainable_parameters = [p for p in model_base.parameters() if p.requires_grad]
    if len(trainable_parameters) == 0:
        raise RuntimeError("No hi ha cap paràmetre entrenable.")
    optimizer = torch.optim.AdamW(trainable_parameters, lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, cfg.max_epochs))

    temp_mean = float(artifacts["temp_mean"])
    temp_std = float(artifacts["temp_std"])
    rain_cap = float(artifacts["rain_cap_99p9"])
    rain_heavy_threshold = float(artifacts.get("rain_heavy_threshold", artifacts.get("rain_pos_extreme_thr", cfg.rain_event_threshold)))

    best_score = float("inf")
    best_path = os.path.join(outdir, model_name)
    bad_epochs = 0

    epoch_iter = range(1, cfg.max_epochs + 1)
    if tqdm is not None:
        epoch_iter = tqdm(epoch_iter, total=cfg.max_epochs, desc=f"Entrenament ({model_name})", dynamic_ncols=True)

    t_start_all = time.time()
    for epoch in epoch_iter:
        model.train()
        train_losses = []
        lr_now = float(optimizer.param_groups[0]["lr"])

        batch_iter = loaders["train"]
        if tqdm is not None:
            batch_iter = tqdm(batch_iter, desc=f"Època {epoch}/{cfg.max_epochs} (train)", leave=False, dynamic_ncols=True)

        for step, batch in enumerate(batch_iter, start=1):
            batch = to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)

            temp_baseline_scaled = (batch["temp_baseline_raw"] - temp_mean) / temp_std
            last_temp_scaled = (batch["last_temp_raw"] - temp_mean) / temp_std

            out = model(
                batch["x_past"],
                batch["t_future"],
                temp_baseline_scaled=temp_baseline_scaled,
                rain_occ_baseline=batch["rain_occ_baseline"],
                rain_mu_baseline=batch["rain_mu_baseline"],
                last_temp_scaled=last_temp_scaled,
                last_rain_raw=batch["last_rain_raw"],
                future_cosz=batch["future_cosz"],
                future_toa=batch["future_toa"],
            )

            lt_el = gaussian_nll(batch["y_temp"], out["temp_mu"], out["temp_sigma"], eps=cfg.eps)
            lr_el = zigamma_nll(
                batch["y_rain"],
                out["rain_logit_p0"],
                out["rain_k"],
                out["rain_theta"],
                eps=cfg.eps,
                cap=rain_cap,
                pos_weight=cfg.rain_pos_weight,
                intensity_alpha=cfg.rain_intensity_alpha,
            )

            temp_mu_raw = out["temp_mu"] * temp_std + temp_mean
            rain_mu = expected_rain_from_params(out["rain_logit_p0"], out["rain_k"], out["rain_theta"])

            loss_temp = masked_mean(lt_el, batch["mask_temp"])
            loss_rain = masked_mean(lr_el, batch["mask_rain"])
            loss_rain_occ = masked_mean(
                rain_occurrence_bce_loss(
                    out["rain_logit_p0"],
                    batch["y_rain_occ"],
                    eps=cfg.eps,
                    focal_gamma=cfg.rain_occ_focal_gamma,
                    pos_weight=cfg.rain_occ_pos_weight,
                ),
                batch["mask_rain"],
            )
            loss_smooth = temperature_smoothness_loss(temp_mu_raw, batch["mask_temp"])
            loss_anchor = temperature_anchor_loss(out["temp_mu"], temp_baseline_scaled, batch["mask_temp"])
            loss_accum = accumulation_consistency_loss(
                rain_mu=rain_mu,
                y_rain=batch["y_rain"],
                mask_rain=batch["mask_rain"],
                horizon_steps=[max(1, int(round(float(h) * artifacts["steps_per_hour"]))) for h in cfg.accumulation_horizons_hours],
            )
            loss_temp_trend = temperature_trend_consistency_loss(
                temp_mu_raw=temp_mu_raw,
                temp_trend_prior_raw=batch["temp_trend_prior_raw"],
                mask_temp=batch["mask_temp"],
                steps_per_hour=float(artifacts["steps_per_hour"]),
            )
            loss_temp_diurnal = temperature_diurnal_trend_loss(
                temp_mu_raw=temp_mu_raw,
                future_toa=batch["future_toa"],
                mask_temp=batch["mask_temp"],
                steps_per_hour=float(artifacts["steps_per_hour"]),
            )
            loss_rain_heavy = rain_heavy_amount_loss(
                rain_mu=rain_mu,
                y_rain=batch["y_rain"],
                mask_rain=batch["mask_rain"],
                heavy_threshold=rain_heavy_threshold,
                under_weight=cfg.rain_heavy_under_weight,
                intensity_power=cfg.rain_heavy_intensity_power,
            )
            horizon_steps_train = [max(1, int(round(float(h) * artifacts["steps_per_hour"]))) for h in cfg.accumulation_horizons_hours]

            loss_rain_heavy_accum = rain_heavy_accumulation_loss(
                rain_mu=rain_mu,
                y_rain=batch["y_rain"],
                mask_rain=batch["mask_rain"],
                horizon_steps=horizon_steps_train,
                step_heavy_threshold=rain_heavy_threshold,
                under_weight=cfg.rain_heavy_under_weight,
            )
            loss_rain_regime_aux = rain_regime_auxiliary_loss(
                regime_logits=out["rain_regime_logits"],
                y_rain=batch["y_rain"],
                mask_rain=batch["mask_rain"],
                last_rain_raw=batch["last_rain_raw"],
                horizon_steps=horizon_steps_train,
                event_threshold=cfg.rain_event_threshold,
                onset_pos_weight=cfg.rain_onset_pos_weight,
                continue_pos_weight=cfg.rain_continue_pos_weight,
                decay_pos_weight=cfg.rain_decay_pos_weight,
            )
            loss_rain_accum_direct = rain_accum_direct_loss(
                accum_direct_log1p=out["rain_accum_direct_log1p"],
                rain_mu=rain_mu,
                y_rain=batch["y_rain"],
                mask_rain=batch["mask_rain"],
                horizon_steps=horizon_steps_train,
                under_weight=cfg.rain_accum_direct_under_weight,
            )
            loss_rain_regime_weighted = rain_regime_weighted_step_loss(
                rain_mu=rain_mu,
                y_rain=batch["y_rain"],
                mask_rain=batch["mask_rain"],
                last_rain_raw=batch["last_rain_raw"],
                event_threshold=cfg.rain_event_threshold,
                preonset_weight=cfg.rain_preonset_sample_weight,
                decay_weight=cfg.rain_decay_sample_weight,
            )
            loss_temp_rain_cooling = temperature_rain_cooling_consistency_loss(
                temp_mu_raw=temp_mu_raw,
                rain_mu=rain_mu,
                p_occ=1.0 - torch.sigmoid(out["rain_logit_p0"]),
                mask_temp=batch["mask_temp"],
                mask_rain=batch["mask_rain"],
                future_toa=batch["future_toa"],
                steps_per_hour=float(artifacts["steps_per_hour"]),
                event_threshold=cfg.rain_event_threshold,
            )

            if bool(getattr(cfg, "rain_finetune_only", False)):
                # Fine-tuning v8: la temperatura i l'encoder compartit estan congelats.
                # No incloem pèrdues tèrmiques ni la pèrdua temp-rain-cooling, perquè
                # podrien forçar artificialment la pluja per compensar una temperatura
                # que no pot moure's. L'objectiu és reajustar només ocurrència,
                # quantitat, acumulació i règims de precipitació.
                loss = (
                    cfg.rain_loss_weight * loss_rain
                    + cfg.rain_occurrence_bce_weight * loss_rain_occ
                    + cfg.physics_rain_accum_weight * loss_accum
                    + cfg.rain_heavy_step_weight * loss_rain_heavy
                    + cfg.rain_heavy_accum_weight * loss_rain_heavy_accum
                    + cfg.rain_regime_aux_weight * loss_rain_regime_aux
                    + cfg.rain_accum_direct_weight * loss_rain_accum_direct
                    + 0.50 * cfg.rain_regime_aux_weight * loss_rain_regime_weighted
                )
            else:
                loss = (
                    loss_temp
                    + cfg.rain_loss_weight * loss_rain
                    + cfg.rain_occurrence_bce_weight * loss_rain_occ
                    + cfg.physics_temp_smooth_weight * loss_smooth
                    + cfg.physics_temp_anchor_weight * loss_anchor
                    + cfg.physics_rain_accum_weight * loss_accum
                    + cfg.physics_temp_trend_weight * loss_temp_trend
                    + cfg.physics_temp_diurnal_trend_weight * loss_temp_diurnal
                    + cfg.rain_heavy_step_weight * loss_rain_heavy
                    + cfg.rain_heavy_accum_weight * loss_rain_heavy_accum
                    + cfg.rain_regime_aux_weight * loss_rain_regime_aux
                    + cfg.rain_accum_direct_weight * loss_rain_accum_direct
                    + 0.50 * cfg.rain_regime_aux_weight * loss_rain_regime_weighted
                    + cfg.physics_temp_rain_cooling_weight * loss_temp_rain_cooling
                )

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Loss no finita a l'època {epoch}, batch {step}: {float(loss.detach().cpu().item())}"
                )

            loss.backward()
            nn.utils.clip_grad_norm_(trainable_parameters, cfg.grad_clip)
            optimizer.step()
            train_losses.append(float(loss.item()))

            if tqdm is not None and hasattr(batch_iter, "set_postfix"):
                if step % 50 == 0 or step == 1:
                    batch_iter.set_postfix({"loss": f"{np.mean(train_losses):.4f}", "lr": f"{lr_now:.2e}"})

        scheduler.step()

        train_loss = float(np.mean(train_losses))
        val_metrics = evaluate(model, loaders["val"], device, cfg, temp_mean, temp_std, rain_cap, artifacts)
        val_loss = float(val_metrics["summary"]["loss"])
        val_score = validation_selection_score(val_metrics, cfg)

        msg = (
            f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} "
            f"| val_score={val_score:.4f} | T_MAE={val_metrics['summary']['temp_mae_mean']:.3f} "
            f"| R_MAE={val_metrics['summary']['rain_mae_mean']:.3f} "
            f"| Brier={val_metrics['summary']['rain_brier_mean']:.3f} "
            f"| SEDI={val_metrics['summary']['rain_sedi_mean']:.3f}"
        )
        print(msg)

        if val_score < best_score - 1e-4:
            best_score = val_score
            bad_epochs = 0
            torch.save({"model_state": model_base.state_dict()}, best_path)
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                print("Early stopping.")
                break

    print(f"[INFO] Entrenament finalitzat en {(time.time() - t_start_all) / 60:.1f} minuts.")

    ckpt = torch.load(best_path, map_location=device)
    model_base.load_state_dict(ckpt["model_state"])

    test_metrics = evaluate(model_base, loaders["test"], device, cfg, temp_mean, temp_std, rain_cap, artifacts)
    return {"test": test_metrics, "artifacts": artifacts, "data_meta": meta}




@torch.no_grad()
def evaluate_ensemble(
    model_dir: str,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    cfg: TrainConfig,
    temp_mean: float,
    temp_std: float,
    rain_cap: float,
    artifacts: Dict[str, object],
) -> Dict[str, object]:
    """
    Avalua la predicció conjunta d'un deep ensemble híbrid.

    Manté la NLL exacta de barreja ZI-Gamma per a precipitació i afegeix els
    mateixos diagnòstics avançats que l'avaluació d'un membre individual.
    """
    member_files = sorted(fn for fn in os.listdir(model_dir) if fn.startswith("best_model_seed") and fn.endswith(".pt"))
    if not member_files:
        raise FileNotFoundError(f"No s'han trobat fitxers best_model_seed*.pt a '{model_dir}'.")

    models: List[nn.Module] = []
    for fn in member_files:
        model = HybridResidualNowcaster(
            in_channels=len(artifacts["feature_cols"]) + len(artifacts["time_cols"]),
            time_feat_dim=len(artifacts["time_cols"]),
            history_steps=int(artifacts["history_steps"]),
            horizon_steps=int(artifacts["horizon_steps"]),
            steps_per_hour=int(artifacts["steps_per_hour"]),
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_layers=cfg.n_layers,
            dropout=cfg.dropout,
            patch_len_hours=cfg.patch_len_hours,
            patch_stride_frac=cfg.patch_stride_frac,
            recent_window_hours=cfg.recent_window_hours,
            recent_patch_len_hours=cfg.recent_patch_len_hours,
            recent_patch_stride_frac=cfg.recent_patch_stride_frac,
            timemixer_scales=cfg.timemixer_scales,
            timemixer_ma_kernel=cfg.timemixer_ma_kernel,
            residual_temp_max_sigma=cfg.residual_temp_max_sigma,
            residual_occ_logit_max=cfg.residual_occ_logit_max,
            residual_log_wet_mean_max=cfg.residual_log_wet_mean_max,
            temp_feature_index=int(artifacts["temp_feature_index"]),
            rain_feature_index=int(artifacts["rain_feature_index"]),
        ).to(device)
        ckpt = torch.load(os.path.join(model_dir, fn), map_location=device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        models.append(model)

    O = int(artifacts["horizon_steps"])
    steps_per_hour = int(artifacts["steps_per_hour"])
    rain_event_threshold = float(cfg.rain_event_threshold)
    heavy_threshold = max(rain_event_threshold, finite_or(artifacts.get("rain_pos_extreme_thr"), rain_event_threshold))
    horizons_steps = [
        max(1, int(round(float(h) * steps_per_hour)))
        for h in cfg.accumulation_horizons_hours
        if float(h) > 0.0 and float(h) <= float(cfg.horizon_hours) + 1e-9
    ]

    n_temp = np.zeros(O, dtype=np.int64)
    sum_abs_temp = np.zeros(O, dtype=np.float64)
    sum_sq_temp = np.zeros(O, dtype=np.float64)
    sum_nll_temp = np.zeros(O, dtype=np.float64)
    sum_crps_temp = np.zeros(O, dtype=np.float64)

    n_rain = np.zeros(O, dtype=np.int64)
    sum_abs_rain = np.zeros(O, dtype=np.float64)
    sum_sq_rain = np.zeros(O, dtype=np.float64)
    sum_nll_rain = np.zeros(O, dtype=np.float64)
    sum_brier = np.zeros(O, dtype=np.float64)
    tp = np.zeros(O, dtype=np.int64)
    fn = np.zeros(O, dtype=np.int64)
    fp = np.zeros(O, dtype=np.int64)
    tn = np.zeros(O, dtype=np.int64)

    sum_crps_rain = np.zeros(O, dtype=np.float64)
    n_crps_rain = np.zeros(O, dtype=np.int64)
    crps_rain_points = 0
    losses: List[float] = []

    baseline_accs = {
        "persistence": init_baseline_accumulators(O),
        "climatology": init_baseline_accumulators(O),
        "daily_persistence_temp": init_baseline_accumulators(O),
        "rain_markov": init_baseline_accumulators(O),
    }
    temp_coverage_acc = init_temp_coverage_accumulators(O)
    reliability_acc = init_reliability_accumulators(O, n_bins=int(getattr(cfg, "reliability_bins", 10)))
    accum_acc = init_accumulation_diagnostics(horizons_steps)
    regime_acc = init_regime_accumulators(["all", "dry_window", "already_raining", "rain_onset", "heavy_future"])

    for batch in loader:
        batch = to_device(batch, device)
        temp_baseline_scaled = (batch["temp_baseline_raw"] - temp_mean) / temp_std
        last_temp_scaled = (batch["last_temp_raw"] - temp_mean) / temp_std

        outs = [
            model(
                batch["x_past"],
                batch["t_future"],
                temp_baseline_scaled=temp_baseline_scaled,
                rain_occ_baseline=batch["rain_occ_baseline"],
                rain_mu_baseline=batch["rain_mu_baseline"],
                last_temp_scaled=last_temp_scaled,
                last_rain_raw=batch["last_rain_raw"],
                future_cosz=batch["future_cosz"],
                future_toa=batch["future_toa"],
            )
            for model in models
        ]

        temp_mu_stack = torch.stack([o["temp_mu"] for o in outs], dim=0)
        temp_sigma_stack = torch.stack([o["temp_sigma"] for o in outs], dim=0)
        temp_mu_ens = temp_mu_stack.mean(dim=0)
        temp_var_ens = (temp_sigma_stack.pow(2) + temp_mu_stack.pow(2)).mean(dim=0) - temp_mu_ens.pow(2)
        temp_sigma_ens = torch.sqrt(torch.clamp(temp_var_ens, min=cfg.eps))

        p0_stack = torch.stack([torch.sigmoid(o["rain_logit_p0"]) for o in outs], dim=0)
        k_stack = torch.stack([o["rain_k"] for o in outs], dim=0)
        theta_stack = torch.stack([o["rain_theta"] for o in outs], dim=0)

        mT = batch["mask_temp"]
        mR = batch["mask_rain"]
        lt_el = gaussian_nll(batch["y_temp"], temp_mu_ens, temp_sigma_ens, eps=cfg.eps)
        lr_el = zigamma_mixture_nll(batch["y_rain"], p0_stack, k_stack, theta_stack, eps=cfg.eps, cap=rain_cap)

        temp_mu_raw = temp_mu_ens * temp_std + temp_mean
        temp_sigma_raw = temp_sigma_ens * temp_std
        temp_true_raw = batch["y_temp_raw"]
        p_occ = (1.0 - p0_stack).mean(dim=0)
        rain_mu = ((1.0 - p0_stack) * (k_stack * theta_stack)).mean(dim=0)

        p_occ_safe = torch.clamp(p_occ, min=cfg.eps, max=1.0 - cfg.eps)
        loss_temp = masked_mean(lt_el, mT)
        loss_rain = masked_mean(lr_el, mR)
        loss_rain_occ = masked_mean(
            rain_occurrence_bce_loss(
                torch.log(torch.clamp(1.0 - p_occ_safe, min=cfg.eps)) - torch.log(p_occ_safe),
                batch["y_rain_occ"],
                eps=cfg.eps,
                focal_gamma=cfg.rain_occ_focal_gamma,
                pos_weight=cfg.rain_occ_pos_weight,
            ),
            mR,
        )
        loss_smooth = temperature_smoothness_loss(temp_mu_raw, mT)
        loss_anchor = temperature_anchor_loss(temp_mu_ens, temp_baseline_scaled, mT)
        loss_accum = accumulation_consistency_loss(rain_mu=rain_mu, y_rain=batch["y_rain"], mask_rain=mR, horizon_steps=horizons_steps)
        loss = (
            loss_temp
            + cfg.rain_loss_weight * loss_rain
            + cfg.rain_occurrence_bce_weight * loss_rain_occ
            + cfg.physics_temp_smooth_weight * loss_smooth
            + cfg.physics_temp_anchor_weight * loss_anchor
            + cfg.physics_rain_accum_weight * loss_accum
        )
        losses.append(float(loss.item()))

        crpsT = crps_gaussian(temp_mu_raw, temp_sigma_raw, temp_true_raw, eps=cfg.eps) if cfg.compute_crps else None

        tmu = temp_mu_raw.detach().cpu().numpy()
        tsig = temp_sigma_raw.detach().cpu().numpy()
        ttr = temp_true_raw.detach().cpu().numpy()
        rmu = rain_mu.detach().cpu().numpy()
        rtr = batch["y_rain"].detach().cpu().numpy()
        occp = p_occ.detach().cpu().numpy()
        occ_true = batch["y_rain_occ"].detach().cpu().numpy()
        mTn = mT.detach().cpu().numpy()
        mRn = mR.detach().cpu().numpy()
        lt_np = lt_el.detach().cpu().numpy()
        lr_np = lr_el.detach().cpu().numpy()
        crpsT_np = crpsT.detach().cpu().numpy() if crpsT is not None else None

        temp_persistence = batch["last_temp_raw"].detach().cpu().numpy()[:, None].repeat(O, axis=1)
        rain_persistence = batch["last_rain_raw"].detach().cpu().numpy()[:, None].repeat(O, axis=1)
        occ_persistence = (rain_persistence >= rain_event_threshold).astype(np.float64)
        temp_climatology = batch["temp_baseline_raw"].detach().cpu().numpy()
        rain_markov_mu = batch["rain_mu_baseline"].detach().cpu().numpy()
        rain_markov_occ = batch["rain_occ_baseline"].detach().cpu().numpy()
        temp_daily = temp_climatology.copy()

        update_baseline_accumulators(baseline_accs["persistence"], temp_persistence, rain_persistence, occ_persistence, ttr, rtr, occ_true, mTn, mRn)
        update_baseline_accumulators(baseline_accs["climatology"], temp_climatology, np.zeros_like(rain_markov_mu), np.zeros_like(rain_markov_occ), ttr, rtr, occ_true, mTn, np.zeros_like(mRn))
        update_baseline_accumulators(baseline_accs["daily_persistence_temp"], temp_daily, np.zeros_like(rain_markov_mu), np.zeros_like(rain_markov_occ), ttr, rtr, occ_true, mTn, np.zeros_like(mRn))
        update_baseline_accumulators(baseline_accs["rain_markov"], np.zeros_like(temp_daily), rain_markov_mu, rain_markov_occ, ttr, rtr, occ_true, np.zeros_like(mTn), mRn)

        update_temp_coverage_accumulators(temp_coverage_acc, tmu, tsig, ttr, mTn)
        update_reliability_accumulators(reliability_acc, occp, occ_true, mRn)
        update_accumulation_diagnostics(accum_acc, rmu, rtr, occp, mRn, event_threshold=rain_event_threshold)
        last_rain_np = batch["last_rain_raw"].detach().cpu().numpy()
        future_sum = np.sum(np.where(mRn > 0.5, rtr, 0.0), axis=1)
        future_max = np.max(np.where(mRn > 0.5, rtr, 0.0), axis=1)
        regime_masks = {
            "all": np.ones_like(last_rain_np, dtype=bool),
            "dry_window": (last_rain_np < rain_event_threshold) & (future_sum < rain_event_threshold),
            "already_raining": last_rain_np >= rain_event_threshold,
            "rain_onset": (last_rain_np < rain_event_threshold) & (future_max >= rain_event_threshold),
            "heavy_future": future_max >= heavy_threshold,
        }
        update_regime_accumulators(regime_acc, regime_masks, tmu, ttr, rmu, rtr, occp, occ_true, mTn, mRn)

        for k in range(O):
            vt = mTn[:, k] > 0.5
            if np.any(vt):
                n_temp[k] += int(vt.sum())
                err = tmu[vt, k] - ttr[vt, k]
                sum_abs_temp[k] += float(np.sum(np.abs(err)))
                sum_sq_temp[k] += float(np.sum(err ** 2))
                sum_nll_temp[k] += float(np.sum(lt_np[vt, k]))
                if crpsT_np is not None:
                    sum_crps_temp[k] += float(np.sum(crpsT_np[vt, k]))
            vr = mRn[:, k] > 0.5
            if np.any(vr):
                n_rain[k] += int(vr.sum())
                errr = rmu[vr, k] - rtr[vr, k]
                sum_abs_rain[k] += float(np.sum(np.abs(errr)))
                sum_sq_rain[k] += float(np.sum(errr ** 2))
                sum_nll_rain[k] += float(np.sum(lr_np[vr, k]))
                y01 = occ_true[vr, k].astype(int)
                p = np.clip(occp[vr, k], 0.0, 1.0)
                sum_brier[k] += float(np.sum((p - y01) ** 2))
                yhat = (p >= 0.5).astype(int)
                tp[k] += int(np.sum((yhat == 1) & (y01 == 1)))
                fn[k] += int(np.sum((yhat == 0) & (y01 == 1)))
                fp[k] += int(np.sum((yhat == 1) & (y01 == 0)))
                tn[k] += int(np.sum((yhat == 0) & (y01 == 0)))

        if cfg.compute_crps and (cfg.crps_rain_max_points > 0) and (crps_rain_points < cfg.crps_rain_max_points):
            r_valid = (mR > 0.5).view(-1)
            if torch.any(r_valid):
                idx_all = torch.where(r_valid)[0]
                remaining = int(cfg.crps_rain_max_points) - crps_rain_points
                take = int(min(remaining, idx_all.numel(), 512))
                if take > 0:
                    perm = torch.randperm(idx_all.numel(), device=idx_all.device)[:take]
                    sel = idx_all[perm]
                    y_sel = batch["y_rain"].view(-1)[sel]
                    p0_flat = p0_stack.view(p0_stack.shape[0], -1)[:, sel]
                    k_flat = k_stack.view(k_stack.shape[0], -1)[:, sel]
                    th_flat = theta_stack.view(theta_stack.shape[0], -1)[:, sel]
                    n_samp = int(cfg.crps_rain_n_samples)
                    cols = torch.arange(sel.numel(), device=device).unsqueeze(0).expand(n_samp, -1)
                    j_idx = torch.randint(low=0, high=p0_flat.shape[0], size=(n_samp, sel.numel()), device=device)
                    p0_s = p0_flat[j_idx, cols]
                    k_s = k_flat[j_idx, cols]
                    th_s = torch.clamp(th_flat[j_idx, cols], min=1e-6)
                    gamma = torch.distributions.Gamma(concentration=k_s, rate=1.0 / th_s)
                    samp = gamma.sample()
                    u = torch.rand_like(samp)
                    samp = torch.where(u < p0_s, torch.zeros_like(samp), samp)
                    crps = fair_crps_from_samples(samp, y_sel)
                    crps_np = crps.detach().cpu().numpy()
                    lead_np = (sel.detach().cpu().numpy() % O).astype(np.int64)
                    for kk, vv in zip(lead_np, crps_np):
                        sum_crps_rain[kk] += float(vv)
                        n_crps_rain[kk] += 1
                    crps_rain_points += int(crps.numel())

    def safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        out = np.full_like(a, np.nan, dtype=np.float64)
        m = b > 0
        out[m] = a[m] / b[m]
        return out

    temp_mae = safe_div(sum_abs_temp, n_temp)
    temp_rmse = np.sqrt(safe_div(sum_sq_temp, n_temp))
    temp_nll = safe_div(sum_nll_temp, n_temp)
    temp_crps = safe_div(sum_crps_temp, n_temp) if cfg.compute_crps else np.full(O, np.nan)
    rain_mae = safe_div(sum_abs_rain, n_rain)
    rain_rmse = np.sqrt(safe_div(sum_sq_rain, n_rain))
    rain_nll = safe_div(sum_nll_rain, n_rain)
    rain_brier = safe_div(sum_brier, n_rain)
    rain_sedi = np.array([sedi(int(tp[i]), int(fn[i]), int(fp[i]), int(tn[i])) if n_rain[i] > 0 else np.nan for i in range(O)], dtype=np.float64)
    rain_crps = safe_div(sum_crps_rain, n_crps_rain) if cfg.compute_crps else np.full(O, np.nan)

    by_lead = {
        "lead_hours": ((np.arange(O, dtype=np.float64) + 1.0) / float(steps_per_hour)).tolist(),
        "temp_mae": temp_mae.tolist(),
        "temp_rmse": temp_rmse.tolist(),
        "temp_nll": temp_nll.tolist(),
        "temp_crps": temp_crps.tolist(),
        "rain_mae": rain_mae.tolist(),
        "rain_rmse": rain_rmse.tolist(),
        "rain_nll": rain_nll.tolist(),
        "rain_brier": rain_brier.tolist(),
        "rain_sedi": rain_sedi.tolist(),
        "rain_crps": rain_crps.tolist(),
        "rain_crps_n": n_crps_rain.tolist(),
        "n_temp": n_temp.tolist(),
        "n_rain": n_rain.tolist(),
    }
    baselines = {
        "persistence": finalize_baseline_metrics(baseline_accs["persistence"], steps_per_hour, "persistence"),
        "climatology": finalize_baseline_metrics(baseline_accs["climatology"], steps_per_hour, "climatology"),
        "daily_persistence_temp": finalize_baseline_metrics(baseline_accs["daily_persistence_temp"], steps_per_hour, "daily_persistence_temp"),
        "rain_markov": finalize_baseline_metrics(baseline_accs["rain_markov"], steps_per_hour, "rain_markov"),
    }
    skill_scores = build_skill_scores(by_lead, baselines, steps_per_hour)
    diagnostics = {
        "temperature_coverage": finalize_temp_coverage_accumulators(temp_coverage_acc, steps_per_hour),
        "rain_reliability": finalize_reliability_accumulators(reliability_acc, steps_per_hour),
        "rain_accumulation": finalize_accumulation_diagnostics(accum_acc, steps_per_hour),
        "regime_metrics": finalize_regime_accumulators(regime_acc),
    }
    summary = {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "temp_mae_mean": safe_nanmean(temp_mae),
        "temp_rmse_mean": safe_nanmean(temp_rmse),
        "temp_nll_mean": safe_nanmean(temp_nll),
        "temp_crps_mean": safe_nanmean(temp_crps) if cfg.compute_crps else float("nan"),
        "rain_mae_mean": safe_nanmean(rain_mae),
        "rain_rmse_mean": safe_nanmean(rain_rmse),
        "rain_nll_mean": safe_nanmean(rain_nll),
        "rain_brier_mean": safe_nanmean(rain_brier),
        "rain_sedi_mean": safe_nanmean(rain_sedi),
        "rain_crps_mean_sampled": safe_nanmean(rain_crps) if cfg.compute_crps else float("nan"),
        "crps_rain_mean_sampled": safe_nanmean(rain_crps) if cfg.compute_crps else float("nan"),
        "crps_rain_points": int(crps_rain_points),
        "n_members": int(len(models)),
        "aggregation": "deep_ensemble_hibrid",
        "temp_coverage80_mean": diagnostics["temperature_coverage"]["summary"]["coverage80_mean"],
        "temp_coverage90_mean": diagnostics["temperature_coverage"]["summary"]["coverage90_mean"],
        "rain_reliability_ece": diagnostics["rain_reliability"]["summary"]["expected_calibration_error"],
        "rain_accum_mae_mean": diagnostics["rain_accumulation"]["summary"]["rain_accum_mae_mean"],
    }
    return {"summary": summary, "by_lead": by_lead, "baselines": baselines, "skill_scores": skill_scores, "diagnostics": diagnostics}


def evaluate_saved_ensemble(model_dir: str, csv: str) -> Dict[str, object]:
    with open(os.path.join(model_dir, "config.json"), "r", encoding="utf-8") as f:
        cfg_raw = json.load(f)
    with open(os.path.join(model_dir, "artifacts.json"), "r", encoding="utf-8") as f:
        artifacts = json.load(f)

    cfg = TrainConfig(**cfg_raw)
    cfg.csv = csv
    df, _ = load_dataset(cfg)
    splits = split_by_time(df, cfg)
    datasets, artifacts = build_datasets(splits, cfg)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    configure_gpu_perf(cfg)
    loaders = make_loaders(datasets, cfg, use_cuda)

    metrics = evaluate_ensemble(
        model_dir=model_dir,
        loader=loaders["test"],
        device=device,
        cfg=cfg,
        temp_mean=float(artifacts["temp_mean"]),
        temp_std=float(artifacts["temp_std"]),
        rain_cap=float(artifacts["rain_cap_99p9"]),
        artifacts=artifacts,
    )

    with open(os.path.join(model_dir, "test_metrics_ensemble.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    with open(os.path.join(model_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    return metrics


# =============================================================================
# Avaluació final millorada: CRPS de pluja robusta i calibratge post-hoc de p_rain
# =============================================================================


# Es conserva una referència a l'avaluació original per no duplicar tota la lògica
# de temperatura, NLL i acumuladors generals. La funció nova de sota reavalua i
# substitueix els diagnòstics probabilistes de pluja que depenen de p_rain i CRPS.
_evaluate_ensemble_uncalibrated_core = evaluate_ensemble


def empirical_crps_from_samples(samples: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    CRPS empírica no negativa a partir de mostres de la distribució predictiva.

    Fórmula finita:
        CRPS = E|X-y| - 0.5 E|X-X'|
    amb l'esperança estimada sobre totes les parelles N x N. A diferència de
    l'estimador fair no esbiaixat, aquest estimador no dona valors negatius per
    soroll de mostreig quan la distribució és gairebé degenerada a zero.
    """
    n = int(samples.shape[0])
    if n <= 0:
        return torch.full_like(y, float("nan"))
    term1 = torch.mean(torch.abs(samples - y.unsqueeze(0)), dim=0)
    if n < 2:
        return torch.clamp(term1, min=0.0)
    dif = torch.abs(samples.unsqueeze(1) - samples.unsqueeze(0))
    term2 = 0.5 * dif.mean(dim=(0, 1))
    return torch.clamp(term1 - term2, min=0.0)


def _pava_increasing(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Pool Adjacent Violators Algorithm per imposar calibratge monòton."""
    v = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    finite = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not np.any(finite):
        return np.clip(np.nan_to_num(v, nan=0.0), 0.0, 1.0)
    # Omplim valors no finits per interpolació simple abans del PAVA.
    x = np.arange(len(v), dtype=np.float64)
    if not np.all(finite):
        v = v.copy()
        v[~finite] = np.interp(x[~finite], x[finite], v[finite])
        w = np.where(finite, w, 1e-6)
    blocks = []
    for i in range(len(v)):
        blocks.append([i, i, float(v[i]), float(max(w[i], 1e-6))])
        while len(blocks) >= 2 and blocks[-2][2] > blocks[-1][2]:
            b2 = blocks.pop()
            b1 = blocks.pop()
            ww = b1[3] + b2[3]
            vv = (b1[2] * b1[3] + b2[2] * b2[3]) / ww
            blocks.append([b1[0], b2[1], vv, ww])
    out = np.zeros_like(v, dtype=np.float64)
    for start, end, vv, _ in blocks:
        out[int(start):int(end) + 1] = vv
    return np.clip(out, 0.0, 1.0)


def fit_rain_probability_calibrator_np(
    prob: np.ndarray,
    obs01: np.ndarray,
    mask: Optional[np.ndarray] = None,
    n_bins: int = 20,
    smoothing: float = 25.0,
) -> Dict[str, object]:
    """
    Ajusta un calibrador post-hoc de probabilitat de pluja amb dades de validació.

    És un calibratge tipus reliability-binning + PAVA:
      - cada bin aprèn la freqüència observada de pluja,
      - s'hi aplica suavitzat cap a la freqüència climatològica de validació,
      - s'imposa monotonia perquè una probabilitat prevista més alta no pugui
        correspondre a una probabilitat calibrada més baixa.

    No usa test i, per tant, no introdueix leakage temporal.
    """
    p = np.clip(np.asarray(prob, dtype=np.float64).reshape(-1), 0.0, 1.0)
    y = np.asarray(obs01, dtype=np.float64).reshape(-1)
    if mask is None:
        m = np.isfinite(p) & np.isfinite(y)
    else:
        m = (np.asarray(mask, dtype=np.float64).reshape(-1) > 0.5) & np.isfinite(p) & np.isfinite(y)
    p = p[m]
    y = np.clip(y[m], 0.0, 1.0)
    n_bins = max(4, int(n_bins))
    smoothing = max(0.0, float(smoothing))
    if p.size == 0:
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        return {
            "method": "identity_empty_validation",
            "bin_edges": edges.tolist(),
            "bin_centers": centers.tolist(),
            "raw_bin_count": np.zeros(n_bins, dtype=int).tolist(),
            "raw_bin_mean_probability": centers.tolist(),
            "raw_bin_observed_frequency": centers.tolist(),
            "calibrated_probability": centers.tolist(),
            "base_event_rate": float("nan"),
        }
    base_rate = float(np.mean(y))
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    idx = np.minimum(np.floor(p * n_bins).astype(np.int64), n_bins - 1)
    counts = np.bincount(idx, minlength=n_bins).astype(np.float64)
    sum_p = np.bincount(idx, weights=p, minlength=n_bins).astype(np.float64)
    sum_y = np.bincount(idx, weights=y, minlength=n_bins).astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_p = np.where(counts > 0, sum_p / counts, centers)
        obs = np.where(counts > 0, sum_y / counts, np.nan)
    smoothed = np.where(
        counts > 0,
        (sum_y + smoothing * base_rate) / np.maximum(counts + smoothing, 1e-12),
        base_rate,
    )
    weights = counts + smoothing
    calibrated = _pava_increasing(smoothed, weights)
    # Evitem probabilitats exactament 0/1 per estabilitat de Brier, logits i productes d'acumulació.
    eps = 1e-5
    calibrated = np.clip(calibrated, eps, 1.0 - eps)
    return {
        "method": "histogram_reliability_pava",
        "n_bins": int(n_bins),
        "smoothing": float(smoothing),
        "n_validation_points": int(p.size),
        "base_event_rate": base_rate,
        "bin_edges": edges.tolist(),
        "bin_centers": centers.tolist(),
        "raw_bin_count": counts.astype(int).tolist(),
        "raw_bin_mean_probability": mean_p.tolist(),
        "raw_bin_observed_frequency": obs.tolist(),
        "calibrated_probability": calibrated.tolist(),
    }


def apply_rain_probability_calibrator_np(prob: np.ndarray, calibrator: Optional[Dict[str, object]]) -> np.ndarray:
    """Aplica el calibrador de probabilitat de pluja a un array arbitrari."""
    p = np.clip(np.asarray(prob, dtype=np.float64), 0.0, 1.0)
    if not calibrator or "bin_edges" not in calibrator or "calibrated_probability" not in calibrator:
        return p.astype(np.float64)
    edges = np.asarray(calibrator["bin_edges"], dtype=np.float64)
    vals = np.asarray(calibrator["calibrated_probability"], dtype=np.float64)
    if edges.ndim != 1 or vals.ndim != 1 or len(edges) != len(vals) + 1:
        return p.astype(np.float64)
    idx = np.searchsorted(edges[1:-1], p, side="right")
    idx = np.clip(idx, 0, len(vals) - 1)
    return vals[idx].reshape(p.shape).astype(np.float64)



def _amount_lead_block_indices(n_leads: int, steps_per_hour: int) -> np.ndarray:
    """Assigna cada lead time a blocs 0-1 h, 1-2 h i 2-3 h."""
    lead_hours = (np.arange(int(n_leads), dtype=np.float64) + 1.0) / max(float(steps_per_hour), 1.0)
    return np.clip(np.floor(lead_hours).astype(np.int64), 0, 2)


def _digitize_edges(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Discretitza valors segons edges interns, retornant índexs vàlids."""
    v = np.asarray(values, dtype=np.float64)
    e = np.asarray(edges, dtype=np.float64)
    idx = np.searchsorted(e[1:-1], v, side="right")
    return np.clip(idx, 0, len(e) - 2).astype(np.int64)


def _default_rain_amount_calibrator(cfg: TrainConfig, steps_per_hour: int) -> Dict[str, object]:
    """Calibrador identitat per a casos sense dades suficients de validació."""
    p_edges = [0.0, 0.05, 0.15, 0.30, 0.50, 1.000001]
    mu_edges = [0.0, 0.02, 0.05, 0.10, 0.30, 1.00, 1.0e9]
    factors = np.ones((3, 2, len(p_edges) - 1, len(mu_edges) - 1), dtype=np.float64)
    return {
        "method": "identity_empty_validation",
        "description": "Calibrador identitat de quantitat de pluja.",
        "deadband_mm": float(getattr(cfg, "rain_amount_deadband", cfg.rain_event_threshold)),
        "event_threshold_mm": float(cfg.rain_event_threshold),
        "steps_per_hour": int(steps_per_hour),
        "p_edges": p_edges,
        "mu_edges": mu_edges,
        "factor_grid": factors.tolist(),
        "lead_fallback_factor": [1.0, 1.0, 1.0],
        "global_factor": 1.0,
        "max_factor": float(getattr(cfg, "rain_amount_calibration_max_factor", 3.0)),
    }


def fit_rain_amount_calibrator_np(
    rain_mu: np.ndarray,
    y_rain: np.ndarray,
    p_raw: np.ndarray,
    p_cal: np.ndarray,
    last_rain: np.ndarray,
    mask: np.ndarray,
    cfg: TrainConfig,
    steps_per_hour: int,
) -> Dict[str, object]:
    """
    Ajusta un calibrador post-hoc de quantitat de pluja amb VALIDACIÓ.

    El calibrador aprèn factors multiplicatius per a la quantitat esperada de
    pluja, però només s'aplicarà en règims humits. La filosofia és corregir la
    infraestimació condicionada a pluja sense generar plugim espuri en finestres
    clarament seques.
    """
    rmu = np.asarray(rain_mu, dtype=np.float64)
    y = np.asarray(y_rain, dtype=np.float64)
    pr = np.clip(np.asarray(p_raw, dtype=np.float64), 0.0, 1.0)
    pc = np.clip(np.asarray(p_cal, dtype=np.float64), 0.0, 1.0)
    m = np.asarray(mask, dtype=np.float64) > 0.5
    lr = np.asarray(last_rain, dtype=np.float64).reshape(-1)
    if rmu.ndim != 2 or y.shape != rmu.shape or pr.shape != rmu.shape or pc.shape != rmu.shape:
        return _default_rain_amount_calibrator(cfg, steps_per_hour)
    N, O = rmu.shape
    if lr.shape[0] != N:
        lr = np.zeros(N, dtype=np.float64)

    thr = float(getattr(cfg, "rain_event_threshold", 0.1))
    deadband = float(getattr(cfg, "rain_amount_deadband", thr))
    p_edges = np.asarray([0.0, 0.05, 0.15, 0.30, 0.50, 1.000001], dtype=np.float64)
    mu_edges = np.asarray([0.0, 0.02, 0.05, 0.10, 0.30, 1.00, 1.0e9], dtype=np.float64)
    nL, nR, nP, nM = 3, 2, len(p_edges) - 1, len(mu_edges) - 1

    wet_apply = (
        (pc >= float(getattr(cfg, "rain_amount_apply_p_cal_threshold", 0.20))) |
        (pr >= float(getattr(cfg, "rain_amount_apply_p_raw_threshold", 0.35))) |
        (rmu >= float(getattr(cfg, "rain_amount_apply_mu_threshold", 0.03))) |
        (lr[:, None] >= thr)
    )
    # Durant l'ajust, incloem també casos on finalment va ploure perquè el
    # calibrador aprengui la infraestimació d'onsets que el model marcava febles.
    fit_candidate = m & (wet_apply | (y >= thr))
    if int(np.sum(fit_candidate)) < int(getattr(cfg, "rain_amount_calibration_min_points", 200)):
        return _default_rain_amount_calibrator(cfg, steps_per_hour)

    lead_block = _amount_lead_block_indices(O, steps_per_hour)
    recent_bin = (lr >= thr).astype(np.int64)
    p_for_bin = np.maximum(pc, 0.65 * pr)
    p_bin = _digitize_edges(p_for_bin, p_edges)
    mu_bin = _digitize_edges(np.maximum(rmu, 0.0), mu_edges)

    count = np.zeros((nL, nR, nP, nM), dtype=np.float64)
    sum_pred = np.zeros_like(count)
    sum_obs = np.zeros_like(count)

    # Bucle explícit: el nombre de leads és petit i és més llegible/segur.
    for j in range(O):
        lb = int(lead_block[j])
        valid_j = fit_candidate[:, j]
        if not np.any(valid_j):
            continue
        rb = recent_bin[valid_j]
        pb = p_bin[valid_j, j]
        mb = mu_bin[valid_j, j]
        pred = np.maximum(rmu[valid_j, j], 0.0)
        obs = np.maximum(y[valid_j, j], 0.0)
        for rr, pp, mm, pv, ov in zip(rb, pb, mb, pred, obs):
            count[lb, int(rr), int(pp), int(mm)] += 1.0
            sum_pred[lb, int(rr), int(pp), int(mm)] += float(pv)
            sum_obs[lb, int(rr), int(pp), int(mm)] += float(ov)

    # Factors fallback per bloc temporal i global.
    eps = 1e-6
    lead_count = count.sum(axis=(1, 2, 3))
    lead_pred = sum_pred.sum(axis=(1, 2, 3))
    lead_obs = sum_obs.sum(axis=(1, 2, 3))
    with np.errstate(divide="ignore", invalid="ignore"):
        lead_ratio = np.where(lead_pred > eps, lead_obs / np.maximum(lead_pred, eps), 1.0)
    global_ratio = float(np.sum(sum_obs) / max(np.sum(sum_pred), eps))
    max_factor = float(getattr(cfg, "rain_amount_calibration_max_factor", 3.0))
    min_factor = float(getattr(cfg, "rain_amount_calibration_min_factor", 1.0))
    min_factor = max(0.0, min_factor)
    lead_fallback = np.clip(np.where(lead_count > 0, lead_ratio, global_ratio), min_factor, max_factor)
    global_factor = float(np.clip(global_ratio, min_factor, max_factor))

    factors = np.ones_like(count)
    min_cell = float(getattr(cfg, "rain_amount_calibration_min_cell_count", 80))
    smooth = float(getattr(cfg, "rain_amount_calibration_smoothing", 120.0))
    for lb in range(nL):
        for rb in range(nR):
            for pb in range(nP):
                for mb in range(nM):
                    n = count[lb, rb, pb, mb]
                    if n <= 0:
                        raw = lead_fallback[lb]
                        cred = 0.0
                    else:
                        raw = sum_obs[lb, rb, pb, mb] / max(sum_pred[lb, rb, pb, mb], eps)
                        # Credibilitat suau: cel·les petites tornen al fallback.
                        cred = n / (n + smooth)
                        if n < min_cell:
                            cred *= n / max(min_cell, 1.0)
                    f = (1.0 - cred) * lead_fallback[lb] + cred * raw
                    factors[lb, rb, pb, mb] = float(np.clip(f, min_factor, max_factor))

    return {
        "method": "validation_conditional_amount_ratio_v9",
        "description": "Factors multiplicatius de quantitat condicionats a règim humit, ajustats només amb validació.",
        "event_threshold_mm": thr,
        "deadband_mm": deadband,
        "steps_per_hour": int(steps_per_hour),
        "p_edges": p_edges.tolist(),
        "mu_edges": mu_edges.tolist(),
        "factor_grid": factors.tolist(),
        "lead_fallback_factor": lead_fallback.tolist(),
        "global_factor": global_factor,
        "max_factor": max_factor,
        "min_factor": min_factor,
        "apply_p_cal_threshold": float(getattr(cfg, "rain_amount_apply_p_cal_threshold", 0.20)),
        "apply_p_raw_threshold": float(getattr(cfg, "rain_amount_apply_p_raw_threshold", 0.35)),
        "apply_mu_threshold": float(getattr(cfg, "rain_amount_apply_mu_threshold", 0.03)),
        "n_validation_candidates": int(np.sum(fit_candidate)),
        "cell_count": count.astype(int).tolist(),
        "cell_sum_pred": sum_pred.tolist(),
        "cell_sum_obs": sum_obs.tolist(),
    }


def apply_rain_amount_calibrator_np(
    rain_mu: np.ndarray,
    p_raw: np.ndarray,
    p_cal: np.ndarray,
    last_rain: np.ndarray,
    calibrator: Optional[Dict[str, object]],
    steps_per_hour: int,
    rain_q: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """Aplica el calibrador v9 de quantitat de pluja i retorna també el factor."""
    rmu = np.asarray(rain_mu, dtype=np.float64)
    pr = np.clip(np.asarray(p_raw, dtype=np.float64), 0.0, 1.0)
    pc = np.clip(np.asarray(p_cal, dtype=np.float64), 0.0, 1.0)
    if calibrator is None or "factor_grid" not in calibrator:
        return rmu.astype(np.float64), None if rain_q is None else np.asarray(rain_q, dtype=np.float64), np.ones_like(rmu, dtype=np.float64)
    if rmu.ndim == 1:
        rmu2 = rmu.reshape(1, -1)
        pr2 = pr.reshape(1, -1)
        pc2 = pc.reshape(1, -1)
        one_dim = True
    else:
        rmu2, pr2, pc2 = rmu, pr, pc
        one_dim = False
    N, O = rmu2.shape
    lr = np.asarray(last_rain, dtype=np.float64).reshape(-1)
    if lr.size == 1 and N > 1:
        lr = np.full(N, float(lr[0]), dtype=np.float64)
    if lr.shape[0] != N:
        lr = np.zeros(N, dtype=np.float64)

    factors_grid = np.asarray(calibrator.get("factor_grid"), dtype=np.float64)
    p_edges = np.asarray(calibrator.get("p_edges"), dtype=np.float64)
    mu_edges = np.asarray(calibrator.get("mu_edges"), dtype=np.float64)
    if factors_grid.ndim != 4 or p_edges.ndim != 1 or mu_edges.ndim != 1:
        return rmu.astype(np.float64), None if rain_q is None else np.asarray(rain_q, dtype=np.float64), np.ones_like(rmu, dtype=np.float64)

    thr = float(calibrator.get("event_threshold_mm", 0.1))
    pcal_thr = float(calibrator.get("apply_p_cal_threshold", 0.20))
    praw_thr = float(calibrator.get("apply_p_raw_threshold", 0.35))
    mu_thr = float(calibrator.get("apply_mu_threshold", 0.03))
    wet_apply = (pc2 >= pcal_thr) | (pr2 >= praw_thr) | (rmu2 >= mu_thr) | (lr[:, None] >= thr)

    lead_block = _amount_lead_block_indices(O, int(steps_per_hour))
    recent_bin = (lr >= thr).astype(np.int64)
    p_for_bin = np.maximum(pc2, 0.65 * pr2)
    p_bin = _digitize_edges(p_for_bin, p_edges)
    mu_bin = _digitize_edges(np.maximum(rmu2, 0.0), mu_edges)

    factor = np.ones_like(rmu2, dtype=np.float64)
    for j in range(O):
        lb = int(np.clip(lead_block[j], 0, factors_grid.shape[0] - 1))
        rb = np.clip(recent_bin, 0, factors_grid.shape[1] - 1)
        pb = np.clip(p_bin[:, j], 0, factors_grid.shape[2] - 1)
        mb = np.clip(mu_bin[:, j], 0, factors_grid.shape[3] - 1)
        factor[:, j] = factors_grid[lb, rb, pb, mb]
    factor = np.where(wet_apply, factor, 1.0)
    out_mu = np.maximum(rmu2, 0.0) * factor

    out_q = None
    if rain_q is not None:
        q = np.asarray(rain_q, dtype=np.float64)
        q2 = q.reshape(1, -1) if q.ndim == 1 else q
        out_q = np.maximum(q2, 0.0) * factor
        if one_dim:
            out_q = out_q.reshape(-1)
    if one_dim:
        return out_mu.reshape(-1).astype(np.float64), out_q, factor.reshape(-1).astype(np.float64)
    return out_mu.astype(np.float64), out_q, factor.astype(np.float64)


@torch.no_grad()
def fit_rain_amount_calibrator_ensemble(
    model_dir: str,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    cfg: TrainConfig,
    temp_mean: float,
    temp_std: float,
    artifacts: Dict[str, object],
    rain_prob_calibrator: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """Ajusta el calibrador v9 de quantitat amb el split de validació."""
    if not bool(getattr(cfg, "calibrate_rain_amount", True)):
        return _default_rain_amount_calibrator(cfg, int(artifacts["steps_per_hour"]))
    models = _load_ensemble_models_for_final_eval(model_dir, cfg, artifacts, device)
    mu_all: List[np.ndarray] = []
    y_all: List[np.ndarray] = []
    praw_all: List[np.ndarray] = []
    pcal_all: List[np.ndarray] = []
    m_all: List[np.ndarray] = []
    lr_all: List[np.ndarray] = []
    max_points = int(getattr(cfg, "rain_amount_calibration_max_points", 600000))
    total = 0
    for batch in loader:
        batch = to_device(batch, device)
        p0_stack, _, _, rain_mu_t = _ensemble_prob_rain_for_batch(models, batch, temp_mean=temp_mean, temp_std=temp_std)
        p_raw = (1.0 - p0_stack).mean(dim=0).detach().cpu().numpy()
        p_cal = apply_rain_probability_calibrator_np(p_raw, rain_prob_calibrator)
        rmu = rain_mu_t.detach().cpu().numpy()
        y = batch["y_rain"].detach().cpu().numpy()
        m = batch["mask_rain"].detach().cpu().numpy()
        lr = batch["last_rain_raw"].detach().cpu().numpy()
        nrows = rmu.shape[0]
        remaining = max_points - total if max_points > 0 else nrows
        if remaining <= 0:
            break
        if nrows > remaining:
            rng = np.random.default_rng(54321 + total)
            sel = rng.choice(nrows, size=int(remaining), replace=False)
            rmu, y, p_raw, p_cal, m, lr = rmu[sel], y[sel], p_raw[sel], p_cal[sel], m[sel], lr[sel]
            nrows = rmu.shape[0]
        mu_all.append(rmu.astype(np.float64)); y_all.append(y.astype(np.float64)); praw_all.append(p_raw.astype(np.float64)); pcal_all.append(p_cal.astype(np.float64)); m_all.append(m.astype(np.float64)); lr_all.append(lr.astype(np.float64))
        total += int(nrows)
        if max_points > 0 and total >= max_points:
            break
    if not mu_all:
        return _default_rain_amount_calibrator(cfg, int(artifacts["steps_per_hour"]))
    return fit_rain_amount_calibrator_np(
        rain_mu=np.concatenate(mu_all, axis=0),
        y_rain=np.concatenate(y_all, axis=0),
        p_raw=np.concatenate(praw_all, axis=0),
        p_cal=np.concatenate(pcal_all, axis=0),
        last_rain=np.concatenate(lr_all, axis=0),
        mask=np.concatenate(m_all, axis=0),
        cfg=cfg,
        steps_per_hour=int(artifacts["steps_per_hour"]),
    )


def _load_ensemble_models_for_final_eval(
    model_dir: str,
    cfg: TrainConfig,
    artifacts: Dict[str, object],
    device: torch.device,
) -> List[nn.Module]:
    member_files = sorted(fn for fn in os.listdir(model_dir) if fn.startswith("best_model_seed") and fn.endswith(".pt"))
    if not member_files and os.path.exists(os.path.join(model_dir, "best_model.pt")):
        member_files = ["best_model.pt"]
    if not member_files:
        raise FileNotFoundError(f"No s'han trobat checkpoints a '{model_dir}'.")
    models: List[nn.Module] = []
    for fn in member_files:
        model = HybridResidualNowcaster(
            in_channels=len(artifacts["feature_cols"]) + len(artifacts["time_cols"]),
            time_feat_dim=len(artifacts["time_cols"]),
            history_steps=int(artifacts["history_steps"]),
            horizon_steps=int(artifacts["horizon_steps"]),
            steps_per_hour=int(artifacts["steps_per_hour"]),
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_layers=cfg.n_layers,
            dropout=cfg.dropout,
            patch_len_hours=cfg.patch_len_hours,
            patch_stride_frac=cfg.patch_stride_frac,
            recent_window_hours=cfg.recent_window_hours,
            recent_patch_len_hours=cfg.recent_patch_len_hours,
            recent_patch_stride_frac=cfg.recent_patch_stride_frac,
            timemixer_scales=cfg.timemixer_scales,
            timemixer_ma_kernel=cfg.timemixer_ma_kernel,
            residual_temp_max_sigma=cfg.residual_temp_max_sigma,
            residual_occ_logit_max=cfg.residual_occ_logit_max,
            residual_log_wet_mean_max=cfg.residual_log_wet_mean_max,
            temp_feature_index=int(artifacts["temp_feature_index"]),
            rain_feature_index=int(artifacts["rain_feature_index"]),
        ).to(device)
        ckpt = torch.load(os.path.join(model_dir, fn), map_location=device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        models.append(model)
    return models


@torch.no_grad()
def _ensemble_prob_rain_for_batch(
    models: List[nn.Module],
    batch: Dict[str, torch.Tensor],
    temp_mean: float,
    temp_std: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Retorna p0_stack, k_stack, theta_stack i esperança de pluja de l'ensemble."""
    temp_baseline_scaled = (batch["temp_baseline_raw"] - temp_mean) / temp_std
    last_temp_scaled = (batch["last_temp_raw"] - temp_mean) / temp_std
    outs = [
        model(
            batch["x_past"],
            batch["t_future"],
            temp_baseline_scaled=temp_baseline_scaled,
            rain_occ_baseline=batch["rain_occ_baseline"],
            rain_mu_baseline=batch["rain_mu_baseline"],
            last_temp_scaled=last_temp_scaled,
            last_rain_raw=batch["last_rain_raw"],
            future_cosz=batch["future_cosz"],
            future_toa=batch["future_toa"],
        )
        for model in models
    ]
    p0_stack = torch.stack([torch.sigmoid(o["rain_logit_p0"]) for o in outs], dim=0)
    k_stack = torch.stack([o["rain_k"] for o in outs], dim=0)
    theta_stack = torch.stack([o["rain_theta"] for o in outs], dim=0)
    rain_mu = ((1.0 - p0_stack) * (k_stack * theta_stack)).mean(dim=0)
    return p0_stack, k_stack, theta_stack, rain_mu


@torch.no_grad()
def fit_rain_probability_calibrator_ensemble(
    model_dir: str,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    cfg: TrainConfig,
    temp_mean: float,
    temp_std: float,
    artifacts: Dict[str, object],
) -> Dict[str, object]:
    """Ajusta el calibrador de p_rain amb el split de validació."""
    models = _load_ensemble_models_for_final_eval(model_dir, cfg, artifacts, device)
    p_all: List[np.ndarray] = []
    y_all: List[np.ndarray] = []
    m_all: List[np.ndarray] = []
    max_points = int(getattr(cfg, "rain_calibration_max_points", 600000))
    total = 0
    for batch in loader:
        batch = to_device(batch, device)
        p0_stack, _, _, _ = _ensemble_prob_rain_for_batch(models, batch, temp_mean=temp_mean, temp_std=temp_std)
        p_occ = (1.0 - p0_stack).mean(dim=0).detach().cpu().numpy().reshape(-1)
        y = batch["y_rain_occ"].detach().cpu().numpy().reshape(-1)
        m = batch["mask_rain"].detach().cpu().numpy().reshape(-1)
        valid = m > 0.5
        p_occ = p_occ[valid]
        y = y[valid]
        m = m[valid]
        if p_occ.size == 0:
            continue
        remaining = max_points - total if max_points > 0 else p_occ.size
        if remaining <= 0:
            break
        if p_occ.size > remaining:
            rng = np.random.default_rng(12345 + total)
            sel = rng.choice(p_occ.size, size=int(remaining), replace=False)
            p_occ = p_occ[sel]
            y = y[sel]
            m = m[sel]
        p_all.append(p_occ.astype(np.float64))
        y_all.append(y.astype(np.float64))
        m_all.append(m.astype(np.float64))
        total += int(p_occ.size)
        if max_points > 0 and total >= max_points:
            break
    if not p_all:
        return fit_rain_probability_calibrator_np(np.array([]), np.array([]), None)
    return fit_rain_probability_calibrator_np(
        np.concatenate(p_all),
        np.concatenate(y_all),
        np.concatenate(m_all),
        n_bins=int(getattr(cfg, "rain_calibration_bins", 20)),
        smoothing=float(getattr(cfg, "rain_calibration_smoothing", 25.0)),
    )


def _finalize_event_scores(sum_brier: np.ndarray, n: np.ndarray, tp: np.ndarray, fn: np.ndarray, fp: np.ndarray, tn: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    with np.errstate(divide="ignore", invalid="ignore"):
        brier = np.where(n > 0, sum_brier / n, np.nan)
    sedi_vals = np.array([
        sedi(int(tp[i]), int(fn[i]), int(fp[i]), int(tn[i])) if n[i] > 0 else np.nan
        for i in range(len(n))
    ], dtype=np.float64)
    return brier, sedi_vals


@torch.no_grad()
def recompute_rain_probability_and_crps_diagnostics_ensemble(
    model_dir: str,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    cfg: TrainConfig,
    temp_mean: float,
    temp_std: float,
    artifacts: Dict[str, object],
    rain_prob_calibrator: Optional[Dict[str, object]] = None,
    rain_amount_calibrator: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """
    Recalcula mètriques de pluja sensibles a probabilitat, quantitat i CRPS amb criteris robustos.
    """
    models = _load_ensemble_models_for_final_eval(model_dir, cfg, artifacts, device)
    O = int(artifacts["horizon_steps"])
    steps_per_hour = int(artifacts["steps_per_hour"])
    rain_event_threshold = float(cfg.rain_event_threshold)
    heavy_threshold = max(rain_event_threshold, finite_or(artifacts.get("rain_pos_extreme_thr"), rain_event_threshold))
    horizons_steps = [
        max(1, int(round(float(h) * steps_per_hour)))
        for h in cfg.accumulation_horizons_hours
        if float(h) > 0.0 and float(h) <= float(cfg.horizon_hours) + 1e-9
    ]

    n = np.zeros(O, dtype=np.int64)
    sum_rain_abs = np.zeros(O, dtype=np.float64)
    sum_rain_sq = np.zeros(O, dtype=np.float64)
    sum_brier_raw = np.zeros(O, dtype=np.float64)
    sum_brier_cal = np.zeros(O, dtype=np.float64)
    tp_raw = np.zeros(O, dtype=np.int64); fn_raw = np.zeros(O, dtype=np.int64); fp_raw = np.zeros(O, dtype=np.int64); tn_raw = np.zeros(O, dtype=np.int64)
    tp_cal = np.zeros(O, dtype=np.int64); fn_cal = np.zeros(O, dtype=np.int64); fp_cal = np.zeros(O, dtype=np.int64); tn_cal = np.zeros(O, dtype=np.int64)

    reliability_raw = init_reliability_accumulators(O, n_bins=int(getattr(cfg, "reliability_bins", 10)))
    reliability_cal = init_reliability_accumulators(O, n_bins=int(getattr(cfg, "reliability_bins", 10)))
    accum_cal = init_accumulation_diagnostics(horizons_steps)
    regime_cal = init_regime_accumulators(["all", "dry_window", "already_raining", "rain_onset", "heavy_future"])

    max_per_lead = int(getattr(cfg, "crps_rain_max_points_per_lead", 0))
    if max_per_lead <= 0:
        max_per_lead = max(2000, int(getattr(cfg, "crps_rain_max_points", 2000)))
    n_samp = max(int(getattr(cfg, "crps_rain_min_samples", 100)), int(getattr(cfg, "crps_rain_n_samples", 30)))
    n_samp = min(max(n_samp, 32), 512)

    sum_crps = np.zeros(O, dtype=np.float64); n_crps = np.zeros(O, dtype=np.int64)
    sum_crps_wet = np.zeros(O, dtype=np.float64); n_crps_wet = np.zeros(O, dtype=np.int64)
    sum_crps_heavy = np.zeros(O, dtype=np.float64); n_crps_heavy = np.zeros(O, dtype=np.int64)

    def _sample_crps_for_indices(p0_stack: torch.Tensor, k_stack: torch.Tensor, theta_stack: torch.Tensor, y_rain: torch.Tensor, lead: int, batch_indices: torch.Tensor) -> np.ndarray:
        if batch_indices.numel() == 0:
            return np.array([], dtype=np.float64)
        # Paràmetres [J, N] per als N punts seleccionats d'un lead concret.
        p0_sel = p0_stack[:, batch_indices, lead]
        k_sel = k_stack[:, batch_indices, lead]
        th_sel = torch.clamp(theta_stack[:, batch_indices, lead], min=1e-6)
        y_sel = y_rain[batch_indices, lead]
        J, N = p0_sel.shape
        cols = torch.arange(N, device=device).unsqueeze(0).expand(n_samp, -1)
        j_idx = torch.randint(low=0, high=J, size=(n_samp, N), device=device)
        p0_s = p0_sel[j_idx, cols]
        k_s = k_sel[j_idx, cols]
        th_s = th_sel[j_idx, cols]
        gamma = torch.distributions.Gamma(concentration=torch.clamp(k_s, min=1e-6), rate=1.0 / th_s)
        samples = gamma.sample()
        u = torch.rand_like(samples)
        samples = torch.where(u < p0_s, torch.zeros_like(samples), samples)
        crps = empirical_crps_from_samples(samples, y_sel)
        return crps.detach().cpu().numpy().astype(np.float64)

    for batch in loader:
        batch = to_device(batch, device)
        p0_stack, k_stack, theta_stack, rain_mu_t = _ensemble_prob_rain_for_batch(models, batch, temp_mean=temp_mean, temp_std=temp_std)
        p_raw_t = (1.0 - p0_stack).mean(dim=0)
        p_raw = p_raw_t.detach().cpu().numpy()
        p_cal = apply_rain_probability_calibrator_np(p_raw, rain_prob_calibrator)
        rmu = rain_mu_t.detach().cpu().numpy()
        if rain_amount_calibrator is not None:
            rmu, _, _ = apply_rain_amount_calibrator_np(
                rmu, p_raw, p_cal, batch["last_rain_raw"].detach().cpu().numpy(),
                rain_amount_calibrator, steps_per_hour=int(artifacts["steps_per_hour"])
            )
        # Sortida operativa v9: mai es reporten quantitats inferiors al llindar instrumental.
        rmu = apply_operational_rain_deadband(rmu, float(getattr(cfg, "rain_amount_deadband", cfg.rain_event_threshold)))
        rtr = batch["y_rain"].detach().cpu().numpy()
        occ_true = batch["y_rain_occ"].detach().cpu().numpy().astype(np.int64)
        mR = batch["mask_rain"].detach().cpu().numpy()

        update_reliability_accumulators(reliability_raw, p_raw, occ_true, mR)
        update_reliability_accumulators(reliability_cal, p_cal, occ_true, mR)
        update_accumulation_diagnostics(accum_cal, rmu, rtr, p_cal, mR, event_threshold=rain_event_threshold)

        last_rain_np = batch["last_rain_raw"].detach().cpu().numpy()
        future_sum = np.sum(np.where(mR > 0.5, rtr, 0.0), axis=1)
        future_max = np.max(np.where(mR > 0.5, rtr, 0.0), axis=1)
        regime_masks = {
            "all": np.ones_like(last_rain_np, dtype=bool),
            "dry_window": (last_rain_np < rain_event_threshold) & (future_sum < rain_event_threshold),
            "already_raining": last_rain_np >= rain_event_threshold,
            "rain_onset": (last_rain_np < rain_event_threshold) & (future_max >= rain_event_threshold),
            "heavy_future": future_max >= heavy_threshold,
        }
        # Només passem arrays de temperatura zero perquè en aquesta passada només es recalculen mètriques de pluja.
        zeros_temp = np.zeros_like(rtr)
        update_regime_accumulators(regime_cal, regime_masks, zeros_temp, zeros_temp, rmu, rtr, p_cal, occ_true, np.zeros_like(mR), mR)

        for lead in range(O):
            valid = mR[:, lead] > 0.5
            if np.any(valid):
                yy = occ_true[valid, lead]
                pr = np.clip(p_raw[valid, lead], 0.0, 1.0)
                pc = np.clip(p_cal[valid, lead], 0.0, 1.0)
                n[lead] += int(valid.sum())
                rr = rmu[valid, lead]
                rt = rtr[valid, lead]
                sum_rain_abs[lead] += float(np.sum(np.abs(rr - rt)))
                sum_rain_sq[lead] += float(np.sum((rr - rt) ** 2))
                sum_brier_raw[lead] += float(np.sum((pr - yy) ** 2))
                sum_brier_cal[lead] += float(np.sum((pc - yy) ** 2))
                yh_raw = (pr >= 0.5).astype(np.int64)
                yh_cal = (pc >= 0.5).astype(np.int64)
                tp_raw[lead] += int(np.sum((yh_raw == 1) & (yy == 1))); fn_raw[lead] += int(np.sum((yh_raw == 0) & (yy == 1)))
                fp_raw[lead] += int(np.sum((yh_raw == 1) & (yy == 0))); tn_raw[lead] += int(np.sum((yh_raw == 0) & (yy == 0)))
                tp_cal[lead] += int(np.sum((yh_cal == 1) & (yy == 1))); fn_cal[lead] += int(np.sum((yh_cal == 0) & (yy == 1)))
                fp_cal[lead] += int(np.sum((yh_cal == 1) & (yy == 0))); tn_cal[lead] += int(np.sum((yh_cal == 0) & (yy == 0)))

            if cfg.compute_crps:
                # Mostra global natural per lead.
                if n_crps[lead] < max_per_lead:
                    valid_idx = torch.where(batch["mask_rain"][:, lead] > 0.5)[0]
                    take = min(int(max_per_lead - n_crps[lead]), int(valid_idx.numel()), 256)
                    if take > 0:
                        perm = torch.randperm(valid_idx.numel(), device=device)[:take]
                        sel = valid_idx[perm]
                        vals = _sample_crps_for_indices(p0_stack, k_stack, theta_stack, batch["y_rain"], lead, sel)
                        sum_crps[lead] += float(np.sum(vals)); n_crps[lead] += int(vals.size)
                # Mostra condicionada a pluja observada.
                if n_crps_wet[lead] < max_per_lead:
                    wet_idx = torch.where((batch["mask_rain"][:, lead] > 0.5) & (batch["y_rain"][:, lead] >= rain_event_threshold))[0]
                    take = min(int(max_per_lead - n_crps_wet[lead]), int(wet_idx.numel()), 256)
                    if take > 0:
                        perm = torch.randperm(wet_idx.numel(), device=device)[:take]
                        sel = wet_idx[perm]
                        vals = _sample_crps_for_indices(p0_stack, k_stack, theta_stack, batch["y_rain"], lead, sel)
                        sum_crps_wet[lead] += float(np.sum(vals)); n_crps_wet[lead] += int(vals.size)
                # Mostra condicionada a pluja intensa observada.
                if n_crps_heavy[lead] < max_per_lead:
                    heavy_idx = torch.where((batch["mask_rain"][:, lead] > 0.5) & (batch["y_rain"][:, lead] >= heavy_threshold))[0]
                    take = min(int(max_per_lead - n_crps_heavy[lead]), int(heavy_idx.numel()), 256)
                    if take > 0:
                        perm = torch.randperm(heavy_idx.numel(), device=device)[:take]
                        sel = heavy_idx[perm]
                        vals = _sample_crps_for_indices(p0_stack, k_stack, theta_stack, batch["y_rain"], lead, sel)
                        sum_crps_heavy[lead] += float(np.sum(vals)); n_crps_heavy[lead] += int(vals.size)

    def div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        out = np.full_like(a, np.nan, dtype=np.float64)
        m = b > 0
        out[m] = a[m] / b[m]
        return out

    brier_raw, sedi_raw = _finalize_event_scores(sum_brier_raw, n, tp_raw, fn_raw, fp_raw, tn_raw)
    brier_cal, sedi_cal = _finalize_event_scores(sum_brier_cal, n, tp_cal, fn_cal, fp_cal, tn_cal)
    rain_mae_calibrated_amount = div(sum_rain_abs, n)
    rain_rmse_calibrated_amount = np.sqrt(div(sum_rain_sq, n))
    rain_crps = div(sum_crps, n_crps)
    rain_crps_wet = div(sum_crps_wet, n_crps_wet)
    rain_crps_heavy = div(sum_crps_heavy, n_crps_heavy)

    return {
        "by_lead": {
            "rain_brier_raw": brier_raw.tolist(),
            "rain_brier_calibrated": brier_cal.tolist(),
            "rain_mae_amount_calibrated": rain_mae_calibrated_amount.tolist(),
            "rain_rmse_amount_calibrated": rain_rmse_calibrated_amount.tolist(),
            "rain_sedi_raw": sedi_raw.tolist(),
            "rain_sedi_calibrated": sedi_cal.tolist(),
            "rain_crps": rain_crps.tolist(),
            "rain_crps_wet": rain_crps_wet.tolist(),
            "rain_crps_heavy": rain_crps_heavy.tolist(),
            "rain_crps_n": n_crps.tolist(),
            "rain_crps_wet_n": n_crps_wet.tolist(),
            "rain_crps_heavy_n": n_crps_heavy.tolist(),
        },
        "summary": {
            "rain_brier_raw_mean": safe_nanmean(brier_raw),
            "rain_brier_calibrated_mean": safe_nanmean(brier_cal),
            "rain_mae_amount_calibrated_mean": safe_nanmean(rain_mae_calibrated_amount),
            "rain_rmse_amount_calibrated_mean": safe_nanmean(rain_rmse_calibrated_amount),
            "rain_sedi_raw_mean": safe_nanmean(sedi_raw),
            "rain_sedi_calibrated_mean": safe_nanmean(sedi_cal),
            "rain_crps_mean_sampled": safe_nanmean(rain_crps),
            "rain_crps_wet_mean_sampled": safe_nanmean(rain_crps_wet),
            "rain_crps_heavy_mean_sampled": safe_nanmean(rain_crps_heavy),
            "crps_rain_points": int(np.sum(n_crps)),
            "crps_rain_wet_points": int(np.sum(n_crps_wet)),
            "crps_rain_heavy_points": int(np.sum(n_crps_heavy)),
        },
        "diagnostics": {
            "rain_reliability_raw": finalize_reliability_accumulators(reliability_raw, steps_per_hour),
            "rain_reliability_calibrated": finalize_reliability_accumulators(reliability_cal, steps_per_hour),
            "rain_accumulation_calibrated": finalize_accumulation_diagnostics(accum_cal, steps_per_hour),
            "regime_metrics_calibrated": finalize_regime_accumulators(regime_cal),
        },
    }


@torch.no_grad()
def evaluate_ensemble(
    model_dir: str,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    cfg: TrainConfig,
    temp_mean: float,
    temp_std: float,
    rain_cap: float,
    artifacts: Dict[str, object],
    rain_prob_calibrator: Optional[Dict[str, object]] = None,
    rain_amount_calibrator: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """
    Avaluació final de l'ensemble.

    Parteix de l'avaluació completa original i substitueix la part probabilista de
    precipitació per una versió robusta:
      - CRPS empírica no negativa per lead time,
      - CRPS condicionada a pluja i a pluja intensa,
      - Brier/fiabilitat amb calibratge post-hoc opcional ajustat en validació.
    """
    metrics = _evaluate_ensemble_uncalibrated_core(
        model_dir=model_dir,
        loader=loader,
        device=device,
        cfg=cfg,
        temp_mean=temp_mean,
        temp_std=temp_std,
        rain_cap=rain_cap,
        artifacts=artifacts,
    )
    rain_diag = recompute_rain_probability_and_crps_diagnostics_ensemble(
        model_dir=model_dir,
        loader=loader,
        device=device,
        cfg=cfg,
        temp_mean=temp_mean,
        temp_std=temp_std,
        artifacts=artifacts,
        rain_prob_calibrator=rain_prob_calibrator,
        rain_amount_calibrator=rain_amount_calibrator,
    )

    use_cal = rain_prob_calibrator is not None
    # Mantenim mètriques raw i calibrades. Les mètriques principals de Brier/SEDI
    # passen a usar la probabilitat calibrada si hi ha calibrador, perquè és la
    # probabilitat operativa que es reportarà. La resta de quantitat de pluja no canvia.
    for key, value in rain_diag["by_lead"].items():
        metrics["by_lead"][key] = value
    if use_cal:
        metrics["by_lead"]["rain_brier"] = rain_diag["by_lead"]["rain_brier_calibrated"]
        metrics["by_lead"]["rain_sedi"] = rain_diag["by_lead"]["rain_sedi_calibrated"]
    else:
        metrics["by_lead"]["rain_brier"] = rain_diag["by_lead"]["rain_brier_raw"]
        metrics["by_lead"]["rain_sedi"] = rain_diag["by_lead"]["rain_sedi_raw"]
    metrics["by_lead"]["rain_crps"] = rain_diag["by_lead"]["rain_crps"]
    metrics["by_lead"]["rain_mae_amount_calibrated"] = rain_diag["by_lead"].get("rain_mae_amount_calibrated", metrics["by_lead"].get("rain_mae"))
    metrics["by_lead"]["rain_rmse_amount_calibrated"] = rain_diag["by_lead"].get("rain_rmse_amount_calibrated", metrics["by_lead"].get("rain_rmse"))
    if rain_amount_calibrator is not None:
        metrics["by_lead"]["rain_mae"] = metrics["by_lead"]["rain_mae_amount_calibrated"]
        metrics["by_lead"]["rain_rmse"] = metrics["by_lead"]["rain_rmse_amount_calibrated"]
    metrics["by_lead"]["rain_crps_wet"] = rain_diag["by_lead"]["rain_crps_wet"]
    metrics["by_lead"]["rain_crps_heavy"] = rain_diag["by_lead"]["rain_crps_heavy"]

    metrics["summary"].update(rain_diag["summary"])
    if rain_amount_calibrator is not None:
        metrics["summary"]["rain_mae_mean"] = rain_diag["summary"].get("rain_mae_amount_calibrated_mean", metrics["summary"].get("rain_mae_mean"))
        metrics["summary"]["rain_rmse_mean"] = rain_diag["summary"].get("rain_rmse_amount_calibrated_mean", metrics["summary"].get("rain_rmse_mean"))
        metrics["summary"]["rain_amount_calibration"] = "validation_conditional_amount_ratio_v9"
    else:
        metrics["summary"]["rain_amount_calibration"] = "none"
    metrics["summary"]["rain_brier_mean"] = rain_diag["summary"]["rain_brier_calibrated_mean" if use_cal else "rain_brier_raw_mean"]
    metrics["summary"]["rain_sedi_mean"] = rain_diag["summary"]["rain_sedi_calibrated_mean" if use_cal else "rain_sedi_raw_mean"]
    metrics["summary"]["crps_rain_mean_sampled"] = rain_diag["summary"]["rain_crps_mean_sampled"]
    metrics["summary"]["rain_probability_calibration"] = "validation_histogram_pava" if use_cal else "none_raw_probability"

    metrics.setdefault("diagnostics", {})
    metrics["diagnostics"].update(rain_diag["diagnostics"])
    if rain_amount_calibrator is not None and "rain_accumulation_calibrated" in metrics["diagnostics"]:
        metrics["diagnostics"]["rain_accumulation"] = metrics["diagnostics"]["rain_accumulation_calibrated"]
        acc_sum = metrics["diagnostics"]["rain_accumulation"].get("summary", {})
        for kk in ["rain_accum_mae_mean", "rain_accum_rmse_mean", "rain_accum_bias_mean", "rain_accum_brier_event_mean", "rain_accum_sedi_event_mean"]:
            if kk in acc_sum:
                metrics["summary"][kk] = acc_sum[kk]
    # Per compatibilitat amb informes previs, rain_reliability apunta a la versió operativa.
    metrics["diagnostics"]["rain_reliability"] = metrics["diagnostics"]["rain_reliability_calibrated" if use_cal else "rain_reliability_raw"]
    metrics["summary"]["rain_reliability_ece"] = metrics["diagnostics"]["rain_reliability"]["summary"]["expected_calibration_error"]
    metrics["summary"]["rain_reliability_ece_raw"] = metrics["diagnostics"]["rain_reliability_raw"]["summary"]["expected_calibration_error"]
    metrics["summary"]["rain_reliability_ece_calibrated"] = metrics["diagnostics"]["rain_reliability_calibrated"]["summary"]["expected_calibration_error"]

    # Skill scores recalculats amb la probabilitat operativa triada.
    metrics["skill_scores"] = build_skill_scores(metrics["by_lead"], metrics["baselines"], int(artifacts["steps_per_hour"]))
    return metrics


def evaluate_saved_ensemble(model_dir: str, csv: str) -> Dict[str, object]:
    """Reavalua un ensemble guardat amb calibratge de p_rain ajustat en validació."""
    with open(os.path.join(model_dir, "config.json"), "r", encoding="utf-8") as f:
        cfg_raw = json.load(f)
    with open(os.path.join(model_dir, "artifacts.json"), "r", encoding="utf-8") as f:
        artifacts = json.load(f)

    cfg = TrainConfig(**cfg_raw)
    cfg.csv = csv
    df, _ = load_dataset(cfg)
    splits = split_by_time(df, cfg)
    datasets, artifacts = build_datasets(splits, cfg)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    configure_gpu_perf(cfg)
    loaders = make_loaders(datasets, cfg, use_cuda)

    calibrator = None
    if bool(getattr(cfg, "calibrate_rain_probability", True)):
        print("[INFO] Ajustant calibratge post-hoc de p_rain amb el split de validació...")
        calibrator = fit_rain_probability_calibrator_ensemble(
            model_dir=model_dir,
            loader=loaders["val"],
            device=device,
            cfg=cfg,
            temp_mean=float(artifacts["temp_mean"]),
            temp_std=float(artifacts["temp_std"]),
            artifacts=artifacts,
        )
        with open(os.path.join(model_dir, "rain_probability_calibrator.json"), "w", encoding="utf-8") as f:
            json.dump(calibrator, f, indent=2, ensure_ascii=False)
        print("[OK] Calibrador de p_rain desat a rain_probability_calibrator.json")

    amount_calibrator = None
    if bool(getattr(cfg, "calibrate_rain_amount", True)):
        print("[INFO] Ajustant calibratge condicional de quantitat de pluja amb el split de validació...")
        amount_calibrator = fit_rain_amount_calibrator_ensemble(
            model_dir=model_dir,
            loader=loaders["val"],
            device=device,
            cfg=cfg,
            temp_mean=float(artifacts["temp_mean"]),
            temp_std=float(artifacts["temp_std"]),
            artifacts=artifacts,
            rain_prob_calibrator=calibrator,
        )
        with open(os.path.join(model_dir, "rain_amount_calibrator.json"), "w", encoding="utf-8") as f:
            json.dump(_json_safe(amount_calibrator), f, indent=2, ensure_ascii=False)
        print("[OK] Calibrador de quantitat desat a rain_amount_calibrator.json")

    metrics = evaluate_ensemble(
        model_dir=model_dir,
        loader=loaders["test"],
        device=device,
        cfg=cfg,
        temp_mean=float(artifacts["temp_mean"]),
        temp_std=float(artifacts["temp_std"]),
        rain_cap=float(artifacts["rain_cap_99p9"]),
        artifacts=artifacts,
        rain_prob_calibrator=calibrator,
        rain_amount_calibrator=amount_calibrator,
    )
    if calibrator is not None:
        metrics.setdefault("diagnostics", {})["rain_probability_calibrator"] = calibrator
    if amount_calibrator is not None:
        metrics.setdefault("diagnostics", {})["rain_amount_calibrator"] = amount_calibrator
    with open(os.path.join(model_dir, "test_metrics_ensemble.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    with open(os.path.join(model_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    return metrics


def train(cfg: TrainConfig, outdir: str) -> None:
    ensure_dir(outdir)

    with open(os.path.join(outdir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2, ensure_ascii=False)

    members = []
    first_res = None

    for j in range(int(cfg.n_ensembles)):
        seed_j = int(cfg.seed + j * cfg.ensemble_seed_stride)
        cfg_j = TrainConfig(**{**asdict(cfg), "seed": seed_j})
        model_name = "best_model.pt" if int(cfg.n_ensembles) == 1 else f"best_model_seed{seed_j}.pt"

        print(f"[INFO] Entrenant membre {j + 1}/{cfg.n_ensembles} amb seed={seed_j} -> {model_name}")
        res = _train_single(cfg_j, outdir=outdir, model_name=model_name)
        if first_res is None:
            first_res = res
            with open(os.path.join(outdir, "data_meta.json"), "w", encoding="utf-8") as f:
                json.dump(res["data_meta"], f, indent=2, ensure_ascii=False)
            with open(os.path.join(outdir, "artifacts.json"), "w", encoding="utf-8") as f:
                json.dump(res["artifacts"], f, indent=2, ensure_ascii=False)

        members.append({"seed": seed_j, "model": model_name, "test": res["test"]})
        metrics_path = os.path.join(outdir, f"test_metrics_seed{seed_j}.json" if int(cfg.n_ensembles) > 1 else "test_metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(res["test"], f, indent=2, ensure_ascii=False)

    if int(cfg.n_ensembles) > 1:
        with open(os.path.join(outdir, "ensemble_members.json"), "w", encoding="utf-8") as f:
            json.dump(members, f, indent=2, ensure_ascii=False)
        ens_metrics = evaluate_saved_ensemble(outdir, cfg.csv)
        print("[OK] Deep ensemble híbrid entrenat i agregat a test_metrics_ensemble.json.")
    else:
        with open(os.path.join(outdir, "test_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(members[0]["test"], f, indent=2, ensure_ascii=False)

    print("[OK] Entrenament completat. Mira test_metrics.json, skill_scores i les línies base guardades.")


# =============================================================================
# Predicció operativa per a un timestamp
# =============================================================================




# =============================================================================
# Postprocessament operatiu per a predicció puntual
# =============================================================================


def _finite_linear_slope_C_per_hour(values: np.ndarray, steps_per_hour: float, window_hours: float) -> float:
    """
    Estima una pendent recent de temperatura [°C/h] de manera causal i robusta.

    Es fa una regressió lineal només sobre els últims `window_hours`. Si hi ha
    massa pocs valors finits, cau a diferència simple últim-primer.
    """
    vals = np.asarray(values, dtype=np.float64)
    n = int(max(2, round(float(window_hours) * float(steps_per_hour))))
    vals = vals[-min(len(vals), n):]
    ok = np.isfinite(vals)
    if ok.sum() < 2:
        return 0.0
    y = vals[ok]
    x = np.arange(len(vals), dtype=np.float64)[ok] / float(steps_per_hour)
    if len(y) >= 4:
        x0 = x - x.mean()
        den = float(np.sum(x0 * x0))
        if den > 1e-9:
            return float(np.sum(x0 * (y - y.mean())) / den)
    dt = float(x[-1] - x[0])
    if dt <= 1e-9:
        return 0.0
    return float((y[-1] - y[0]) / dt)


def _recent_temperature_slope_C_per_hour(
    temp_hist: np.ndarray,
    steps_per_hour: float,
    main_window_hours: float = 3.0,
    max_abs_slope: float = 2.0,
) -> float:
    """
    Combina pendents recents d'1 h, `main_window_hours` i 6 h.

    La idea meteorològica és no reaccionar a un sol pas sorollós de 10 minuts,
    sinó capturar si la capa límit ja fa estona que s'escalfa o es refreda.
    """
    candidates = []
    for wh, ww in ((1.0, 0.50), (float(main_window_hours), 0.35), (6.0, 0.15)):
        if len(temp_hist) >= max(2, int(round(wh * steps_per_hour))):
            candidates.append((ww, _finite_linear_slope_C_per_hour(temp_hist, steps_per_hour, wh)))
    if not candidates:
        slope = 0.0
    else:
        wsum = sum(w for w, _ in candidates)
        slope = sum(w * v for w, v in candidates) / max(wsum, 1e-9)
    return float(np.clip(slope, -float(max_abs_slope), float(max_abs_slope)))


def _temperature_trend_diurnal_gate(
    slope_C_per_hour: float,
    future_toa: np.ndarray,
) -> np.ndarray:
    """
    Calcula una confiança [0,1] perquè la tendència recent continuï.

    - De nit o amb irradiància decreixent, un refredament recent és físicament
      plausible i rep més pes.
    - Amb irradiància creixent al matí, un escalfament recent rep més pes.
    - Si la tendència recent contradiu clarament el cicle radiatiu, el pes baixa.
    """
    toa = np.asarray(future_toa, dtype=np.float64)
    if toa.size == 0:
        return np.zeros(0, dtype=np.float32)
    toa_norm = np.clip(toa / 800.0, 0.0, 1.0)
    d_toa = np.diff(np.r_[toa[0], toa])

    night_or_low_sun = toa < 80.0
    solar_decreasing = d_toa <= 0.0
    solar_increasing = d_toa > 0.0

    if slope_C_per_hour < -0.05:
        gate = np.where(night_or_low_sun | solar_decreasing, 1.0, 0.35 + 0.25 * (1.0 - toa_norm))
    elif slope_C_per_hour > 0.05:
        gate = np.where(solar_increasing & (toa > 30.0), 1.0, 0.30 + 0.20 * toa_norm)
    else:
        gate = np.zeros_like(toa, dtype=np.float64)
    return np.clip(gate, 0.0, 1.0).astype(np.float32)


def apply_operational_temperature_trend_postprocess(
    temp_mu: np.ndarray,
    temp_lo80: np.ndarray,
    temp_hi80: np.ndarray,
    temp_hist: np.ndarray,
    future_toa: np.ndarray,
    steps_per_hour: float,
    window_hours: float = 3.0,
    tau_hours: float = 1.5,
    max_abs_slope_C_per_hour: float = 2.0,
    max_blend_weight: float = 0.45,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Correcció operativa suau de temperatura.

    No imposa monotonia estricta. Només barreja la sortida del model amb una
    projecció causal de la tendència recent quan aquesta és prou persistent i
    coherent amb el forçament radiatiu diürn.
    """
    temp_mu = np.asarray(temp_mu, dtype=np.float64)
    temp_lo80 = np.asarray(temp_lo80, dtype=np.float64)
    temp_hi80 = np.asarray(temp_hi80, dtype=np.float64)
    hist = np.asarray(temp_hist, dtype=np.float64)
    O = len(temp_mu)
    if O == 0 or len(hist) < 2 or max_blend_weight <= 0.0:
        return temp_mu.astype(np.float32), temp_lo80.astype(np.float32), temp_hi80.astype(np.float32), np.zeros(O, dtype=np.float32), 0.0

    slope = _recent_temperature_slope_C_per_hour(
        hist,
        steps_per_hour=steps_per_hour,
        main_window_hours=window_hours,
        max_abs_slope=max_abs_slope_C_per_hour,
    )
    lead_hours = (np.arange(O, dtype=np.float64) + 1.0) / float(steps_per_hour)
    trend_decay = np.exp(-lead_hours / max(float(tau_hours), 1e-3))
    trend_target = float(hist[-1]) + slope * lead_hours * trend_decay

    slope_conf = np.clip(abs(slope) / 0.35, 0.0, 1.0)
    diurnal_gate = _temperature_trend_diurnal_gate(slope, future_toa).astype(np.float64)
    lead_gate = np.exp(-lead_hours / 2.0)
    w = np.clip(float(max_blend_weight) * slope_conf * diurnal_gate * lead_gate, 0.0, float(max_blend_weight))

    adj = w * (trend_target - temp_mu)
    # Límit de seguretat: cap postprocessament pot moure més d'1 °C una predicció puntual.
    adj = np.clip(adj, -1.0, 1.0)

    temp_mu2 = temp_mu + adj
    temp_lo2 = temp_lo80 + adj
    temp_hi2 = temp_hi80 + adj
    return temp_mu2.astype(np.float32), temp_lo2.astype(np.float32), temp_hi2.astype(np.float32), adj.astype(np.float32), float(slope)


def apply_operational_rain_deadband(
    rain_values: np.ndarray,
    threshold_mm: float,
) -> np.ndarray:
    """Posa a zero la precipitació prevista inferior al llindar operatiu."""
    arr = np.asarray(rain_values, dtype=np.float64).copy()
    thr = float(threshold_mm)
    if thr > 0.0:
        arr[arr < thr] = 0.0
    return arr.astype(np.float32)


@torch.no_grad()
def predict_one(model_dir: str, csv: str, at: Optional[str] = None) -> pd.DataFrame:
    with open(os.path.join(model_dir, "config.json"), "r", encoding="utf-8") as f:
        cfg_raw = json.load(f)
    with open(os.path.join(model_dir, "artifacts.json"), "r", encoding="utf-8") as f:
        art = json.load(f)

    cfg = TrainConfig(**cfg_raw)
    cfg.csv = csv

    df, _ = load_dataset(cfg)
    H = int(art["history_steps"])
    O = int(art["horizon_steps"])
    feat_cols = art["feature_cols"]
    time_cols = art["time_cols"]
    seg_col = art["segment_col"]

    meteo_scaler = RobustScaler.from_state_dict(art["meteo_scaler"])
    temp_mean = float(art["temp_mean"])
    temp_std = float(art["temp_std"])

    if at is None:
        t0 = df.index.max()
    else:
        t_req = pd.Timestamp(at)
        candidates = df.index[df.index <= t_req]
        if len(candidates) == 0:
            raise ValueError(f"No hi ha cap timestamp al CSV anterior o igual a '{at}'.")
        t0 = candidates.max()

    seg_id = df.loc[t0, seg_col]
    df_seg = df[df[seg_col] == seg_id].copy()
    df_hist = df_seg.loc[:t0].tail(H)
    if len(df_hist) < H:
        raise ValueError(f"No hi ha prou historial en el segment actual per predir a {t0}. Es necessiten {H} passos.")

    X_meteo = meteo_scaler.transform(df_hist[feat_cols].to_numpy(dtype=np.float32))
    X_time = df_hist[time_cols].to_numpy(dtype=np.float32)
    x_past = np.concatenate([X_meteo, X_time], axis=1).astype(np.float32)

    dt = pd.Timedelta(cfg.freq)
    future_index = pd.date_range(t0 + dt, periods=O, freq=dt, tz=df.index.tz)
    t_future = make_time_features(
        future_index,
        latitude_deg=cfg.station_latitude_deg,
        longitude_deg=cfg.station_longitude_deg,
        use_solar_geometry=cfg.use_solar_geometry,
    ).astype(np.float32)

    baseline_ctx = {
        k: (np.asarray(v, dtype=np.float64) if isinstance(v, list) else v)
        for k, v in art["baseline_climatology"].items()
    }
    future_month = future_index.month.to_numpy(dtype=np.int64)
    future_slot = time_slot_of_day(future_index, cfg.freq).astype(np.int64)

    prevday_temp_future = np.full(O, np.nan, dtype=np.float32)
    steps_per_day = int(baseline_ctx["steps_per_day"])
    prevday_times = future_index - steps_per_day * dt
    if np.all(np.isin(prevday_times.astype("datetime64[ns]"), df_seg.index.astype("datetime64[ns]"))):
        prevday_temp_future = df_seg.loc[prevday_times, cfg.temp_col].to_numpy(dtype=np.float32)

    temp_clim = climatology_lookup(
        future_month,
        future_slot,
        np.asarray(baseline_ctx["temp_month_slot"], dtype=np.float64),
        np.asarray(baseline_ctx["temp_slot"], dtype=np.float64),
        float(baseline_ctx["temp_global"]),
    )
    solar_cosz = df.loc[future_index, "solar_cos_zenith"].to_numpy(dtype=np.float32) if "solar_cos_zenith" in df.columns and np.all(np.isin(future_index.astype("datetime64[ns]"), df.index.astype("datetime64[ns]"))) else np.clip(t_future[:, time_cols.index("solar_cos_zenith")] if "solar_cos_zenith" in time_cols else np.zeros(O, dtype=np.float32), -1.0, 1.0)
    solar_toa = 1361.0 * np.clip(t_future[:, time_cols.index("solar_toa_sw")] if "solar_toa_sw" in time_cols else np.zeros(O, dtype=np.float32), 0.0, None)

    lead_hours = (np.arange(O, dtype=np.float64) + 1.0) / float(art["steps_per_hour"])
    solar_gate = np.clip(solar_toa / 1000.0, 0.0, 1.0)
    w_persist = np.exp(-lead_hours / 1.5) * (1.0 - 0.15 * solar_gate)
    remainder = np.clip(1.0 - w_persist, 0.0, 1.0)
    temp_prevday = np.where(np.isfinite(prevday_temp_future), prevday_temp_future, temp_clim)
    solar_w = float(np.clip(cfg.solar_temp_baseline_weight, 0.0, 1.0))
    base_prevday = np.clip(0.45 + solar_w * solar_gate + 0.05 * np.clip(solar_cosz, 0.0, 1.0), 0.20, 0.85)
    w_prevday = remainder * base_prevday
    w_clim = remainder - w_prevday

    use_trend_runtime = bool(getattr(cfg, "use_temp_trend_baseline", False)) and str(art.get("temperature_baseline_version", "")).startswith("trend_v")
    temp_history_raw = df_hist[cfg.temp_col].to_numpy(dtype=np.float32)
    current_toa = 1361.0 * float(df_hist["solar_toa_sw"].iloc[-1]) if "solar_toa_sw" in df_hist.columns else 0.0
    slope = robust_recent_temperature_slope(
        temp_history_raw,
        steps_per_hour=float(art["steps_per_hour"]),
        short_hours=cfg.temp_trend_short_hours,
        mid_hours=cfg.temp_trend_mid_hours,
        long_hours=cfg.temp_trend_long_hours,
        max_abs_slope=cfg.temp_trend_max_abs_slope_C_per_hour,
    ) if use_trend_runtime else 0.0
    temp_trend_prior, _ = damped_temperature_trend_prior(
        last_temp=float(df_hist[cfg.temp_col].iloc[-1]),
        slope_c_per_h=slope,
        future_toa=solar_toa,
        current_toa=current_toa,
        steps_per_hour=float(art["steps_per_hour"]),
        trend_weight=cfg.temp_trend_baseline_weight if use_trend_runtime else 0.0,
        tau_hours=cfg.temp_trend_tau_hours,
        max_delta_c=cfg.temp_trend_max_delta_C,
    )
    temp_baseline_raw = (w_persist * temp_trend_prior + w_prevday * temp_prevday + w_clim * temp_clim).astype(np.float32)

    last_rain = float(df_hist[cfg.rain_col].iloc[-1])
    wet_state = float(last_rain >= cfg.rain_event_threshold)
    p_prev = wet_state
    rain_occ_baseline = np.zeros(O, dtype=np.float32)
    rain_mu_baseline = np.zeros(O, dtype=np.float32)
    for k in range(O):
        mk = future_month[k:k + 1]
        sk = future_slot[k:k + 1]
        p01 = climatology_lookup(mk, sk, np.asarray(baseline_ctx["markov_p01_month_slot"], dtype=np.float64), np.asarray(baseline_ctx["markov_p01_slot"], dtype=np.float64), float(baseline_ctx["markov_p01_global"]), clip_min=0.0, clip_max=1.0)[0]
        p11 = climatology_lookup(mk, sk, np.asarray(baseline_ctx["markov_p11_month_slot"], dtype=np.float64), np.asarray(baseline_ctx["markov_p11_slot"], dtype=np.float64), float(baseline_ctx["markov_p11_global"]), clip_min=0.0, clip_max=1.0)[0]
        p_prev = p_prev * p11 + (1.0 - p_prev) * p01
        rain_occ_baseline[k] = p_prev
        wet_amount = climatology_lookup(mk, sk, np.asarray(baseline_ctx["rain_amt_wet_month_slot"], dtype=np.float64), np.asarray(baseline_ctx["rain_amt_wet_slot"], dtype=np.float64), float(baseline_ctx["rain_amt_wet_global"]), clip_min=0.0)[0]
        rain_mu_baseline[k] = p_prev * wet_amount

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    member_files = sorted([fn for fn in os.listdir(model_dir) if fn.startswith("best_model_seed") and fn.endswith(".pt")])
    if not member_files:
        member_files = ["best_model.pt"]

    x_past_t = torch.from_numpy(x_past).unsqueeze(0).to(device)
    t_future_t = torch.from_numpy(t_future).unsqueeze(0).to(device)
    temp_base_t = torch.from_numpy(temp_baseline_raw).unsqueeze(0).to(device)
    rain_occ_base_t = torch.from_numpy(rain_occ_baseline).unsqueeze(0).to(device)
    rain_mu_base_t = torch.from_numpy(rain_mu_baseline).unsqueeze(0).to(device)
    last_temp_scaled_t = torch.tensor([(float(df_hist[cfg.temp_col].iloc[-1]) - temp_mean) / temp_std], dtype=torch.float32, device=device)
    last_rain_t = torch.tensor([last_rain], dtype=torch.float32, device=device)
    future_cosz_t = torch.from_numpy(np.clip(solar_cosz, -1.0, 1.0).astype(np.float32)).unsqueeze(0).to(device)
    future_toa_t = torch.from_numpy(np.clip(solar_toa, 0.0, None).astype(np.float32)).unsqueeze(0).to(device)

    mu_list, sig_list, p0_list, k_list, th_list = [], [], [], [], []
    regime_list, accum_direct_list = [], []

    for fn in member_files:
        model = HybridResidualNowcaster(
            in_channels=len(feat_cols) + len(time_cols),
            time_feat_dim=len(time_cols),
            history_steps=H,
            horizon_steps=O,
            steps_per_hour=int(art["steps_per_hour"]),
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_layers=cfg.n_layers,
            dropout=cfg.dropout,
            patch_len_hours=cfg.patch_len_hours,
            patch_stride_frac=cfg.patch_stride_frac,
            recent_window_hours=cfg.recent_window_hours,
            recent_patch_len_hours=cfg.recent_patch_len_hours,
            recent_patch_stride_frac=cfg.recent_patch_stride_frac,
            timemixer_scales=cfg.timemixer_scales,
            timemixer_ma_kernel=cfg.timemixer_ma_kernel,
            residual_temp_max_sigma=cfg.residual_temp_max_sigma,
            residual_occ_logit_max=cfg.residual_occ_logit_max,
            residual_log_wet_mean_max=cfg.residual_log_wet_mean_max,
            temp_feature_index=int(art["temp_feature_index"]),
            rain_feature_index=int(art["rain_feature_index"]),
        ).to(device)
        ckpt = torch.load(os.path.join(model_dir, fn), map_location=device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        out = model(
            x_past_t,
            t_future_t,
            temp_baseline_scaled=(temp_base_t - temp_mean) / temp_std,
            rain_occ_baseline=rain_occ_base_t,
            rain_mu_baseline=rain_mu_base_t,
            last_temp_scaled=last_temp_scaled_t,
            last_rain_raw=last_rain_t,
            future_cosz=future_cosz_t,
            future_toa=future_toa_t,
        )

        mu_list.append(out["temp_mu"].squeeze(0) * temp_std + temp_mean)
        sig_list.append(out["temp_sigma"].squeeze(0) * temp_std)
        p0_list.append(torch.sigmoid(out["rain_logit_p0"].squeeze(0)))
        k_list.append(out["rain_k"].squeeze(0))
        th_list.append(out["rain_theta"].squeeze(0))
        if "rain_regime_logits" in out:
            regime_list.append(torch.sigmoid(out["rain_regime_logits"].squeeze(0)))
        if "rain_accum_direct_log1p" in out:
            accum_direct_list.append(torch.expm1(torch.clamp(out["rain_accum_direct_log1p"].squeeze(0), min=0.0, max=math.log1p(500.0))))

    J = len(mu_list)
    mu_stack = torch.stack(mu_list, dim=0)
    sig_stack = torch.stack(sig_list, dim=0)
    mu_ens = mu_stack.mean(dim=0)
    var_ens = (sig_stack.pow(2) + mu_stack.pow(2)).mean(dim=0) - mu_ens.pow(2)
    sig_ens = torch.sqrt(torch.clamp(var_ens, min=1e-6))

    z80 = 1.2815515655446004
    temp_mu = mu_ens.detach().cpu().numpy()
    temp_lo = (mu_ens - z80 * sig_ens).detach().cpu().numpy()
    temp_hi = (mu_ens + z80 * sig_ens).detach().cpu().numpy()

    p0_stack = torch.stack(p0_list, dim=0)
    k_stack = torch.stack(k_list, dim=0)
    th_stack = torch.stack(th_list, dim=0)

    p_rain_raw = (1.0 - p0_stack).mean(dim=0).detach().cpu().numpy()
    cal_path = os.path.join(model_dir, "rain_probability_calibrator.json")
    if os.path.exists(cal_path):
        try:
            with open(cal_path, "r", encoding="utf-8") as f:
                rain_prob_calibrator = json.load(f)
            p_rain = apply_rain_probability_calibrator_np(p_rain_raw, rain_prob_calibrator)
        except Exception as exc:
            print(f"[AVÍS] No s'ha pogut aplicar el calibratge de pluja ({type(exc).__name__}: {exc}). S'usa p_rain cru.")
            p_rain = p_rain_raw
    else:
        p_rain = p_rain_raw
    rain_mu = ((1.0 - p0_stack) * (k_stack * th_stack)).mean(dim=0).detach().cpu().numpy()

    n_samp = 800
    j_idx = torch.randint(low=0, high=J, size=(n_samp,), device=device)
    p0_s = p0_stack[j_idx, :]
    k_s = k_stack[j_idx, :]
    th_s = torch.clamp(th_stack[j_idx, :], min=1e-6)
    gamma = torch.distributions.Gamma(concentration=k_s, rate=1.0 / th_s)
    samp = gamma.sample()
    u = torch.rand_like(samp)
    samp = torch.where(u < p0_s, torch.zeros_like(samp), samp)
    rain_q90 = torch.quantile(samp, 0.90, dim=0).detach().cpu().numpy()

    if regime_list:
        regime_mean = torch.stack(regime_list, dim=0).mean(dim=0).detach().cpu().numpy()  # [3,3]
    else:
        regime_mean = np.full((3, 3), np.nan, dtype=np.float32)
    if accum_direct_list:
        accum_direct_mean = torch.stack(accum_direct_list, dim=0).mean(dim=0).detach().cpu().numpy()
    else:
        accum_direct_mean = np.full(3, np.nan, dtype=np.float32)

    # ------------------------------------------------------------------
    # Postprocessament operatiu
    # ------------------------------------------------------------------
    # 1) Pluja: conservem la sortida crua del model i publiquem una sortida
    #    operativa sense plugim numèric inferior al llindar instrumental.
    rain_mu_raw_model = rain_mu.copy()
    rain_q90_raw_model = rain_q90.copy()

    amount_cal_path = os.path.join(model_dir, "rain_amount_calibrator.json")
    rain_amount_factor = np.ones_like(rain_mu_raw_model, dtype=np.float32)
    rain_mu_amount_calibrated = rain_mu_raw_model.copy()
    rain_q90_amount_calibrated = rain_q90_raw_model.copy()
    if os.path.exists(amount_cal_path):
        try:
            with open(amount_cal_path, "r", encoding="utf-8") as f:
                rain_amount_calibrator = json.load(f)
            rain_mu_amount_calibrated, rain_q90_amount_calibrated, rain_amount_factor = apply_rain_amount_calibrator_np(
                rain_mu_raw_model,
                p_rain_raw,
                p_rain,
                np.array([last_rain], dtype=np.float64),
                rain_amount_calibrator,
                steps_per_hour=int(art["steps_per_hour"]),
                rain_q=rain_q90_raw_model,
            )
            rain_mu_amount_calibrated = rain_mu_amount_calibrated.astype(np.float32)
            rain_q90_amount_calibrated = rain_q90_amount_calibrated.astype(np.float32)
            rain_amount_factor = rain_amount_factor.astype(np.float32)
        except Exception as exc:
            print(f"[AVÍS] No s'ha pogut aplicar el calibratge de quantitat ({type(exc).__name__}: {exc}). S'usa rain_mu cru.")
    rain_deadband = float(getattr(cfg, "rain_amount_deadband", cfg.rain_event_threshold))
    rain_mu_operational = apply_operational_rain_deadband(rain_mu_amount_calibrated, rain_deadband)
    rain_q90_operational = apply_operational_rain_deadband(rain_q90_amount_calibrated, rain_deadband)

    # 2) Temperatura: en v4b NO fem cap correcció posterior.
    #    Conservem la sortida crua del model entrenat com a sortida operativa.
    #    Això evita empitjoraments puntuals observats amb el postprocessament trend-aware.
    temp_mu_raw_model = temp_mu.copy()
    temp_lo_raw_model = temp_lo.copy()
    temp_hi_raw_model = temp_hi.copy()
    temp_trend_adjustment = np.zeros_like(temp_mu_raw_model, dtype=np.float32)
    temp_recent_slope_C_per_hour = float("nan")
    temp_mu = temp_mu_raw_model.copy()
    temp_lo = temp_lo_raw_model.copy()
    temp_hi = temp_hi_raw_model.copy()

    return pd.DataFrame(
        {
            "TIMESTAMP": future_index.astype(str),
            "temp_baseline": temp_baseline_raw,
            "temp_mu_raw_model": temp_mu_raw_model,
            "temp_mu": temp_mu,
            "temp_lo80_raw_model": temp_lo_raw_model,
            "temp_hi80_raw_model": temp_hi_raw_model,
            "temp_lo80": temp_lo,
            "temp_hi80": temp_hi,
            "temp_trend_adjustment": temp_trend_adjustment,
            "temp_recent_slope_C_per_hour": [temp_recent_slope_C_per_hour] * O,
            "p_rain_baseline": rain_occ_baseline,
            "p_rain_raw": p_rain_raw,
            "p_rain": p_rain,
            "rain_mu_baseline": rain_mu_baseline,
            "rain_mu_raw_model": rain_mu_raw_model,
            "rain_mu_amount_calibrated": rain_mu_amount_calibrated,
            "rain_amount_factor": rain_amount_factor,
            "rain_mu": rain_mu_operational,
            "rain_q90_raw_model": rain_q90_raw_model,
            "rain_q90_amount_calibrated": rain_q90_amount_calibrated,
            "rain_q90": rain_q90_operational,
            "rain_deadband_mm": [rain_deadband] * O,
            "p_onset_1h": [float(regime_mean[0, 0])] * O,
            "p_onset_2h": [float(regime_mean[0, 1])] * O,
            "p_onset_3h": [float(regime_mean[0, 2])] * O,
            "p_continue_1h": [float(regime_mean[1, 0])] * O,
            "p_continue_2h": [float(regime_mean[1, 1])] * O,
            "p_continue_3h": [float(regime_mean[1, 2])] * O,
            "p_decay_1h": [float(regime_mean[2, 0])] * O,
            "p_decay_2h": [float(regime_mean[2, 1])] * O,
            "p_decay_3h": [float(regime_mean[2, 2])] * O,
            "rain_accum_direct_1h": [float(accum_direct_mean[0])] * O,
            "rain_accum_direct_2h": [float(accum_direct_mean[1])] * O,
            "rain_accum_direct_3h": [float(accum_direct_mean[2])] * O,
            "solar_toa_sw": solar_toa,
            "solar_cos_zenith": solar_cosz,
            "n_members": [J] * O,
        }
    )



# =============================================================================
# Avaluació avançada final: llindars de validació, IC block-bootstrap,
# mètriques categòriques, Brier decomposition i estratificació temporal
# =============================================================================


def _json_safe(obj):
    """Converteix objectes NumPy/Pandas a tipus serialitzables en JSON."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return v
    if isinstance(obj, (pd.Timestamp,)):
        return str(obj)
    return obj


def contingency_counts_np(y_true: np.ndarray, y_score: np.ndarray, threshold: float, mask: Optional[np.ndarray] = None) -> Tuple[int, int, int, int]:
    """Retorna TP, FN, FP, TN per a una predicció categòrica score >= threshold."""
    y = np.asarray(y_true).astype(np.int64).reshape(-1)
    s = np.asarray(y_score, dtype=np.float64).reshape(-1)
    if mask is None:
        m = np.isfinite(s) & np.isfinite(y)
    else:
        m = (np.asarray(mask).reshape(-1).astype(bool)) & np.isfinite(s) & np.isfinite(y)
    if not np.any(m):
        return 0, 0, 0, 0
    yy = y[m]
    yh = (s[m] >= float(threshold)).astype(np.int64)
    tp = int(np.sum((yh == 1) & (yy == 1)))
    fn = int(np.sum((yh == 0) & (yy == 1)))
    fp = int(np.sum((yh == 1) & (yy == 0)))
    tn = int(np.sum((yh == 0) & (yy == 0)))
    return tp, fn, fp, tn


def contingency_metrics_np(tp: int, fn: int, fp: int, tn: int) -> Dict[str, float]:
    """Mètriques categòriques clàssiques per verificació de precipitació."""
    tp = int(tp); fn = int(fn); fp = int(fp); tn = int(tn)
    n = tp + fn + fp + tn
    pod = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    pofd = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
    far = fp / (tp + fp) if (tp + fp) > 0 else float("nan")
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    csi = tp / (tp + fn + fp) if (tp + fn + fp) > 0 else float("nan")
    fbias = (tp + fp) / (tp + fn) if (tp + fn) > 0 else float("nan")
    acc = (tp + tn) / n if n > 0 else float("nan")
    random_hits = ((tp + fp) * (tp + fn) / n) if n > 0 else float("nan")
    ets_den = tp + fn + fp - random_hits if np.isfinite(random_hits) else float("nan")
    ets = (tp - random_hits) / ets_den if np.isfinite(ets_den) and abs(ets_den) > 1e-12 else float("nan")
    sedi_v = sedi(tp, fn, fp, tn) if n > 0 else float("nan")
    return {
        "n": int(n),
        "tp": int(tp),
        "fn": int(fn),
        "fp": int(fp),
        "tn": int(tn),
        "pod": float(pod),
        "pofd": float(pofd),
        "far": float(far),
        "precision": float(precision),
        "csi": float(csi),
        "ets": float(ets),
        "frequency_bias": float(fbias),
        "accuracy": float(acc),
        "sedi": float(sedi_v),
    }


def categorical_metrics_for_threshold_np(y_true: np.ndarray, y_score: np.ndarray, threshold: float, mask: Optional[np.ndarray] = None) -> Dict[str, object]:
    tp, fn, fp, tn = contingency_counts_np(y_true, y_score, threshold, mask=mask)
    out = contingency_metrics_np(tp, fn, fp, tn)
    out["decision_threshold"] = float(threshold)
    return out


def threshold_grid_from_scores(scores: np.ndarray) -> np.ndarray:
    """Graella robusta de llindars candidata, combinant linspace i quantils."""
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    s = s[np.isfinite(s)]
    if s.size == 0:
        return np.linspace(0.01, 0.99, 99)
    base = np.linspace(0.01, 0.99, 99)
    qs = np.quantile(s, np.linspace(0.01, 0.99, 99))
    grid = np.unique(np.clip(np.concatenate([base, qs]), 0.0, 1.0))
    return grid


def optimize_threshold_np(
    y_true: np.ndarray,
    y_score: np.ndarray,
    mask: Optional[np.ndarray] = None,
    metric: str = "sedi",
    min_events: int = 5,
) -> Dict[str, object]:
    """Selecciona el llindar que maximitza una mètrica en validació."""
    y = np.asarray(y_true).astype(np.int64).reshape(-1)
    s = np.asarray(y_score, dtype=np.float64).reshape(-1)
    if mask is None:
        m = np.isfinite(s) & np.isfinite(y)
    else:
        m = (np.asarray(mask).reshape(-1).astype(bool)) & np.isfinite(s) & np.isfinite(y)
    if not np.any(m) or int(np.sum(y[m] == 1)) < int(min_events):
        return {"threshold": 0.5, "metric": metric, "score": float("nan"), "n": int(np.sum(m)), "n_events": int(np.sum(y[m] == 1))}
    grid = threshold_grid_from_scores(s[m])
    best_thr = 0.5
    best_score = -float("inf")
    best_metrics = None
    for thr in grid:
        met = categorical_metrics_for_threshold_np(y[m], s[m], float(thr))
        score = finite_or(met.get(metric), -1e9)
        # Desempat conservador: llindar més alt si la mètrica és pràcticament igual.
        if score > best_score + 1e-12 or (abs(score - best_score) <= 1e-12 and thr > best_thr):
            best_score = score
            best_thr = float(thr)
            best_metrics = met
    out = dict(best_metrics or {})
    out.update({"threshold": float(best_thr), "metric": metric, "score": float(best_score), "n": int(np.sum(m)), "n_events": int(np.sum(y[m] == 1))})
    return out


def optimize_thresholds_by_lead_np(
    lead_idx: np.ndarray,
    y_true: np.ndarray,
    y_score: np.ndarray,
    mask: np.ndarray,
    n_leads: int,
    metric: str = "sedi",
) -> Dict[str, object]:
    """Llindar òptim independent per lead time, ajustat només en validació."""
    lead_idx = np.asarray(lead_idx, dtype=np.int64).reshape(-1)
    thresholds = []
    scores = []
    details = []
    for k in range(int(n_leads)):
        mk = (lead_idx == k) & np.asarray(mask, dtype=bool).reshape(-1)
        opt = optimize_threshold_np(y_true, y_score, mk, metric=metric)
        thresholds.append(float(opt["threshold"]))
        scores.append(float(opt.get("score", float("nan"))))
        details.append(opt)
    return {"thresholds": thresholds, "scores": scores, "details": details, "metric": metric}


def brier_decomposition_np(prob: np.ndarray, obs01: np.ndarray, mask: Optional[np.ndarray] = None, n_bins: int = 10) -> Dict[str, object]:
    """Descomposició de Murphy del Brier Score: reliability - resolution + uncertainty."""
    p = np.clip(np.asarray(prob, dtype=np.float64).reshape(-1), 0.0, 1.0)
    y = np.asarray(obs01, dtype=np.float64).reshape(-1)
    if mask is None:
        m = np.isfinite(p) & np.isfinite(y)
    else:
        m = (np.asarray(mask).reshape(-1).astype(bool)) & np.isfinite(p) & np.isfinite(y)
    p = p[m]
    y = y[m]
    n = int(p.size)
    n_bins = max(2, int(n_bins))
    if n == 0:
        return {"n": 0, "brier": float("nan"), "reliability": float("nan"), "resolution": float("nan"), "uncertainty": float("nan")}
    ybar = float(np.mean(y))
    brier = float(np.mean((p - y) ** 2))
    idx = np.minimum((p * n_bins).astype(np.int64), n_bins - 1)
    counts = np.bincount(idx, minlength=n_bins).astype(np.float64)
    sum_p = np.bincount(idx, weights=p, minlength=n_bins).astype(np.float64)
    sum_y = np.bincount(idx, weights=y, minlength=n_bins).astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        pbar = np.where(counts > 0, sum_p / counts, np.nan)
        obar = np.where(counts > 0, sum_y / counts, np.nan)
    w = counts / max(float(n), 1.0)
    reliability = float(np.nansum(w * (pbar - obar) ** 2))
    resolution = float(np.nansum(w * (obar - ybar) ** 2))
    uncertainty = float(ybar * (1.0 - ybar))
    return {
        "n": int(n),
        "event_rate": float(ybar),
        "brier": brier,
        "reliability": reliability,
        "resolution": resolution,
        "uncertainty": uncertainty,
        "reconstructed_brier": float(reliability - resolution + uncertainty),
        "n_bins": int(n_bins),
        "bin_count": counts.astype(int).tolist(),
        "bin_mean_probability": pbar.tolist(),
        "bin_observed_frequency": obar.tolist(),
    }


def brier_decomposition_by_lead_np(lead_idx: np.ndarray, prob: np.ndarray, obs01: np.ndarray, mask: np.ndarray, n_leads: int, n_bins: int = 10) -> Dict[str, object]:
    out = []
    lead_idx = np.asarray(lead_idx, dtype=np.int64).reshape(-1)
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    for k in range(int(n_leads)):
        mk = mask & (lead_idx == k)
        out.append(brier_decomposition_np(prob, obs01, mk, n_bins=n_bins))
    return {"by_lead": out}


def gamma_mixture_exceedance_np(p0_stack: np.ndarray, k_stack: np.ndarray, theta_stack: np.ndarray, thresholds: List[float]) -> Optional[Dict[str, np.ndarray]]:
    """Probabilitat P(Y >= threshold) per a una barreja ZI-Gamma, si SciPy és disponible."""
    try:
        from scipy.special import gammaincc  # type: ignore
    except Exception:
        return None
    p0 = np.clip(np.asarray(p0_stack, dtype=np.float64), 1e-8, 1.0 - 1e-8)
    kk = np.clip(np.asarray(k_stack, dtype=np.float64), 1e-8, None)
    th = np.clip(np.asarray(theta_stack, dtype=np.float64), 1e-8, None)
    out: Dict[str, np.ndarray] = {}
    for thr in thresholds:
        x = float(thr) / th
        sf = gammaincc(kk, x)
        prob = np.mean((1.0 - p0) * sf, axis=0)
        out[str(float(thr))] = np.clip(prob, 0.0, 1.0).astype(np.float64)
    return out


def _ensemble_full_prediction_for_batch(models: List[nn.Module], batch: Dict[str, torch.Tensor], temp_mean: float, temp_std: float, cfg: TrainConfig) -> Dict[str, torch.Tensor]:
    """Predicció completa de l'ensemble per a temperatura i pluja."""
    temp_baseline_scaled = (batch["temp_baseline_raw"] - temp_mean) / temp_std
    last_temp_scaled = (batch["last_temp_raw"] - temp_mean) / temp_std
    outs = [
        model(
            batch["x_past"],
            batch["t_future"],
            temp_baseline_scaled=temp_baseline_scaled,
            rain_occ_baseline=batch["rain_occ_baseline"],
            rain_mu_baseline=batch["rain_mu_baseline"],
            last_temp_scaled=last_temp_scaled,
            last_rain_raw=batch["last_rain_raw"],
            future_cosz=batch["future_cosz"],
            future_toa=batch["future_toa"],
        )
        for model in models
    ]
    temp_mu_stack = torch.stack([o["temp_mu"] for o in outs], dim=0)
    temp_sigma_stack = torch.stack([o["temp_sigma"] for o in outs], dim=0)
    temp_mu = temp_mu_stack.mean(dim=0)
    temp_var = (temp_sigma_stack.pow(2) + temp_mu_stack.pow(2)).mean(dim=0) - temp_mu.pow(2)
    temp_sigma = torch.sqrt(torch.clamp(temp_var, min=cfg.eps))
    p0_stack = torch.stack([torch.sigmoid(o["rain_logit_p0"]) for o in outs], dim=0)
    k_stack = torch.stack([o["rain_k"] for o in outs], dim=0)
    theta_stack = torch.stack([o["rain_theta"] for o in outs], dim=0)
    p_raw = (1.0 - p0_stack).mean(dim=0)
    rain_mu = ((1.0 - p0_stack) * (k_stack * theta_stack)).mean(dim=0)
    return {
        "temp_mu_raw": temp_mu * temp_std + temp_mean,
        "temp_sigma_raw": temp_sigma * temp_std,
        "p_raw": p_raw,
        "rain_mu": rain_mu,
        "p0_stack": p0_stack,
        "k_stack": k_stack,
        "theta_stack": theta_stack,
    }


@torch.no_grad()
def collect_advanced_eval_points_ensemble(
    model_dir: str,
    loader: torch.utils.data.DataLoader,
    split_df: pd.DataFrame,
    dataset: WindowDataset,
    device: torch.device,
    cfg: TrainConfig,
    temp_mean: float,
    temp_std: float,
    artifacts: Dict[str, object],
    calibrator: Optional[Dict[str, object]],
    amount_calibrator: Optional[Dict[str, object]] = None,
    rain_thresholds: Tuple[float, ...] = (0.1, 0.5, 1.0),
    compute_exceedance: bool = True,
) -> Dict[str, np.ndarray]:
    """Recull prediccions punt a punt de l'ensemble per fer l'avaluació avançada."""
    models = _load_ensemble_models_for_final_eval(model_dir, cfg, artifacts, device)
    O = int(artifacts["horizon_steps"])
    steps_per_hour = int(artifacts["steps_per_hour"])
    arrays: Dict[str, List[np.ndarray]] = {
        "forecast_time_ns": [], "init_time_ns": [], "lead_idx": [], "lead_hours": [],
        "temp_true": [], "temp_mu": [], "temp_sigma": [], "temp_abs_error": [], "temp_sq_error": [], "temp_crps": [], "mask_temp": [],
        "rain_true": [], "rain_mu": [], "rain_abs_error": [], "rain_sq_error": [], "mask_rain": [],
        "occ_true": [], "p_raw": [], "p_calibrated": [],
    }
    exceed_store: Dict[str, List[np.ndarray]] = {}
    for thr in rain_thresholds:
        exceed_store[f"p_exceed_{float(thr)}"] = []
        exceed_store[f"event_{float(thr)}"] = []

    cursor = 0
    time_values = split_df.index.to_numpy(dtype="datetime64[ns]")
    lead_offsets = np.arange(1, O + 1, dtype=np.int64)
    for batch in loader:
        batch = to_device(batch, device)
        B = int(batch["y_rain"].shape[0])
        valid_idx = dataset.valid[cursor:cursor + B]
        cursor += B
        if len(valid_idx) != B:
            raise RuntimeError("Desalineació entre DataLoader i WindowDataset: el test loader ha de tenir shuffle=False.")

        pred = _ensemble_full_prediction_for_batch(models, batch, temp_mean=temp_mean, temp_std=temp_std, cfg=cfg)
        temp_mu = pred["temp_mu_raw"]
        temp_sigma = pred["temp_sigma_raw"]
        temp_true = batch["y_temp_raw"]
        temp_crps = crps_gaussian(temp_mu, temp_sigma, temp_true, eps=cfg.eps)
        rain_mu = pred["rain_mu"]
        p_raw = pred["p_raw"].detach().cpu().numpy()
        p_cal = apply_rain_probability_calibrator_np(p_raw, calibrator)

        tmu = temp_mu.detach().cpu().numpy()
        tsig = temp_sigma.detach().cpu().numpy()
        ttr = temp_true.detach().cpu().numpy()
        tcrps = temp_crps.detach().cpu().numpy()
        rmu = rain_mu.detach().cpu().numpy()
        if amount_calibrator is not None:
            rmu, _, _ = apply_rain_amount_calibrator_np(
                rmu, p_raw, p_cal, batch["last_rain_raw"].detach().cpu().numpy(),
                amount_calibrator, steps_per_hour=int(artifacts["steps_per_hour"])
            )
        rmu = apply_operational_rain_deadband(rmu, float(getattr(cfg, "rain_amount_deadband", cfg.rain_event_threshold)))
        rtr = batch["y_rain"].detach().cpu().numpy()
        mT = batch["mask_temp"].detach().cpu().numpy()
        mR = batch["mask_rain"].detach().cpu().numpy()
        occ = batch["y_rain_occ"].detach().cpu().numpy().astype(np.int64)

        future_idx = valid_idx[:, None] + lead_offsets[None, :]
        forecast_time = time_values[future_idx]
        init_time = time_values[valid_idx][:, None].repeat(O, axis=1)
        lead_idx = np.arange(O, dtype=np.int64)[None, :].repeat(B, axis=0)
        lead_hours = ((lead_idx.astype(np.float64) + 1.0) / float(steps_per_hour))

        arrays["forecast_time_ns"].append(forecast_time.reshape(-1).astype("datetime64[ns]").astype(np.int64))
        arrays["init_time_ns"].append(init_time.reshape(-1).astype("datetime64[ns]").astype(np.int64))
        arrays["lead_idx"].append(lead_idx.reshape(-1))
        arrays["lead_hours"].append(lead_hours.reshape(-1))
        arrays["temp_true"].append(ttr.reshape(-1))
        arrays["temp_mu"].append(tmu.reshape(-1))
        arrays["temp_sigma"].append(tsig.reshape(-1))
        arrays["temp_abs_error"].append(np.abs(tmu - ttr).reshape(-1))
        arrays["temp_sq_error"].append(((tmu - ttr) ** 2).reshape(-1))
        arrays["temp_crps"].append(tcrps.reshape(-1))
        arrays["mask_temp"].append((mT.reshape(-1) > 0.5).astype(np.int8))
        arrays["rain_true"].append(rtr.reshape(-1))
        arrays["rain_mu"].append(rmu.reshape(-1))
        arrays["rain_abs_error"].append(np.abs(rmu - rtr).reshape(-1))
        arrays["rain_sq_error"].append(((rmu - rtr) ** 2).reshape(-1))
        arrays["mask_rain"].append((mR.reshape(-1) > 0.5).astype(np.int8))
        arrays["occ_true"].append(occ.reshape(-1))
        arrays["p_raw"].append(p_raw.reshape(-1))
        arrays["p_calibrated"].append(p_cal.reshape(-1))

        # Probabilitats d'excedència per llindars d'intensitat, si SciPy és disponible.
        exceed = None
        if compute_exceedance:
            exceed = gamma_mixture_exceedance_np(
                pred["p0_stack"].detach().cpu().numpy(),
                pred["k_stack"].detach().cpu().numpy(),
                pred["theta_stack"].detach().cpu().numpy(),
                list(rain_thresholds),
            )
        for thr in rain_thresholds:
            key = str(float(thr))
            if abs(float(thr) - float(cfg.rain_event_threshold)) < 1e-9:
                p_thr = p_cal
            elif exceed is not None and key in exceed:
                p_thr = exceed[key]
            else:
                p_thr = (rmu >= float(thr)).astype(np.float64)
            exceed_store[f"p_exceed_{float(thr)}"].append(np.asarray(p_thr, dtype=np.float64).reshape(-1))
            exceed_store[f"event_{float(thr)}"].append((rtr >= float(thr)).astype(np.int64).reshape(-1))

    out: Dict[str, np.ndarray] = {k: np.concatenate(v, axis=0) for k, v in arrays.items()}
    for k, v in exceed_store.items():
        out[k] = np.concatenate(v, axis=0)
    return out


def score_by_lead_with_threshold_np(points: Dict[str, np.ndarray], score_key: str, event_key: str, threshold: float, n_leads: int) -> Dict[str, object]:
    lead = points["lead_idx"]
    mask = points["mask_rain"].astype(bool)
    global_metrics = categorical_metrics_for_threshold_np(points[event_key], points[score_key], threshold, mask)
    by_lead = []
    for k in range(int(n_leads)):
        mk = mask & (lead == k)
        by_lead.append(categorical_metrics_for_threshold_np(points[event_key], points[score_key], threshold, mk))
    return {"global": global_metrics, "by_lead": by_lead}


def score_by_lead_with_perlead_thresholds_np(points: Dict[str, np.ndarray], score_key: str, event_key: str, thresholds: List[float], n_leads: int) -> Dict[str, object]:
    lead = points["lead_idx"]
    mask = points["mask_rain"].astype(bool)
    by_lead = []
    total = {"tp": 0, "fn": 0, "fp": 0, "tn": 0}
    for k in range(int(n_leads)):
        mk = mask & (lead == k)
        thr = float(thresholds[k]) if k < len(thresholds) else 0.5
        tp, fn, fp, tn = contingency_counts_np(points[event_key], points[score_key], thr, mk)
        total["tp"] += tp; total["fn"] += fn; total["fp"] += fp; total["tn"] += tn
        met = contingency_metrics_np(tp, fn, fp, tn)
        met["decision_threshold"] = thr
        by_lead.append(met)
    glob = contingency_metrics_np(total["tp"], total["fn"], total["fp"], total["tn"])
    glob["decision_threshold"] = "per_lead"
    return {"global": glob, "by_lead": by_lead}


def build_threshold_optimization_report(val_points: Dict[str, np.ndarray], rain_thresholds: Tuple[float, ...], n_leads: int) -> Dict[str, object]:
    """Ajusta llindars en validació per ocurrència i intensitats de precipitació."""
    report: Dict[str, object] = {"rain_event_thresholds_mm_per_10min": [float(x) for x in rain_thresholds], "occurrence": {}, "intensity": {}}
    mask = val_points["mask_rain"].astype(bool)
    for score_key in ["p_raw", "p_calibrated"]:
        report["occurrence"][score_key] = {}
        for metric in ["sedi", "csi"]:
            glob = optimize_threshold_np(val_points["occ_true"], val_points[score_key], mask, metric=metric)
            per = optimize_thresholds_by_lead_np(val_points["lead_idx"], val_points["occ_true"], val_points[score_key], mask, n_leads=n_leads, metric=metric)
            report["occurrence"][score_key][metric] = {"global": glob, "per_lead": per}
    for thr in rain_thresholds:
        score_key = f"p_exceed_{float(thr)}"
        event_key = f"event_{float(thr)}"
        report["intensity"][str(float(thr))] = {}
        for metric in ["sedi", "csi"]:
            glob = optimize_threshold_np(val_points[event_key], val_points[score_key], mask, metric=metric)
            per = optimize_thresholds_by_lead_np(val_points["lead_idx"], val_points[event_key], val_points[score_key], mask, n_leads=n_leads, metric=metric)
            report["intensity"][str(float(thr))][metric] = {"global": glob, "per_lead": per, "score_key": score_key, "event_key": event_key}
    return report


def apply_threshold_report_to_test(points: Dict[str, np.ndarray], threshold_report: Dict[str, object], n_leads: int) -> Dict[str, object]:
    """Aplica al test els llindars que s'han ajustat en validació."""
    out: Dict[str, object] = {"occurrence": {}, "intensity": {}}
    for score_key in ["p_raw", "p_calibrated"]:
        out["occurrence"][score_key] = {}
        for metric in ["sedi", "csi"]:
            spec = threshold_report["occurrence"][score_key][metric]
            thr_g = float(spec["global"]["threshold"])
            out["occurrence"][score_key][metric] = {
                "global_threshold": score_by_lead_with_threshold_np(points, score_key, "occ_true", thr_g, n_leads=n_leads),
                "per_lead_threshold": score_by_lead_with_perlead_thresholds_np(points, score_key, "occ_true", spec["per_lead"]["thresholds"], n_leads=n_leads),
            }
    for thr_s, spec_thr in threshold_report["intensity"].items():
        out["intensity"][thr_s] = {}
        score_key = f"p_exceed_{float(thr_s)}"
        event_key = f"event_{float(thr_s)}"
        for metric in ["sedi", "csi"]:
            spec = spec_thr[metric]
            thr_g = float(spec["global"]["threshold"])
            out["intensity"][thr_s][metric] = {
                "global_threshold": score_by_lead_with_threshold_np(points, score_key, event_key, thr_g, n_leads=n_leads),
                "per_lead_threshold": score_by_lead_with_perlead_thresholds_np(points, score_key, event_key, spec["per_lead"]["thresholds"], n_leads=n_leads),
            }
    return out


def season_labels_from_ns(time_ns: np.ndarray) -> np.ndarray:
    dt = pd.to_datetime(time_ns)
    month = dt.month.to_numpy()
    out = np.full(month.shape, "", dtype=object)
    out[np.isin(month, [12, 1, 2])] = "DJF_hivern"
    out[np.isin(month, [3, 4, 5])] = "MAM_primavera"
    out[np.isin(month, [6, 7, 8])] = "JJA_estiu"
    out[np.isin(month, [9, 10, 11])] = "SON_tardor"
    return out


def diurnal_labels_from_ns(time_ns: np.ndarray) -> np.ndarray:
    dt = pd.to_datetime(time_ns)
    hour = dt.hour.to_numpy()
    out = np.full(hour.shape, "", dtype=object)
    out[(hour >= 0) & (hour < 6)] = "nit_00_06"
    out[(hour >= 6) & (hour < 12)] = "mati_06_12"
    out[(hour >= 12) & (hour < 18)] = "tarda_12_18"
    out[(hour >= 18) & (hour < 24)] = "vespre_18_24"
    return out


def summarize_subset_np(points: Dict[str, np.ndarray], subset_mask: np.ndarray, rain_score_key: str, rain_threshold: float) -> Dict[str, object]:
    subset = np.asarray(subset_mask, dtype=bool)
    mt = subset & points["mask_temp"].astype(bool)
    mr = subset & points["mask_rain"].astype(bool)
    temp_abs = points["temp_abs_error"][mt]
    temp_sq = points["temp_sq_error"][mt]
    temp_crps = points["temp_crps"][mt]
    rain_abs = points["rain_abs_error"][mr]
    rain_sq = points["rain_sq_error"][mr]
    p = points[rain_score_key][mr]
    y = points["occ_true"][mr]
    brier = np.mean((p - y) ** 2) if p.size else float("nan")
    cat = categorical_metrics_for_threshold_np(y, p, rain_threshold) if p.size else contingency_metrics_np(0, 0, 0, 0)
    return {
        "n_temp_points": int(mt.sum()),
        "n_rain_points": int(mr.sum()),
        "event_rate": float(np.mean(y)) if p.size else float("nan"),
        "temp_mae": float(np.mean(temp_abs)) if temp_abs.size else float("nan"),
        "temp_rmse": float(math.sqrt(np.mean(temp_sq))) if temp_sq.size else float("nan"),
        "temp_crps": float(np.mean(temp_crps)) if temp_crps.size else float("nan"),
        "rain_mae": float(np.mean(rain_abs)) if rain_abs.size else float("nan"),
        "rain_rmse": float(math.sqrt(np.mean(rain_sq))) if rain_sq.size else float("nan"),
        "rain_brier": float(brier),
        "rain_categorical": cat,
    }


def stratified_temporal_metrics_np(points: Dict[str, np.ndarray], rain_score_key: str, rain_threshold: float) -> Dict[str, object]:
    """Mètriques per mesos, estació de l'any, període del dia i lead time."""
    time_ns = points["forecast_time_ns"]
    dt = pd.to_datetime(time_ns)
    month = dt.month.to_numpy()
    hour = dt.hour.to_numpy()
    season = season_labels_from_ns(time_ns)
    diurnal = diurnal_labels_from_ns(time_ns)
    lead_idx = points["lead_idx"]
    out: Dict[str, object] = {"by_month": {}, "by_season": {}, "by_diurnal_period": {}, "by_lead": {}}
    for m in range(1, 13):
        out["by_month"][f"{m:02d}"] = summarize_subset_np(points, month == m, rain_score_key, rain_threshold)
    for s in ["DJF_hivern", "MAM_primavera", "JJA_estiu", "SON_tardor"]:
        out["by_season"][s] = summarize_subset_np(points, season == s, rain_score_key, rain_threshold)
    for d in ["nit_00_06", "mati_06_12", "tarda_12_18", "vespre_18_24"]:
        out["by_diurnal_period"][d] = summarize_subset_np(points, diurnal == d, rain_score_key, rain_threshold)
    for k in range(int(np.nanmax(lead_idx)) + 1):
        out["by_lead"][str(k + 1)] = summarize_subset_np(points, lead_idx == k, rain_score_key, rain_threshold)
    return out


def block_bootstrap_confidence_intervals_np(
    points: Dict[str, np.ndarray],
    raw_threshold: float,
    calibrated_threshold: float,
    n_boot: int = 500,
    block_days: int = 3,
    seed: int = 12345,
) -> Dict[str, object]:
    """Intervals de confiança al 95% amb block bootstrap temporal."""
    time_ns = points["forecast_time_ns"]
    day = (time_ns // (24 * 3600 * 10**9)).astype(np.int64)
    block_id = day // max(1, int(block_days))
    blocks, inv = np.unique(block_id, return_inverse=True)
    nb = len(blocks)
    if nb == 0:
        return {"n_blocks": 0}

    mt = points["mask_temp"].astype(bool)
    mr = points["mask_rain"].astype(bool)
    y = points["occ_true"].astype(np.int64)
    pr = points["p_raw"]
    pc = points["p_calibrated"]
    yh_raw = (pr >= float(raw_threshold)).astype(np.int64)
    yh_cal = (pc >= float(calibrated_threshold)).astype(np.int64)

    def bc_sum(x, valid):
        return np.bincount(inv, weights=np.where(valid, x, 0.0), minlength=nb).astype(np.float64)
    def bc_count(valid):
        return np.bincount(inv, weights=valid.astype(np.float64), minlength=nb).astype(np.float64)
    def bc_bool(cond):
        return np.bincount(inv, weights=cond.astype(np.float64), minlength=nb).astype(np.float64)

    block = {
        "temp_abs_sum": bc_sum(points["temp_abs_error"], mt),
        "temp_sq_sum": bc_sum(points["temp_sq_error"], mt),
        "temp_crps_sum": bc_sum(points["temp_crps"], mt),
        "temp_n": bc_count(mt),
        "rain_abs_sum": bc_sum(points["rain_abs_error"], mr),
        "rain_sq_sum": bc_sum(points["rain_sq_error"], mr),
        "brier_raw_sum": bc_sum((pr - y) ** 2, mr),
        "brier_cal_sum": bc_sum((pc - y) ** 2, mr),
        "rain_n": bc_count(mr),
        "tp_raw": bc_bool(mr & (yh_raw == 1) & (y == 1)),
        "fn_raw": bc_bool(mr & (yh_raw == 0) & (y == 1)),
        "fp_raw": bc_bool(mr & (yh_raw == 1) & (y == 0)),
        "tn_raw": bc_bool(mr & (yh_raw == 0) & (y == 0)),
        "tp_cal": bc_bool(mr & (yh_cal == 1) & (y == 1)),
        "fn_cal": bc_bool(mr & (yh_cal == 0) & (y == 1)),
        "fp_cal": bc_bool(mr & (yh_cal == 1) & (y == 0)),
        "tn_cal": bc_bool(mr & (yh_cal == 0) & (y == 0)),
    }

    def aggregate(sel):
        tn = np.sum(block["temp_n"][sel]); rn = np.sum(block["rain_n"][sel])
        tp_r = int(np.sum(block["tp_raw"][sel])); fn_r = int(np.sum(block["fn_raw"][sel])); fp_r = int(np.sum(block["fp_raw"][sel])); tn_r = int(np.sum(block["tn_raw"][sel]))
        tp_c = int(np.sum(block["tp_cal"][sel])); fn_c = int(np.sum(block["fn_cal"][sel])); fp_c = int(np.sum(block["fp_cal"][sel])); tn_c = int(np.sum(block["tn_cal"][sel]))
        met_raw = contingency_metrics_np(tp_r, fn_r, fp_r, tn_r)
        met_cal = contingency_metrics_np(tp_c, fn_c, fp_c, tn_c)
        return {
            "temp_mae": np.sum(block["temp_abs_sum"][sel]) / tn if tn > 0 else np.nan,
            "temp_rmse": math.sqrt(np.sum(block["temp_sq_sum"][sel]) / tn) if tn > 0 else np.nan,
            "temp_crps": np.sum(block["temp_crps_sum"][sel]) / tn if tn > 0 else np.nan,
            "rain_mae": np.sum(block["rain_abs_sum"][sel]) / rn if rn > 0 else np.nan,
            "rain_rmse": math.sqrt(np.sum(block["rain_sq_sum"][sel]) / rn) if rn > 0 else np.nan,
            "rain_brier_raw": np.sum(block["brier_raw_sum"][sel]) / rn if rn > 0 else np.nan,
            "rain_brier_calibrated": np.sum(block["brier_cal_sum"][sel]) / rn if rn > 0 else np.nan,
            "rain_sedi_raw": met_raw["sedi"],
            "rain_sedi_calibrated": met_cal["sedi"],
            "rain_csi_raw": met_raw["csi"],
            "rain_csi_calibrated": met_cal["csi"],
            "rain_far_raw": met_raw["far"],
            "rain_far_calibrated": met_cal["far"],
            "rain_pod_raw": met_raw["pod"],
            "rain_pod_calibrated": met_cal["pod"],
        }

    base = aggregate(np.arange(nb))
    rng = np.random.default_rng(int(seed))
    vals: Dict[str, List[float]] = {k: [] for k in base}
    n_boot = max(50, int(n_boot))
    for _ in range(n_boot):
        sel = rng.integers(0, nb, size=nb)
        met = aggregate(sel)
        for k, v in met.items():
            vals[k].append(float(v))
    ci = {}
    for k, arr in vals.items():
        a = np.asarray(arr, dtype=np.float64)
        ci[k] = {
            "estimate": float(base[k]),
            "ci95_low": float(np.nanpercentile(a, 2.5)),
            "ci95_high": float(np.nanpercentile(a, 97.5)),
            "bootstrap_std": float(np.nanstd(a, ddof=1)),
        }
    return {"n_blocks": int(nb), "block_days": int(block_days), "n_bootstrap": int(n_boot), "metrics": ci}


def advanced_evaluation_summary_markdown(report: Dict[str, object]) -> str:
    """Genera un resum Markdown compacte de l'avaluació avançada."""
    base = report.get("base_metrics_summary", {})
    cat = report.get("categorical_test", {}).get("occurrence", {})
    raw_sedi = cat.get("p_raw", {}).get("sedi", {}).get("global_threshold", {}).get("global", {})
    cal_sedi = cat.get("p_calibrated", {}).get("sedi", {}).get("global_threshold", {}).get("global", {})
    bd = report.get("brier_decomposition", {})
    lines = []
    lines.append("# Avaluació avançada del model Hybrid-TimeMixer no-RS\n")
    lines.append("## Resum principal\n")
    for k in ["temp_mae_mean", "temp_rmse_mean", "temp_crps_mean", "rain_mae_mean", "rain_rmse_mean", "rain_brier_raw_mean", "rain_brier_calibrated_mean", "rain_sedi_raw_mean", "rain_sedi_calibrated_mean", "rain_crps_mean_sampled", "rain_crps_wet_mean_sampled", "rain_crps_heavy_mean_sampled"]:
        if k in base:
            lines.append(f"- `{k}`: {base[k]}\n")
    lines.append("\n## Llindars optimitzats en validació per ocurrència de pluja\n")
    if raw_sedi:
        lines.append(f"- Probabilitat raw, SEDI: llindar={raw_sedi.get('decision_threshold')}, SEDI test={raw_sedi.get('sedi')}, POD={raw_sedi.get('pod')}, FAR={raw_sedi.get('far')}, CSI={raw_sedi.get('csi')}\n")
    if cal_sedi:
        lines.append(f"- Probabilitat calibrada, SEDI: llindar={cal_sedi.get('decision_threshold')}, SEDI test={cal_sedi.get('sedi')}, POD={cal_sedi.get('pod')}, FAR={cal_sedi.get('far')}, CSI={cal_sedi.get('csi')}\n")
    lines.append("\n## Descomposició de Brier\n")
    for key in ["raw", "calibrated"]:
        if key in bd:
            b = bd[key].get("global", {})
            lines.append(f"- {key}: BS={b.get('brier')}, reliability={b.get('reliability')}, resolution={b.get('resolution')}, uncertainty={b.get('uncertainty')}\n")
    lines.append("\nAquest fitxer és un resum; els detalls complets són a `advanced_evaluation_metrics.json`.\n")
    return "".join(lines)


@torch.no_grad()
def evaluate_saved_ensemble_advanced(
    model_dir: str,
    csv: str,
    n_bootstrap: int = 500,
    block_days: int = 3,
    rain_thresholds: Tuple[float, ...] = (0.1, 0.5, 1.0),
    output_json: Optional[str] = None,
) -> Dict[str, object]:
    """
    Avaluació avançada final d'un ensemble ja entrenat.

    S'ajusten llindars només amb validació i s'apliquen a test. També es calculen
    IC 95% per block bootstrap, Brier decomposition i mètriques estratificades.
    """
    with open(os.path.join(model_dir, "config.json"), "r", encoding="utf-8") as f:
        cfg_raw = json.load(f)
    with open(os.path.join(model_dir, "artifacts.json"), "r", encoding="utf-8") as f:
        artifacts_saved = json.load(f)
    cfg = TrainConfig(**cfg_raw)
    cfg.csv = csv
    df, _ = load_dataset(cfg)
    splits = split_by_time(df, cfg)
    datasets, artifacts = build_datasets(splits, cfg)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    configure_gpu_perf(cfg)
    loaders = make_loaders(datasets, cfg, use_cuda)

    calibrator_path = os.path.join(model_dir, "rain_probability_calibrator.json")
    calibrator = None
    if os.path.exists(calibrator_path):
        with open(calibrator_path, "r", encoding="utf-8") as f:
            calibrator = json.load(f)
    elif bool(getattr(cfg, "calibrate_rain_probability", True)):
        calibrator = fit_rain_probability_calibrator_ensemble(
            model_dir=model_dir,
            loader=loaders["val"],
            device=device,
            cfg=cfg,
            temp_mean=float(artifacts["temp_mean"]),
            temp_std=float(artifacts["temp_std"]),
            artifacts=artifacts,
        )
        with open(calibrator_path, "w", encoding="utf-8") as f:
            json.dump(_json_safe(calibrator), f, indent=2, ensure_ascii=False)

    amount_calibrator_path = os.path.join(model_dir, "rain_amount_calibrator.json")
    amount_calibrator = None
    if os.path.exists(amount_calibrator_path):
        with open(amount_calibrator_path, "r", encoding="utf-8") as f:
            amount_calibrator = json.load(f)
    elif bool(getattr(cfg, "calibrate_rain_amount", True)):
        amount_calibrator = fit_rain_amount_calibrator_ensemble(
            model_dir=model_dir,
            loader=loaders["val"],
            device=device,
            cfg=cfg,
            temp_mean=float(artifacts["temp_mean"]),
            temp_std=float(artifacts["temp_std"]),
            artifacts=artifacts,
            rain_prob_calibrator=calibrator,
        )
        with open(amount_calibrator_path, "w", encoding="utf-8") as f:
            json.dump(_json_safe(amount_calibrator), f, indent=2, ensure_ascii=False)

    print("[INFO] Recollint prediccions de validació per optimitzar llindars...")
    val_points = collect_advanced_eval_points_ensemble(
        model_dir, loaders["val"], splits["val"], datasets["val"], device, cfg,
        temp_mean=float(artifacts["temp_mean"]), temp_std=float(artifacts["temp_std"]),
        artifacts=artifacts, calibrator=calibrator, amount_calibrator=amount_calibrator, rain_thresholds=rain_thresholds,
    )
    print("[INFO] Recollint prediccions de test per a l'avaluació avançada...")
    test_points = collect_advanced_eval_points_ensemble(
        model_dir, loaders["test"], splits["test"], datasets["test"], device, cfg,
        temp_mean=float(artifacts["temp_mean"]), temp_std=float(artifacts["temp_std"]),
        artifacts=artifacts, calibrator=calibrator, amount_calibrator=amount_calibrator, rain_thresholds=rain_thresholds,
    )

    O = int(artifacts["horizon_steps"])
    threshold_report = build_threshold_optimization_report(val_points, rain_thresholds, n_leads=O)
    categorical_test = apply_threshold_report_to_test(test_points, threshold_report, n_leads=O)

    raw_thr = float(threshold_report["occurrence"]["p_raw"]["sedi"]["global"]["threshold"])
    cal_thr = float(threshold_report["occurrence"]["p_calibrated"]["sedi"]["global"]["threshold"])

    mask_rain = test_points["mask_rain"].astype(bool)
    brier_decomp = {
        "raw": {
            "global": brier_decomposition_np(test_points["p_raw"], test_points["occ_true"], mask_rain, n_bins=int(cfg.reliability_bins)),
            "by_lead": brier_decomposition_by_lead_np(test_points["lead_idx"], test_points["p_raw"], test_points["occ_true"], mask_rain, n_leads=O, n_bins=int(cfg.reliability_bins)),
        },
        "calibrated": {
            "global": brier_decomposition_np(test_points["p_calibrated"], test_points["occ_true"], mask_rain, n_bins=int(cfg.reliability_bins)),
            "by_lead": brier_decomposition_by_lead_np(test_points["lead_idx"], test_points["p_calibrated"], test_points["occ_true"], mask_rain, n_leads=O, n_bins=int(cfg.reliability_bins)),
        },
    }

    bootstrap_ci = block_bootstrap_confidence_intervals_np(
        test_points,
        raw_threshold=raw_thr,
        calibrated_threshold=cal_thr,
        n_boot=int(n_bootstrap),
        block_days=int(block_days),
        seed=int(cfg.seed) + 9001,
    )

    stratified = stratified_temporal_metrics_np(test_points, rain_score_key="p_calibrated", rain_threshold=cal_thr)
    stratified_raw = stratified_temporal_metrics_np(test_points, rain_score_key="p_raw", rain_threshold=raw_thr)

    # Mètriques deterministes per llindars d'intensitat basades en quantitat prevista.
    intensity_deterministic: Dict[str, object] = {}
    for thr in rain_thresholds:
        event_key = f"event_{float(thr)}"
        score_key = f"p_exceed_{float(thr)}"
        intensity_deterministic[str(float(thr))] = {
            "probabilistic_at_validation_thresholds": categorical_test["intensity"].get(str(float(thr)), {}),
            "deterministic_rain_mu_ge_threshold": score_by_lead_with_threshold_np(
                {**test_points, "rain_mu_score": test_points["rain_mu"]},
                "rain_mu_score",
                event_key,
                float(thr),
                n_leads=O,
            ),
            "brier_decomposition_probability_exceedance": brier_decomposition_np(test_points[score_key], test_points[event_key], mask_rain, n_bins=int(cfg.reliability_bins)),
        }

    base_metrics_summary = {}
    metrics_path = os.path.join(model_dir, "test_metrics_ensemble.json")
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                base_metrics_summary = json.load(f).get("summary", {})
        except Exception:
            base_metrics_summary = {}

    report: Dict[str, object] = {
        "advanced_evaluation_version": "v3_thresholds_bootstrap_brier_temporal",
        "model_dir": model_dir,
        "csv": csv,
        "n_test_points_flat": int(len(test_points["occ_true"])),
        "n_validation_points_flat": int(len(val_points["occ_true"])),
        "rain_thresholds_mm_per_10min": [float(x) for x in rain_thresholds],
        "base_metrics_summary": base_metrics_summary,
        "thresholds_selected_on_validation": threshold_report,
        "categorical_test": categorical_test,
        "brier_decomposition": brier_decomp,
        "block_bootstrap_ci95": bootstrap_ci,
        "temporal_stratification_calibrated_probability": stratified,
        "temporal_stratification_raw_probability": stratified_raw,
        "intensity_threshold_diagnostics": intensity_deterministic,
        "notes": {
            "probability_calibration": "El calibratge de p_rain s'ajusta amb validació i s'aplica a test.",
            "categorical_thresholds": "Els llindars de decisió s'optimitzen exclusivament en validació.",
            "bootstrap": "Els intervals de confiança usen block bootstrap temporal per respectar l'autocorrelació.",
            "brier_decomposition": "Brier = reliability - resolution + uncertainty.",
        },
    }

    if output_json is None:
        output_json = os.path.join(model_dir, "advanced_evaluation_metrics.json")
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(_json_safe(report), f, indent=2, ensure_ascii=False)
    summary_path = os.path.join(os.path.dirname(output_json), "advanced_evaluation_summary.md")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(advanced_evaluation_summary_markdown(report))
    print(f"[OK] Avaluació avançada desada a: {output_json}")
    print(f"[OK] Resum Markdown desat a: {summary_path}")
    return report




def _is_stage_complete(model_dir: str, n_ensembles: int, seed: int, stride: int) -> bool:
    """Comprova si un directori conté una fase d'entrenament aparentment completa."""
    if not model_dir or not os.path.isdir(model_dir):
        return False
    if not os.path.isfile(os.path.join(model_dir, "ensemble_members.json")):
        return False
    if not os.path.isfile(os.path.join(model_dir, "config.json")):
        return False
    for i in range(int(n_ensembles)):
        member_seed = int(seed) + i * int(stride)
        ckpt = os.path.join(model_dir, f"best_model_seed{member_seed}.pt")
        if not os.path.isfile(ckpt):
            return False
    return True



def train_v9_end_to_end(cfg: TrainConfig, outdir: str) -> None:
    """
    Entrenament v9 autocontingut en dues fases.

    Fase 1
    ------
    Entrena el model complet CC-compact amb la mateixa arquitectura CC-compact de la v9.
    Aquesta fase aprèn la representació compartida i la millor temperatura.

    Fase 2
    ------
    Inicialitza des dels checkpoints de la fase 1, congela encoder + branca de
    temperatura i reajusta només la branca de precipitació. El resultat final
    queda a `outdir`.

    Aquest mode permet entrenar la v9 sense disposar prèviament de cap carpeta entrenada.
    """
    stage1_outdir = cfg.v9_stage1_outdir or f"{outdir}__stage1_base"

    if cfg.init_model_dir:
        raise ValueError(
            "En mode --v9-end-to-end no s'ha d'indicar --init-model-dir. "
            "La fase 2 s'inicialitza automàticament des de la fase 1."
        )
    if cfg.rain_finetune_only:
        raise ValueError(
            "En mode --v9-end-to-end no indiquis --rain-finetune-only manualment. "
            "El codi l'activa només a la fase 2."
        )

    print("\n" + "=" * 90)
    print("[V9 END-TO-END] FASE 1/2: entrenament complet CC-compact")
    print(f"[V9 END-TO-END] OUTDIR fase 1: {stage1_outdir}")
    print("=" * 90 + "\n")

    stage1_cfg = copy.deepcopy(cfg)
    stage1_cfg.v9_end_to_end = False
    stage1_cfg.v9_stage1_outdir = None
    stage1_cfg.init_model_dir = None
    stage1_cfg.rain_finetune_only = False
    stage1_cfg.max_epochs = int(cfg.v9_stage1_max_epochs)
    stage1_cfg.patience = int(cfg.v9_stage1_patience)
    if cfg.v9_stage1_lr is not None:
        stage1_cfg.lr = float(cfg.v9_stage1_lr)

    if cfg.v9_skip_stage1_if_complete and _is_stage_complete(
        stage1_outdir,
        n_ensembles=cfg.n_ensembles,
        seed=cfg.seed,
        stride=cfg.ensemble_seed_stride,
    ):
        print(f"[V9 END-TO-END] Fase 1 ja existent i completa. Es reutilitza: {stage1_outdir}")
    else:
        train(stage1_cfg, outdir=stage1_outdir)

    print("\n" + "=" * 90)
    print("[V9 END-TO-END] FASE 2/2: fine-tuning només de precipitació")
    print(f"[V9 END-TO-END] INIT_MODEL_DIR fase 2: {stage1_outdir}")
    print(f"[V9 END-TO-END] OUTDIR final v9: {outdir}")
    print("=" * 90 + "\n")

    stage2_cfg = copy.deepcopy(cfg)
    stage2_cfg.v9_end_to_end = False
    stage2_cfg.v9_stage1_outdir = None
    stage2_cfg.init_model_dir = stage1_outdir
    stage2_cfg.rain_finetune_only = True
    stage2_cfg.max_epochs = int(cfg.v9_stage2_max_epochs)
    stage2_cfg.patience = int(cfg.v9_stage2_patience)
    if cfg.v9_stage2_lr is not None:
        stage2_cfg.lr = float(cfg.v9_stage2_lr)

    # Durant el fine-tuning només de pluja, aquestes pèrdues de temperatura no són necessàries:
    # els pesos de temperatura estan congelats i l'objectiu és reajustar probabilitat/quantitat.
    stage2_cfg.physics_temp_smooth_weight = 0.0
    stage2_cfg.physics_temp_anchor_weight = 0.0
    stage2_cfg.physics_temp_trend_weight = 0.0
    stage2_cfg.physics_temp_diurnal_trend_weight = 0.0
    stage2_cfg.physics_temp_rain_cooling_weight = 0.0
    stage2_cfg.val_temp_crps_weight = 0.0

    # Hiperparàmetres específics de la fase v9 rain-head fine-tune.
    # Es fan més conservadors que a la fase base per reduir residus de pluja seca
    # i millorar la discriminació sense desestabilitzar la quantitat.
    stage2_cfg.rain_regime_aux_weight = 0.025
    stage2_cfg.rain_accum_direct_weight = 0.035
    stage2_cfg.rain_preonset_sample_weight = 0.05
    stage2_cfg.rain_decay_sample_weight = 0.05
    stage2_cfg.rain_pos_weight = 0.35
    stage2_cfg.rain_intensity_alpha = 0.12
    stage2_cfg.rain_occurrence_bce_weight = 0.25
    stage2_cfg.rain_occ_pos_weight = 2.0
    stage2_cfg.rain_occ_focal_gamma = 1.0
    stage2_cfg.physics_rain_accum_weight = 0.18
    stage2_cfg.rain_heavy_step_weight = 0.06
    stage2_cfg.rain_heavy_accum_weight = 0.04
    stage2_cfg.rain_heavy_under_weight = 1.35
    stage2_cfg.val_rain_mae_weight = 0.70
    stage2_cfg.val_rain_brier_weight = 4.00
    stage2_cfg.val_rain_nll_weight = 0.20
    stage2_cfg.val_rain_sedi_weight = 0.20
    stage2_cfg.val_rain_crps_weight = 0.10

    train(stage2_cfg, outdir=outdir)

    print("\n" + "=" * 90)
    print("[V9 END-TO-END] Entrenament complet acabat.")
    print(f"[V9 END-TO-END] Resultat final: {outdir}")
    print(f"[V9 END-TO-END] Base interna fase 1: {stage1_outdir}")
    print("=" * 90 + "\n")


# =============================================================================
# CLI
# =============================================================================



def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    p_tr = sub.add_parser("train", help="Entrenament del model híbrid.")
    p_tr.add_argument("--csv", required=True)
    p_tr.add_argument("--datetime-col", required=True)
    p_tr.add_argument("--temp-col", required=True)
    p_tr.add_argument("--rain-col", required=True)
    p_tr.add_argument("--segment-col", default="segment_id")
    p_tr.add_argument("--feature-set", choices=["compact", "full"], default="compact")
    p_tr.add_argument("--timezone", default="naive")
    p_tr.add_argument("--freq", default="10min")
    p_tr.add_argument("--history-hours", type=float, default=48.0)
    p_tr.add_argument("--horizon-hours", type=float, default=3.0)
    p_tr.add_argument("--train-end", default="2023-01-01")
    p_tr.add_argument("--val-end", default="2024-01-01")
    p_tr.add_argument("--test-end", default="2025-01-01")
    p_tr.add_argument("--seed", type=int, default=42)
    p_tr.add_argument("--batch-size", type=int, default=128)
    p_tr.add_argument("--num-workers", type=int, default=0)
    p_tr.add_argument("--max-epochs", type=int, default=30)
    p_tr.add_argument("--patience", type=int, default=6)
    p_tr.add_argument("--lr", type=float, default=3e-4)
    p_tr.add_argument("--weight-decay", type=float, default=1e-4)
    p_tr.add_argument("--grad-clip", type=float, default=1.0)
    p_tr.add_argument("--d-model", type=int, default=256)
    p_tr.add_argument("--n-heads", type=int, default=8)
    p_tr.add_argument("--n-layers", type=int, default=4)
    p_tr.add_argument("--dropout", type=float, default=0.15)
    p_tr.add_argument("--patch-len-hours", type=float, default=1.0)
    p_tr.add_argument("--patch-stride-frac", type=float, default=0.5)
    p_tr.add_argument("--recent-window-hours", type=float, default=6.0)
    p_tr.add_argument("--recent-patch-len-hours", type=float, default=0.5)
    p_tr.add_argument("--recent-patch-stride-frac", type=float, default=0.5)
    p_tr.add_argument("--timemixer-scales", default="1,2,4,8")
    p_tr.add_argument("--timemixer-ma-kernel", type=int, default=5)
    p_tr.add_argument("--residual-temp-max-sigma", type=float, default=3.0)
    p_tr.add_argument("--residual-occ-logit-max", type=float, default=4.0)
    p_tr.add_argument("--residual-log-wet-mean-max", type=float, default=3.0)
    p_tr.add_argument("--rain-loss-weight", type=float, default=1.0)
    p_tr.add_argument("--n-ensembles", type=int, default=1)
    p_tr.add_argument("--ensemble-seed-stride", type=int, default=1)
    p_tr.add_argument("--station-latitude-deg", type=float, default=None)
    p_tr.add_argument("--station-longitude-deg", type=float, default=None)
    p_tr.add_argument("--no-solar-geometry", action="store_true")
    p_tr.add_argument("--rain-pos-weight", type=float, default=0.0)
    p_tr.add_argument("--rain-intensity-alpha", type=float, default=0.0)
    p_tr.add_argument("--rain-occurrence-bce-weight", type=float, default=0.0)
    p_tr.add_argument("--rain-occ-pos-weight", type=float, default=0.0)
    p_tr.add_argument("--rain-occ-focal-gamma", type=float, default=0.0)
    p_tr.add_argument("--physics-temp-smooth-weight", type=float, default=0.01)
    p_tr.add_argument("--physics-temp-anchor-weight", type=float, default=0.02)
    p_tr.add_argument("--physics-rain-accum-weight", type=float, default=0.20)
    p_tr.add_argument("--physics-temp-trend-weight", type=float, default=0.005)
    p_tr.add_argument("--physics-temp-diurnal-trend-weight", type=float, default=0.0)
    p_tr.add_argument("--solar-temp-baseline-weight", type=float, default=0.25)
    p_tr.add_argument("--no-temp-trend-baseline", action="store_true")
    p_tr.add_argument("--temp-trend-baseline-weight", type=float, default=0.15)
    p_tr.add_argument("--temp-trend-short-hours", type=float, default=1.0)
    p_tr.add_argument("--temp-trend-mid-hours", type=float, default=3.0)
    p_tr.add_argument("--temp-trend-long-hours", type=float, default=6.0)
    p_tr.add_argument("--temp-trend-tau-hours", type=float, default=1.75)
    p_tr.add_argument("--temp-trend-max-abs-slope-C-per-hour", dest="temp_trend_max_abs_slope_C_per_hour", type=float, default=2.0)
    p_tr.add_argument("--temp-trend-max-delta-C", dest="temp_trend_max_delta_C", type=float, default=1.25)
    p_tr.add_argument("--rain-heavy-quantile", type=float, default=0.97)
    p_tr.add_argument("--rain-heavy-step-weight", type=float, default=0.08)
    p_tr.add_argument("--rain-heavy-accum-weight", type=float, default=0.06)
    p_tr.add_argument("--rain-heavy-under-weight", type=float, default=1.50)
    p_tr.add_argument("--rain-heavy-intensity-power", type=float, default=0.35)
    p_tr.add_argument("--no-crps", action="store_true")
    p_tr.add_argument("--crps-rain-n-samples", type=int, default=30)
    p_tr.add_argument("--crps-rain-max-points", type=int, default=2000)
    p_tr.add_argument("--crps-rain-max-points-per-lead", type=int, default=2000)
    p_tr.add_argument("--crps-rain-min-samples", type=int, default=100)
    p_tr.add_argument("--no-rain-prob-calibration", action="store_true")
    p_tr.add_argument("--rain-calibration-bins", type=int, default=20)
    p_tr.add_argument("--rain-calibration-smoothing", type=float, default=25.0)
    p_tr.add_argument("--rain-calibration-max-points", type=int, default=600000)
    p_tr.add_argument("--extreme-temp-q", type=float, default=0.99)
    p_tr.add_argument("--extreme-rain-q", type=float, default=0.99)
    p_tr.add_argument("--rain-event-threshold", type=float, default=0.1)
    p_tr.add_argument("--rain-amount-deadband", type=float, default=0.1, help="Llindar operatiu: rain_mu/rain_q90 < llindar es publica com 0 mm.")
    p_tr.add_argument("--no-rain-amount-calibration", action="store_true", help="Desactiva el calibratge condicional de quantitat v9.")
    p_tr.add_argument("--rain-amount-calibration-max-points", type=int, default=600000)
    p_tr.add_argument("--rain-amount-calibration-smoothing", type=float, default=120.0)
    p_tr.add_argument("--rain-amount-calibration-min-cell-count", type=int, default=80)
    p_tr.add_argument("--rain-amount-calibration-min-points", type=int, default=200)
    p_tr.add_argument("--rain-amount-calibration-min-factor", type=float, default=1.0)
    p_tr.add_argument("--rain-amount-calibration-max-factor", type=float, default=3.0)
    p_tr.add_argument("--rain-amount-apply-p-cal-threshold", type=float, default=0.20)
    p_tr.add_argument("--rain-amount-apply-p-raw-threshold", type=float, default=0.35)
    p_tr.add_argument("--rain-amount-apply-mu-threshold", type=float, default=0.03)
    p_tr.add_argument("--rain-cap-quantile", type=float, default=0.999)

    # V7: features i pèrdues auxiliars de règim.
    p_tr.add_argument("--no-v7-regime-features", action="store_true")
    p_tr.add_argument("--radiation-column", default="Rad_Global_Avg")
    p_tr.add_argument("--humidity-column", default="Hum_Rel_1_Avg")
    p_tr.add_argument("--pressure-column", default="Pres_Atm_Avg")
    p_tr.add_argument("--wind-speed-column", default="Vel_Viento_WVc(1)")
    p_tr.add_argument("--wind-dir-column", default="Dir_Viento_Avg")
    p_tr.add_argument("--rain-regime-aux-weight", type=float, default=0.06)
    p_tr.add_argument("--rain-onset-pos-weight", type=float, default=4.0)
    p_tr.add_argument("--rain-continue-pos-weight", type=float, default=1.5)
    p_tr.add_argument("--rain-decay-pos-weight", type=float, default=2.0)
    p_tr.add_argument("--rain-accum-direct-weight", type=float, default=0.08)
    p_tr.add_argument("--rain-accum-direct-under-weight", type=float, default=1.35)
    p_tr.add_argument("--rain-preonset-sample-weight", type=float, default=0.20)
    p_tr.add_argument("--rain-decay-sample-weight", type=float, default=0.15)
    p_tr.add_argument("--physics-temp-rain-cooling-weight", type=float, default=0.003)
    p_tr.add_argument("--val-temp-crps-weight", type=float, default=0.20)
    p_tr.add_argument("--val-rain-mae-weight", type=float, default=0.55)
    p_tr.add_argument("--val-rain-brier-weight", type=float, default=3.50)
    p_tr.add_argument("--val-rain-nll-weight", type=float, default=0.25)
    p_tr.add_argument("--val-rain-sedi-weight", type=float, default=0.20)
    p_tr.add_argument("--val-rain-crps-weight", type=float, default=0.10)
    p_tr.add_argument("--reliability-bins", type=int, default=10)
    p_tr.add_argument("--init-model-dir", default=None, help="Directori amb checkpoints compatibles per inicialitzar el fine-tuning.")
    p_tr.add_argument("--rain-finetune-only", action="store_true", help="Congela la temperatura/encoder i entrena només la branca de precipitació.")

    # Mode v9 end-to-end: no necessita cap execució prèvia.
    # Primer entrena una base CC-compact completa, després fa fine-tuning de pluja i finalment calibra la quantitat.
    p_tr.add_argument("--v9-end-to-end", action="store_true", help="Executa dues fases automàtiques: base completa + fine-tuning rain-head + calibratge de quantitat.")
    p_tr.add_argument("--v9-stage1-outdir", default=None, help="Directori de la fase 1. Per defecte: <outdir>__stage1_base.")
    p_tr.add_argument("--v9-stage1-max-epochs", type=int, default=30)
    p_tr.add_argument("--v9-stage1-patience", type=int, default=6)
    p_tr.add_argument("--v9-stage1-lr", type=float, default=None)
    p_tr.add_argument("--v9-stage2-max-epochs", type=int, default=12)
    p_tr.add_argument("--v9-stage2-patience", type=int, default=4)
    p_tr.add_argument("--v9-stage2-lr", type=float, default=None)
    p_tr.add_argument("--v9-skip-stage1-if-complete", action="store_true", help="Reutilitza la fase 1 si ja hi ha ensemble_members.json i best_model_seed*.pt.")
    p_tr.add_argument("--outdir", required=True)
    p_tr.add_argument("--no-tf32", action="store_true")
    p_tr.add_argument("--compile", action="store_true")
    p_tr.add_argument("--compile-mode", default="max-autotune")

    p_ens = sub.add_parser("evaluate-ensemble", help="Avalua l'ensemble híbrid agregat guardat.")
    p_ens.add_argument("--model-dir", required=True)
    p_ens.add_argument("--csv", required=True)

    p_adv = sub.add_parser("evaluate-advanced", help="Avaluació avançada final: llindars de validació, block-bootstrap, Brier decomposition i estratificació temporal.")
    p_adv.add_argument("--model-dir", required=True)
    p_adv.add_argument("--csv", required=True)
    p_adv.add_argument("--n-bootstrap", type=int, default=500)
    p_adv.add_argument("--block-days", type=int, default=3)
    p_adv.add_argument("--rain-thresholds", default="0.1,0.5,1.0")
    p_adv.add_argument("--output-json", default=None)

    p_pred = sub.add_parser("predict", help="Predicció amb el model híbrid entrenat.")
    p_pred.add_argument("--model-dir", required=True)
    p_pred.add_argument("--csv", required=True)
    p_pred.add_argument("--at", default=None)

    return p



def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.cmd == "train":
        cfg = TrainConfig(
            csv=args.csv,
            datetime_col=args.datetime_col,
            temp_col=args.temp_col,
            rain_col=args.rain_col,
            segment_col=args.segment_col,
            feature_set=args.feature_set,
            timezone=args.timezone,
            freq=args.freq,
            history_hours=args.history_hours,
            horizon_hours=args.horizon_hours,
            train_end=args.train_end,
            val_end=args.val_end,
            test_end=args.test_end,
            seed=args.seed,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_epochs=args.max_epochs,
            patience=args.patience,
            lr=args.lr,
            weight_decay=args.weight_decay,
            grad_clip=args.grad_clip,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            dropout=args.dropout,
            patch_len_hours=args.patch_len_hours,
            patch_stride_frac=args.patch_stride_frac,
            recent_window_hours=args.recent_window_hours,
            recent_patch_len_hours=args.recent_patch_len_hours,
            recent_patch_stride_frac=args.recent_patch_stride_frac,
            timemixer_scales=tuple(int(v) for v in str(args.timemixer_scales).split(",") if str(v).strip()),
            timemixer_ma_kernel=args.timemixer_ma_kernel,
            residual_temp_max_sigma=args.residual_temp_max_sigma,
            residual_occ_logit_max=args.residual_occ_logit_max,
            residual_log_wet_mean_max=args.residual_log_wet_mean_max,
            rain_loss_weight=args.rain_loss_weight,
            rain_pos_weight=args.rain_pos_weight,
            rain_intensity_alpha=args.rain_intensity_alpha,
            rain_occurrence_bce_weight=args.rain_occurrence_bce_weight,
            rain_occ_pos_weight=args.rain_occ_pos_weight,
            rain_occ_focal_gamma=args.rain_occ_focal_gamma,
            physics_temp_smooth_weight=args.physics_temp_smooth_weight,
            physics_temp_anchor_weight=args.physics_temp_anchor_weight,
            physics_rain_accum_weight=args.physics_rain_accum_weight,
            physics_temp_trend_weight=args.physics_temp_trend_weight,
            physics_temp_diurnal_trend_weight=args.physics_temp_diurnal_trend_weight,
            use_temp_trend_baseline=not args.no_temp_trend_baseline,
            temp_trend_baseline_weight=args.temp_trend_baseline_weight,
            temp_trend_short_hours=args.temp_trend_short_hours,
            temp_trend_mid_hours=args.temp_trend_mid_hours,
            temp_trend_long_hours=args.temp_trend_long_hours,
            temp_trend_tau_hours=args.temp_trend_tau_hours,
            temp_trend_max_abs_slope_C_per_hour=getattr(args, "temp_trend_max_abs_slope_C_per_hour", args.temp_trend_max_abs_slope_c_per_hour if hasattr(args, "temp_trend_max_abs_slope_c_per_hour") else 2.0),
            temp_trend_max_delta_C=args.temp_trend_max_delta_C,
            rain_heavy_quantile=args.rain_heavy_quantile,
            rain_heavy_step_weight=args.rain_heavy_step_weight,
            rain_heavy_accum_weight=args.rain_heavy_accum_weight,
            rain_heavy_under_weight=args.rain_heavy_under_weight,
            rain_heavy_intensity_power=args.rain_heavy_intensity_power,
            use_solar_geometry=not args.no_solar_geometry,
            station_latitude_deg=args.station_latitude_deg,
            station_longitude_deg=args.station_longitude_deg,
            solar_temp_baseline_weight=args.solar_temp_baseline_weight,
            n_ensembles=args.n_ensembles,
            ensemble_seed_stride=args.ensemble_seed_stride,
            compute_crps=not args.no_crps,
            crps_rain_n_samples=args.crps_rain_n_samples,
            crps_rain_max_points=args.crps_rain_max_points,
            crps_rain_max_points_per_lead=args.crps_rain_max_points_per_lead,
            crps_rain_min_samples=args.crps_rain_min_samples,
            calibrate_rain_probability=not args.no_rain_prob_calibration,
            rain_calibration_bins=args.rain_calibration_bins,
            rain_calibration_smoothing=args.rain_calibration_smoothing,
            rain_calibration_max_points=args.rain_calibration_max_points,
            extreme_temp_q=args.extreme_temp_q,
            extreme_rain_q=args.extreme_rain_q,
            rain_event_threshold=args.rain_event_threshold,
            rain_amount_deadband=args.rain_amount_deadband,
            calibrate_rain_amount=not args.no_rain_amount_calibration,
            rain_amount_calibration_max_points=args.rain_amount_calibration_max_points,
            rain_amount_calibration_smoothing=args.rain_amount_calibration_smoothing,
            rain_amount_calibration_min_cell_count=args.rain_amount_calibration_min_cell_count,
            rain_amount_calibration_min_points=args.rain_amount_calibration_min_points,
            rain_amount_calibration_min_factor=args.rain_amount_calibration_min_factor,
            rain_amount_calibration_max_factor=args.rain_amount_calibration_max_factor,
            rain_amount_apply_p_cal_threshold=args.rain_amount_apply_p_cal_threshold,
            rain_amount_apply_p_raw_threshold=args.rain_amount_apply_p_raw_threshold,
            rain_amount_apply_mu_threshold=args.rain_amount_apply_mu_threshold,
            rain_cap_quantile=args.rain_cap_quantile,
            use_v7_regime_features=not args.no_v7_regime_features,
            radiation_column=args.radiation_column,
            humidity_column=args.humidity_column,
            pressure_column=args.pressure_column,
            wind_speed_column=args.wind_speed_column,
            wind_dir_column=args.wind_dir_column,
            rain_regime_aux_weight=args.rain_regime_aux_weight,
            rain_onset_pos_weight=args.rain_onset_pos_weight,
            rain_continue_pos_weight=args.rain_continue_pos_weight,
            rain_decay_pos_weight=args.rain_decay_pos_weight,
            rain_accum_direct_weight=args.rain_accum_direct_weight,
            rain_accum_direct_under_weight=args.rain_accum_direct_under_weight,
            rain_preonset_sample_weight=args.rain_preonset_sample_weight,
            rain_decay_sample_weight=args.rain_decay_sample_weight,
            physics_temp_rain_cooling_weight=args.physics_temp_rain_cooling_weight,
            val_temp_crps_weight=args.val_temp_crps_weight,
            val_rain_mae_weight=args.val_rain_mae_weight,
            val_rain_brier_weight=args.val_rain_brier_weight,
            val_rain_nll_weight=args.val_rain_nll_weight,
            val_rain_sedi_weight=args.val_rain_sedi_weight,
            val_rain_crps_weight=args.val_rain_crps_weight,
            reliability_bins=args.reliability_bins,
            init_model_dir=args.init_model_dir,
            rain_finetune_only=args.rain_finetune_only,
            v9_end_to_end=args.v9_end_to_end,
            v9_stage1_outdir=args.v9_stage1_outdir,
            v9_stage1_max_epochs=args.v9_stage1_max_epochs,
            v9_stage1_patience=args.v9_stage1_patience,
            v9_stage1_lr=args.v9_stage1_lr,
            v9_stage2_max_epochs=args.v9_stage2_max_epochs,
            v9_stage2_patience=args.v9_stage2_patience,
            v9_stage2_lr=args.v9_stage2_lr,
            v9_skip_stage1_if_complete=args.v9_skip_stage1_if_complete,
            use_tf32=not args.no_tf32,
            compile=args.compile,
            compile_mode=args.compile_mode,
        )
        if cfg.v9_end_to_end:
            train_v9_end_to_end(cfg, outdir=args.outdir)
        else:
            train(cfg, outdir=args.outdir)

    elif args.cmd == "evaluate-ensemble":
        metrics = evaluate_saved_ensemble(args.model_dir, args.csv)
        print(json.dumps(metrics["summary"], indent=2, ensure_ascii=False))

    elif args.cmd == "evaluate-advanced":
        rain_thresholds = tuple(float(v) for v in str(args.rain_thresholds).split(",") if str(v).strip())
        report = evaluate_saved_ensemble_advanced(
            model_dir=args.model_dir,
            csv=args.csv,
            n_bootstrap=args.n_bootstrap,
            block_days=args.block_days,
            rain_thresholds=rain_thresholds,
            output_json=args.output_json,
        )
        compact = {
            "advanced_evaluation_version": report.get("advanced_evaluation_version"),
            "base_metrics_summary": report.get("base_metrics_summary", {}),
            "occurrence_raw_sedi_test": report.get("categorical_test", {}).get("occurrence", {}).get("p_raw", {}).get("sedi", {}).get("global_threshold", {}).get("global", {}),
            "occurrence_calibrated_sedi_test": report.get("categorical_test", {}).get("occurrence", {}).get("p_calibrated", {}).get("sedi", {}).get("global_threshold", {}).get("global", {}),
            "block_bootstrap_ci95": report.get("block_bootstrap_ci95", {}).get("metrics", {}),
        }
        print(json.dumps(_json_safe(compact), indent=2, ensure_ascii=False))

    elif args.cmd == "predict":
        out_df = predict_one(args.model_dir, args.csv, at=args.at)
        print(out_df.to_string(index=False))

    else:
        raise RuntimeError("Comanda desconeguda.")


if __name__ == "__main__":
    main()
