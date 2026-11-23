#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Animació professional del nowcasting local de precipitació del model Hybrid-TimeMixer v9.

Objectiu
--------
Generar una animació de l'any 2024 on, per a cada emissió de nowcast disponible,
es mostri una finestra divulgativa de:
  - 6 hores d'observacions reals anteriors, visibles a pantalla;
  - 3 hores de precipitació observada posterior, per verificar el pronòstic;
  - les trajectòries de precipitació dels membres de l'ensemble;
  - la predicció operativa de l'ensemble v9;
  - la línia base de precipitació del model.

Rigor metodològic
------------------
- No reentrena ni reescala el model.
- Carrega `config.json`, `artifacts.json`, els checkpoints i, quan existeixen,
  els calibradors de probabilitat i quantitat de precipitació guardats amb la v9.
- Reutilitza `load_dataset`, `WindowDataset` i `HybridResidualNowcaster` del codi
  original v9, de manera que les features i la línia base són idèntiques a les
  utilitzades a l'avaluació.
- La corba operativa de precipitació aplica el calibratge de quantitat v9 i el
  deadband instrumental/operatiu guardat en la configuració del model.
- Els membres es representen amb el mateix factor de calibratge de quantitat de
  l'ensemble per preservar-ne la dispersió de manera visualment coherent.
- Les observacions futures només es mostren com a verificació visual, no com a entrada.

Sortides
--------
- `rain_nowcasts_<start>_<end>.npz`: prediccions i verificacions reutilitzables.
- `rain_metrics_<start>_<end>.csv`: MAE/RMSE/skill per lead time.
- `rain_summary_<start>_<end>.json`: metadades i mètriques globals.
- `rain_nowcasting_<start>_<end>.mp4` o `.gif`: animació.

Compatibilitat Windows/Conda
-----------------------------
La inferència PyTorch i la renderització Matplotlib/FFmpeg s'executen en processos
separats. Això evita el conflicte `OMP: Error #15` de Windows/Conda.

Exemple recomanat: inferència completa i vídeo amb un frame per hora
--------------------------------------------------------------------
python animate_precipitation_nowcasting_v9_2024.py ^
  --model-script model_nowcasting_hibrid_timemixer_FINAL_no_rs_v9_amount_calibrated_end_to_end.py ^
  --model-results final_hybrid_timemixer_FINAL_no_rs_v9_amount_calibrated_end_to_end_ens10_v1.zip ^
  --dataset CR3000_Tabla1_2011_2024_dataset_nowcasting_no_rs.zip ^
  --output-dir outputs/animation_precipitation_2024 ^
  --start "2024-01-01 00:00:00" --end "2025-01-01 00:00:00" ^
  --forecast-stride-steps 1 --render-stride 6 --fps 30 --format mp4

La primera execució necessita calcular la inferència de precipitació. Per tornar a
renderitzar sense recalcular-la, afegeix `--reuse-cache --render-only`.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
import sys
import zipfile
from dataclasses import asdict
from types import SimpleNamespace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# IMPORTANT:
# Matplotlib i PyTorch s'importen de manera diferida i en processos separats.
# Això evita a Windows el conflicte entre dues còpies de libiomp5md.dll
# durant el flux inferència (PyTorch) -> renderització (Matplotlib/FFmpeg).


# =============================================================================
# Utilitats generals
# =============================================================================


def log(message: str) -> None:
    print(message, flush=True)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


def safe_slug(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z_-]+", "_", text.strip()).strip("_")


def parse_timestamp(value: str) -> pd.Timestamp:
    out = pd.Timestamp(value)
    if pd.isna(out):
        raise ValueError(f"Timestamp invàlid: {value!r}")
    return out


def to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def finite_minmax(*arrays: np.ndarray) -> Tuple[float, float]:
    finite_values: List[np.ndarray] = []
    for array in arrays:
        values = np.asarray(array, dtype=float)
        values = values[np.isfinite(values)]
        if values.size:
            finite_values.append(values)
    if not finite_values:
        return 0.0, 1.0
    joined = np.concatenate(finite_values)
    return float(joined.min()), float(joined.max())


def find_single_csv_in_zip(source: Path, destination: Path) -> Path:
    ensure_dir(destination)
    with zipfile.ZipFile(source, "r") as zf:
        candidates = [name for name in zf.namelist() if name.lower().endswith(".csv")]
        if not candidates:
            raise FileNotFoundError(f"El ZIP no conté cap CSV: {source}")
        preferred = [name for name in candidates if "dataset_nowcasting" in Path(name).name]
        member = preferred[0] if preferred else candidates[0]
        output = destination / Path(member).name
        if not output.exists():
            log(f"[INFO] Extraient dataset: {output.name}")
            with zf.open(member, "r") as src, output.open("wb") as dst:
                shutil.copyfileobj(src, dst)
        return output


def resolve_dataset_path(source: Path, cache_dir: Path) -> Path:
    if source.is_file() and source.suffix.lower() == ".csv":
        return source.resolve()
    if source.is_file() and source.suffix.lower() == ".zip":
        return find_single_csv_in_zip(source.resolve(), cache_dir / "dataset")
    raise FileNotFoundError(f"No trobo un CSV o ZIP de dataset vàlid: {source}")


def find_model_root(directory: Path) -> Path:
    if (directory / "config.json").exists() and (directory / "artifacts.json").exists():
        return directory
    matches = sorted(directory.rglob("artifacts.json"))
    for artifacts_path in matches:
        candidate = artifacts_path.parent
        if (candidate / "config.json").exists():
            return candidate
    raise FileNotFoundError(f"No s'han trobat config.json i artifacts.json dins de {directory}")


def resolve_model_dir(source: Path, cache_dir: Path) -> Path:
    if source.is_dir():
        return find_model_root(source.resolve())
    if source.is_file() and source.suffix.lower() == ".zip":
        destination = cache_dir / "model_results"
        marker = destination / ".extracted"
        if not marker.exists():
            if destination.exists():
                shutil.rmtree(destination)
            ensure_dir(destination)
            log(f"[INFO] Extraient resultats de l'ensemble: {source.name}")
            with zipfile.ZipFile(source.resolve(), "r") as zf:
                zf.extractall(destination)
            marker.write_text(str(source.resolve()), encoding="utf-8")
        return find_model_root(destination)
    raise FileNotFoundError(f"No trobo un directori o ZIP de model vàlid: {source}")


def import_model_module(model_script: Path):
    if not model_script.exists():
        raise FileNotFoundError(f"No trobo el codi del model: {model_script}")
    module_name = "tfm_model_v9_precipitation_animation"
    spec = importlib.util.spec_from_file_location(module_name, str(model_script.resolve()))
    if spec is None or spec.loader is None:
        raise ImportError(f"No s'ha pogut importar el model des de {model_script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def sort_checkpoint_files(files: Iterable[Path]) -> List[Path]:
    def seed_key(path: Path) -> Tuple[int, str]:
        match = re.search(r"seed(\d+)", path.name)
        return (int(match.group(1)) if match else 10**9, path.name)
    return sorted(files, key=seed_key)


# =============================================================================
# Construcció del dataset d'inferència 2024, coherent amb el model v9
# =============================================================================


def load_observations_only(csv_path: Path, cfg: Any) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Llegeix només temps i precipitació per renderitzar des d'una memòria cau existent."""
    df = pd.read_csv(csv_path, usecols=[cfg.datetime_col, cfg.rain_col], low_memory=False)
    df[cfg.datetime_col] = pd.to_datetime(df[cfg.datetime_col], errors="coerce")
    df = df.dropna(subset=[cfg.datetime_col]).sort_values(cfg.datetime_col)
    df = df.drop_duplicates(subset=[cfg.datetime_col], keep="last").set_index(cfg.datetime_col)
    df[cfg.rain_col] = pd.to_numeric(df[cfg.rain_col], errors="coerce").clip(lower=0.0)
    meta = {
        "rows_observations_only": int(len(df)),
        "time_start": str(df.index.min()),
        "time_end": str(df.index.max()),
        "render_from_cache": True,
    }
    return df, meta

def build_animation_dataset(
    model_module: Any,
    df: pd.DataFrame,
    cfg: Any,
    artifacts: Dict[str, Any],
    start: pd.Timestamp,
    end: pd.Timestamp,
    forecast_stride_steps: int,
) -> Tuple[Any, np.ndarray]:
    """
    Crea un WindowDataset sobre la sèrie completa, filtrant després els temps
    d'emissió. Això permet que les primeres prediccions del 2024 utilitzin les
    48 hores reals prèvies de finals de 2023, sense leakage.
    """
    H = int(artifacts["history_steps"])
    O = int(artifacts["horizon_steps"])
    steps_per_hour = int(artifacts["steps_per_hour"])
    feature_cols = list(artifacts["feature_cols"])
    time_cols = list(artifacts["time_cols"])

    missing_features = [c for c in feature_cols + time_cols if c not in df.columns]
    if missing_features:
        raise ValueError(f"Falten variables requerides pels artefactes: {missing_features}")

    scaler = model_module.RobustScaler.from_state_dict(artifacts["meteo_scaler"])
    X_meteo = scaler.transform(df[feature_cols].to_numpy(dtype=np.float32))
    X_time = df[time_cols].to_numpy(dtype=np.float32)

    temp_mean = float(artifacts["temp_mean"])
    temp_std = float(artifacts["temp_std"])
    y_temp_raw = df[cfg.temp_col].to_numpy(dtype=np.float32)
    y_temp_scaled = ((y_temp_raw - temp_mean) / temp_std).astype(np.float32)
    y_rain = np.clip(df[cfg.rain_col].to_numpy(dtype=np.float32), 0.0, None).astype(np.float32)

    miss_temp = f"{cfg.temp_col}__miss"
    miss_rain = f"{cfg.rain_col}__miss"
    mask_temp = (
        1.0 - df[miss_temp].to_numpy(dtype=np.float32)
        if miss_temp in df.columns else np.ones(len(df), dtype=np.float32)
    )
    mask_rain = (
        1.0 - df[miss_rain].to_numpy(dtype=np.float32)
        if miss_rain in df.columns else np.ones(len(df), dtype=np.float32)
    )

    valid_idx = model_module.valid_t_end_indices_per_segment(
        df, cfg, history_steps=H, horizon_steps=O
    )
    issue_times_all = df.index.to_numpy(dtype="datetime64[ns]")[valid_idx]
    issue_mask = (
        (issue_times_all >= np.datetime64(start.to_datetime64()))
        & (issue_times_all < np.datetime64(end.to_datetime64()))
    )
    valid_idx = valid_idx[issue_mask]
    if forecast_stride_steps > 1:
        valid_idx = valid_idx[::forecast_stride_steps]
    if valid_idx.size == 0:
        raise ValueError("No hi ha cap emissió de nowcast vàlida en l'interval seleccionat.")

    issue_times = df.index.to_numpy(dtype="datetime64[ns]")[valid_idx]
    month_all = df.index.month.to_numpy(dtype=np.int64)
    slot_all = model_module.time_slot_of_day(df.index, cfg.freq).astype(np.int64)
    segment_all = df[cfg.segment_col].to_numpy(dtype=np.int64)

    if cfg.use_solar_geometry and "solar_cos_zenith" in df.columns:
        future_cosz = df["solar_cos_zenith"].to_numpy(dtype=np.float32)
        future_toa = (
            1361.0 * np.clip(df["solar_toa_sw"].to_numpy(dtype=np.float32), 0.0, None)
        ).astype(np.float32)
    else:
        future_cosz = np.zeros(len(df), dtype=np.float32)
        future_toa = np.zeros(len(df), dtype=np.float32)

    dataset = model_module.WindowDataset(
        X_meteo=X_meteo,
        X_time=X_time,
        y_temp_scaled=y_temp_scaled,
        y_temp_raw=y_temp_raw,
        y_rain=y_rain,
        month_all=month_all,
        slot_all=slot_all,
        segment_all=segment_all,
        steps_per_day=int(24 * steps_per_hour),
        history_steps=H,
        horizon_steps=O,
        valid_t_idx=valid_idx,
        baseline_ctx=artifacts["baseline_climatology"],
        future_cosz_all=future_cosz,
        future_toa_all=future_toa,
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
        mask_temp_all=mask_temp,
        mask_rain_all=mask_rain,
        rain_event_threshold=cfg.rain_event_threshold,
    )
    return dataset, issue_times


# =============================================================================
# Inferència d'ensemble en lots
# =============================================================================


def build_network(model_module: Any, cfg: Any, artifacts: Dict[str, Any], device: torch.device):
    return model_module.HybridResidualNowcaster(
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


def load_checkpoint(model: torch.nn.Module, checkpoint: Path, device: torch.device) -> None:
    try:
        payload = torch.load(str(checkpoint), map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(str(checkpoint), map_location=device)
    state = payload["model_state"] if isinstance(payload, dict) and "model_state" in payload else payload
    model.load_state_dict(state)


def run_batched_ensemble_inference(
    model_module: Any,
    model_dir: Path,
    cfg: Any,
    artifacts: Dict[str, Any],
    dataset: Any,
    issue_times: np.ndarray,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    max_members: Optional[int],
    use_amp: bool,
) -> Dict[str, np.ndarray]:
    """
    Calcula les trajectòries de quantitat de pluja dels membres i la sortida
    operativa de l'ensemble, incloent calibratge de quantitat i deadband v9.
    """
    checkpoint_files = sort_checkpoint_files(model_dir.glob("best_model_seed*.pt"))
    if not checkpoint_files:
        fallback = model_dir / "best_model.pt"
        if not fallback.exists():
            raise FileNotFoundError(f"No s'han trobat checkpoints dins de {model_dir}")
        checkpoint_files = [fallback]
    if max_members is not None:
        checkpoint_files = checkpoint_files[: max(1, int(max_members))]

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
        drop_last=False,
    )

    n_members = len(checkpoint_files)
    n_cases = len(dataset)
    horizon_steps = int(artifacts["horizon_steps"])
    temp_mean = float(artifacts["temp_mean"])
    temp_std = float(artifacts["temp_std"])

    member_forecasts_raw = np.empty((n_members, n_cases, horizon_steps), dtype=np.float32)
    member_prob_raw = np.empty((n_members, n_cases, horizon_steps), dtype=np.float32)
    observations = np.empty((n_cases, horizon_steps), dtype=np.float32)
    baselines = np.empty((n_cases, horizon_steps), dtype=np.float32)
    masks = np.empty((n_cases, horizon_steps), dtype=np.float32)
    last_rain = np.empty((n_cases,), dtype=np.float32)

    prob_calibrator = read_json(model_dir / "rain_probability_calibrator.json") if (model_dir / "rain_probability_calibrator.json").exists() else None
    amount_calibrator = read_json(model_dir / "rain_amount_calibrator.json") if (model_dir / "rain_amount_calibrator.json").exists() else None
    deadband_mm = float(getattr(cfg, "rain_amount_deadband", cfg.rain_event_threshold))

    amp_enabled = bool(use_amp and device.type == "cuda")
    if hasattr(model_module, "configure_gpu_perf"):
        model_module.configure_gpu_perf(cfg)

    log(
        f"[INFO] Inferència de precipitació: {n_cases:,} emissions, {n_members} membres, "
        f"{horizon_steps} passos futurs, dispositiu={device}."
    )

    for member_id, checkpoint in enumerate(checkpoint_files):
        log(f"[INFO] Membre {member_id + 1}/{n_members}: {checkpoint.name}")
        model = build_network(model_module, cfg, artifacts, device)
        load_checkpoint(model, checkpoint, device)
        model.eval()
        cursor = 0

        with torch.inference_mode():
            for batch in loader:
                batch = to_device(batch, device)
                n_batch = int(batch["x_past"].shape[0])
                temp_baseline_scaled = (batch["temp_baseline_raw"] - temp_mean) / temp_std
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                    out = model(
                        batch["x_past"],
                        batch["t_future"],
                        temp_baseline_scaled=temp_baseline_scaled,
                        rain_occ_baseline=batch["rain_occ_baseline"],
                        rain_mu_baseline=batch["rain_mu_baseline"],
                        last_temp_scaled=(batch["last_temp_raw"] - temp_mean) / temp_std,
                        last_rain_raw=batch["last_rain_raw"],
                        future_cosz=batch["future_cosz"],
                        future_toa=batch["future_toa"],
                    )
                p_occ = 1.0 - torch.sigmoid(out["rain_logit_p0"])
                rain_mu_raw = p_occ * (out["rain_k"] * out["rain_theta"])
                member_prob_raw[member_id, cursor:cursor + n_batch, :] = p_occ.float().cpu().numpy()
                member_forecasts_raw[member_id, cursor:cursor + n_batch, :] = rain_mu_raw.float().cpu().numpy()
                if member_id == 0:
                    observations[cursor:cursor + n_batch, :] = batch["y_rain"].cpu().numpy()
                    baselines[cursor:cursor + n_batch, :] = batch["rain_mu_baseline"].cpu().numpy()
                    masks[cursor:cursor + n_batch, :] = batch["mask_rain"].cpu().numpy()
                    last_rain[cursor:cursor + n_batch] = batch["last_rain_raw"].reshape(-1).cpu().numpy()
                cursor += n_batch
        if cursor != n_cases:
            raise RuntimeError(f"Nombre de mostres inconsistent: {cursor} != {n_cases}")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    ensemble_raw = member_forecasts_raw.mean(axis=0).astype(np.float32)
    p_rain_raw = member_prob_raw.mean(axis=0).astype(np.float32)
    p_rain_cal = model_module.apply_rain_probability_calibrator_np(p_rain_raw, prob_calibrator).astype(np.float32)
    ensemble_cal, _, amount_factor = model_module.apply_rain_amount_calibrator_np(
        rain_mu=ensemble_raw,
        p_raw=p_rain_raw,
        p_cal=p_rain_cal,
        last_rain=last_rain,
        calibrator=amount_calibrator,
        steps_per_hour=int(artifacts["steps_per_hour"]),
    )
    ensemble_operational = model_module.apply_operational_rain_deadband(ensemble_cal, deadband_mm)

    # Les trajectòries individuals són una representació de dispersió dels 10 models.
    # S'hi aplica el factor de quantitat de l'ensemble per fer-les coherents amb
    # la mitjana operativa que es mostra en primer pla.
    member_forecasts = np.maximum(member_forecasts_raw, 0.0) * amount_factor[None, :, :]
    member_forecasts = model_module.apply_operational_rain_deadband(member_forecasts, deadband_mm)
    ensemble_q10 = np.quantile(member_forecasts, 0.10, axis=0).astype(np.float32)
    ensemble_q90 = np.quantile(member_forecasts, 0.90, axis=0).astype(np.float32)

    return {
        "issue_times": issue_times.astype("datetime64[ns]"),
        "member_forecasts": member_forecasts.astype(np.float32),
        "member_forecasts_raw": member_forecasts_raw.astype(np.float32),
        "ensemble_mean": ensemble_operational.astype(np.float32),
        "ensemble_mean_raw": ensemble_raw.astype(np.float32),
        "ensemble_q10": ensemble_q10,
        "ensemble_q90": ensemble_q90,
        "observations": observations,
        "baseline": baselines,
        "mask": masks,
        "p_rain_raw": p_rain_raw,
        "p_rain_calibrated": p_rain_cal,
        "rain_amount_factor": np.asarray(amount_factor, dtype=np.float32),
        "rain_deadband_mm": np.asarray([deadband_mm], dtype=np.float32),
        "checkpoint_names": np.asarray([p.name for p in checkpoint_files], dtype="U128"),
    }


# =============================================================================
# Mètriques i memòria cau

# =============================================================================
# Mètriques i memòria cau
# =============================================================================


def compute_precipitation_metrics(pred: Dict[str, np.ndarray], steps_per_hour: int) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    model = pred["ensemble_mean"].astype(float)
    baseline = pred["baseline"].astype(float)
    observed = pred["observations"].astype(float)
    valid = pred["mask"] > 0.5
    horizon_steps = model.shape[1]
    rows: List[Dict[str, float]] = []

    def scores(forecast: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> Tuple[float, float]:
        err = (forecast - truth)[mask]
        if err.size == 0:
            return float("nan"), float("nan")
        return float(np.mean(np.abs(err))), float(np.sqrt(np.mean(err ** 2)))

    for lead in range(horizon_steps):
        mae_model, rmse_model = scores(model[:, lead], observed[:, lead], valid[:, lead])
        mae_base, rmse_base = scores(baseline[:, lead], observed[:, lead], valid[:, lead])
        skill = 1.0 - rmse_model / rmse_base if np.isfinite(rmse_base) and rmse_base > 0 else float("nan")
        rows.append({
            "lead_steps": int(lead + 1),
            "lead_minutes": int(round((lead + 1) * 60.0 / steps_per_hour)),
            "n_valid": int(valid[:, lead].sum()),
            "mae_model_mm_10min": mae_model,
            "rmse_model_mm_10min": rmse_model,
            "mae_baseline_mm_10min": mae_base,
            "rmse_baseline_mm_10min": rmse_base,
            "rmse_skill_vs_baseline": skill,
        })

    metrics_by_lead = pd.DataFrame(rows)
    mae_model_all, rmse_model_all = scores(model, observed, valid)
    mae_base_all, rmse_base_all = scores(baseline, observed, valid)
    skill_all = 1.0 - rmse_model_all / rmse_base_all if np.isfinite(rmse_base_all) and rmse_base_all > 0 else float("nan")

    valid_accum = valid.all(axis=1)
    obs_accum = observed.sum(axis=1)
    mod_accum = model.sum(axis=1)
    bas_accum = baseline.sum(axis=1)
    mae_acc_mod, rmse_acc_mod = scores(mod_accum, obs_accum, valid_accum)
    mae_acc_bas, rmse_acc_bas = scores(bas_accum, obs_accum, valid_accum)
    skill_acc = 1.0 - rmse_acc_mod / rmse_acc_bas if np.isfinite(rmse_acc_bas) and rmse_acc_bas > 0 else float("nan")

    summary: Dict[str, Any] = {
        "n_initializations": int(model.shape[0]),
        "horizon_steps": int(horizon_steps),
        "horizon_hours": float(horizon_steps / steps_per_hour),
        "n_members": int(pred["member_forecasts"].shape[0]),
        "rain_deadband_mm": float(np.asarray(pred.get("rain_deadband_mm", [np.nan])).reshape(-1)[0]),
        "mae_model_mm_10min_all_leads": mae_model_all,
        "rmse_model_mm_10min_all_leads": rmse_model_all,
        "mae_baseline_mm_10min_all_leads": mae_base_all,
        "rmse_baseline_mm_10min_all_leads": rmse_base_all,
        "rmse_skill_vs_baseline_all_leads": skill_all,
        "mae_model_accum_3h_mm": mae_acc_mod,
        "rmse_model_accum_3h_mm": rmse_acc_mod,
        "mae_baseline_accum_3h_mm": mae_acc_bas,
        "rmse_baseline_accum_3h_mm": rmse_acc_bas,
        "rmse_skill_vs_baseline_accum_3h": skill_acc,
    }
    return metrics_by_lead, summary

def save_prediction_cache(path: Path, pred: Dict[str, np.ndarray]) -> None:
    ensure_dir(path.parent)
    log(f"[INFO] Desant memòria cau de prediccions: {path}")
    np.savez_compressed(path, **pred)


def load_prediction_cache(path: Path) -> Dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"No existeix la memòria cau requerida: {path}")
    log(f"[INFO] Reutilitzant memòria cau de prediccions: {path}")
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


# =============================================================================
# Renderització professional de l'animació
# =============================================================================


def create_precipitation_animation(
    output_path: Path,
    df: pd.DataFrame,
    cfg: Any,
    artifacts: Dict[str, Any],
    predictions: Dict[str, np.ndarray],
    metrics_summary: Dict[str, Any],
    render_stride: int,
    fps: int,
    dpi: int,
    bitrate: int,
    output_format: str,
    max_frames: Optional[int],
    allow_large_gif: bool,
    display_history_hours: float,
    title_text: str,
) -> None:
    # Importació local: en --render-only el procés no carrega PyTorch.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.animation as animation
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from matplotlib.ticker import MaxNLocator

    issue_times = pd.to_datetime(predictions["issue_times"])
    members = predictions["member_forecasts"]
    mean_pred = predictions["ensemble_mean"]
    q10 = predictions["ensemble_q10"]
    q90 = predictions["ensemble_q90"]
    truth = predictions["observations"]
    baseline = predictions["baseline"]
    mask = predictions["mask"] > 0.5

    O = int(artifacts["horizon_steps"])
    dt = pd.Timedelta(cfg.freq)
    display_history_duration = pd.Timedelta(hours=float(display_history_hours))
    future_offsets = np.arange(1, O + 1) * dt
    frame_ids = np.arange(0, len(issue_times), max(1, int(render_stride)), dtype=int)
    if max_frames is not None:
        frame_ids = frame_ids[: int(max_frames)]
    if frame_ids.size == 0:
        raise ValueError("No hi ha frames per renderitzar.")
    if output_format == "gif" and frame_ids.size > 3000 and not allow_large_gif:
        raise ValueError("Un GIF amb més de 3.000 frames serà molt pesant. Usa MP4 o incrementa --render-stride.")

    obs_series = df[cfg.rain_col].astype(float).clip(lower=0.0)
    obs_times = obs_series.index.to_numpy(dtype="datetime64[ns]")
    obs_values = obs_series.to_numpy(dtype=np.float32)
    bar_width = float(dt / pd.Timedelta(days=1)) * 0.82

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10.5,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#d1d5db",
        "axes.linewidth": 1.0,
    })

    fig = plt.figure(figsize=(15.5, 7.4), constrained_layout=False)
    ax = fig.add_axes([0.07, 0.14, 0.90, 0.74])
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#ffffff")

    t0_first = issue_times[frame_ids[0]]
    future_first = pd.DatetimeIndex([t0_first + offset for offset in future_offsets])
    x_future_first = mdates.date2num(future_first.to_pydatetime())

    # Element fictici només per a la llegenda; les barres reals es redibuixen a cada frame.
    obs_legend = ax.bar([], [], width=bar_width, color="#64748b", alpha=0.55, label="Observacions", zorder=5)
    bars_container = [None]
    member_lines = [
        ax.plot([], [], color="#7dd3fc", linewidth=1.05, alpha=0.32,
                drawstyle="steps-mid", label="Ensemble" if i == 0 else None, zorder=3)[0]
        for i in range(members.shape[0])
    ]
    mean_line, = ax.plot([], [], color="#0369a1", linewidth=3.0, marker="o", markersize=3.0,
                         drawstyle="steps-mid", label="Mitjana ensemble", zorder=8)
    baseline_line, = ax.plot([], [], color="#f59e0b", linewidth=2.4, linestyle="--",
                             drawstyle="steps-mid", label="Línia base", zorder=8)
    now_line = ax.axvline(t0_first, color="#64748b", linewidth=1.4, linestyle=":", zorder=4)
    forecast_span = ax.axvspan(t0_first, future_first[-1], color="#e0f2fe", alpha=0.42, zorder=1)
    uncertainty = Polygon(np.empty((0, 2)), closed=True, facecolor="#38bdf8", edgecolor="none", alpha=0.14, zorder=2)
    ax.add_patch(uncertainty)

    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Temps")
    ax.set_ylabel("Precipitació (mm / 10 min)")
    ax.grid(axis="y", color="#e2e8f0", linewidth=0.9)
    ax.grid(axis="x", color="#f1f5f9", linewidth=0.75)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, min_n_ticks=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M\n%d/%m"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=6, maxticks=10))
    ax.set_title(title_text, fontsize=17, pad=16, color="#0f172a")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=4,
              frameon=False, fontsize=10, handlelength=2.6, columnspacing=1.5)

    def update(frame_position: int):
        sample_id = int(frame_ids[frame_position])
        t0 = issue_times[sample_id]
        future_times = pd.DatetimeIndex([t0 + offset for offset in future_offsets])
        x_future = mdates.date2num(future_times.to_pydatetime())
        x_left = t0 - display_history_duration
        x_right = future_times[-1]

        left_idx = int(np.searchsorted(obs_times, np.datetime64(x_left), side="left"))
        right_idx = int(np.searchsorted(obs_times, np.datetime64(x_right), side="right"))
        visible_times = pd.to_datetime(obs_times[left_idx:right_idx])
        visible_values = obs_values[left_idx:right_idx]
        if bars_container[0] is not None:
            bars_container[0].remove()
        bars_container[0] = ax.bar(
            visible_times, visible_values, width=bar_width, color="#64748b", alpha=0.50,
            edgecolor="none", zorder=5
        )

        for member_id, line in enumerate(member_lines):
            line.set_data(future_times, members[member_id, sample_id, :])
        mean_line.set_data(future_times, mean_pred[sample_id, :])
        baseline_line.set_data(future_times, baseline[sample_id, :])

        vertices = np.vstack([
            np.column_stack([x_future, q10[sample_id, :]]),
            np.column_stack([x_future[::-1], q90[sample_id, ::-1]]),
        ])
        uncertainty.set_xy(vertices)
        now_line.set_xdata([t0, t0])
        x0_num = mdates.date2num(t0)
        x1_num = mdates.date2num(x_right)
        forecast_span.set_x(x0_num)
        forecast_span.set_width(x1_num - x0_num)
        ax.set_xlim(x_left, x_right)

        frame_values = np.concatenate([
            visible_values[np.isfinite(visible_values)],
            members[:, sample_id, :].reshape(-1),
            mean_pred[sample_id, :], baseline[sample_id, :],
        ])
        frame_values = frame_values[np.isfinite(frame_values)]
        local_max = float(frame_values.max()) if frame_values.size else 0.0
        upper = max(0.5, np.ceil((local_max * 1.22) / 0.5) * 0.5)
        ax.set_ylim(0.0, upper)
        return [mean_line, baseline_line, now_line, uncertainty, forecast_span, *member_lines]

    log(f"[INFO] Renderitzant {frame_ids.size:,} frames a {output_path.name}")
    anim = animation.FuncAnimation(fig, update, frames=frame_ids.size, interval=1000 / fps, blit=False)
    ensure_dir(output_path.parent)
    if output_format == "mp4":
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("No s'ha trobat ffmpeg. Instal·la'l o carrega'l al sistema abans de generar MP4.")
        writer = animation.FFMpegWriter(
            fps=int(fps), bitrate=int(bitrate),
            metadata={"title": "Nowcasting local de precipitació (2024)", "artist": "TFM - Nil Farrés"},
            extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
        )
    else:
        writer = animation.PillowWriter(fps=int(fps))
    anim.save(str(output_path), writer=writer, dpi=int(dpi))
    plt.close(fig)
    log(f"[OK] Animació creada: {output_path}")


# =============================================================================
# CLI i programa principal

# =============================================================================
# CLI i programa principal
# =============================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Crea una animació anual professional del nowcasting de precipitació v9.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-script", required=True, type=Path,
                        help="Fitxer .py original del model v9 sense radiosondatge.")
    parser.add_argument("--model-results", required=True, type=Path,
                        help="Directori o ZIP de resultats amb checkpoints, config.json i artifacts.json.")
    parser.add_argument("--dataset", required=True, type=Path,
                        help="CSV o ZIP del dataset final de nowcasting sense radiosondatge.")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="Directori de sortida per a animació, mètriques i memòria cau.")
    parser.add_argument("--start", default="2024-01-01 00:00:00",
                        help="Primer temps d'emissió inclòs.")
    parser.add_argument("--end", default="2025-01-01 00:00:00",
                        help="Temps d'emissió final exclòs.")
    parser.add_argument("--forecast-stride-steps", type=int, default=1,
                        help="Submostreig de la inferència; 1 conserva prediccions iniciades cada 10 minuts.")
    parser.add_argument("--render-stride", type=int, default=1,
                        help="Submostreig només de frames del vídeo; les prediccions cached no es perden.")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Mida del lot d'inferència; redueix-la si falta memòria GPU.")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="Workers de DataLoader; a MN5 es pot provar 2 o 4.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--amp", action="store_true",
                        help="Activa mixed precision CUDA; més ràpid, amb petites diferències numèriques.")
    parser.add_argument("--max-members", type=int, default=None,
                        help="Només per proves ràpides; omet-ho per utilitzar tot l'ensemble.")
    parser.add_argument("--reuse-cache", action="store_true",
                        help="No repeteix inferència si ja existeix el fitxer NPZ corresponent.")
    parser.add_argument("--render-only", action="store_true",
                        help="Renderitza exclusivament des de la memòria cau NPZ; no importa PyTorch ni el model. Evita conflictes OpenMP a Windows.")
    parser.add_argument("--no-render", action="store_true",
                        help="Calcula/cacheja prediccions i mètriques però no crea vídeo.")
    parser.add_argument("--format", choices=["mp4", "gif"], default="mp4")
    parser.add_argument("--fps", type=int, default=60,
                        help="Frames per segon del vídeo.")
    parser.add_argument("--dpi", type=int, default=125,
                        help="Resolució de renderització; 110-150 és adequat per a MP4.")
    parser.add_argument("--bitrate", type=int, default=4500,
                        help="Bitrate MP4 en kbit/s aproximats.")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Limita frames renderitzats; útil només per proves.")
    parser.add_argument("--allow-large-gif", action="store_true")
    parser.add_argument(
        "--display-history-hours",
        type=float,
        default=6.0,
        help="Hores de precipitació observada visibles abans de l'emissió del nowcast.",
    )
    parser.add_argument(
        "--title-text",
        type=str,
        default="Nowcasting local de precipitació (2024)",
        help="Títol únic que es mostra a l'animació.",
    )
    return parser


def choose_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("S'ha demanat CUDA però PyTorch no detecta cap GPU.")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    args = build_parser().parse_args()
    if args.forecast_stride_steps < 1 or args.render_stride < 1:
        raise ValueError("--forecast-stride-steps i --render-stride han de ser >= 1.")
    if args.fps < 1 or args.dpi < 50:
        raise ValueError("Valors de renderització invàlids.")
    if args.display_history_hours <= 0:
        raise ValueError("--display-history-hours ha de ser > 0.")

    start = parse_timestamp(args.start)
    end = parse_timestamp(args.end)
    if start >= end:
        raise ValueError("--start ha de ser anterior a --end.")

    output_dir = args.output_dir.resolve()
    cache_dir = output_dir / "_cache"
    ensure_dir(output_dir)
    ensure_dir(cache_dir)
    model_dir = resolve_model_dir(args.model_results, cache_dir)
    dataset_csv = resolve_dataset_path(args.dataset, cache_dir)

    # Aquests JSON són simples metadades; es poden llegir sense importar PyTorch.
    artifacts = read_json(model_dir / "artifacts.json")
    cfg_raw = read_json(model_dir / "config.json")
    forecast_slug = f"{safe_slug(str(start))}_{safe_slug(str(end))}"
    member_tag = f"m{args.max_members}" if args.max_members is not None else "mall"
    cache_tag = f"{forecast_slug}_fstride{int(args.forecast_stride_steps)}_{member_tag}"
    cache_path = output_dir / f"rain_nowcasts_{cache_tag}.npz"
    metrics_path = output_dir / f"rain_metrics_{forecast_slug}.csv"
    summary_path = output_dir / f"rain_summary_{forecast_slug}.json"
    video_path = output_dir / f"rain_nowcasting_{forecast_slug}.{args.format}"

    inference_performed = False
    use_existing_cache = bool(args.render_only or (args.reuse_cache and cache_path.exists()))

    if args.render_only and not cache_path.exists():
        raise FileNotFoundError(
            f"--render-only requereix la memòria cau ja calculada, però no existeix: {cache_path}"
        )

    if use_existing_cache:
        # Camí lleuger de renderització: no s'importa ni el codi del model ni PyTorch.
        # A Windows, això evita carregar una segona runtime OpenMP al mateix procés.
        cfg = SimpleNamespace(**cfg_raw)
        predictions = load_prediction_cache(cache_path)
        if args.no_render:
            df = pd.DataFrame()
            data_meta = {"render_from_cache": True, "observations_not_read": True}
        else:
            log(f"[INFO] Llegint només observacions per renderitzar: {dataset_csv.name}")
            df, data_meta = load_observations_only(dataset_csv, cfg)
    else:
        # Camí d'inferència: PyTorch i el model només s'importen quan fan falta.
        # La renderització posterior es llançarà en un procés net independent.
        import torch as _torch
        globals()["torch"] = _torch
        model_module = import_model_module(args.model_script)
        cfg = model_module.TrainConfig(**cfg_raw)
        cfg.csv = str(dataset_csv)
        log(f"[INFO] Llegint i derivant variables amb el pipeline original: {dataset_csv.name}")
        df, data_meta = model_module.load_dataset(cfg)
        dataset, issue_times = build_animation_dataset(
            model_module=model_module,
            df=df,
            cfg=cfg,
            artifacts=artifacts,
            start=start,
            end=end,
            forecast_stride_steps=int(args.forecast_stride_steps),
        )
        device = choose_device(args.device)
        predictions = run_batched_ensemble_inference(
            model_module=model_module,
            model_dir=model_dir,
            cfg=cfg,
            artifacts=artifacts,
            dataset=dataset,
            issue_times=issue_times,
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            device=device,
            max_members=args.max_members,
            use_amp=bool(args.amp),
        )
        save_prediction_cache(cache_path, predictions)
        inference_performed = True

    metrics_by_lead, metrics_summary = compute_precipitation_metrics(
        predictions, steps_per_hour=int(artifacts["steps_per_hour"])
    )
    metrics_by_lead.to_csv(metrics_path, index=False)
    summary = {
        "model_results": str(args.model_results),
        "dataset": str(args.dataset),
        "model_script": str(args.model_script),
        "start_inclusive": str(start),
        "end_exclusive": str(end),
        "forecast_stride_steps": int(args.forecast_stride_steps),
        "forecast_frequency_original": str(cfg.freq),
        "render_stride": int(args.render_stride),
        "display_history_hours": float(args.display_history_hours),
        "title_text": str(args.title_text),
        "data_metadata": data_meta,
        "rain_amount_calibration_applied": bool((model_dir / "rain_amount_calibrator.json").exists()),
        "rain_probability_calibration_applied": bool((model_dir / "rain_probability_calibrator.json").exists()),
        "metrics": metrics_summary,
    }
    write_json(summary_path, summary)
    log("[RESULTAT] " + json.dumps(metrics_summary, ensure_ascii=False))
    log(f"[OK] Mètriques: {metrics_path}")
    log(f"[OK] Resum: {summary_path}")

    if args.no_render:
        return

    if inference_performed:
        # No renderitzem en el procés que ha carregat PyTorch: en entorns Conda/
        # Windows això pot activar dues versions de libiomp5md.dll. El procés fill
        # carregarà només la memòria cau i Matplotlib/FFmpeg.
        log("[INFO] La inferència ja és completa. Iniciant la renderització en un procés separat sense PyTorch.")
        child_command = [
            sys.executable,
            str(Path(__file__).resolve()),
            *sys.argv[1:],
            "--render-only",
            "--reuse-cache",
        ]
        subprocess.run(child_command, check=True)
        return

    create_precipitation_animation(
        output_path=video_path,
        df=df,
        cfg=cfg,
        artifacts=artifacts,
        predictions=predictions,
        metrics_summary=metrics_summary,
        render_stride=int(args.render_stride),
        fps=int(args.fps),
        dpi=int(args.dpi),
        bitrate=int(args.bitrate),
        output_format=str(args.format),
        max_frames=args.max_frames,
        allow_large_gif=bool(args.allow_large_gif),
        display_history_hours=float(args.display_history_hours),
        title_text=str(args.title_text),
    )


if __name__ == "__main__":
    main()
