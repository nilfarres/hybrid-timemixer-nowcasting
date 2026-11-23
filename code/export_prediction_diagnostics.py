#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagnòstics professionals per al model de nowcasting local UB.

Aquest script substitueix i amplia `export_prediction_diagnostics.py`.
Està pensat específicament per al model:
    model_nowcasting_hibrid_timemixer_FINAL_no_rs_v3_advanced_eval.py

i per als resultats:
    final_hybrid_timemixer_FINAL_no_rs_ens10.zip

i també continua sent compatible amb directoris ja descomprimits.

Funcionalitats principals
------------------------
1) Accepta com a entrada un directori de model o un ZIP de resultats.
2) Accepta com a dataset un CSV o un ZIP que contingui un únic CSV.
3) Genera gràfiques i taules a partir dels JSON ja guardats:
   - mètriques per lead time,
   - comparació amb línies base,
   - skill scores,
   - calibratge/reliability de precipitació,
   - cobertura i amplada dels intervals de temperatura,
   - diagnòstics de precipitació acumulada,
   - diagnòstics per règim meteorològic,
   - bootstrap i avaluació avançada si existeix `advanced_evaluation_metrics.json`.
4) Opcionalment executa prediccions operatives per timestamps concrets i les compara
   amb les observacions reals del CSV.
5) Opcionalment selecciona casos de test representatius: sec, inici de pluja,
   ja plou, episodi més intens i casos distribuïts temporalment.

Exemples d'ús
-------------
Només gràfiques globals a partir del ZIP de resultats:

python export_prediction_diagnostics.py \
  --model-dir final_hybrid_timemixer_FINAL_no_rs_ens10.zip \
  --outdir diagnostics_final_no_rs

Gràfiques globals + prediccions per casos concrets:

python export_prediction_diagnostics.py \
  --model-script model_nowcasting_hibrid_timemixer_FINAL_no_rs_v3_advanced_eval.py \
  --model-dir final_hybrid_timemixer_FINAL_no_rs_ens10.zip \
  --csv CR3000_Tabla1_2011_2024_dataset_nowcasting_no_rs.zip \
  --at "2024-03-08 12:00:00" \
  --at "2024-09-01 18:00:00" \
  --outdir diagnostics_final_no_rs

Gràfiques globals + selecció automàtica de casos de test:

python export_prediction_diagnostics.py \
  --model-script model_nowcasting_hibrid_timemixer_FINAL_no_rs_v3_advanced_eval.py \
  --model-dir final_hybrid_timemixer_FINAL_no_rs_ens10.zip \
  --csv CR3000_Tabla1_2011_2024_dataset_nowcasting_no_rs.zip \
  --auto-sample-cases 8 \
  --outdir diagnostics_final_no_rs
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import shutil
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# Constants i utilitats bàsiques
# =============================================================================

DEFAULT_RAIN_THRESHOLD_MM = 0.1
DEFAULT_DPI = 180
JSON_CANDIDATES = {
    "config": ["config.json"],
    "artifacts": ["artifacts.json"],
    "metrics": ["test_metrics_ensemble.json", "test_metrics.json"],
    "advanced": ["advanced_evaluation_metrics.json"],
    "calibrator": ["rain_probability_calibrator.json"],
    "members": ["ensemble_members.json"],
}


def _log(msg: str) -> None:
    print(msg, flush=True)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def slugify_timestamp(ts: Any) -> str:
    return (
        str(ts)
        .strip()
        .replace(":", "-")
        .replace("/", "-")
        .replace(" ", "_")
        .replace("+", "p")
    )


def safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        v = float(x)
    except Exception:
        return default
    return v if np.isfinite(v) else default


def finite_array(x: Any) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    return arr[np.isfinite(arr)]


def finite_mean(x: Iterable[float]) -> float:
    arr = finite_array(list(x))
    if arr.size == 0:
        return float("nan")
    return float(arr.mean())


def finite_rmse(errors: Iterable[float]) -> float:
    arr = finite_array(list(errors))
    if arr.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(arr ** 2)))


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


def write_text(text: str, path: Path) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def savefig(path: Path) -> None:
    ensure_dir(path.parent)
    plt.tight_layout()
    plt.savefig(path, dpi=DEFAULT_DPI)
    plt.close()


def _format_time_axis(ax: plt.Axes) -> None:
    ax.tick_params(axis="x", rotation=45)


def _has_columns(df: pd.DataFrame, cols: Sequence[str]) -> bool:
    return all(c in df.columns for c in cols)


# =============================================================================
# Lectura des de directoris o ZIPs
# =============================================================================


def is_zip_path(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() == ".zip"


def _zip_member_by_basename(zf: zipfile.ZipFile, candidates: Sequence[str]) -> Optional[str]:
    names = zf.namelist()
    for cand in candidates:
        if cand in names:
            return cand
    cand_set = {Path(c).name for c in candidates}
    matches = [name for name in names if Path(name).name in cand_set]
    if not matches:
        return None
    matches = sorted(matches, key=lambda s: (len(Path(s).parts), s))
    return matches[0]


def read_json_from_container(container: Path, candidates: Sequence[str]) -> Optional[dict]:
    """Llegeix un JSON d'un directori o d'un ZIP, si existeix."""
    if container.is_dir():
        for cand in candidates:
            p = container / cand
            if p.exists():
                with open(p, "r", encoding="utf-8") as f:
                    return json.load(f)
        for p in container.rglob("*.json"):
            if p.name in {Path(c).name for c in candidates}:
                with open(p, "r", encoding="utf-8") as f:
                    return json.load(f)
        return None

    if is_zip_path(container):
        with zipfile.ZipFile(container, "r") as zf:
            member = _zip_member_by_basename(zf, candidates)
            if member is None:
                return None
            with zf.open(member, "r") as f:
                return json.loads(f.read().decode("utf-8"))

    return None


def list_container_files(container: Path) -> List[str]:
    if container.is_dir():
        return [str(p.relative_to(container)) for p in container.rglob("*") if p.is_file()]
    if is_zip_path(container):
        with zipfile.ZipFile(container, "r") as zf:
            return zf.namelist()
    return []


def extract_zip_to_dir(zip_path: Path, dest_dir: Path, force: bool = False) -> Path:
    ensure_dir(dest_dir)
    marker = dest_dir / ".extracted_from_zip"
    if force and dest_dir.exists():
        shutil.rmtree(dest_dir)
        ensure_dir(dest_dir)
    if marker.exists() and not force:
        return dest_dir
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)
    marker.write_text(str(zip_path.resolve()), encoding="utf-8")
    return dest_dir


def extract_single_csv_from_zip(zip_path: Path, dest_dir: Path, force: bool = False) -> Path:
    ensure_dir(dest_dir)
    with zipfile.ZipFile(zip_path, "r") as zf:
        csv_members = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not csv_members:
            raise FileNotFoundError(f"El ZIP no conté cap CSV: {zip_path}")
        if len(csv_members) > 1:
            preferred = [n for n in csv_members if "dataset_nowcasting" in Path(n).name]
            csv_member = preferred[0] if preferred else csv_members[0]
        else:
            csv_member = csv_members[0]
        out = dest_dir / Path(csv_member).name
        if out.exists() and not force:
            return out
        with zf.open(csv_member, "r") as src, open(out, "wb") as dst:
            shutil.copyfileobj(src, dst)
    return out


def resolve_csv_path(csv_arg: Optional[str], workdir: Path, force_extract: bool = False) -> Optional[Path]:
    if not csv_arg:
        return None
    p = Path(csv_arg).expanduser().resolve()
    if p.is_file() and p.suffix.lower() == ".csv":
        return p
    if is_zip_path(p):
        return extract_single_csv_from_zip(p, workdir / "dataset_csv", force=force_extract)
    raise FileNotFoundError(f"No trobo un CSV o ZIP vàlid: {p}")


def resolve_model_dir_for_prediction(model_arg: str, workdir: Path, force_extract: bool = False) -> Path:
    p = Path(model_arg).expanduser().resolve()
    if p.is_dir():
        return p
    if is_zip_path(p):
        return extract_zip_to_dir(p, workdir / "model_dir", force=force_extract)
    raise FileNotFoundError(f"No trobo un directori o ZIP de model vàlid: {p}")


def load_saved_result_jsons(model_container: Path) -> Dict[str, Optional[dict]]:
    return {key: read_json_from_container(model_container, cands) for key, cands in JSON_CANDIDATES.items()}


# =============================================================================
# Càrrega dinàmica del model només quan cal predir
# =============================================================================


def load_model_module(model_script: Path):
    if not model_script.exists():
        raise FileNotFoundError(f"No trobo el fitxer del model: {model_script}")
    spec = importlib.util.spec_from_file_location("model_nowcasting_dynamic", model_script)
    if spec is None or spec.loader is None:
        raise ImportError(f"No s'ha pogut carregar dinàmicament: {model_script}")
    module = importlib.util.module_from_spec(spec)

    # Important per a Python 3.11/3.12 i :
    # dataclasses consulta sys.modules[cls.__module__] mentre processa la classe.
    # Si carreguem el fitxer amb importlib però no registrem el mòdul abans
    # d'executar-lo, cls.__module__ apunta a un nom que encara no existeix a
    # sys.modules i apareix l'error:
    # AttributeError: 'NoneType' object has no attribute '__dict__'
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    required = ["predict_one", "TrainConfig"]
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise AttributeError(f"Al fitxer {model_script} falten objectes necessaris: {missing}")
    return module


# =============================================================================
# Lectura de dades observades
# =============================================================================


def get_config_value(config: Optional[dict], key: str, default: Any) -> Any:
    if not isinstance(config, dict):
        return default
    return config.get(key, default)


def load_real_csv(csv_path: Path, datetime_col: str) -> pd.DataFrame:
    if not csv_path or not csv_path.exists():
        raise FileNotFoundError(f"No trobo el CSV real: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)
    if datetime_col not in df.columns:
        raise ValueError(f"La columna temporal '{datetime_col}' no existeix al CSV")
    df[datetime_col] = pd.to_datetime(df[datetime_col], errors="coerce")
    df = df.dropna(subset=[datetime_col]).sort_values(datetime_col)
    df = df.drop_duplicates(subset=[datetime_col], keep="last")
    return df.reset_index(drop=True)


def infer_steps_from_config(config: Optional[dict], artifacts: Optional[dict]) -> Tuple[pd.Timedelta, int, int, int]:
    freq = str(get_config_value(config, "freq", "10min"))
    dt = pd.Timedelta(freq)
    if artifacts and "history_steps" in artifacts and "horizon_steps" in artifacts:
        steps_per_hour = int(artifacts.get("steps_per_hour", round(pd.Timedelta("1h") / dt)))
        return dt, int(artifacts["history_steps"]), int(artifacts["horizon_steps"]), steps_per_hour
    history_hours = float(get_config_value(config, "history_hours", 48.0))
    horizon_hours = float(get_config_value(config, "horizon_hours", 3.0))
    steps_per_hour = max(1, int(round(pd.Timedelta("1h") / dt)))
    return dt, int(round(history_hours * steps_per_hour)), int(round(horizon_hours * steps_per_hour)), steps_per_hour


# =============================================================================
# Taules i gràfiques a partir de mètriques guardades
# =============================================================================


def dict_of_lists_to_frame(d: Optional[dict]) -> pd.DataFrame:
    if not isinstance(d, dict):
        return pd.DataFrame()
    lengths = {k: len(v) for k, v in d.items() if isinstance(v, list)}
    if not lengths:
        return pd.DataFrame()
    n = max(lengths.values())
    out: Dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, list) and len(v) == n:
            out[k] = v
    return pd.DataFrame(out)


def write_metrics_tables(metrics: dict, advanced: Optional[dict], outdir: Path) -> Dict[str, str]:
    tables_dir = outdir / "tables"
    ensure_dir(tables_dir)
    paths: Dict[str, str] = {}

    summary = metrics.get("summary", {}) if isinstance(metrics, dict) else {}
    if summary:
        p = tables_dir / "summary_metrics.csv"
        pd.DataFrame([summary]).to_csv(p, index=False)
        paths["summary_metrics"] = str(p)

    by_lead = dict_of_lists_to_frame(metrics.get("by_lead", {}))
    if not by_lead.empty:
        p = tables_dir / "metrics_by_lead.csv"
        by_lead.to_csv(p, index=False)
        paths["metrics_by_lead"] = str(p)

    for bname, bobj in (metrics.get("baselines", {}) or {}).items():
        df = dict_of_lists_to_frame(bobj.get("by_lead", {}) if isinstance(bobj, dict) else {})
        if not df.empty:
            p = tables_dir / f"baseline_{bname}_by_lead.csv"
            df.to_csv(p, index=False)
            paths[f"baseline_{bname}_by_lead"] = str(p)

    skill = metrics.get("skill_scores", {}) or {}
    if isinstance(skill, dict):
        skill_by_lead = dict_of_lists_to_frame(skill.get("by_lead", {}))
        if not skill_by_lead.empty:
            p = tables_dir / "skill_scores_by_lead.csv"
            skill_by_lead.to_csv(p, index=False)
            paths["skill_scores_by_lead"] = str(p)
        if isinstance(skill.get("summary"), dict):
            p = tables_dir / "skill_scores_summary.csv"
            pd.DataFrame([skill["summary"]]).to_csv(p, index=False)
            paths["skill_scores_summary"] = str(p)

    diagnostics = metrics.get("diagnostics", {}) or {}
    temp_cov = diagnostics.get("temperature_coverage", {})
    temp_cov_df = dict_of_lists_to_frame(temp_cov.get("by_lead", {}) if isinstance(temp_cov, dict) else {})
    if not temp_cov_df.empty:
        p = tables_dir / "temperature_coverage_by_lead.csv"
        temp_cov_df.to_csv(p, index=False)
        paths["temperature_coverage_by_lead"] = str(p)

    for rel_name in ["rain_reliability_raw", "rain_reliability_calibrated", "rain_reliability"]:
        rel = diagnostics.get(rel_name, {})
        bins = rel.get("global_bins", {}) if isinstance(rel, dict) else {}
        df = dict_of_lists_to_frame(bins)
        if not df.empty:
            p = tables_dir / f"{rel_name}_global_bins.csv"
            df.to_csv(p, index=False)
            paths[f"{rel_name}_global_bins"] = str(p)

    accum = diagnostics.get("rain_accumulation", {})
    accum_df = dict_of_lists_to_frame(accum.get("by_horizon", {}) if isinstance(accum, dict) else {})
    if not accum_df.empty:
        p = tables_dir / "rain_accumulation_by_horizon.csv"
        accum_df.to_csv(p, index=False)
        paths["rain_accumulation_by_horizon"] = str(p)

    regime = diagnostics.get("regime_metrics", {})
    if isinstance(regime, dict) and regime:
        rows = []
        for name, vals in regime.items():
            if isinstance(vals, dict):
                row = {"regime": name}
                row.update(vals)
                rows.append(row)
        if rows:
            p = tables_dir / "regime_metrics.csv"
            pd.DataFrame(rows).to_csv(p, index=False)
            paths["regime_metrics"] = str(p)

    if isinstance(advanced, dict):
        base = advanced.get("base_metrics_summary")
        if isinstance(base, dict):
            p = tables_dir / "advanced_base_metrics_summary.csv"
            pd.DataFrame([base]).to_csv(p, index=False)
            paths["advanced_base_metrics_summary"] = str(p)
        boot = advanced.get("block_bootstrap_ci95", {}).get("metrics", {})
        boot_rows = []
        if isinstance(boot, dict):
            for metric_name, vals in boot.items():
                if isinstance(vals, dict):
                    row = {"metric": metric_name}
                    row.update(vals)
                    boot_rows.append(row)
        if boot_rows:
            p = tables_dir / "advanced_block_bootstrap_ci95.csv"
            pd.DataFrame(boot_rows).to_csv(p, index=False)
            paths["advanced_block_bootstrap_ci95"] = str(p)

        for strat_name in ["temporal_stratification_calibrated_probability", "temporal_stratification_raw_probability"]:
            strat = advanced.get(strat_name, {})
            if isinstance(strat, dict):
                for group_name, group_obj in strat.items():
                    rows = []
                    if isinstance(group_obj, dict):
                        for label, vals in group_obj.items():
                            if isinstance(vals, dict):
                                row = {"group": label}
                                for k, v in vals.items():
                                    if k == "rain_categorical" and isinstance(v, dict):
                                        for kk, vv in v.items():
                                            row[f"rain_cat_{kk}"] = vv
                                    elif not isinstance(v, (dict, list)):
                                        row[k] = v
                                rows.append(row)
                    if rows:
                        p = tables_dir / f"{strat_name}_{group_name}.csv"
                        pd.DataFrame(rows).to_csv(p, index=False)
                        paths[f"{strat_name}_{group_name}"] = str(p)

    return paths


def line_plot(
    df: pd.DataFrame,
    x: str,
    y_cols: Sequence[str],
    labels: Optional[Sequence[str]],
    title: str,
    xlabel: str,
    ylabel: str,
    outpath: Path,
    hlines: Optional[Sequence[Tuple[float, str]]] = None,
) -> Optional[str]:
    cols = [c for c in y_cols if c in df.columns]
    if not cols or x not in df.columns:
        return None
    plt.figure(figsize=(9, 5))
    for i, col in enumerate(cols):
        lab = labels[i] if labels and i < len(labels) else col
        plt.plot(df[x], df[col], marker="o", label=lab)
    if hlines:
        for y, lab in hlines:
            plt.axhline(y, linestyle="--", linewidth=1, label=lab)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    savefig(outpath)
    return str(outpath)


def bar_plot(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    xlabel: str,
    ylabel: str,
    outpath: Path,
    rotation: int = 30,
) -> Optional[str]:
    if df.empty or x not in df.columns or y not in df.columns:
        return None
    sub = df[[x, y]].copy()
    sub[y] = pd.to_numeric(sub[y], errors="coerce")
    sub = sub.dropna(subset=[y])
    if sub.empty:
        return None
    plt.figure(figsize=(9, 5))
    plt.bar(sub[x].astype(str), sub[y])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=rotation, ha="right")
    savefig(outpath)
    return str(outpath)


def plot_lead_metrics(metrics: dict, figures_dir: Path) -> Dict[str, str]:
    paths: Dict[str, str] = {}
    by_lead = dict_of_lists_to_frame(metrics.get("by_lead", {}))
    if by_lead.empty or "lead_hours" not in by_lead.columns:
        return paths

    p = line_plot(
        by_lead,
        "lead_hours",
        ["temp_mae", "temp_rmse", "temp_crps"],
        ["MAE", "RMSE", "CRPS"],
        "Temperatura: error i CRPS per lead time",
        "Lead time (h)",
        "°C",
        figures_dir / "lead_temperature_mae_rmse_crps.png",
    )
    if p: paths["lead_temperature_mae_rmse_crps"] = p

    p = line_plot(
        by_lead,
        "lead_hours",
        ["temp_nll"],
        ["NLL"],
        "Temperatura: NLL per lead time",
        "Lead time (h)",
        "NLL",
        figures_dir / "lead_temperature_nll.png",
    )
    if p: paths["lead_temperature_nll"] = p

    p = line_plot(
        by_lead,
        "lead_hours",
        ["rain_mae", "rain_rmse"],
        ["MAE", "RMSE"],
        "Precipitació: error de quantitat per lead time",
        "Lead time (h)",
        "mm / 10 min",
        figures_dir / "lead_rain_mae_rmse.png",
    )
    if p: paths["lead_rain_mae_rmse"] = p

    p = line_plot(
        by_lead,
        "lead_hours",
        ["rain_brier", "rain_brier_raw", "rain_brier_calibrated"],
        ["Brier operatiu", "Brier cru", "Brier calibrat"],
        "Precipitació: Brier score per lead time",
        "Lead time (h)",
        "Brier score",
        figures_dir / "lead_rain_brier_raw_calibrated.png",
    )
    if p: paths["lead_rain_brier_raw_calibrated"] = p

    p = line_plot(
        by_lead,
        "lead_hours",
        ["rain_sedi", "rain_sedi_raw", "rain_sedi_calibrated"],
        ["SEDI operatiu", "SEDI cru", "SEDI calibrat"],
        "Precipitació: SEDI per lead time",
        "Lead time (h)",
        "SEDI",
        figures_dir / "lead_rain_sedi_raw_calibrated.png",
    )
    if p: paths["lead_rain_sedi_raw_calibrated"] = p

    p = line_plot(
        by_lead,
        "lead_hours",
        ["rain_crps", "rain_crps_wet", "rain_crps_heavy"],
        ["CRPS global", "CRPS si plou", "CRPS pluja intensa"],
        "Precipitació probabilista: CRPS per lead time",
        "Lead time (h)",
        "CRPS / mm",
        figures_dir / "lead_rain_crps.png",
    )
    if p: paths["lead_rain_crps"] = p

    return paths


def plot_baseline_comparisons(metrics: dict, figures_dir: Path) -> Dict[str, str]:
    paths: Dict[str, str] = {}
    by_lead = dict_of_lists_to_frame(metrics.get("by_lead", {}))
    if by_lead.empty or "lead_hours" not in by_lead.columns:
        return paths

    baselines = metrics.get("baselines", {}) or {}

    def make_comparison(metric: str, basename: str, title: str, ylabel: str, baseline_names: Sequence[str]) -> None:
        df = pd.DataFrame({"lead_hours": by_lead["lead_hours"], "Model": by_lead.get(metric)})
        for b in baseline_names:
            bdf = dict_of_lists_to_frame((baselines.get(b, {}) or {}).get("by_lead", {}))
            if not bdf.empty and metric in bdf.columns:
                df[b] = bdf[metric]
        p = line_plot(
            df,
            "lead_hours",
            [c for c in df.columns if c != "lead_hours"],
            [c for c in df.columns if c != "lead_hours"],
            title,
            "Lead time (h)",
            ylabel,
            figures_dir / f"baseline_{basename}.png",
        )
        if p:
            paths[f"baseline_{basename}"] = p

    make_comparison("temp_mae", "temperature_mae", "Temperatura: MAE del model vs línies base", "°C", ["persistence", "climatology", "daily_persistence_temp"])
    make_comparison("temp_rmse", "temperature_rmse", "Temperatura: RMSE del model vs línies base", "°C", ["persistence", "climatology", "daily_persistence_temp"])
    make_comparison("rain_mae", "rain_mae", "Precipitació: MAE del model vs línies base", "mm / 10 min", ["persistence", "climatology", "rain_markov"])
    make_comparison("rain_brier", "rain_brier", "Precipitació: Brier del model vs línies base", "Brier score", ["persistence", "climatology", "rain_markov"])
    make_comparison("rain_sedi", "rain_sedi", "Precipitació: SEDI del model vs línies base", "SEDI", ["persistence", "climatology", "rain_markov"])
    return paths


def plot_skill_scores(metrics: dict, figures_dir: Path) -> Dict[str, str]:
    paths: Dict[str, str] = {}
    skill = metrics.get("skill_scores", {}) or {}
    df = dict_of_lists_to_frame(skill.get("by_lead", {}) if isinstance(skill, dict) else {})
    if df.empty or "lead_hours" not in df.columns:
        return paths
    p = line_plot(
        df,
        "lead_hours",
        ["temp_mae_skill_vs_persistence", "temp_mae_skill_vs_climatology", "temp_mae_skill_vs_daily_persistence"],
        ["vs persistència", "vs climatologia", "vs persistència diària"],
        "Skill de temperatura per lead time",
        "Lead time (h)",
        "Skill MAE",
        figures_dir / "skill_temperature_by_lead.png",
        hlines=[(0.0, "sense skill")],
    )
    if p: paths["skill_temperature_by_lead"] = p
    p = line_plot(
        df,
        "lead_hours",
        ["rain_amount_mae_skill_vs_markov", "rain_occurrence_bss_vs_markov"],
        ["quantitat vs Markov", "BSS ocurrència vs Markov"],
        "Skill de precipitació per lead time",
        "Lead time (h)",
        "Skill",
        figures_dir / "skill_rain_by_lead.png",
        hlines=[(0.0, "sense skill")],
    )
    if p: paths["skill_rain_by_lead"] = p
    return paths


def plot_temperature_coverage(metrics: dict, figures_dir: Path) -> Dict[str, str]:
    paths: Dict[str, str] = {}
    diag = metrics.get("diagnostics", {}) or {}
    temp_cov = diag.get("temperature_coverage", {})
    df = dict_of_lists_to_frame(temp_cov.get("by_lead", {}) if isinstance(temp_cov, dict) else {})
    if df.empty:
        return paths
    p = line_plot(
        df,
        "lead_hours",
        ["coverage80", "coverage90"],
        ["Cobertura interval 80%", "Cobertura interval 90%"],
        "Calibratge dels intervals de temperatura",
        "Lead time (h)",
        "Cobertura empírica",
        figures_dir / "temperature_interval_coverage.png",
        hlines=[(0.8, "objectiu 80%"), (0.9, "objectiu 90%")],
    )
    if p: paths["temperature_interval_coverage"] = p
    p = line_plot(
        df,
        "lead_hours",
        ["width80_C", "width90_C"],
        ["Amplada interval 80%", "Amplada interval 90%"],
        "Amplada dels intervals predictius de temperatura",
        "Lead time (h)",
        "°C",
        figures_dir / "temperature_interval_width.png",
    )
    if p: paths["temperature_interval_width"] = p
    p = line_plot(
        df,
        "lead_hours",
        ["pit_mean"],
        ["PIT mitjà"],
        "Temperatura: PIT mitjà per lead time",
        "Lead time (h)",
        "PIT mitjà",
        figures_dir / "temperature_pit_mean.png",
        hlines=[(0.5, "calibratge ideal")],
    )
    if p: paths["temperature_pit_mean"] = p
    return paths


def plot_reliability(metrics: dict, figures_dir: Path) -> Dict[str, str]:
    paths: Dict[str, str] = {}
    diag = metrics.get("diagnostics", {}) or {}
    for rel_name, title_suffix in [
        ("rain_reliability_raw", "probabilitat crua"),
        ("rain_reliability_calibrated", "probabilitat calibrada"),
        ("rain_reliability", "probabilitat operativa"),
    ]:
        rel = diag.get(rel_name, {})
        df = dict_of_lists_to_frame(rel.get("global_bins", {}) if isinstance(rel, dict) else {})
        if df.empty or not _has_columns(df, ["mean_forecast_probability", "observed_frequency"]):
            continue
        plt.figure(figsize=(6, 6))
        plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="calibratge perfecte")
        plt.plot(df["mean_forecast_probability"], df["observed_frequency"], marker="o", label=title_suffix)
        plt.xlabel("Probabilitat prevista mitjana")
        plt.ylabel("Freqüència observada")
        plt.title(f"Diagrama de reliability de precipitació ({title_suffix})")
        plt.grid(True, alpha=0.3)
        plt.legend()
        p = figures_dir / f"reliability_{rel_name}.png"
        savefig(p)
        paths[f"reliability_{rel_name}"] = str(p)

        if "count" in df.columns:
            p2 = bar_plot(
                df,
                "bin_center",
                "count",
                f"Mostres per bin de probabilitat ({title_suffix})",
                "Bin de probabilitat",
                "Nombre de punts",
                figures_dir / f"reliability_counts_{rel_name}.png",
                rotation=0,
            )
            if p2:
                paths[f"reliability_counts_{rel_name}"] = p2
    return paths


def plot_rain_accumulation(metrics: dict, figures_dir: Path) -> Dict[str, str]:
    paths: Dict[str, str] = {}
    diag = metrics.get("diagnostics", {}) or {}
    accum = diag.get("rain_accumulation", {})
    df = dict_of_lists_to_frame(accum.get("by_horizon", {}) if isinstance(accum, dict) else {})
    if df.empty:
        return paths
    p = line_plot(
        df,
        "horizon_hours",
        ["mae", "rmse"],
        ["MAE acumulat", "RMSE acumulat"],
        "Precipitació acumulada: error per horitzó",
        "Horitzó acumulat (h)",
        "mm acumulats",
        figures_dir / "rain_accumulation_error.png",
    )
    if p: paths["rain_accumulation_error"] = p
    p = line_plot(
        df,
        "horizon_hours",
        ["bias"],
        ["Biaix"],
        "Precipitació acumulada: biaix per horitzó",
        "Horitzó acumulat (h)",
        "mm acumulats",
        figures_dir / "rain_accumulation_bias.png",
        hlines=[(0.0, "sense biaix")],
    )
    if p: paths["rain_accumulation_bias"] = p
    p = line_plot(
        df,
        "horizon_hours",
        ["brier_event", "sedi_event"],
        ["Brier event acumulat", "SEDI event acumulat"],
        "Ocurrència de pluja acumulada per horitzó",
        "Horitzó acumulat (h)",
        "Mètrica",
        figures_dir / "rain_accumulation_occurrence.png",
    )
    if p: paths["rain_accumulation_occurrence"] = p
    return paths


def plot_regime_metrics(metrics: dict, figures_dir: Path) -> Dict[str, str]:
    paths: Dict[str, str] = {}
    regime = (metrics.get("diagnostics", {}) or {}).get("regime_metrics", {})
    if not isinstance(regime, dict) or not regime:
        return paths
    rows = []
    for name, vals in regime.items():
        if isinstance(vals, dict):
            row = {"regime": name}
            row.update(vals)
            rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return paths
    for metric, ylabel, fname in [
        ("temp_mae", "°C", "regime_temp_mae.png"),
        ("rain_mae", "mm / 10 min", "regime_rain_mae.png"),
        ("rain_brier", "Brier", "regime_rain_brier.png"),
        ("rain_sedi", "SEDI", "regime_rain_sedi.png"),
    ]:
        p = bar_plot(
            df,
            "regime",
            metric,
            f"Mètrica per règim: {metric}",
            "Règim meteorològic",
            ylabel,
            figures_dir / fname,
        )
        if p:
            paths[fname.replace(".png", "")] = p
    return paths


def plot_advanced_metrics(advanced: Optional[dict], figures_dir: Path) -> Dict[str, str]:
    paths: Dict[str, str] = {}
    if not isinstance(advanced, dict):
        return paths

    brier = advanced.get("brier_decomposition", {})
    if isinstance(brier, dict):
        rows = []
        for source, vals in brier.items():
            if isinstance(vals, dict):
                for k, v in vals.items():
                    if not isinstance(v, (dict, list)):
                        rows.append({"source": source, "component": k, "value": v})
        df = pd.DataFrame(rows)
        if not df.empty:
            for source in df["source"].unique():
                sub = df[df["source"] == source]
                p = bar_plot(
                    sub,
                    "component",
                    "value",
                    f"Descomposició del Brier ({source})",
                    "Component",
                    "Valor",
                    figures_dir / f"advanced_brier_decomposition_{source}.png",
                )
                if p:
                    paths[f"advanced_brier_decomposition_{source}"] = p

    boot = advanced.get("block_bootstrap_ci95", {}).get("metrics", {})
    rows = []
    if isinstance(boot, dict):
        for metric_name, vals in boot.items():
            if isinstance(vals, dict):
                est = vals.get("estimate", vals.get("mean", vals.get("median", np.nan)))
                lo = vals.get("ci95_low", vals.get("low", vals.get("q025", np.nan)))
                hi = vals.get("ci95_high", vals.get("high", vals.get("q975", np.nan)))
                rows.append({"metric": metric_name, "estimate": est, "low": lo, "high": hi})
    boot_df = pd.DataFrame(rows)
    if not boot_df.empty:
        boot_df["estimate"] = pd.to_numeric(boot_df["estimate"], errors="coerce")
        boot_df["low"] = pd.to_numeric(boot_df["low"], errors="coerce")
        boot_df["high"] = pd.to_numeric(boot_df["high"], errors="coerce")
        boot_df = boot_df.dropna(subset=["estimate"])
        if not boot_df.empty:
            plt.figure(figsize=(9, max(4, 0.35 * len(boot_df))))
            y = np.arange(len(boot_df))
            xerr_low = boot_df["estimate"] - boot_df["low"]
            xerr_high = boot_df["high"] - boot_df["estimate"]
            xerr = np.vstack([np.maximum(xerr_low, 0), np.maximum(xerr_high, 0)])
            plt.errorbar(boot_df["estimate"], y, xerr=xerr, fmt="o")
            plt.yticks(y, boot_df["metric"])
            plt.xlabel("Estimació i IC95%")
            plt.title("Intervals de confiança per bootstrap de blocs")
            plt.grid(True, axis="x", alpha=0.3)
            p = figures_dir / "advanced_block_bootstrap_ci95.png"
            savefig(p)
            paths["advanced_block_bootstrap_ci95"] = str(p)

    strat = advanced.get("temporal_stratification_calibrated_probability", {})
    if isinstance(strat, dict):
        for group_name, group_obj in strat.items():
            rows = []
            if isinstance(group_obj, dict):
                for label, vals in group_obj.items():
                    if isinstance(vals, dict):
                        row = {"group": label}
                        for k in ["temp_mae", "rain_brier", "rain_mae", "event_rate"]:
                            row[k] = vals.get(k, np.nan)
                        cat = vals.get("rain_categorical", {})
                        if isinstance(cat, dict):
                            for k in ["pod", "far", "csi", "sedi", "frequency_bias"]:
                                row[f"rain_cat_{k}"] = cat.get(k, np.nan)
                        rows.append(row)
            df = pd.DataFrame(rows)
            if not df.empty:
                for metric in ["temp_mae", "rain_brier", "rain_cat_sedi", "event_rate"]:
                    p = bar_plot(
                        df,
                        "group",
                        metric,
                        f"Estratificació temporal: {metric} ({group_name})",
                        group_name,
                        metric,
                        figures_dir / f"advanced_strat_{group_name}_{metric}.png",
                    )
                    if p:
                        paths[f"advanced_strat_{group_name}_{metric}"] = p
    return paths


def create_metrics_diagnostics(model_container: Path, outdir: Path) -> Dict[str, Any]:
    ensure_dir(outdir)
    figures_dir = outdir / "figures" / "global_metrics"
    ensure_dir(figures_dir)

    jsons = load_saved_result_jsons(model_container)
    metrics = jsons.get("metrics")
    advanced = jsons.get("advanced")
    if not isinstance(metrics, dict):
        raise FileNotFoundError(
            "No he trobat test_metrics_ensemble.json ni test_metrics.json dins del directori/ZIP del model."
        )

    tables = write_metrics_tables(metrics, advanced, outdir)
    figures: Dict[str, str] = {}
    for maker in [
        plot_lead_metrics,
        plot_baseline_comparisons,
        plot_skill_scores,
        plot_temperature_coverage,
        plot_reliability,
        plot_rain_accumulation,
        plot_regime_metrics,
    ]:
        figures.update(maker(metrics, figures_dir))
    figures.update(plot_advanced_metrics(advanced, figures_dir))

    summary_md = build_global_summary_markdown(metrics, advanced, tables, figures)
    summary_path = outdir / "diagnostic_summary_global.md"
    write_text(summary_md, summary_path)

    manifest = {
        "model_container": str(model_container),
        "tables": tables,
        "figures": figures,
        "summary_markdown": str(summary_path),
        "available_files_in_model_container": list_container_files(model_container),
    }
    write_json(manifest, outdir / "manifest_global_metrics_diagnostics.json")
    return manifest


def build_global_summary_markdown(metrics: dict, advanced: Optional[dict], tables: Dict[str, str], figures: Dict[str, str]) -> str:
    s = metrics.get("summary", {}) if isinstance(metrics, dict) else {}
    skill = metrics.get("skill_scores", {}).get("summary", {}) if isinstance(metrics.get("skill_scores"), dict) else {}
    lines = [
        "# Diagnòstic global del model de nowcasting",
        "",
        "## Resum numèric principal",
        "",
    ]
    principal = [
        ("Loss", s.get("loss")),
        ("MAE temperatura mitjà", s.get("temp_mae_mean")),
        ("RMSE temperatura mitjà", s.get("temp_rmse_mean")),
        ("CRPS temperatura mitjà", s.get("temp_crps_mean")),
        ("MAE precipitació mitjà", s.get("rain_mae_mean")),
        ("RMSE precipitació mitjà", s.get("rain_rmse_mean")),
        ("Brier precipitació mitjà", s.get("rain_brier_mean")),
        ("SEDI precipitació mitjà", s.get("rain_sedi_mean")),
        ("CRPS precipitació mostrejat", s.get("crps_rain_mean_sampled", s.get("rain_crps_mean_sampled"))),
        ("Cobertura T 80%", s.get("temp_coverage80_mean")),
        ("Cobertura T 90%", s.get("temp_coverage90_mean")),
        ("ECE reliability pluja", s.get("rain_reliability_ece")),
        ("Skill T vs persistència", skill.get("temp_mae_skill_vs_persistence_mean")),
        ("BSS pluja vs Markov", skill.get("rain_occurrence_bss_vs_markov_mean")),
    ]
    for name, val in principal:
        if val is not None:
            lines.append(f"- **{name}:** {safe_float(val):.6g}")
    if isinstance(advanced, dict):
        lines.extend([
            "",
            "## Avaluació avançada detectada",
            "",
            f"- Versió: `{advanced.get('advanced_evaluation_version', 'n/d')}`",
            f"- Punts test plans: `{advanced.get('n_test_points_flat', 'n/d')}`",
            f"- Bootstrap: `{advanced.get('block_bootstrap_ci95', {}).get('n_bootstrap', 'n/d')}` remostreigs, blocs de `{advanced.get('block_bootstrap_ci95', {}).get('block_days', 'n/d')}` dies.",
        ])
    lines.extend([
        "",
        "## Fitxers generats",
        "",
        f"- Taules: {len(tables)}",
        f"- Figures: {len(figures)}",
        "",
        "Les figures principals són dins de `figures/global_metrics/` i les taules dins de `tables/`.",
    ])
    return "\n".join(lines) + "\n"


# =============================================================================
# Predicció operativa i comparació amb observacions
# =============================================================================


def run_prediction(module: Any, model_dir: Path, csv_path: Path, at: str) -> pd.DataFrame:
    pred_df = module.predict_one(str(model_dir), str(csv_path), at=at)
    pred_df = pred_df.copy()
    if "TIMESTAMP" not in pred_df.columns:
        raise ValueError("La sortida de predict_one no conté la columna TIMESTAMP")
    pred_df["TIMESTAMP"] = pd.to_datetime(pred_df["TIMESTAMP"], errors="coerce")
    pred_df = pred_df.dropna(subset=["TIMESTAMP"]).reset_index(drop=True)
    return pred_df


def actual_init_time_from_prediction(pred_df: pd.DataFrame, requested_at: str, freq: str) -> pd.Timestamp:
    if pred_df.empty:
        return pd.Timestamp(requested_at)
    first_future = pd.Timestamp(pred_df["TIMESTAMP"].iloc[0])
    return first_future - pd.Timedelta(freq)


def build_comparison_dataframe(
    pred_df: pd.DataFrame,
    real_df: pd.DataFrame,
    datetime_col: str,
    temp_col: str,
    rain_col: str,
    init_time: pd.Timestamp,
    rain_threshold: float,
) -> pd.DataFrame:
    cols_keep = [datetime_col, temp_col, rain_col]
    missing = [c for c in cols_keep if c not in real_df.columns]
    if missing:
        raise ValueError(f"Al CSV real falten columnes necessàries: {missing}")

    real_small = real_df[cols_keep].copy()
    real_small = real_small.rename(columns={datetime_col: "TIMESTAMP", temp_col: "temp_real", rain_col: "rain_real"})
    out = pred_df.merge(real_small, on="TIMESTAMP", how="left")

    if out[["temp_real", "rain_real"]].isna().all(axis=None):
        raise ValueError(
            "No s'han trobat observacions reals per als timestamps predits. "
            "Revisa el CSV, la zona horària o tria un timestamp prou anterior al final del dataset."
        )

    out["init_time"] = pd.Timestamp(init_time)
    out["lead_step"] = np.arange(1, len(out) + 1, dtype=int)
    out["lead_minutes"] = (out["TIMESTAMP"] - pd.Timestamp(init_time)).dt.total_seconds() / 60.0
    out["lead_hours"] = out["lead_minutes"] / 60.0

    out["temp_error"] = out["temp_mu"] - out["temp_real"]
    out["temp_abs_error"] = out["temp_error"].abs()
    out["temp_sq_error"] = out["temp_error"] ** 2
    if {"temp_lo80", "temp_hi80"}.issubset(out.columns):
        out["temp_inside_80"] = ((out["temp_real"] >= out["temp_lo80"]) & (out["temp_real"] <= out["temp_hi80"])).astype(float)
        out["temp_width80"] = out["temp_hi80"] - out["temp_lo80"]

    out["rain_error"] = out["rain_mu"] - out["rain_real"]
    out["rain_abs_error"] = out["rain_error"].abs()
    out["rain_sq_error"] = out["rain_error"] ** 2
    out["rain_occ_real"] = (out["rain_real"] >= rain_threshold).astype(float)
    if "p_rain" in out.columns:
        out["rain_brier"] = (out["p_rain"] - out["rain_occ_real"]) ** 2
        out["rain_occ_pred_05"] = (out["p_rain"] >= 0.5).astype(float)
    if "p_rain_raw" in out.columns:
        out["rain_brier_raw"] = (out["p_rain_raw"] - out["rain_occ_real"]) ** 2
    if "p_rain_baseline" in out.columns:
        out["rain_brier_baseline"] = (out["p_rain_baseline"] - out["rain_occ_real"]) ** 2
    return out


def contingency_counts(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> Tuple[int, int, int, int]:
    y = np.asarray(y_true).astype(int)
    s = np.asarray(y_score, dtype=float)
    mask = np.isfinite(y) & np.isfinite(s)
    if not np.any(mask):
        return 0, 0, 0, 0
    y = y[mask]
    pred = (s[mask] >= threshold).astype(int)
    tp = int(np.sum((pred == 1) & (y == 1)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    return tp, fn, fp, tn


def categorical_metrics(tp: int, fn: int, fp: int, tn: int) -> Dict[str, float]:
    n = tp + fn + fp + tn
    pod = tp / (tp + fn) if (tp + fn) else float("nan")
    pofd = fp / (fp + tn) if (fp + tn) else float("nan")
    far = fp / (tp + fp) if (tp + fp) else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    csi = tp / (tp + fn + fp) if (tp + fn + fp) else float("nan")
    fbias = (tp + fp) / (tp + fn) if (tp + fn) else float("nan")
    acc = (tp + tn) / n if n else float("nan")
    random_hits = ((tp + fp) * (tp + fn) / n) if n else float("nan")
    ets_den = tp + fn + fp - random_hits if np.isfinite(random_hits) else float("nan")
    ets = (tp - random_hits) / ets_den if np.isfinite(ets_den) and abs(ets_den) > 1e-12 else float("nan")
    return {
        "n": n,
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "pod": pod,
        "pofd": pofd,
        "far": far,
        "precision": precision,
        "csi": csi,
        "frequency_bias": fbias,
        "accuracy": acc,
        "ets": ets,
    }


def accumulation_summary(df: pd.DataFrame, horizons_hours: Sequence[float]) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for h in horizons_hours:
        sub = df[df["lead_hours"] <= float(h)]
        if sub.empty:
            continue
        obs = finite_mean([sub["rain_real"].sum()])
        pred = finite_mean([sub["rain_mu"].sum()])
        row = {
            "horizon_hours": float(h),
            "n_steps": int(len(sub)),
            "rain_real_accum_mm": obs,
            "rain_pred_accum_mm": pred,
            "bias_mm": pred - obs if np.isfinite(pred) and np.isfinite(obs) else float("nan"),
            "abs_error_mm": abs(pred - obs) if np.isfinite(pred) and np.isfinite(obs) else float("nan"),
        }
        if "rain_q90" in sub.columns:
            row["rain_q90_sum_mm"] = finite_mean([sub["rain_q90"].sum()])
        rows.append(row)
    return rows


def build_prediction_summary(df: pd.DataFrame, rain_threshold: float, probability_threshold: float) -> dict:
    summary = {
        "init_time": str(pd.Timestamp(df["init_time"].iloc[0])),
        "start_pred_time": str(pd.Timestamp(df["TIMESTAMP"].min())),
        "end_pred_time": str(pd.Timestamp(df["TIMESTAMP"].max())),
        "n_steps": int(len(df)),
        "n_members": int(df["n_members"].iloc[0]) if "n_members" in df.columns and pd.notna(df["n_members"].iloc[0]) else None,
        "rain_event_threshold_mm_per_step": float(rain_threshold),
        "decision_probability_threshold": float(probability_threshold),
        "temp_mae_window": finite_mean(df["temp_abs_error"]),
        "temp_rmse_window": finite_rmse(df["temp_error"]),
        "temp_bias_window": finite_mean(df["temp_error"]),
        "rain_mae_window": finite_mean(df["rain_abs_error"]),
        "rain_rmse_window": finite_rmse(df["rain_error"]),
        "rain_bias_window": finite_mean(df["rain_error"]),
        "rain_real_total_mm": finite_mean([df["rain_real"].sum()]),
        "rain_pred_total_mm": finite_mean([df["rain_mu"].sum()]),
        "max_temp_abs_error": finite_mean([df["temp_abs_error"].max()]),
        "max_rain_abs_error": finite_mean([df["rain_abs_error"].max()]),
    }
    if "temp_inside_80" in df.columns:
        summary["temp_coverage80_window"] = finite_mean(df["temp_inside_80"])
        summary["temp_width80_mean_window"] = finite_mean(df["temp_width80"])
    if "p_rain" in df.columns:
        summary.update(
            {
                "mean_p_rain": finite_mean(df["p_rain"]),
                "max_p_rain": finite_mean([df["p_rain"].max()]),
                "rain_brier_window": finite_mean(df["rain_brier"]),
                "n_real_wet_steps": int((df["rain_occ_real"] > 0.5).sum()),
                "n_pred_wet_steps_probability_threshold": int((df["p_rain"] >= probability_threshold).sum()),
            }
        )
        tp, fn, fp, tn = contingency_counts(df["rain_occ_real"].to_numpy(), df["p_rain"].to_numpy(), probability_threshold)
        summary["rain_categorical"] = categorical_metrics(tp, fn, fp, tn)
    return summary


def make_prediction_plots(df: pd.DataFrame, case_dir: Path, label: str) -> Dict[str, str]:
    figs: Dict[str, str] = {}

    plt.figure(figsize=(10, 5))
    plt.plot(df["TIMESTAMP"], df["temp_real"], marker="o", label="Temperatura observada")
    if "temp_baseline" in df.columns:
        plt.plot(df["TIMESTAMP"], df["temp_baseline"], marker=".", label="Línia base T")
    plt.plot(df["TIMESTAMP"], df["temp_mu"], marker="s", label="Temperatura predita")
    if {"temp_lo80", "temp_hi80"}.issubset(df.columns):
        plt.fill_between(df["TIMESTAMP"], df["temp_lo80"], df["temp_hi80"], alpha=0.2, label="Interval 80%")
    plt.xlabel("Temps")
    plt.ylabel("Temperatura (°C)")
    plt.title("Temperatura observada vs prevista")
    plt.grid(True, alpha=0.3)
    plt.legend()
    _format_time_axis(plt.gca())
    p = case_dir / f"temperatura_observada_vs_prevista_{label}.png"
    savefig(p)
    figs["temperature_prediction"] = str(p)

    plt.figure(figsize=(10, 5))
    plt.plot(df["lead_hours"], df["temp_error"], marker="o", label="Error signat")
    plt.plot(df["lead_hours"], df["temp_abs_error"], marker="s", label="Error absolut")
    plt.axhline(0.0, linewidth=1)
    plt.xlabel("Lead time (h)")
    plt.ylabel("Error (°C)")
    plt.title("Error de temperatura per lead time")
    plt.grid(True, alpha=0.3)
    plt.legend()
    p = case_dir / f"error_temperatura_{label}.png"
    savefig(p)
    figs["temperature_error"] = str(p)

    plt.figure(figsize=(10, 5))
    width = 0.006
    x = pd.to_datetime(df["TIMESTAMP"])
    plt.bar(x, df["rain_real"], width=width, label="Precipitació observada")
    plt.plot(x, df["rain_mu"], marker="s", label="Precipitació predita mitjana")
    if "rain_mu_baseline" in df.columns:
        plt.plot(x, df["rain_mu_baseline"], marker=".", label="Línia base quantitat")
    if "rain_q90" in df.columns:
        plt.plot(x, df["rain_q90"], marker="^", label="Quantil 90%")
    plt.xlabel("Temps")
    plt.ylabel("Precipitació (mm / pas)")
    plt.title("Precipitació observada vs prevista")
    plt.grid(True, alpha=0.3)
    plt.legend()
    _format_time_axis(plt.gca())
    p = case_dir / f"precipitacio_observada_vs_prevista_{label}.png"
    savefig(p)
    figs["rain_prediction"] = str(p)

    plt.figure(figsize=(10, 5))
    if "p_rain_baseline" in df.columns:
        plt.plot(df["TIMESTAMP"], df["p_rain_baseline"], marker=".", label="Probabilitat base")
    if "p_rain_raw" in df.columns:
        plt.plot(df["TIMESTAMP"], df["p_rain_raw"], marker="o", label="Probabilitat crua")
    if "p_rain" in df.columns:
        plt.plot(df["TIMESTAMP"], df["p_rain"], marker="s", label="Probabilitat calibrada/operativa")
    plt.step(df["TIMESTAMP"], df["rain_occ_real"], where="mid", label="Ocurrència observada")
    plt.ylim(-0.05, 1.05)
    plt.xlabel("Temps")
    plt.ylabel("Probabilitat / ocurrència")
    plt.title("Probabilitat de precipitació")
    plt.grid(True, alpha=0.3)
    plt.legend()
    _format_time_axis(plt.gca())
    p = case_dir / f"probabilitat_precipitacio_{label}.png"
    savefig(p)
    figs["rain_probability"] = str(p)

    plt.figure(figsize=(10, 5))
    plt.plot(df["lead_hours"], df["rain_real"].cumsum(), marker="o", label="Acumulada observada")
    plt.plot(df["lead_hours"], df["rain_mu"].cumsum(), marker="s", label="Acumulada prevista mitjana")
    if "rain_q90" in df.columns:
        plt.plot(df["lead_hours"], df["rain_q90"].cumsum(), marker="^", label="Suma q90 per pas")
    plt.xlabel("Lead time (h)")
    plt.ylabel("Precipitació acumulada (mm)")
    plt.title("Precipitació acumulada dins de la finestra")
    plt.grid(True, alpha=0.3)
    plt.legend()
    p = case_dir / f"precipitacio_acumulada_{label}.png"
    savefig(p)
    figs["rain_accumulation"] = str(p)

    if "solar_toa_sw" in df.columns:
        plt.figure(figsize=(10, 5))
        plt.plot(df["TIMESTAMP"], df["solar_toa_sw"], marker="o", label="Radiació solar TOA")
        if "solar_cos_zenith" in df.columns:
            plt.plot(df["TIMESTAMP"], df["solar_cos_zenith"], marker="s", label="cos(zenit solar)")
        plt.xlabel("Temps")
        plt.ylabel("Variable solar")
        plt.title("Diagnòstic de geometria solar usada pel model")
        plt.grid(True, alpha=0.3)
        plt.legend()
        _format_time_axis(plt.gca())
        p = case_dir / f"geometria_solar_{label}.png"
        savefig(p)
        figs["solar_geometry"] = str(p)

    return figs


def save_prediction_txt(pred_df: pd.DataFrame, outpath: Path) -> None:
    write_text(pred_df.to_string(index=False) + "\n", outpath)


def process_one_timestamp(
    module: Any,
    model_dir: Path,
    csv_path: Path,
    real_df: pd.DataFrame,
    config: Optional[dict],
    datetime_col: str,
    temp_col: str,
    rain_col: str,
    at: str,
    outdir: Path,
    rain_threshold: float,
    probability_threshold: float,
) -> dict:
    pred_df = run_prediction(module, model_dir=model_dir, csv_path=csv_path, at=at)
    freq = str(get_config_value(config, "freq", "10min"))
    init_time = actual_init_time_from_prediction(pred_df, at, freq)
    comp_df = build_comparison_dataframe(pred_df, real_df, datetime_col, temp_col, rain_col, init_time, rain_threshold)

    label = slugify_timestamp(str(init_time))
    case_dir = outdir / "cases" / label
    ensure_dir(case_dir)

    pred_csv = case_dir / f"prediction_{label}.csv"
    pred_txt = case_dir / f"prediction_{label}.txt"
    comp_csv = case_dir / f"prediction_vs_observations_{label}.csv"
    summary_json = case_dir / f"summary_{label}.json"
    accum_json = case_dir / f"rain_accumulation_{label}.json"

    pred_df.to_csv(pred_csv, index=False)
    save_prediction_txt(pred_df, pred_txt)
    comp_df.to_csv(comp_csv, index=False)

    summary = build_prediction_summary(comp_df, rain_threshold, probability_threshold)
    horizons = get_config_value(config, "accumulation_horizons_hours", [1.0, 2.0, 3.0])
    accum = accumulation_summary(comp_df, horizons)
    write_json(summary, summary_json)
    write_json(accum, accum_json)

    figures = make_prediction_plots(comp_df, case_dir, label)

    return {
        "requested_at": str(at),
        "actual_init_time": str(init_time),
        "case_dir": str(case_dir),
        "prediction_csv": str(pred_csv),
        "prediction_txt": str(pred_txt),
        "comparison_csv": str(comp_csv),
        "summary_json": str(summary_json),
        "rain_accumulation_json": str(accum_json),
        "figures": figures,
        "summary": summary,
        "rain_accumulation": accum,
    }


# =============================================================================
# Selecció automàtica de casos
# =============================================================================


def valid_case_indices(df: pd.DataFrame, config: Optional[dict], artifacts: Optional[dict], datetime_col: str, segment_col: str) -> np.ndarray:
    dt, H, O, _ = infer_steps_from_config(config, artifacts)
    n = len(df)
    if n < H + O + 1:
        return np.array([], dtype=int)

    train_end = pd.Timestamp(get_config_value(config, "val_end", "2024-01-01"))
    test_end = pd.Timestamp(get_config_value(config, "test_end", "2025-01-01"))
    t = pd.to_datetime(df[datetime_col])
    candidates = []
    seg = df[segment_col].to_numpy() if segment_col in df.columns else np.zeros(n, dtype=int)
    for i in range(H - 1, n - O):
        ti = t.iloc[i]
        if not (ti >= train_end and ti < test_end):
            continue
        if t.iloc[i + O] >= test_end:
            continue
        if seg[i - H + 1] != seg[i] or seg[i + O] != seg[i]:
            continue
        candidates.append(i)
    return np.array(candidates, dtype=int)


def choose_auto_cases(
    df: pd.DataFrame,
    config: Optional[dict],
    artifacts: Optional[dict],
    datetime_col: str,
    rain_col: str,
    segment_col: str,
    rain_threshold: float,
    n_cases: int,
) -> List[str]:
    if n_cases <= 0:
        return []
    idx = valid_case_indices(df, config, artifacts, datetime_col, segment_col)
    if idx.size == 0:
        return []
    _, _, O, steps_per_hour = infer_steps_from_config(config, artifacts)
    rain = pd.to_numeric(df[rain_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)

    selected: List[int] = []

    def add(i: Optional[int]) -> None:
        if i is None:
            return
        if i not in selected:
            selected.append(int(i))

    # Cas amb màxima precipitació acumulada futura.
    future_sums = np.array([np.nansum(rain[i + 1:i + O + 1]) for i in idx], dtype=float)
    add(idx[int(np.nanargmax(future_sums))])

    # Inici de pluja: sec en l'última hora, pluja futura.
    past_steps = max(1, int(steps_per_hour))
    onset = []
    for i in idx:
        past_wet = np.nansum(rain[max(0, i - past_steps + 1):i + 1] >= rain_threshold)
        fut_wet = np.nansum(rain[i + 1:i + O + 1] >= rain_threshold)
        if past_wet == 0 and fut_wet > 0:
            onset.append(i)
    if onset:
        add(onset[len(onset) // 2])

    # Ja plou a la inicialització.
    already = [i for i in idx if rain[i] >= rain_threshold]
    if already:
        add(already[len(already) // 2])

    # Cas sec persistent.
    dry = []
    for i in idx:
        past_wet = np.nansum(rain[max(0, i - past_steps + 1):i + 1] >= rain_threshold)
        fut_wet = np.nansum(rain[i + 1:i + O + 1] >= rain_threshold)
        if past_wet == 0 and fut_wet == 0:
            dry.append(i)
    if dry:
        add(dry[len(dry) // 2])

    # Completa amb casos uniformement distribuïts.
    if len(selected) < n_cases:
        positions = np.linspace(0, idx.size - 1, num=n_cases, dtype=int)
        for pos in positions:
            add(idx[pos])
            if len(selected) >= n_cases:
                break

    selected = selected[:n_cases]
    return [str(pd.Timestamp(df[datetime_col].iloc[i])) for i in selected]


# =============================================================================
# Diagnòstic agregat de casos predits
# =============================================================================


def create_cases_aggregate(case_results: List[dict], outdir: Path) -> Optional[dict]:
    frames = []
    for res in case_results:
        p = Path(res["comparison_csv"])
        if p.exists():
            frames.append(pd.read_csv(p, parse_dates=["TIMESTAMP", "init_time"]))
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    agg_dir = outdir / "cases_aggregate"
    ensure_dir(agg_dir)
    df.to_csv(agg_dir / "all_prediction_cases_vs_observations.csv", index=False)

    rows = []
    for lead, g in df.groupby("lead_step"):
        row = {
            "lead_step": int(lead),
            "lead_hours": finite_mean(g["lead_hours"]),
            "n": int(len(g)),
            "temp_mae": finite_mean(g["temp_abs_error"]),
            "temp_rmse": finite_rmse(g["temp_error"]),
            "rain_mae": finite_mean(g["rain_abs_error"]),
            "rain_rmse": finite_rmse(g["rain_error"]),
        }
        if "rain_brier" in g.columns:
            row["rain_brier"] = finite_mean(g["rain_brier"])
        if "temp_inside_80" in g.columns:
            row["temp_coverage80"] = finite_mean(g["temp_inside_80"])
        rows.append(row)
    bylead = pd.DataFrame(rows)
    bylead.to_csv(agg_dir / "aggregate_metrics_by_lead.csv", index=False)

    figures = {}
    p = line_plot(
        bylead,
        "lead_hours",
        ["temp_mae", "temp_rmse"],
        ["MAE T", "RMSE T"],
        "Casos predits: error de temperatura per lead",
        "Lead time (h)",
        "°C",
        agg_dir / "aggregate_temperature_error_by_lead.png",
    )
    if p: figures["aggregate_temperature_error_by_lead"] = p
    p = line_plot(
        bylead,
        "lead_hours",
        ["rain_mae", "rain_rmse", "rain_brier"],
        ["MAE pluja", "RMSE pluja", "Brier pluja"],
        "Casos predits: diagnòstic de precipitació per lead",
        "Lead time (h)",
        "Mètrica",
        agg_dir / "aggregate_rain_metrics_by_lead.png",
    )
    if p: figures["aggregate_rain_metrics_by_lead"] = p

    summary = {
        "n_cases": len(frames),
        "n_points": int(len(df)),
        "temp_mae": finite_mean(df["temp_abs_error"]),
        "temp_rmse": finite_rmse(df["temp_error"]),
        "rain_mae": finite_mean(df["rain_abs_error"]),
        "rain_rmse": finite_rmse(df["rain_error"]),
        "rain_brier": finite_mean(df["rain_brier"]) if "rain_brier" in df.columns else float("nan"),
    }
    write_json(summary, agg_dir / "aggregate_summary.json")
    return {"aggregate_dir": str(agg_dir), "summary": summary, "figures": figures}


# =============================================================================
# CLI
# =============================================================================


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Genera diagnòstics professionals del model de nowcasting UB a partir de JSONs i/o prediccions."
    )
    p.add_argument("--model-dir", required=True, help="Directori del model o ZIP de resultats final_hybrid...zip")
    p.add_argument("--model-script", default=None, help="Ruta al fitxer .py del model. Només és necessari per executar prediccions.")
    p.add_argument("--csv", default=None, help="CSV del dataset nowcasting o ZIP que el conté. Necessari per executar prediccions.")
    p.add_argument("--outdir", required=True, help="Directori on es desaran taules, JSON i figures")
    p.add_argument("--at", action="append", default=None, help="Timestamp d'inicialització. Es pot repetir.")
    p.add_argument("--auto-sample-cases", type=int, default=0, help="Nombre de casos de test a seleccionar automàticament i predir.")
    p.add_argument("--rain-threshold", type=float, default=None, help="Llindar de pluja per pas, en mm. Per defecte s'agafa del config o 0.1.")
    p.add_argument("--probability-threshold", type=float, default=0.5, help="Llindar de probabilitat per mètriques categòriques locals.")
    p.add_argument("--workdir", default=None, help="Directori temporal d'extracció. Per defecte: OUTDIR/_work")
    p.add_argument("--force-extract", action="store_true", help="Força tornar a extreure ZIPs encara que ja hi hagi cache.")
    p.add_argument("--skip-global-metrics", action="store_true", help="No generar gràfiques globals a partir dels JSON de resultats.")
    p.add_argument("--skip-predictions", action="store_true", help="No executar prediccions encara que s'indiqui --at o --auto-sample-cases.")
    return p


def main() -> None:
    args = build_parser().parse_args()

    model_container = Path(args.model_dir).expanduser().resolve()
    if not model_container.exists():
        raise FileNotFoundError(f"No trobo --model-dir: {model_container}")

    outdir = Path(args.outdir).expanduser().resolve()
    ensure_dir(outdir)
    workdir = Path(args.workdir).expanduser().resolve() if args.workdir else outdir / "_work"
    ensure_dir(workdir)

    result_jsons = load_saved_result_jsons(model_container)
    config = result_jsons.get("config")
    artifacts = result_jsons.get("artifacts")

    rain_threshold = args.rain_threshold
    if rain_threshold is None:
        rain_threshold = float(get_config_value(config, "rain_event_threshold", DEFAULT_RAIN_THRESHOLD_MM))

    global_manifest = None
    if not args.skip_global_metrics:
        _log("[INFO] Generant diagnòstics globals a partir dels JSON guardats...")
        global_manifest = create_metrics_diagnostics(model_container, outdir)
        _log(f"[OK] Diagnòstics globals generats: {global_manifest['summary_markdown']}")

    run_predictions = (bool(args.at) or args.auto_sample_cases > 0) and not args.skip_predictions
    prediction_manifest: List[dict] = []
    aggregate_manifest = None

    if run_predictions:
        if not args.model_script:
            raise ValueError("Per executar prediccions cal indicar --model-script.")
        if not args.csv:
            raise ValueError("Per executar prediccions cal indicar --csv.")

        _log("[INFO] Preparant model i dataset per executar prediccions...")
        model_dir_for_prediction = resolve_model_dir_for_prediction(args.model_dir, workdir, force_extract=args.force_extract)
        csv_path = resolve_csv_path(args.csv, workdir, force_extract=args.force_extract)
        assert csv_path is not None

        model_script = Path(args.model_script).expanduser().resolve()
        module = load_model_module(model_script)

        datetime_col = str(get_config_value(config, "datetime_col", "TIMESTAMP"))
        temp_col = str(get_config_value(config, "temp_col", "Temp_1_Avg"))
        rain_col = str(get_config_value(config, "rain_col", "Lluvia_Tot"))
        segment_col = str(get_config_value(config, "segment_col", "segment_id"))

        real_df = load_real_csv(csv_path, datetime_col=datetime_col)

        ats: List[str] = list(args.at or [])
        if args.auto_sample_cases > 0:
            _log(f"[INFO] Seleccionant {args.auto_sample_cases} casos representatius del test...")
            auto_ats = choose_auto_cases(
                real_df,
                config,
                artifacts,
                datetime_col=datetime_col,
                rain_col=rain_col,
                segment_col=segment_col,
                rain_threshold=rain_threshold,
                n_cases=args.auto_sample_cases,
            )
            ats.extend(auto_ats)
            _log(f"[OK] Casos automàtics seleccionats: {len(auto_ats)}")

        # Elimina duplicats preservant l'ordre.
        seen = set()
        ats = [a for a in ats if not (a in seen or seen.add(a))]

        for at in ats:
            _log(f"[INFO] Predient i diagnosticant cas: {at}")
            res = process_one_timestamp(
                module=module,
                model_dir=model_dir_for_prediction,
                csv_path=csv_path,
                real_df=real_df,
                config=config,
                datetime_col=datetime_col,
                temp_col=temp_col,
                rain_col=rain_col,
                at=at,
                outdir=outdir,
                rain_threshold=rain_threshold,
                probability_threshold=args.probability_threshold,
            )
            prediction_manifest.append(res)
            _log(
                f"[OK] {res['actual_init_time']} | "
                f"MAE T={res['summary']['temp_mae_window']:.3f} °C | "
                f"MAE pluja={res['summary']['rain_mae_window']:.4f} mm"
            )

        if prediction_manifest:
            aggregate_manifest = create_cases_aggregate(prediction_manifest, outdir)

    manifest = {
        "model_container": str(model_container),
        "outdir": str(outdir),
        "rain_threshold_mm_per_step": rain_threshold,
        "probability_threshold": args.probability_threshold,
        "global_metrics": global_manifest,
        "prediction_cases": prediction_manifest,
        "prediction_cases_aggregate": aggregate_manifest,
    }
    manifest_path = outdir / "manifest_prediction_diagnostics.json"
    write_json(manifest, manifest_path)
    _log(f"[OK] Manifest general desat a: {manifest_path}")


if __name__ == "__main__":
    main()
