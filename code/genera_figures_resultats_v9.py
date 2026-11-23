#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Genera les figures dels apartats de resultats i conclusions del TFM.

Entrades:
- ZIP de la v9 sense radiosondatge.
- ZIP de la v9 amb índexs convectius de radiosondatge.
- ZIP de diagnòstics finals de la v9 sense radiosondatge.

Sortides, dins del directori indicat amb --outdir:
- fig_resultats_temp_mae_horitzo.pdf i .png
- fig_resultats_rain_brier_horitzo.pdf i .png
- fig_resultats_rain_reliability_no_rs.pdf i .png
- fig_resultats_rain_acumulats.pdf i .png
- fig_resultats_cas_intens_2024-11-04.png

Exemple:
python genera_figures_resultats_v9.py \
  --no-rs-zip final_hybrid_timemixer_FINAL_no_rs_v9_amount_calibrated_end_to_end_ens10_v1.zip \
  --rs-zip final_hybrid_timemixer_FINAL_RS_indices_v9_amount_calibrated_end_to_end_ens10_v1.zip \
  --diagnostics-zip diagnostics_final_no_rs_v9_amount_calibrated_end_to_end_ens10_v1.zip \
  --outdir imatges
"""

from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


METRICS_FILE = "test_metrics_ensemble.json"
INTENSE_CASE_FILE = "precipitacio_acumulada_2024-11-04_07-30-00.png"


def find_member(zf: zipfile.ZipFile, filename: str) -> str:
    """Troba un fitxer dins d'un ZIP, encara que sigui en una subcarpeta."""
    candidates = [
        name for name in zf.namelist()
        if name == filename or name.endswith("/" + filename)
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No s'ha trobat '{filename}' dins del ZIP '{zf.filename}'."
        )
    return candidates[0]


def read_json_from_zip(zip_path: Path, filename: str) -> dict[str, Any]:
    """Llegeix un fitxer JSON comprimit dins d'un ZIP."""
    if not zip_path.is_file():
        raise FileNotFoundError(f"No existeix el fitxer ZIP: {zip_path}")
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(find_member(zf, filename)) as fh:
            return json.load(fh)


def get(data: dict[str, Any], *keys: str) -> Any:
    """Recorre una seqüència de claus amb un error entenedor si en falta una."""
    value: Any = data
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            raise KeyError(f"No s'ha trobat la clau: {' -> '.join(keys)}")
        value = value[key]
    return value


def save_plot(fig: plt.Figure, outdir: Path, basename: str) -> None:
    """Desa cada gràfic en PDF vectorial per a Overleaf i en PNG."""
    fig.tight_layout()
    fig.savefig(outdir / f"{basename}.pdf", bbox_inches="tight")
    fig.savefig(outdir / f"{basename}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_temperature_mae(no_rs: dict[str, Any], rs: dict[str, Any], outdir: Path) -> None:
    """MAE de temperatura per horitzó: superfície, RS i persistència."""
    lead = np.asarray(get(no_rs, "by_lead", "lead_hours"), dtype=float)
    mae_no_rs = np.asarray(get(no_rs, "by_lead", "temp_mae"), dtype=float)
    mae_rs = np.asarray(get(rs, "by_lead", "temp_mae"), dtype=float)
    mae_persist = np.asarray(
        get(no_rs, "baselines", "persistence", "by_lead", "temp_mae"), dtype=float
    )

    fig, ax = plt.subplots(figsize=(6.2, 3.9))
    ax.plot(lead, mae_no_rs, marker="o", linewidth=1.8, label="Superfície")
    ax.plot(lead, mae_rs, marker="s", linewidth=1.8, label="Superfície + RS")
    ax.plot(lead, mae_persist, marker="^", linewidth=1.5, label="Persistència")
    ax.set_xlabel("Horitzó de predicció (h)")
    ax.set_ylabel("MAE de temperatura (°C)")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    save_plot(fig, outdir, "fig_resultats_temp_mae_horitzo")


def plot_rain_brier(no_rs: dict[str, Any], rs: dict[str, Any], outdir: Path) -> None:
    """Brier score calibrat de precipitació per horitzó."""
    lead = np.asarray(get(no_rs, "by_lead", "lead_hours"), dtype=float)
    brier_no_rs = np.asarray(get(no_rs, "by_lead", "rain_brier_calibrated"), dtype=float)
    brier_rs = np.asarray(get(rs, "by_lead", "rain_brier_calibrated"), dtype=float)
    brier_markov = np.asarray(
        get(no_rs, "baselines", "rain_markov", "by_lead", "rain_brier"), dtype=float
    )

    fig, ax = plt.subplots(figsize=(6.2, 3.9))
    ax.plot(lead, brier_no_rs, marker="o", linewidth=1.8, label="Superfície")
    ax.plot(lead, brier_rs, marker="s", linewidth=1.8, label="Superfície + RS")
    ax.plot(lead, brier_markov, marker="^", linewidth=1.5, label="Markov (superfície)")
    ax.set_xlabel("Horitzó de predicció (h)")
    ax.set_ylabel("Brier score de precipitació")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    save_plot(fig, outdir, "fig_resultats_rain_brier_horitzo")


def plot_rain_reliability(no_rs: dict[str, Any], outdir: Path) -> None:
    """Diagrama de fiabilitat de la precipitació abans i després del calibratge."""
    raw = get(no_rs, "diagnostics", "rain_reliability_raw", "global_bins")
    calibrated = get(no_rs, "diagnostics", "rain_reliability_calibrated", "global_bins")

    p_raw = np.asarray(raw["mean_forecast_probability"], dtype=float)
    o_raw = np.asarray(raw["observed_frequency"], dtype=float)
    p_cal = np.asarray(calibrated["mean_forecast_probability"], dtype=float)
    o_cal = np.asarray(calibrated["observed_frequency"], dtype=float)
    valid_raw = np.isfinite(p_raw) & np.isfinite(o_raw)
    valid_cal = np.isfinite(p_cal) & np.isfinite(o_cal)

    fig, ax = plt.subplots(figsize=(5.4, 4.3))
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.2, label="Calibratge perfecte")
    ax.plot(p_raw[valid_raw], o_raw[valid_raw], marker="o", linewidth=1.6, label="Abans de calibrar")
    ax.plot(p_cal[valid_cal], o_cal[valid_cal], marker="s", linewidth=1.6, label="Després de calibrar")
    ax.set_xlabel("Probabilitat prevista de pluja")
    ax.set_ylabel("Freqüència observada")
    ax.set_xlim(0, 0.65)
    ax.set_ylim(0, 0.65)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=9)
    save_plot(fig, outdir, "fig_resultats_rain_reliability_no_rs")


def plot_rain_accumulation(no_rs: dict[str, Any], rs: dict[str, Any], outdir: Path) -> None:
    """MAE de precipitació acumulada a 1, 2 i 3 hores."""
    accum_no_rs = get(no_rs, "diagnostics", "rain_accumulation_calibrated", "by_horizon")
    accum_rs = get(rs, "diagnostics", "rain_accumulation_calibrated", "by_horizon")

    hours = np.asarray(accum_no_rs["horizon_hours"], dtype=float)
    mae_no_rs = np.asarray(accum_no_rs["mae"], dtype=float)
    mae_rs = np.asarray(accum_rs["mae"], dtype=float)

    fig, ax = plt.subplots(figsize=(5.4, 4.3))
    ax.plot(hours, mae_no_rs, marker="o", linewidth=1.8, label="Superfície")
    ax.plot(hours, mae_rs, marker="s", linewidth=1.8, label="Superfície + RS")
    ax.set_xlabel("Horitzó d'acumulació (h)")
    ax.set_ylabel("MAE de precipitació acumulada (mm)")
    ax.set_xticks(hours)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    save_plot(fig, outdir, "fig_resultats_rain_acumulats")


def extract_intense_case(diagnostics_zip: Path, outdir: Path) -> None:
    """Extreu la figura diagnòstica del cas intens utilitzat a l'informe."""
    if not diagnostics_zip.is_file():
        raise FileNotFoundError(f"No existeix el fitxer ZIP: {diagnostics_zip}")
    destination = outdir / "fig_resultats_cas_intens_2024-11-04.png"
    with zipfile.ZipFile(diagnostics_zip) as zf:
        member = find_member(zf, INTENSE_CASE_FILE)
        with zf.open(member) as source, destination.open("wb") as target:
            shutil.copyfileobj(source, target)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Genera les figures dels resultats del model v9.")
    parser.add_argument(
        "--no-rs-zip", type=Path,
        default=Path("final_hybrid_timemixer_FINAL_no_rs_v9_amount_calibrated_end_to_end_ens10_v1.zip"),
        help="ZIP de resultats de la v9 sense radiosondatge.",
    )
    parser.add_argument(
        "--rs-zip", type=Path,
        default=Path("final_hybrid_timemixer_FINAL_RS_indices_v9_amount_calibrated_end_to_end_ens10_v1.zip"),
        help="ZIP de resultats de la v9 amb índexs convectius.",
    )
    parser.add_argument(
        "--diagnostics-zip", type=Path,
        default=Path("diagnostics_final_no_rs_v9_amount_calibrated_end_to_end_ens10_v1.zip"),
        help="ZIP amb els diagnòstics finals de la v9 sense radiosondatge.",
    )
    parser.add_argument(
        "--outdir", type=Path, default=Path("imatges"),
        help="Directori on s'escriuran les figures.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    no_rs = read_json_from_zip(args.no_rs_zip, METRICS_FILE)
    rs = read_json_from_zip(args.rs_zip, METRICS_FILE)

    plot_temperature_mae(no_rs, rs, args.outdir)
    plot_rain_brier(no_rs, rs, args.outdir)
    plot_rain_reliability(no_rs, args.outdir)
    plot_rain_accumulation(no_rs, rs, args.outdir)
    extract_intense_case(args.diagnostics_zip, args.outdir)

    print("Figures generades correctament:")
    for name in [
        "fig_resultats_temp_mae_horitzo.pdf",
        "fig_resultats_rain_brier_horitzo.pdf",
        "fig_resultats_rain_reliability_no_rs.pdf",
        "fig_resultats_rain_acumulats.pdf",
        "fig_resultats_cas_intens_2024-11-04.png",
    ]:
        print(f" - {args.outdir / name}")


if __name__ == "__main__":
    main()
