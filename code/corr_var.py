#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Càlcul de correlacions entre variables meteorològiques i selecció de features per a l'entrenament del model de nowcasting.

IMPORTANT:
----------
- Aquest codi s'ha d'executar després d'haver executat el codi "imputa_NaN.py".
- Totes les correlacions es calculen només amb train (TIMESTAMP < train_end).
- El fitxer csv final per entrenament s'exporta complet (train + val/test), però les decisions de correlació es basen únicament en train.

Sortides:
---------
- correlacio_features_matrix_TRAIN.csv
- correlacio_features_heatmap_TRAIN.png
- correlacio_amb_Temp_1_Avg_TRAIN.csv
- correlacio_amb_Lluvia_Tot_TRAIN.csv
- correlacio_amb_rain_occ_TRAIN.csv
- correlacio_amb_rain_log1p_TRAIN.csv
- csv final: ..._features_nowcasting.csv

Exemple d'ús:
-------------
python corr_var.py --input "Dataset_final/CR3000_Tabla1_2011_2024_imputed.csv" --output-dir "Dataset_final" --train-end "2023-01-01" --corr-method spearman
"""

from __future__ import annotations

import argparse
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# Defaults i configuració
# =============================================================================

DEFAULT_INPUT_CSV = "Dataset_final/CR3000_Tabla1_2011_2024_imputed.csv"
DEFAULT_OUTPUT_DIR = "Dataset_final"
DEFAULT_OUTPUT_CSV = os.path.join(
    DEFAULT_OUTPUT_DIR,
    "CR3000_Tabla1_2011_2024_features_nowcasting.csv"
)

DEFAULT_TIMESTAMP_COL = "TIMESTAMP"
DEFAULT_TARGET_COLS = ["Lluvia_Tot", "Temp_1_Avg"]

# Features desitjades (ajustades a la sortida del codi imputa_NaN.py)
DEFAULT_SELECTED_FEATURES_DESIRED = [
    # ---------------------------------------------------------------------
    # Subconjunt òptim per maximitzar aprenentatge i generalització.
    #
    # Criteri:
    # - mantenir senyal físic fort i relativament independent.
    # - reduir colinearitat termodinàmica i radiativa.
    # - evitar duplicar informació ja present en derivades o variables de calendari.
    #
    # S'elimina explícitament:
    # - VP_Tot            : molt solapada amb Temp_1_Avg + Hum_Rel_1_Avg + VPD_Tot,
    # - Vel_Viento_Max    : molt colinear amb Vel_Viento_WVc(1) i, per aquest cas, aporta menys senyal marginal per a pluja que la velocitat mitjana vectorial.
    # - Vel_Viento_WVc(3) : menys imprescindible que la velocitat mitjana.
    # - paquet radiatiu redundant/derivat: Rad_Difusa_Avg, NetRs_Avg, NetRl_Avg, Albedo_Avg, UVA_Avg, UVB_Avg.
    # ---------------------------------------------------------------------

    # Termodinàmica
    "Hum_Rel_1_Avg",
    "Pres_Atm_Avg",
    "VPD_Tot",

    # Vent
    "Vel_Viento_WVc(1)",

    # Direcció del vent (al dataset imputat és sin/cos)
    "wind_dir_sin",
    "wind_dir_cos",

    # Radiació
    "Rad_Global_Avg",
]


# =============================================================================
# Utilitats i ajudes per a la lectura, preparació i càlcul de correlacions
# =============================================================================

def ensure_dir(path: str) -> None:
    """
    Crea el directori si no existeix.

    Això evita errors quan es volen escriure fitxers a un directori encara no creat.
    """
    os.makedirs(path, exist_ok=True)


def safe_select_features(df: pd.DataFrame, desired: List[str]) -> List[str]:
    """
    Selecciona les features desitjades que realment existeixen al DataFrame.

    El csv imputat pot variar lleugerament en columnes (segons configuracions).
    Si una feature no hi és, s'ignora i s'avisa.

    Retorna:
    - Llista de features presents, en el mateix ordre que "desired".
    """
    present: List[str] = []
    missing: List[str] = []

    for c in desired:
        if c in df.columns:
            present.append(c)
        else:
            missing.append(c)

    if missing:
        print("[AVÍS] Features desitjades que no existeixen i s'ignoren:", missing)

    return present


def save_corr_heatmap(corr_matrix: pd.DataFrame, out_png: str, title: str) -> None:
    """
    Desa un mapa de calor de la matriu de correlacions.

    Visualització:
    - vmin/vmax fixats a [-1, 1] per comparar fàcilment entre execucions.
    - cmap='coolwarm' per distingir correlació positiva/negativa.
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr_matrix.values, vmin=-1, vmax=1, cmap="coolwarm")

    ax.set_xticks(np.arange(len(corr_matrix.columns)))
    ax.set_yticks(np.arange(len(corr_matrix.index)))
    ax.set_xticklabels(corr_matrix.columns, rotation=90)
    ax.set_yticklabels(corr_matrix.index)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Correlació", fontsize=10)

    plt.title(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close(fig)


def miss_flag_cols_for_features(df: pd.DataFrame, features: List[str]) -> List[str]:
    """
    Retorna només les columnes de flags __miss associades a les features seleccionades.

    El codi "imputa_NaN.py" crea, per cada feature imputada, una columna <feature>__miss que val 1 si a l'original era NaN, i 0 altrament.
    Aquí només conservem les flags de les features que efectivament entren al model.
    """
    cols: List[str] = []
    for f in features:
        c = f"{f}__miss"
        if c in df.columns:
            cols.append(c)
    return cols


# =============================================================================
# Lectura i preparació del dataset
# =============================================================================

def read_input_csv(input_path: str, timestamp_col: str) -> pd.DataFrame:
    """
    Llegeix el csv i prepara la columna temporal:

    - Llegeix el csv.
    - Converteix la columna de timestamp a datetime (errors='coerce' -> NaT).
    - Elimina files amb timestamp invàlid.
    - Ordena per temps (requisit bàsic per a qualsevol tractament temporal).
    """
    print(f"Llegint: {input_path}")
    df = pd.read_csv(input_path)

    if timestamp_col not in df.columns:
        raise ValueError(f"No s'ha trobat la columna temporal '{timestamp_col}' al CSV.")

    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
    df = df.dropna(subset=[timestamp_col]).sort_values(timestamp_col)

    return df


def validate_targets_exist(df: pd.DataFrame, target_cols: List[str]) -> None:
    """
    Verificació mínima: els targets han d'existir al csv.

    Això evita seguir endavant i fallar més tard amb errors menys clars.
    """
    for tgt in target_cols:
        if tgt not in df.columns:
            raise ValueError(f"No s'ha trobat el target '{tgt}' al csv.")


def build_training_dataframe(
    df: pd.DataFrame,
    timestamp_col: str,
    target_cols: List[str],
    selected_features: List[str],
    include_segment_id: bool
) -> pd.DataFrame:
    """
    Construeix el dataframe.

    Columnes incloses (ordre preservat, sense duplicats):
    - TIMESTAMP
    - Targets (Lluvia_Tot, Temp_1_Avg)
    - Features seleccionades
    - segment_id (opcional, si existeix i està activat)
    - Flags __miss de les features seleccionades

    També crea dues variables auxiliars de pluja (per a l'anàlisi de correlacions):
    - rain_occ: ocurrència (1 si pluja > 0, 0 si no)
    - rain_log1p: log(1 + pluja), útil per reduir asimetries
    """
    # segment_id (opcional): només s'inclou si el fitxer el té
    seg_cols = ["segment_id"] if (include_segment_id and "segment_id" in df.columns) else []

    # Flags __miss: només les de les features seleccionades
    miss_cols = miss_flag_cols_for_features(df, selected_features)

    # Conjunt final de columnes a conservar
    keep_cols = [timestamp_col] + target_cols + selected_features + seg_cols + miss_cols
    keep_cols = list(dict.fromkeys(keep_cols))  # elimina duplicats preservant ordre

    df_out = df[keep_cols].copy()

    # Variables auxiliars per a pluja (per a les correlacions)
    df_out["rain_occ"] = (df_out["Lluvia_Tot"] > 0.0).astype(int)
    df_out["rain_log1p"] = np.log1p(df_out["Lluvia_Tot"].clip(lower=0.0))

    return df_out


# =============================================================================
# Correlacions (train-only)
# =============================================================================

def subset_train(df_out: pd.DataFrame, timestamp_col: str, train_end: pd.Timestamp) -> Tuple[pd.DataFrame, int, int]:
    """
    Extreu el subconjunt train per calcular correlacions sense leakage.

    Retorna:
    - df_corr: dataframe train-only
    - n_corr: files en train
    - n_all: files totals
    """
    df_corr = df_out.loc[df_out[timestamp_col] < train_end].copy()
    return df_corr, len(df_corr), len(df_out)


def compute_and_save_feature_matrix(
    df_corr: pd.DataFrame,
    selected_features: List[str],
    corr_method: str,
    output_dir: str
) -> None:
    """
    Calcula i desa:
    - Matriu de correlació features-feature (train)
    - Mapa de calor corresponent

    Es calcula només sobre "selected_features" per coherència amb la selecció del model.
    """
    print("\nCalculant correlacions entre features (train)...")
    corr_feat = df_corr[selected_features].corr(method=corr_method)

    corr_feat_path = os.path.join(output_dir, "correlacio_features_matrix_TRAIN.csv")
    corr_feat.to_csv(corr_feat_path)
    print(f"[OK] Matriu features-feature (train): {corr_feat_path}")

    heatmap_path = os.path.join(output_dir, "correlacio_features_heatmap_TRAIN.png")
    save_corr_heatmap(
        corr_feat,
        heatmap_path,
        f"Matriu de correlacions ({corr_method}) entre features (train)"
    )
    print(f"[OK] Heatmap (train): {heatmap_path}")


def compute_and_save_target_correlations(
    df_corr: pd.DataFrame,
    selected_features: List[str],
    corr_method: str,
    output_dir: str,
    target_name: str,
    out_filename: str
) -> None:
    """
    Calcula i desa la correlació de cada feature amb un target concret (train).

    Procediment:
    - Construeix un sub-dataframe amb features + target.
    - Calcula la matriu de correlacions.
    - Extreu la columna del target i elimina l'autocorrelació.
    - Ordena de major a menor i desa a csv.
    """
    corr_cols = list(dict.fromkeys(selected_features + [target_name]))
    corr_series = df_corr[corr_cols].corr(method=corr_method)[target_name].drop(target_name)
    corr_series = corr_series.sort_values(ascending=False)

    out_path = os.path.join(output_dir, out_filename)
    corr_series.to_csv(out_path)
    print(f"[OK] {out_path}")


# =============================================================================
# CLI i lògica d'execució
# =============================================================================

def parse_args() -> argparse.Namespace:
    """
    Defineix i parseja arguments de línia d'ordres.

    Millora respecte de la versió anterior:
    - si no s'indica --output-csv, es construeix automàticament dins de --output-dir
      per evitar discrepàncies entre els fitxers de correlació i el csv final.
    """
    p = argparse.ArgumentParser(
        description="Correlacions (train-only) + selecció de features per nowcasting."
    )

    p.add_argument("--input", default=DEFAULT_INPUT_CSV)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument(
        "--output-csv",
        default=None,
        help=(
            "Ruta del csv final. Si no s'indica, s'escriurà dins de --output-dir "
            "amb el nom CR3000_Tabla1_2011_2024_features_nowcasting.csv."
        ),
    )

    p.add_argument("--timestamp-col", default=DEFAULT_TIMESTAMP_COL)
    p.add_argument("--train-end", default="2023-01-01", help="Tall temporal de train (YYYY-MM-DD).")
    p.add_argument("--corr-method", default="pearson", choices=["pearson", "spearman", "kendall"])

    # Opcions per incloure o no segment_id al csv final
    p.add_argument("--include-segment-id", action="store_true", help="Inclou segment_id al csv final.")
    p.add_argument("--no-include-segment-id", action="store_true", help="No incloure segment_id al csv final.")

    return p.parse_args()


def resolve_output_csv(args: argparse.Namespace) -> str:
    """
    Resol la ruta final del csv de sortida.

    Prioritat:
    1) --output-csv, si l'usuari l'ha indicat.
    2) Si no, es crea automàticament dins de --output-dir.
    """
    if args.output_csv is not None and str(args.output_csv).strip() != "":
        return args.output_csv
    return os.path.join(args.output_dir, "CR3000_Tabla1_2011_2024_features_nowcasting.csv")


def decide_include_segment_id(args: argparse.Namespace) -> bool:
    """
    Resol la lògica d'inclusió de segment_id.

    Regles:
    - Default: True
    - Si --include-segment-id: True
    - Si --no-include-segment-id: False (té prioritat si ambdós es posen)
    """
    include_segment_id = True
    if args.include_segment_id:
        include_segment_id = True
    if args.no_include_segment_id:
        include_segment_id = False
    return include_segment_id


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()

    ensure_dir(args.output_dir)
    output_csv = resolve_output_csv(args)
    output_csv_dir = os.path.dirname(output_csv)
    if output_csv_dir:
        ensure_dir(output_csv_dir)

    timestamp_col = args.timestamp_col
    train_end = pd.Timestamp(args.train_end)
    corr_method = args.corr_method

    # 1) Lectura i ordenació temporal
    df = read_input_csv(args.input, timestamp_col)

    # 2) Verificació de targets
    validate_targets_exist(df, DEFAULT_TARGET_COLS)

    # 3) Selecció de features (només les que existeixen)
    selected_features = safe_select_features(df, DEFAULT_SELECTED_FEATURES_DESIRED)
    if not selected_features:
        raise RuntimeError("No s'ha trobat cap feature seleccionada al csv.")

    # 4) Decideix si incloure segment_id i construeix el dataframe final
    include_segment_id = decide_include_segment_id(args)
    df_out = build_training_dataframe(
        df=df,
        timestamp_col=timestamp_col,
        target_cols=DEFAULT_TARGET_COLS,
        selected_features=selected_features,
        include_segment_id=include_segment_id
    )

    # 5) Correlacions train-only (anti-leakage metodològic)
    df_corr, n_corr, n_all = subset_train(df_out, timestamp_col, train_end)
    print(f"\n[INFO] Correlacions en train-only: {timestamp_col} < {train_end.date()}  -> {n_corr} files (de {n_all})")

    if n_corr < 10000:
        print("[AVÍS] train sembla petit per una anàlisi robusta de correlacions. Revisa train_end si cal.")

    # 6) Matriu features-feature (train) + mapa de calor
    compute_and_save_feature_matrix(
        df_corr=df_corr,
        selected_features=selected_features,
        corr_method=corr_method,
        output_dir=args.output_dir
    )

    # 7) Correlacions features -> temperatura (train)
    print("\nCorrelacions features -> Temp_1_Avg (train)")
    compute_and_save_target_correlations(
        df_corr=df_corr,
        selected_features=selected_features,
        corr_method=corr_method,
        output_dir=args.output_dir,
        target_name="Temp_1_Avg",
        out_filename="correlacio_amb_Temp_1_Avg_TRAIN.csv"
    )

    # 8) Correlacions features -> pluja (3 vistes) (train)
    print("\nCorrelacions del subconjunt final -> Lluvia_Tot / rain_occ / rain_log1p (train)")
    for tgt_name in ["Lluvia_Tot", "rain_occ", "rain_log1p"]:
        compute_and_save_target_correlations(
            df_corr=df_corr,
            selected_features=selected_features,
            corr_method=corr_method,
            output_dir=args.output_dir,
            target_name=tgt_name,
            out_filename=f"correlacio_amb_{tgt_name}_TRAIN.csv"
        )

    # 9) Export del csv final per entrenament
    df_out.to_csv(output_csv, index=False)
    print(f"\n[OK] csv final: {output_csv}")
    print(f"Columnes ({len(df_out.columns)}):")
    for c in df_out.columns:
        print(f"  - {c}")


if __name__ == "__main__":
    main()