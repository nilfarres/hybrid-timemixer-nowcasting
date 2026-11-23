#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script inicial del pipeline del TFM.

Construeix el CSV net a partir dels fitxers anuals del datalogger
`CR3000_Tabla1_YYYY.dat` en format TOA5, mantenint un format de sortida
coherent amb la resta d'scripts del projecte:

- `Dataset_csv/CR3000_Tabla1_2011_2024_clean.csv`
- `Dataset_csv/CR3000_Tabla1_2011_2024.csv` (opcional)
- `Dataset_csv/csv_clean_report.json`

Què fa aquest script:
1) Localitza els fitxers d'entrada, tant si estan en un directori com en un ZIP.
2) Llegeix cada any, fins i tot si la capçalera no és exactament
   igual entre fitxers o si hi ha petites anomalies de format.
3) Normalitza noms de columna i el camp temporal `TIMESTAMP`.
4) Concatena tots els anys, audita l'índex temporal i detecta duplicats.
5) Elimina duplicats segons una política configurable i desa el CSV net.
6) Escriu un informe JSON amb un resum del procés.

Exemples d'ús:
----------------
python csv_clean.py --data-dir "private/logger"
python csv_clean.py --data-zip "private/logger.zip"
"""

from __future__ import annotations

import argparse
import io
import json
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


# ----------------------------- Configuració base -----------------------------


@dataclass(frozen=True)
class TOA5Config:
    """Configuració general de lectura dels fitxers TOA5 del CR3000."""

    timestamp_col: str = "TIMESTAMP"
    encoding: str = "latin-1"
    na_values: tuple[str, ...] = (
        "NAN",
        "NaN",
        "nan",
        "NA",
        "",
    )
    max_header_scan_lines: int = 40
    expected_sep_default: str = ","


@dataclass(frozen=True)
class InputSource:
    """Representa la font d'entrada: directori o ZIP."""

    kind: str  # "dir" o "zip"
    path: Path


# ----------------------------- Utilitats generals -----------------------------


def _log(msg: str) -> None:
    print(msg)


def _project_root_from_cwd() -> Path:
    """Assumeix que l'script s'executa des de l'arrel del projecte."""
    return Path.cwd().resolve()


def ensure_dataset_csv_dir(project_root: Path, explicit_output_dir: Optional[str] = None) -> Path:
    """Crea i retorna el directori canònic de sortida dels CSV del projecte."""
    if explicit_output_dir:
        out_dir = Path(explicit_output_dir).expanduser().resolve()
    else:
        out_dir = project_root / "Dataset_csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _format_dt(ts: pd.Timestamp | None) -> str | None:
    if ts is None or pd.isna(ts):
        return None
    return pd.Timestamp(ts).isoformat(sep=" ")


# ----------------------------- Detecció de l'entrada -----------------------------


def _build_probe_names(years: Iterable[int], template: str) -> list[str]:
    years_list = list(years)
    if not years_list:
        return []
    return [template.format(year=y) for y in years_list[: min(3, len(years_list))]]


def _find_matching_member(zf: zipfile.ZipFile, target_name: str) -> Optional[str]:
    """
    Retorna el membre del ZIP que acaba amb `target_name`.
    Això permet tolerar ZIPs amb subdirectoris interns.
    """
    candidates = [name for name in zf.namelist() if name.endswith(target_name)]
    if not candidates:
        return None
    if len(candidates) > 1:
        raise ValueError(f"Hi ha més d'un membre al ZIP que coincideix amb {target_name}: {candidates}")
    return candidates[0]


def resolve_input_source(
    project_root: Path,
    explicit_data_dir: Optional[str],
    explicit_data_zip: Optional[str],
    years: Iterable[int],
    template: str,
) -> InputSource:
    """
    Determina la font d'entrada dels fitxers `.dat`.

    Prioritat:
    1) `--data-dir`
    2) `--data-zip`
    3) Variables d'entorn `TFM_UB_DAT_DIR`, `UB_DAT_DIR`, `TFM_UB_DAT_ZIP`, `UB_DAT_ZIP`
    4) Cerca en ubicacions típiques dins del projecte
    5) Cerca recursiva dins del projecte
    """
    probe_names = _build_probe_names(years, template)

    if explicit_data_dir and explicit_data_zip:
        raise ValueError("Només es pot indicar una font d'entrada: --data-dir o --data-zip.")

    # 1) Directori explícit
    if explicit_data_dir:
        p = Path(explicit_data_dir).expanduser().resolve()
        if not p.exists() or not p.is_dir():
            raise FileNotFoundError(f"--data-dir no existeix o no és un directori: {p}")
        return InputSource(kind="dir", path=p)

    # 2) ZIP explícit
    if explicit_data_zip:
        p = Path(explicit_data_zip).expanduser().resolve()
        if not p.exists() or not p.is_file():
            raise FileNotFoundError(f"--data-zip no existeix o no és un fitxer: {p}")
        return InputSource(kind="zip", path=p)

    # 3) Variables d'entorn
    for env_key, kind in (
        ("TFM_UB_DAT_DIR", "dir"),
        ("UB_DAT_DIR", "dir"),
        ("TFM_UB_DAT_ZIP", "zip"),
        ("UB_DAT_ZIP", "zip"),
    ):
        v = os.environ.get(env_key, "").strip()
        if not v:
            continue
        p = Path(v).expanduser().resolve()
        if p.exists():
            _log(f"[INFO] Font d'entrada obtinguda de {env_key}: {p}")
            return InputSource(kind=kind, path=p)

    # 4) Ubicacions típiques dins del projecte
    candidates_dirs = [
        project_root / "DADES_UB",
        project_root / "Dades_UB",
        project_root / "data" / "DADES_UB",
        project_root / "data" / "dades_ub",
        project_root / "Dataset_dat",
        project_root / "Dataset_dat" / "DADES_UB",
        project_root / "Dataset_dat" / "ub",
    ]
    candidates_zips = [
        project_root / "CR3000_Tabla1.zip",
        project_root / "data" / "CR3000_Tabla1.zip",
        project_root / "Dataset_dat" / "CR3000_Tabla1.zip",
    ]

    for c in candidates_dirs:
        if c.exists() and c.is_dir() and all((c / name).exists() for name in probe_names):
            _log(f"[INFO] Directori dels .dat autodetectat: {c}")
            return InputSource(kind="dir", path=c)

    for c in candidates_zips:
        if c.exists() and c.is_file():
            _log(f"[INFO] ZIP dels .dat autodetectat: {c}")
            return InputSource(kind="zip", path=c)

    # 5) Cerca recursiva dins del projecte
    for p in project_root.rglob("*.dat"):
        if p.name in probe_names:
            _log(f"[INFO] Directori dels .dat autodetectat per cerca: {p.parent}")
            return InputSource(kind="dir", path=p.parent)

    for p in project_root.rglob("*.zip"):
        if p.name.lower() == "cr3000_tabla1.zip":
            _log(f"[INFO] ZIP dels .dat autodetectat per cerca: {p}")
            return InputSource(kind="zip", path=p)

    raise FileNotFoundError(
        "No s'ha pogut autodetectar la font dels fitxers .dat. "
        "Indica --data-dir o --data-zip."
    )


# ----------------------------- Lectura robusta TOA5 -----------------------------


def _read_text_from_source(source: InputSource, file_name: str, encoding: str) -> str:
    """Llegeix el text complet d'un fitxer anual des d'un directori o un ZIP."""
    if source.kind == "dir":
        fpath = source.path / file_name
        if not fpath.exists():
            raise FileNotFoundError(f"No s'ha trobat el fitxer {fpath}")
        return fpath.read_text(encoding=encoding, errors="replace")

    if source.kind == "zip":
        with zipfile.ZipFile(source.path, "r") as zf:
            member = _find_matching_member(zf, file_name)
            if member is None:
                raise FileNotFoundError(f"No s'ha trobat {file_name} dins del ZIP {source.path}")
            return zf.read(member).decode(encoding, errors="replace")

    raise ValueError(f"Tipus de font desconegut: {source.kind}")


def _detect_header_and_sep(lines: list[str], timestamp_col: str, default_sep: str = ",") -> tuple[Optional[int], str]:
    """
    Detecta la línia de capçalera real i el separador probable.

    Retorna:
    - `header_line`: índex 0-based de la línia de capçalera o `None` si no es detecta.
    - `sep`: `,` o `;`.
    """
    for i, line in enumerate(lines):
        if timestamp_col in line:
            sep = ";" if line.count(";") > line.count(",") else ","
            return i, sep
    return None, default_sep


def _normalize_columns(columns: Iterable[object]) -> list[str]:
    """Normalitza els noms de columna."""
    out: list[str] = []
    for c in columns:
        name = str(c)
        name = name.lstrip("\ufeff").strip().strip('"').strip()
        out.append(name)
    return out


def _parse_timestamp_series(ts_raw: pd.Series) -> pd.Series:
    """
    Converteix una sèrie textual de timestamps a `datetime64[ns]`.

    L'estratègia és robusta a petites irregularitats de format.
    """
    s = ts_raw.astype(str).str.strip().str.strip('"').str.replace("\x00", "", regex=False)

    ts = pd.to_datetime(s, format="%Y-%m-%d %H:%M:%S", errors="coerce")

    mask = ts.isna()
    if mask.any():
        ts_frac = pd.to_datetime(s[mask], format="%Y-%m-%d %H:%M:%S.%f", errors="coerce")
        ts.loc[mask] = ts_frac

    mask = ts.isna()
    if mask.any():
        ts_fallback = pd.to_datetime(s[mask], errors="coerce", utc=True)
        try:
            ts_fallback = ts_fallback.dt.tz_convert(None)
        except Exception:
            pass
        ts.loc[mask] = ts_fallback

    return ts


def _coerce_object_columns_to_numeric(df: pd.DataFrame, exclude: Iterable[str]) -> pd.DataFrame:
    """
    Intenta convertir a numèric les columnes de tipus objecte que no són de text real.

    En aquests fitxers, excepte `TIMESTAMP`, totes les columnes haurien de ser numèriques.
    La funció és conservadora: només substitueix la columna si la conversió és clarament útil.
    """
    exclude_set = set(exclude)
    for col in df.columns:
        if col in exclude_set:
            continue
        if df[col].dtype != object:
            continue

        raw = df[col].astype(str).str.strip().str.strip('"')
        conv = pd.to_numeric(raw, errors="coerce")

        non_empty = raw.ne("") & raw.ne("nan") & raw.ne("NaN")
        if non_empty.sum() == 0:
            df[col] = conv
            continue

        frac_ok = float(conv.notna()[non_empty].mean())
        if frac_ok >= 0.90:
            df[col] = conv

    return df


def read_toa5_year(
    source: InputSource,
    file_name: str,
    cfg: TOA5Config,
) -> pd.DataFrame:
    """
    Llegeix robustament un fitxer anual TOA5 i retorna un dataframe indexat per `TIMESTAMP`.

    Aquesta funció tolera:
    - capçaleres TOA5 amb longitud variable,
    - absència de la primera línia típica `TOA5`,
    - petites anomalies de cometes o separador,
    - i algunes irregularitats puntuals al camp temporal.
    """
    text = _read_text_from_source(source, file_name, cfg.encoding)
    lines = text.splitlines()
    scan_lines = lines[: cfg.max_header_scan_lines]
    header_line, sep = _detect_header_and_sep(scan_lines, cfg.timestamp_col, cfg.expected_sep_default)

    if header_line is None:
        raise ValueError(
            f"No s'ha pogut detectar la capçalera real del fitxer {file_name}. "
            f"No s'ha trobat la columna {cfg.timestamp_col}."
        )

    csv_text = "\n".join(lines[header_line:])
    df = pd.read_csv(
        io.StringIO(csv_text),
        sep=sep,
        na_values=list(cfg.na_values),
        keep_default_na=True,
        engine="python",
        dtype=str,
        on_bad_lines="skip",
    )

    df.columns = _normalize_columns(df.columns)
    if cfg.timestamp_col not in df.columns:
        raise ValueError(
            f"Després de llegir {file_name}, no s'ha trobat la columna {cfg.timestamp_col}. "
            f"Columnes llegides: {list(df.columns)}"
        )

    ts = _parse_timestamp_series(df[cfg.timestamp_col])
    df = df.loc[ts.notna()].copy()
    ts = ts.loc[ts.notna()].copy()
    df[cfg.timestamp_col] = ts.values

    df = _coerce_object_columns_to_numeric(df, exclude=[cfg.timestamp_col])
    df = df.set_index(cfg.timestamp_col).sort_index()
    return df


# ----------------------------- Concatenació anual -----------------------------


def _columns_signature(df: pd.DataFrame) -> tuple[str, ...]:
    return tuple(map(str, df.columns.tolist()))


def load_years_concat(
    source: InputSource,
    years: Iterable[int],
    template: str,
    cfg: TOA5Config,
    strict_columns: bool = False,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    """
    Llegeix i concatena tots els anys indicats.

    Si `strict_columns=True`, falla quan detecta discrepàncies d'esquema entre anys.
    Si `strict_columns=False`, fa una unió per noms de columna.
    """
    frames: list[pd.DataFrame] = []
    per_year_summary: list[dict[str, object]] = []
    ref_signature: Optional[tuple[str, ...]] = None

    for year in years:
        fname = template.format(year=year)
        _log(f"\n[INFO] Llegint {fname}...")
        df_y = read_toa5_year(source=source, file_name=fname, cfg=cfg)

        sig = _columns_signature(df_y)
        if ref_signature is None:
            ref_signature = sig
        else:
            if sig != ref_signature:
                missing_vs_ref = [c for c in ref_signature if c not in sig]
                extra_vs_ref = [c for c in sig if c not in ref_signature]
                msg = (
                    f"Discrepància de columnes a {fname}. "
                    f"Falten respecte la referència: {missing_vs_ref} | "
                    f"Extra: {extra_vs_ref}"
                )
                if strict_columns:
                    raise ValueError(msg)
                _log(f"[WARN] {msg}")

        summary = {
            "year": int(year),
            "file_name": fname,
            "n_rows": int(df_y.shape[0]),
            "n_cols": int(df_y.shape[1]),
            "first_timestamp": _format_dt(df_y.index.min()),
            "last_timestamp": _format_dt(df_y.index.max()),
            "columns": list(map(str, df_y.columns)),
        }
        per_year_summary.append(summary)

        _log(
            f"[INFO] {fname}: rows={df_y.shape[0]} | cols={df_y.shape[1]} | "
            f"primer={df_y.index.min()} | últim={df_y.index.max()}"
        )
        frames.append(df_y)

    if not frames:
        raise ValueError("No s'ha pogut llegir cap fitxer anual.")

    df_all = pd.concat(frames, axis=0, join="outer", sort=False)
    df_all = df_all.sort_index()
    return df_all, per_year_summary


# ----------------------------- Auditoria temporal -----------------------------


def audit_time_index(df: pd.DataFrame, label: str) -> dict[str, object]:
    """Calcula un resum de qualitat de l'índex temporal del dataframe."""
    idx = pd.DatetimeIndex(df.index)

    summary = {
        "label": label,
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "first_timestamp": _format_dt(idx.min()),
        "last_timestamp": _format_dt(idx.max()),
        "is_monotonic_increasing": bool(idx.is_monotonic_increasing),
        "is_unique": bool(idx.is_unique),
        "n_duplicated_timestamps": int(idx.duplicated(keep=False).sum()),
        "top_time_deltas": {},
    }

    if len(idx) >= 2:
        deltas = pd.Series(idx[1:] - idx[:-1])
        top = deltas.value_counts(dropna=False).head(10)
        summary["top_time_deltas"] = {str(k): int(v) for k, v in top.items()}

    _log(f"\n[INFO] Auditoria temporal ({label}):")
    _log(f"- Registres:            {summary['n_rows']}")
    _log(f"- Variables:            {summary['n_cols']}")
    _log(f"- Primer timestamp:     {summary['first_timestamp']}")
    _log(f"- Últim timestamp:      {summary['last_timestamp']}")
    _log(f"- Índex únic?:         {summary['is_unique']}")
    _log(f"- Índex ordenat?:      {summary['is_monotonic_increasing']}")
    _log(f"- Duplicats (n):       {summary['n_duplicated_timestamps']}")

    if summary["top_time_deltas"]:
        _log("\n[INFO] Principals intervals entre registres:")
        for delta_str, count in summary["top_time_deltas"].items():
            _log(f"- {delta_str}: {count}")

    return summary


def duplicated_rows_identical(grp: pd.DataFrame) -> bool:
    """Retorna `True` si totes les files d'un mateix timestamp són idèntiques."""
    nunics = grp.nunique(dropna=False)
    return bool((nunics <= 1).all())


def _diff_columns_in_duplicate_group(grp: pd.DataFrame) -> list[str]:
    """Llista les columnes que difereixen dins d'un grup duplicat."""
    return [col for col in grp.columns if grp[col].nunique(dropna=False) > 1]


def analyze_and_drop_duplicates(
    df: pd.DataFrame,
    keep: str = "first",
    examples: int = 0,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """
    Analitza els timestamps duplicats i elimina duplicats segons la política indicada.

    Paràmetres:
    - `keep='first'`: conserva la primera fila de cada timestamp duplicat.
    - `keep='last'`: conserva l'última fila.
    """
    if keep not in {"first", "last"}:
        raise ValueError("`keep` ha de ser 'first' o 'last'.")

    mask_dups = df.index.duplicated(keep=False)
    if not bool(mask_dups.any()):
        _log("\n[INFO] No s'han detectat timestamps duplicats.")
        info = {
            "n_rows_involved": 0,
            "n_duplicated_timestamps": 0,
            "n_identical_duplicate_groups": 0,
            "n_conflicting_duplicate_groups": 0,
            "conflicting_examples": [],
            "keep_policy": keep,
        }
        return df, info

    df_dups = df.loc[mask_dups].copy()
    same_flags = df_dups.groupby(level=0).apply(duplicated_rows_identical)
    conflicting = same_flags[~same_flags]

    _log(
        f"\n[INFO] Duplicats detectats: files implicades = {df_dups.shape[0]} | "
        f"timestamps afectats = {df_dups.index.nunique()}"
    )
    _log("[INFO] Resum de grups duplicats:")
    _log(f"- Grups idèntics:      {int(same_flags.sum())}")
    _log(f"- Grups conflictius:   {int((~same_flags).sum())}")

    conflicting_examples: list[dict[str, object]] = []
    if not conflicting.empty:
        _log(f"[WARN] Hi ha {int(conflicting.shape[0])} timestamps duplicats amb discrepàncies reals.")
        for ts in conflicting.index[: max(0, examples)]:
            grp = df_dups.loc[[ts]] if isinstance(ts, pd.Timestamp) else df_dups.loc[ts]
            if isinstance(grp, pd.Series):
                grp = grp.to_frame().T
            conflicting_examples.append(
                {
                    "timestamp": _format_dt(pd.Timestamp(ts)),
                    "n_rows": int(grp.shape[0]),
                    "diff_columns": _diff_columns_in_duplicate_group(grp),
                }
            )

        if examples > 0:
            _log("\n[INFO] Exemples de duplicats conflictius:")
            for item in conflicting_examples:
                _log(
                    f"- {item['timestamp']} | files={item['n_rows']} | "
                    f"columnes que difereixen={item['diff_columns']}"
                )

    df_clean = df.loc[~df.index.duplicated(keep=keep)].copy()
    _log("\n[INFO] Després de netejar duplicats:")
    _log(f"- Política aplicada:   keep='{keep}'")
    _log(f"- Registres:           {df_clean.shape[0]}")
    _log(f"- Índex únic?:         {bool(df_clean.index.is_unique)}")

    info = {
        "n_rows_involved": int(df_dups.shape[0]),
        "n_duplicated_timestamps": int(df_dups.index.nunique()),
        "n_identical_duplicate_groups": int(same_flags.sum()),
        "n_conflicting_duplicate_groups": int((~same_flags).sum()),
        "conflicting_examples": conflicting_examples,
        "keep_policy": keep,
    }
    return df_clean, info


# ----------------------------- CLI -----------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Genera el CSV net del període 2011-2024 a partir dels fitxers TOA5 anuals.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", type=str, default=None, help="Directori que conté els fitxers .dat anuals.")
    parser.add_argument("--data-zip", type=str, default=None, help="ZIP que conté els fitxers .dat anuals.")
    parser.add_argument("--output-dir", type=str, default=None, help="Directori on es desaran els CSV de sortida.")
    parser.add_argument("--start-year", type=int, default=2011, help="Any inicial inclòs.")
    parser.add_argument("--end-year", type=int, default=2024, help="Any final inclòs.")
    parser.add_argument(
        "--file-template",
        type=str,
        default="CR3000_Tabla1_{year}.dat",
        help="Plantilla del nom del fitxer anual.",
    )
    parser.add_argument(
        "--end-exclusive",
        type=str,
        default=None,
        help=(
            "Timestamp exclusiu superior del dataset final. "
            "Si no s'indica, s'utilitza automàticament l'1 de gener de l'any següent a end-year."
        ),
    )
    parser.add_argument(
        "--strict-columns",
        action="store_true",
        help="Falla si detecta discrepàncies de columnes entre anys.",
    )
    parser.add_argument(
        "--duplicate-policy",
        choices=["first", "last"],
        default="first",
        help="Política aplicada en eliminar timestamps duplicats.",
    )
    parser.add_argument(
        "--examples-dups",
        type=int,
        default=5,
        help="Nombre màxim d'exemples de duplicats conflictius a mostrar i guardar a l'informe.",
    )
    parser.add_argument(
        "--write-raw",
        action="store_true",
        help="Si s'activa, desa també el CSV concatenat abans d'eliminar duplicats.",
    )
    parser.add_argument(
        "--clean-name",
        type=str,
        default="CR3000_Tabla1_2011_2024_clean.csv",
        help="Nom del CSV net de sortida.",
    )
    parser.add_argument(
        "--raw-name",
        type=str,
        default="CR3000_Tabla1_2011_2024.csv",
        help="Nom del CSV raw concatenat, si `--write-raw` està activat.",
    )
    parser.add_argument(
        "--report-name",
        type=str,
        default="csv_clean_report.json",
        help="Nom del fitxer JSON d'informe.",
    )
    return parser


# ----------------------------- Main -----------------------------


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.start_year > args.end_year:
        raise ValueError("`start-year` no pot ser superior a `end-year`.")

    project_root = _project_root_from_cwd()
    output_dir = ensure_dataset_csv_dir(project_root, args.output_dir)
    years = range(args.start_year, args.end_year + 1)
    cfg = TOA5Config()

    source = resolve_input_source(
        project_root=project_root,
        explicit_data_dir=args.data_dir,
        explicit_data_zip=args.data_zip,
        years=years,
        template=args.file_template,
    )

    if args.end_exclusive is None:
        end_exclusive = pd.Timestamp(year=args.end_year + 1, month=1, day=1)
    else:
        end_exclusive = pd.Timestamp(args.end_exclusive)

    _log(f"[INFO] Font d'entrada: {source.kind} -> {source.path}")
    _log(f"[INFO] Directori de sortida: {output_dir}")
    _log(f"[INFO] Finestra final: [{args.start_year}-01-01, {end_exclusive})")

    # 1) Llegeix i concatena tots els anys.
    df_all, per_year_summary = load_years_concat(
        source=source,
        years=years,
        template=args.file_template,
        cfg=cfg,
        strict_columns=bool(args.strict_columns),
    )

    # 2) Retalla el dataset final amb el límit superior exclusiu.
    df_all = df_all.loc[df_all.index < end_exclusive].copy()
    _log(
        f"\n[INFO] Conjunt concatenat dins del rang final: shape={df_all.shape} | "
        f"primer={df_all.index.min()} | últim={df_all.index.max()}"
    )

    # 3) Audita el dataframe raw.
    raw_audit = audit_time_index(df_all, label="raw")

    # 4) Desa, opcionalment, el raw.
    if bool(args.write_raw):
        raw_path = output_dir / args.raw_name
        df_all.to_csv(raw_path, index_label=cfg.timestamp_col, encoding="utf-8")
        _log(f"[OK] CSV raw desat a: {raw_path}")
    else:
        raw_path = None

    # 5) Analitza i elimina duplicats.
    df_clean, dup_info = analyze_and_drop_duplicates(
        df_all,
        keep=args.duplicate_policy,
        examples=int(args.examples_dups),
    )

    # 6) Audita el dataframe net.
    clean_audit = audit_time_index(df_clean, label="clean")

    # 7) Desa el CSV net exactament amb el format esperat pel següent script.
    clean_path = output_dir / args.clean_name
    df_clean.to_csv(clean_path, index_label=cfg.timestamp_col, encoding="utf-8")
    _log(f"\n[OK] CSV net desat a: {clean_path}")

    # 8) Informe JSON de traçabilitat.
    report = {
        "input_source": {"kind": source.kind, "path": str(source.path)},
        "years_requested": list(years),
        "file_template": args.file_template,
        "end_exclusive": _format_dt(end_exclusive),
        "per_year_summary": per_year_summary,
        "raw_audit": raw_audit,
        "duplicates": dup_info,
        "clean_audit": clean_audit,
        "outputs": {
            "clean_csv": str(clean_path),
            "raw_csv": str(raw_path) if raw_path is not None else None,
        },
    }
    report_path = output_dir / args.report_name
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    _log(f"[OK] Informe JSON desat a: {report_path}")

    _log(
        "\n[INFO] Següent pas recomanat: executa `imputa_NaN.py` sobre "
        f"{clean_path.name}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())