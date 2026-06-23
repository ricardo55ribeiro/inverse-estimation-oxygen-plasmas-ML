from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, ScalarFormatter


# ======================================================================================
# GLOBAL SETTINGS
# ======================================================================================

SCHEME = "O2_novib"
OUTPUT_SCHEME = "O2_novib_noisy"
NOISY_TRAINING_EXPERIMENT_NAME = "NoisyTraining_PIC1_Robustness"
ARCHITECTURE = "30, 30, 30"

BASE_RESULTS_DIR = Path("Results_NN")

MEAN_COL = "test_mse_scaled_mean"
STD_COL = "test_mse_scaled_std"
RAW_METRIC_COL = "test_mse_scaled"

SAVE_PDF = True
SAVE_PNG = True
SAVE_DATA_CSV = True
SAVE_MANIFEST = True
DPI = 300

NOISE_LEVELS_PERCENT = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]

# O2(X) / O2(a) / O2(b) / O2(Hz) / O2+(X) / O(3P) 
# O(1D) / O+(gnd) / O-(gnd) / O3(X) / O3(exc)
REPRESENTATIVE_COMBINATIONS = [
    ["O2(a)", "O2(b)", "O2(Hz)"],
    ["O2(a)", "O3(X)"],
    ["O2(a)", "O2(b)", "O3(X)"],
    ["O2(a)", "O(3P)", "O3(X)"],
    ["O2(a)", "O2(b)"],
    ["O2(X)", "O2(b)", "O(3P)"],
    ["O2(X)", "O2(a)", "O(3P)"],
]

COMBO_COLORS = {
    ("O2(a)", "O2(b)", "O2(Hz)"): "#0072B2",
    ("O2(a)", "O3(X)"): "#E69F00",
    ("O2(a)", "O2(b)", "O3(X)"): "#009E73",
    ("O2(a)", "O(3P)", "O3(X)"): "#CC79A7",
    ("O2(a)", "O2(b)"): "#D55E00",
    ("O2(X)", "O2(b)", "O(3P)"): "#56B4E9",
    ("O2(X)", "O2(a)", "O(3P)"): "#000000",
}

COMBO_MARKERS = {
    ("O2(a)", "O2(b)", "O2(Hz)"): "o",
    ("O2(a)", "O3(X)"): "s",
    ("O2(a)", "O2(b)", "O3(X)"): "^",
    ("O2(a)", "O(3P)", "O3(X)"): "v",
    ("O2(a)", "O2(b)"): "D",
    ("O2(X)", "O2(b)", "O(3P)"): "P",
    ("O2(X)", "O2(a)", "O(3P)"): "X",
}

INNER_SUBSET_ORDER = [
    ["O2(a)"],
    ["O2(b)"],
    ["O2(Hz)"],
    ["O2(a)", "O2(b)"],
    ["O2(a)", "O2(Hz)"],
    ["O2(b)", "O2(Hz)"],
    ["O2(a)", "O2(b)", "O2(Hz)"],
]

SUBSET_COLORS = {
    ("O2(a)",): "#0072B2",
    ("O2(b)",): "#E69F00",
    ("O2(Hz)",): "#009E73",
    ("O2(a)", "O2(b)"): "#CC79A7",
    ("O2(a)", "O2(Hz)"): "#D55E00",
    ("O2(b)", "O2(Hz)"): "#56B4E9",
    ("O2(a)", "O2(b)", "O2(Hz)"): "#000000",
}

SUBSET_MARKERS = {
    ("O2(a)",): "o",
    ("O2(b)",): "s",
    ("O2(Hz)",): "^",
    ("O2(a)", "O2(b)"): "v",
    ("O2(a)", "O2(Hz)"): "D",
    ("O2(b)", "O2(Hz)"): "P",
    ("O2(a)", "O2(b)", "O2(Hz)"): "X",
}

# Shared paper-figure style settings for the three representative two-panel plots.
FIGSIZE_TWO_PANEL = (10.7, 4.05)
SUBPLOTS_ADJUST_TWO_PANEL = {
    "left": 0.075,
    "right": 0.985,
    "top": 0.965,
    "bottom": 0.250,
    "wspace": 0.18,
}
LINEWIDTH = 1.6
MARKERSIZE = 4.2
CAPSIZE = 2.2
ERRORBAR_LINEWIDTH = 1.0
GRID_ALPHA = 0.30
LEGEND_FONTSIZE = 8


# ======================================================================================
# GENERAL HELPERS
# ======================================================================================


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)


def normalize_architecture_text(value) -> str:
    text = str(value).strip()
    text = text.strip("[]()")
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if parts and all(re.fullmatch(r"\d+", p) for p in parts):
        return ", ".join(parts)
    return str(value).strip()


def split_species_text(text) -> tuple[str, ...]:
    """Parse CSV species strings without splitting O2+(X)."""
    if pd.isna(text):
        return tuple()
    s = str(text).strip()
    if not s or s.lower() in {"clean_baseline", "clean baseline"}:
        return tuple()
    s = s.replace("[", "").replace("]", "").replace("'", "").replace('"', "")
    parts = re.split(r"\s*,\s*|\s+\+\s+", s)
    return tuple(part.strip() for part in parts if part.strip())


def species_key(species) -> tuple[str, ...]:
    """Order-insensitive key for matching species combinations in CSV rows."""
    if isinstance(species, (list, tuple)):
        values = tuple(str(x).strip() for x in species if str(x).strip())
    else:
        values = split_species_text(species)
    return tuple(sorted(values))


def ordered_species_tuple(species) -> tuple[str, ...]:
    lookup = {species_key(combo): tuple(combo) for combo in REPRESENTATIVE_COMBINATIONS}
    key = species_key(species)
    if key in lookup:
        return lookup[key]
    if isinstance(species, (list, tuple)):
        return tuple(str(x).strip() for x in species if str(x).strip())
    return split_species_text(species)


def species_label(species, joiner: str = ", ") -> str:
    return joiner.join(ordered_species_tuple(species))


def subset_label(species) -> str:
    if isinstance(species, (list, tuple)):
        return " + ".join(str(x).strip() for x in species if str(x).strip())
    return " + ".join(split_species_text(species))


def ordered_subset_key(species) -> tuple[str, ...]:
    lookup = {species_key(combo): tuple(combo) for combo in INNER_SUBSET_ORDER}
    key = species_key(species)
    if key in lookup:
        return lookup[key]
    if isinstance(species, (list, tuple)):
        return tuple(str(x).strip() for x in species if str(x).strip())
    return split_species_text(species)


def format_noise(value) -> str:
    return f"{float(value):g}"


def noise_is_requested(series: pd.Series) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(float)
    mask = np.zeros(values.shape, dtype=bool)
    for requested in NOISE_LEVELS_PERCENT:
        mask |= np.isclose(values, float(requested), rtol=0.0, atol=1e-12)
    return mask


def find_first_existing(candidates: Iterable[Path], label: str) -> Path:
    for path in candidates:
        if path.exists():
            return path
    searched = "\n".join(f"  - {p}" for p in candidates)
    raise FileNotFoundError(f"Could not find {label}. Searched:\n{searched}")


def ensure_mean_std_columns(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Ensure a results table has test_mse_scaled_mean/std columns."""
    df = df.copy()
    required_base = {"hidden_size", "kept_species", "noise_percent"}
    missing = [col for col in required_base if col not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")

    df["hidden_size"] = df["hidden_size"].apply(normalize_architecture_text)
    df["noise_percent"] = pd.to_numeric(df["noise_percent"], errors="coerce")

    if MEAN_COL in df.columns:
        df[MEAN_COL] = pd.to_numeric(df[MEAN_COL], errors="coerce")
        if STD_COL in df.columns:
            df[STD_COL] = pd.to_numeric(df[STD_COL], errors="coerce")
        else:
            df[STD_COL] = np.nan
        return df

    if RAW_METRIC_COL not in df.columns:
        raise ValueError(f"{path} contains neither {MEAN_COL!r} nor {RAW_METRIC_COL!r}.")

    df[RAW_METRIC_COL] = pd.to_numeric(df[RAW_METRIC_COL], errors="coerce")
    group_cols = ["hidden_size", "kept_species", "noise_percent"]
    optional_first_cols = ["species_config_name", "num_species_kept", "architecture"]
    agg_dict = {RAW_METRIC_COL: ["mean", "std"]}
    for col in optional_first_cols:
        if col in df.columns:
            agg_dict[col] = "first"

    out = df.groupby(group_cols, as_index=False).agg(agg_dict)
    out.columns = [
        col if isinstance(col, str) else "_".join([c for c in col if c])
        for col in out.columns.to_flat_index()
    ]
    out = out.rename(
        columns={
            f"{RAW_METRIC_COL}_mean": MEAN_COL,
            f"{RAW_METRIC_COL}_std": STD_COL,
        }
    )
    return out


def load_representative_noise_df(path: Path, source_name: str, architecture: str = ARCHITECTURE) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_csv(path)
    df = ensure_mean_std_columns(df, path)
    df[MEAN_COL] = pd.to_numeric(df[MEAN_COL], errors="coerce")
    df[STD_COL] = pd.to_numeric(df[STD_COL], errors="coerce")
    df["noise_percent"] = pd.to_numeric(df["noise_percent"], errors="coerce")

    df = df[
        (df["hidden_size"].astype(str) == architecture)
        & noise_is_requested(df["noise_percent"])
    ].dropna(subset=["noise_percent", MEAN_COL]).copy()

    if df.empty:
        raise ValueError(f"No rows found in {path} for architecture {architecture!r}.")

    df["species_key"] = df["kept_species"].apply(species_key)
    wanted_keys = {species_key(combo) for combo in REPRESENTATIVE_COMBINATIONS}
    df = df[df["species_key"].isin(wanted_keys)].copy()
    if df.empty:
        raise ValueError(f"No representative combinations found in {path}.")

    if df.duplicated(["species_key", "noise_percent"]).any():
        print(f"WARNING: duplicate rows found in {path}; averaging duplicates.")
        df = (
            df.groupby(["species_key", "noise_percent"], as_index=False)
            .agg(
                kept_species=("kept_species", "first"),
                hidden_size=("hidden_size", "first"),
                **{MEAN_COL: (MEAN_COL, "mean"), STD_COL: (STD_COL, "mean")},
            )
        )

    order = {species_key(combo): idx for idx, combo in enumerate(REPRESENTATIVE_COMBINATIONS)}
    labels = {species_key(combo): species_label(combo) for combo in REPRESENTATIVE_COMBINATIONS}
    df["combination_order"] = df["species_key"].apply(lambda key: order[key])
    df["species_label"] = df["species_key"].apply(lambda key: labels[key])
    df["source"] = source_name

    return df.sort_values(["combination_order", "noise_percent"]).reset_index(drop=True)


def yerr_for_log_axis(y, yerr, cap_fraction: float = 0.95):
    y = np.asarray(y, dtype=float)
    yerr = np.asarray(yerr, dtype=float)
    yerr = np.where(np.isnan(yerr), 0.0, yerr)
    yerr = np.where(yerr < 0.0, 0.0, yerr)
    lower = np.minimum(yerr, np.maximum(0.0, y * cap_fraction))
    upper = yerr
    return np.vstack([lower, upper])


def save_standard_figure(fig, output_dir: Path, basename: str) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    if SAVE_PDF:
        paths["pdf"] = output_dir / f"{basename}.pdf"
        fig.savefig(paths["pdf"], bbox_inches="tight")
    if SAVE_PNG:
        paths["png"] = output_dir / f"{basename}.png"
        fig.savefig(paths["png"], dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return paths


def combo_legend_handles() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            color=COMBO_COLORS[tuple(combo)],
            marker=COMBO_MARKERS[tuple(combo)],
            linestyle="-",
            linewidth=LINEWIDTH,
            markersize=MARKERSIZE,
            label=species_label(combo),
        )
        for combo in REPRESENTATIVE_COMBINATIONS
    ]


def plot_combo_lines(
    ax,
    df: pd.DataFrame,
    y_col: str,
    yerr_col: str,
    *,
    errorbars: bool,
    log_y: bool,
    ylabel: str,
) -> None:
    for combo in REPRESENTATIVE_COMBINATIONS:
        key = species_key(combo)
        group = df[df["species_key"] == key].sort_values("noise_percent")
        if group.empty:
            continue

        combo_tuple = tuple(combo)
        x = group["noise_percent"].to_numpy(float)
        y = group[y_col].to_numpy(float)
        if yerr_col in group.columns:
            yerr = group[yerr_col].fillna(0.0).to_numpy(float)
        else:
            yerr = np.zeros_like(y)

        if errorbars:
            ax.errorbar(
                x,
                y,
                yerr=yerr_for_log_axis(y, yerr) if log_y else yerr,
                color=COMBO_COLORS[combo_tuple],
                marker=COMBO_MARKERS[combo_tuple],
                linestyle="-",
                linewidth=LINEWIDTH,
                markersize=MARKERSIZE,
                capsize=CAPSIZE,
                elinewidth=ERRORBAR_LINEWIDTH,
            )
        else:
            ax.plot(
                x,
                y,
                color=COMBO_COLORS[combo_tuple],
                marker=COMBO_MARKERS[combo_tuple],
                linestyle="-",
                linewidth=LINEWIDTH,
                markersize=MARKERSIZE,
            )

    xticks = sorted(df["noise_percent"].dropna().unique())
    ax.set_xticks(xticks)
    ax.set_xticklabels([format_noise(x) for x in xticks])
    ax.set_xlabel("Noise level (%)")
    ax.set_ylabel(ylabel, labelpad=2)
    if log_y:
        yvals = pd.to_numeric(df[y_col], errors="coerce").dropna().to_numpy(float)
        if yvals.size == 0 or np.all(yvals > 0.0):
            ax.set_yscale("log")
        else:
            print(f"WARNING: non-positive values found for {y_col}; using linear y-axis.")
    ax.grid(True, which="both", alpha=GRID_ALPHA)


# ======================================================================================
# FIGURE 1: ERROR BY NUMBER OF SPECIES
# ======================================================================================

SPECIES_MIN = 2
ZOOM_MIN = 3
FULLRUN_PARENT = BASE_RESULTS_DIR / SCHEME / "FullRun_2to11species"
ANALYSIS_DIR = FULLRUN_PARENT / "Comparative_Analysis"
BASELINE_PREFIX = "11__"
EXPERIMENT_REGEX = re.compile(r"^\d+__")
IGNORED_DIR_NAMES = {"Comparative_Analysis", "Comparative_Plots", "Plots"}
SPECIES_COUNT_ARCHITECTURES = ["30, 30", "30, 30, 30", "50, 50"]
ARCH_COLORS = {"30, 30": "blue", "30, 30, 30": "green", "50, 50": "red"}


def parse_hidden_size(value) -> tuple[int, ...]:
    if isinstance(value, tuple):
        return tuple(int(x) for x in value)
    if isinstance(value, list):
        return tuple(int(x) for x in value)
    if pd.isna(value):
        return tuple()

    text = str(value).strip()
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple)):
            return tuple(int(x) for x in parsed)
    except Exception:
        pass

    text = text.strip("[]()")
    parts = [p.strip() for p in text.split(",") if p.strip()]
    return tuple(int(p) for p in parts)


def hidden_size_to_label(hidden_size_tuple) -> str:
    return ", ".join(map(str, hidden_size_tuple))


def hidden_size_sort_key(hidden_size_tuple):
    if not hidden_size_tuple:
        return (999, 999, ())
    return (len(hidden_size_tuple), hidden_size_tuple[0], hidden_size_tuple)


def is_valid_experiment_dir(path: Path) -> bool:
    return path.is_dir() and bool(EXPERIMENT_REGEX.match(path.name)) and path.name not in IGNORED_DIR_NAMES


def get_color_for_architecture(label: str) -> str:
    return ARCH_COLORS.get(label, "black")


def summarize_summary_df(df: pd.DataFrame, experiment_folder: str, run_timestamp: str) -> pd.DataFrame:
    if "hidden_size" not in df.columns:
        raise RuntimeError(f"summary.csv in {experiment_folder} does not contain 'hidden_size'.")

    numeric_candidates = [
        "input_size",
        "num_species_kept",
        "num_pressure_conditions",
        "test_mse",
        "test_rmse",
        "test_mse_unscaled",
        "test_rmse_unscaled",
        "training_time_s",
        "epochs_ran",
        "best_val_loss",
        "mean_rel_error_k1",
        "mean_rel_error_k2",
        "mean_rel_error_k3",
        "max_rel_error_k1",
        "max_rel_error_k2",
        "max_rel_error_k3",
    ]
    for col in numeric_candidates:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["hidden_size_tuple"] = df["hidden_size"].apply(parse_hidden_size)
    df["hidden_size_label"] = df["hidden_size_tuple"].apply(hidden_size_to_label)

    if "num_species_kept" not in df.columns or df["num_species_kept"].isna().all():
        if "input_size" in df.columns:
            n_pressures = 2
            if "num_pressure_conditions" in df.columns:
                vals = df["num_pressure_conditions"].dropna().unique()
                if len(vals) == 1:
                    n_pressures = int(vals[0])
            df["num_species_kept"] = df["input_size"] / n_pressures
        else:
            raise RuntimeError(f"Could not determine num_species_kept for {experiment_folder}.")

    df["num_species_kept"] = pd.to_numeric(df["num_species_kept"], errors="coerce").round().astype("Int64")
    df["experiment_folder"] = experiment_folder
    df["run_timestamp"] = run_timestamp

    group_cols = ["experiment_folder", "run_timestamp", "hidden_size_label", "hidden_size_tuple", "num_species_kept"]
    metric_cols = [
        "test_mse",
        "test_rmse",
        "test_mse_unscaled",
        "test_rmse_unscaled",
        "training_time_s",
        "epochs_ran",
        "best_val_loss",
        "mean_rel_error_k1",
        "mean_rel_error_k2",
        "mean_rel_error_k3",
        "max_rel_error_k1",
        "max_rel_error_k2",
        "max_rel_error_k3",
    ]
    metric_cols = [c for c in metric_cols if c in df.columns]

    agg_dict = {col: ["mean", "std"] for col in metric_cols}
    grouped = df.groupby(group_cols, as_index=False).agg(agg_dict)
    grouped.columns = [
        col if isinstance(col, str) else "_".join([c for c in col if c])
        for col in grouped.columns.to_flat_index()
    ]
    grouped = grouped.rename(columns={f"{col}_mean": col for col in metric_cols})
    grouped["source_mode"] = "summary_aggregated" if len(df) > len(grouped) else "summary_single"
    return grouped


def summarize_seed_aggregate_df(df: pd.DataFrame, experiment_folder: str, run_timestamp: str) -> pd.DataFrame:
    if "hidden_size" in df.columns:
        hidden_size_col = "hidden_size"
    elif "hidden_size_str" in df.columns:
        hidden_size_col = "hidden_size_str"
    else:
        raise RuntimeError(
            f"seed_aggregate_summary.csv does not contain 'hidden_size' or 'hidden_size_str'. "
            f"Columns found: {list(df.columns)}"
        )

    numeric_cols = [col for col in df.columns if col not in {"scheme", "experiment_name", "hidden_size", "hidden_size_str"}]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["hidden_size_tuple"] = df[hidden_size_col].apply(parse_hidden_size)
    df["hidden_size_label"] = df["hidden_size_tuple"].apply(hidden_size_to_label)
    df["num_species_kept"] = pd.to_numeric(df["num_species_kept"], errors="coerce").round().astype("Int64")

    out = pd.DataFrame(
        {
            "experiment_folder": experiment_folder,
            "run_timestamp": run_timestamp,
            "hidden_size_label": df["hidden_size_label"],
            "hidden_size_tuple": df["hidden_size_tuple"],
            "num_species_kept": df["num_species_kept"],
            "source_mode": "seed_aggregate",
        }
    )

    mean_std_pairs = [
        "test_mse",
        "test_rmse",
        "test_mse_unscaled",
        "test_rmse_unscaled",
        "training_time_s",
        "epochs_ran",
        "best_val_loss",
        "mean_rel_error_k1",
        "mean_rel_error_k2",
        "mean_rel_error_k3",
        "max_rel_error_k1",
        "max_rel_error_k2",
        "max_rel_error_k3",
    ]
    for metric in mean_std_pairs:
        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"
        if mean_col in df.columns:
            out[metric] = df[mean_col]
        if std_col in df.columns:
            out[f"{metric}_std"] = df[std_col]
    return out


def collect_species_count_tables(fullrun_parent: Path = FULLRUN_PARENT) -> list[pd.DataFrame]:
    scheme_root = BASE_RESULTS_DIR / SCHEME
    if not scheme_root.exists():
        raise FileNotFoundError(f"Scheme root not found: {scheme_root}")
    if not fullrun_parent.exists():
        raise FileNotFoundError(f"FullRun folder not found: {fullrun_parent}")

    candidates: dict[str, dict] = {}
    for csv_path in fullrun_parent.rglob("*.csv"):
        if csv_path.name not in {"summary.csv", "seed_aggregate_summary.csv"}:
            continue
        experiment_dir = csv_path.parent
        if not is_valid_experiment_dir(experiment_dir):
            continue

        experiment_name = experiment_dir.name
        info = {
            "path": csv_path,
            "run_timestamp": "",
            "source_priority": 1 if csv_path.name == "seed_aggregate_summary.csv" else 0,
            "mtime": csv_path.stat().st_mtime,
        }
        prev = candidates.get(experiment_name)
        if prev is None or (info["source_priority"], info["mtime"]) > (prev["source_priority"], prev["mtime"]):
            candidates[experiment_name] = info

    if not candidates:
        raise RuntimeError("No suitable summary.csv or seed_aggregate_summary.csv files were found.")

    tables = []
    for experiment_name, info in sorted(candidates.items()):
        df = pd.read_csv(info["path"])
        if info["path"].name == "seed_aggregate_summary.csv":
            tables.append(summarize_seed_aggregate_df(df, experiment_name, info["run_timestamp"]))
        else:
            tables.append(summarize_summary_df(df, experiment_name, info["run_timestamp"]))
    return tables


def load_species_count_results() -> pd.DataFrame:
    all_df = pd.concat(collect_species_count_tables(), ignore_index=True)
    all_df["hidden_size_tuple"] = all_df["hidden_size_tuple"].apply(parse_hidden_size)
    all_df["hidden_size_label"] = all_df["hidden_size_tuple"].apply(hidden_size_to_label)
    all_df["arch_sort_key"] = all_df["hidden_size_tuple"].apply(hidden_size_sort_key)

    rel_cols = [c for c in all_df.columns if re.fullmatch(r"mean_rel_error_k\d+", c)]
    if rel_cols:
        all_df["mean_rel_error_avg"] = all_df[rel_cols].mean(axis=1)

    rel_std_cols = [c for c in all_df.columns if re.fullmatch(r"mean_rel_error_k\d+_std", c)]
    if rel_std_cols:
        sq = sum(np.square(all_df[c].fillna(0.0)) for c in rel_std_cols)
        counts = sum(all_df[c].notna().astype(int) for c in rel_std_cols)
        with np.errstate(invalid="ignore", divide="ignore"):
            all_df["mean_rel_error_avg_std"] = np.sqrt(sq) / counts.replace(0, np.nan)

    keep_cols = [c for c in all_df.columns if c.startswith("test_") or c.startswith("mean_rel_error") or c.startswith("max_rel_error")]
    for col in keep_cols + ["training_time_s", "epochs_ran", "best_val_loss"]:
        if col in all_df.columns:
            all_df[col] = pd.to_numeric(all_df[col], errors="coerce")

    all_df = all_df.replace([np.inf, -np.inf], np.nan)
    all_df = all_df[all_df["hidden_size_label"].isin(SPECIES_COUNT_ARCHITECTURES)].copy()
    if all_df.empty:
        raise RuntimeError("No rows found for the requested architectures.")
    return all_df


def propagate_ratio_std(numerator, numerator_std, denominator, denominator_std):
    numerator = pd.to_numeric(numerator, errors="coerce")
    denominator = pd.to_numeric(denominator, errors="coerce")
    numerator_std = pd.to_numeric(numerator_std, errors="coerce")
    denominator_std = pd.to_numeric(denominator_std, errors="coerce")
    ratio = numerator / denominator
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_num = numerator_std / numerator
        rel_den = denominator_std / denominator
        ratio_std = ratio * np.sqrt(np.square(rel_num) + np.square(rel_den))
    return ratio, ratio_std.where(
        numerator_std.notna() & denominator_std.notna() & (numerator > 0) & (denominator > 0),
        np.nan,
    )


def compute_relative_deterioration(df: pd.DataFrame) -> pd.DataFrame:
    baseline_mask = df["experiment_folder"].astype(str).str.startswith(BASELINE_PREFIX)
    baseline_df = df[baseline_mask].copy()
    if baseline_df.empty:
        raise RuntimeError(f"No baseline experiment found. Expected a folder starting with {BASELINE_PREFIX!r}.")

    baseline_df = baseline_df[
        [
            "hidden_size_label",
            "num_species_kept",
            "test_mse",
            *(["test_mse_std"] if "test_mse_std" in baseline_df.columns else []),
        ]
    ].rename(
        columns={
            "num_species_kept": "baseline_num_species",
            "test_mse": "baseline_test_mse",
            "test_mse_std": "baseline_test_mse_std",
        }
    )

    merged = df.merge(baseline_df, on="hidden_size_label", how="left")
    if merged["baseline_test_mse"].isna().any():
        missing_archs = merged.loc[merged["baseline_test_mse"].isna(), "hidden_size_label"].dropna().unique().tolist()
        raise RuntimeError(f"Missing baseline for some architectures: {missing_archs}")

    if "test_mse_std" not in merged.columns:
        merged["test_mse_std"] = np.nan
    if "baseline_test_mse_std" not in merged.columns:
        merged["baseline_test_mse_std"] = np.nan

    ratio, ratio_std = propagate_ratio_std(
        merged["test_mse"], merged["test_mse_std"], merged["baseline_test_mse"], merged["baseline_test_mse_std"]
    )
    merged["mse_ratio_vs_baseline"] = ratio
    merged["mse_ratio_vs_baseline_std"] = ratio_std
    merged["relative_deterioration_test_mse_pct"] = 100.0 * (ratio - 1.0)
    merged["relative_deterioration_test_mse_pct_std"] = 100.0 * ratio_std

    baseline_self_mask = merged["experiment_folder"].astype(str).str.startswith(BASELINE_PREFIX)
    merged.loc[baseline_self_mask, "mse_ratio_vs_baseline"] = 1.0
    merged.loc[baseline_self_mask, "mse_ratio_vs_baseline_std"] = 0.0
    merged.loc[baseline_self_mask, "relative_deterioration_test_mse_pct"] = 0.0
    merged.loc[baseline_self_mask, "relative_deterioration_test_mse_pct_std"] = 0.0
    return merged.replace([np.inf, -np.inf], np.nan)


def sanitize_yerr_for_log(y, yerr):
    if yerr is None:
        return None
    y = np.asarray(y, dtype=float)
    yerr = np.asarray(yerr, dtype=float)
    yerr = np.where(np.isnan(yerr), 0.0, yerr)
    yerr = np.where(yerr < 0, 0.0, yerr)
    return np.minimum(yerr, np.maximum(0.0, 0.999 * y))


def make_single_arch_plot(
    df_arch: pd.DataFrame,
    arch_label: str,
    y_col: str,
    y_label: str,
    filename: Path,
    title: Optional[str] = None,
    yscale: Optional[str] = None,
    min_species: Optional[int] = None,
    yerr_col: Optional[str] = None,
) -> None:
    if y_col not in df_arch.columns:
        return

    plot_df = df_arch.copy()
    if min_species is not None:
        plot_df = plot_df[plot_df["num_species_kept"] >= min_species].copy()
    plot_df = plot_df.replace([np.inf, -np.inf], np.nan)
    plot_df = plot_df.dropna(subset=["num_species_kept", y_col]).sort_values("num_species_kept")
    if plot_df.empty:
        return

    x = plot_df["num_species_kept"].to_numpy(dtype=float)
    y = plot_df[y_col].to_numpy(dtype=float)
    color = get_color_for_architecture(arch_label)

    use_errorbars = False
    yerr = None
    if yerr_col is not None and yerr_col in plot_df.columns:
        yerr_series = pd.to_numeric(plot_df[yerr_col], errors="coerce")
        if yerr_series.notna().any():
            yerr = yerr_series.to_numpy(dtype=float)
            if yscale == "log":
                yerr = sanitize_yerr_for_log(y, yerr)
            else:
                yerr = np.where(np.isnan(yerr), 0.0, yerr)
                yerr = np.where(yerr < 0, 0.0, yerr)
            use_errorbars = np.any(yerr > 0)

    plt.figure(figsize=(8, 6))
    if use_errorbars:
        plt.errorbar(x, y, yerr=yerr, marker="o", linewidth=1.8, elinewidth=1.2, capsize=4, color=color)
    else:
        plt.plot(x, y, marker="o", linewidth=1.8, color=color)

    plt.xlabel("Number of Species")
    plt.ylabel(y_label)
    if title:
        plt.title(title)
    if yscale == "log":
        vals = plot_df[y_col].dropna()
        if not vals.empty and (vals > 0).all():
            plt.yscale("log")
    elif yscale == "symlog":
        plt.yscale("symlog", linthresh=1)
    plt.xticks(sorted(plot_df["num_species_kept"].dropna().unique()))
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    filename.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(filename, dpi=300)
    plt.close()


def save_species_count_reports(analysis_dir: Path, arch_label: str, df_arch: pd.DataFrame) -> None:
    arch_dir = analysis_dir / arch_label
    arch_dir.mkdir(parents=True, exist_ok=True)

    cols = [
        "experiment_folder",
        "run_timestamp",
        "source_mode",
        "hidden_size_label",
        "num_species_kept",
        "test_mse",
        "test_mse_std",
        "baseline_num_species",
        "baseline_test_mse",
        "baseline_test_mse_std",
        "mse_ratio_vs_baseline",
        "mse_ratio_vs_baseline_std",
        "relative_deterioration_test_mse_pct",
        "relative_deterioration_test_mse_pct_std",
        "mean_rel_error_avg",
        "mean_rel_error_avg_std",
        "mean_rel_error_k1",
        "mean_rel_error_k1_std",
        "mean_rel_error_k2",
        "mean_rel_error_k2_std",
        "mean_rel_error_k3",
        "mean_rel_error_k3_std",
        "max_rel_error_k1",
        "max_rel_error_k2",
        "max_rel_error_k3",
        "training_time_s",
        "training_time_s_std",
    ]
    cols = [c for c in cols if c in df_arch.columns]
    df_arch[cols].sort_values("num_species_kept").to_csv(arch_dir / "relative_deterioration_report.csv", index=False)

    lines = [
        f"Architecture report: {arch_label}",
        "",
        "Definitions:",
        "  MSE ratio = MSE_current / MSE_baseline",
        "  Relative deterioration (%) = 100 * (MSE_current - MSE_baseline) / MSE_baseline",
        f"  Baseline = experiment folder starting with '{BASELINE_PREFIX}' (11 kept species)",
        "  Error bars are shown only when a standard deviation is available.",
        "",
    ]

    baseline_candidates = df_arch[df_arch["experiment_folder"].astype(str).str.startswith(BASELINE_PREFIX)]
    if not baseline_candidates.empty:
        baseline_row = baseline_candidates.iloc[0]
        baseline_std_str = ""
        if "baseline_test_mse_std" in baseline_row and pd.notna(baseline_row["baseline_test_mse_std"]):
            baseline_std_str = f" ± {baseline_row['baseline_test_mse_std']:.6e}"
        lines.extend(
            [
                f"Baseline experiment: {baseline_row['experiment_folder']}",
                f"Baseline number of species: {int(baseline_row['baseline_num_species'])}",
                f"Baseline test MSE: {baseline_row['baseline_test_mse']:.6e}{baseline_std_str}",
                "",
            ]
        )

    for _, row in df_arch.sort_values("num_species_kept").iterrows():
        ratio = row.get("mse_ratio_vs_baseline", np.nan)
        ratio_std = row.get("mse_ratio_vs_baseline_std", np.nan)
        signed_det = row.get("relative_deterioration_test_mse_pct", np.nan)
        signed_det_std = row.get("relative_deterioration_test_mse_pct_std", np.nan)
        mse_std = row.get("test_mse_std", np.nan)

        ratio_str = "nan" if pd.isna(ratio) else f"{ratio:.4f}"
        if pd.notna(ratio_std):
            ratio_str += f" ± {ratio_std:.4f}"
        signed_det_str = "nan" if pd.isna(signed_det) else f"{signed_det:+.2f}%"
        if pd.notna(signed_det_std):
            signed_det_str += f" ± {signed_det_std:.2f}%"
        mse_str = f"{row['test_mse']:.6e}"
        if pd.notna(mse_std):
            mse_str += f" ± {mse_std:.6e}"

        lines.append(
            f"Species: {int(row['num_species_kept']):>2d} | "
            f"Experiment: {row['experiment_folder']} | "
            f"Source: {row.get('source_mode', 'unknown')} | "
            f"Test MSE: {mse_str} | "
            f"MSE ratio: {ratio_str} | "
            f"Relative deterioration: {signed_det_str}"
        )

    with open(arch_dir / "relative_deterioration_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def make_species_count_architecture_plots(arch_dir: Path, arch_label: str, df_arch: pd.DataFrame) -> None:
    make_single_arch_plot(
        df_arch,
        arch_label,
        "test_mse",
        "Test MSE",
        arch_dir / "test_mse_vs_num_species.pdf",
        title=f"{arch_label} — Test MSE vs number of species ({SPECIES_MIN} to 11 species)",
        yscale="log",
        yerr_col="test_mse_std",
    )
    make_single_arch_plot(
        df_arch,
        arch_label,
        "test_mse",
        "Test MSE",
        arch_dir / "test_mse_vs_num_species_zoom.pdf",
        title=f"{arch_label} — Test MSE vs number of species ({ZOOM_MIN} to 11 species)",
        yscale="log",
        min_species=ZOOM_MIN,
        yerr_col="test_mse_std",
    )
    make_single_arch_plot(
        df_arch,
        arch_label,
        "relative_deterioration_test_mse_pct",
        "Relative deterioration in test MSE (%)",
        arch_dir / "relative_deterioration_vs_num_species.pdf",
        title=f"{arch_label} — Relative deterioration vs number of species ({SPECIES_MIN} to 11 species)",
        yscale="symlog",
    )
    make_single_arch_plot(
        df_arch,
        arch_label,
        "relative_deterioration_test_mse_pct",
        "Relative deterioration in test MSE (%)",
        arch_dir / "relative_deterioration_vs_num_species_zoom.pdf",
        title=f"{arch_label} — Relative deterioration vs number of species ({ZOOM_MIN} to 11 species)",
        min_species=ZOOM_MIN,
    )

    if "mean_rel_error_avg" in df_arch.columns:
        make_single_arch_plot(
            df_arch,
            arch_label,
            "mean_rel_error_avg",
            "Mean relative error (average over k's)",
            arch_dir / "mean_relative_error_avg_vs_num_species.pdf",
            title=f"{arch_label} — Mean relative error vs number of species ({SPECIES_MIN} to 11 species)",
            yscale="log",
            yerr_col="mean_rel_error_avg_std",
        )
        make_single_arch_plot(
            df_arch,
            arch_label,
            "mean_rel_error_avg",
            "Mean relative error (average over k's)",
            arch_dir / "mean_relative_error_avg_vs_num_species_zoom.pdf",
            title=f"{arch_label} — Mean relative error vs number of species ({ZOOM_MIN} to 11 species)",
            yscale="log",
            min_species=ZOOM_MIN,
            yerr_col="mean_rel_error_avg_std",
        )

    for k in [1, 2, 3]:
        col = f"mean_rel_error_k{k}"
        std_col = f"mean_rel_error_k{k}_std"
        if col in df_arch.columns:
            make_single_arch_plot(
                df_arch,
                arch_label,
                col,
                f"Mean relative error k{k}",
                arch_dir / f"mean_relative_error_k{k}_vs_num_species.pdf",
                title=f"{arch_label} — Mean relative error k{k} vs number of species ({SPECIES_MIN} to 11 species)",
                yscale="log",
                yerr_col=std_col,
            )
            make_single_arch_plot(
                df_arch,
                arch_label,
                col,
                f"Mean relative error k{k}",
                arch_dir / f"mean_relative_error_k{k}_vs_num_species_zoom.pdf",
                title=f"{arch_label} — Mean relative error k{k} vs number of species ({ZOOM_MIN} to 11 species)",
                yscale="log",
                min_species=ZOOM_MIN,
                yerr_col=std_col,
            )


def run_species_count_figure() -> dict[str, Path]:
    df = load_species_count_results()
    df = compute_relative_deterioration(df)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    aggregated_csv = ANALYSIS_DIR / "aggregated_results.csv"
    df.sort_values(["hidden_size_label", "num_species_kept"]).to_csv(aggregated_csv, index=False)

    for arch_label in SPECIES_COUNT_ARCHITECTURES:
        df_arch = df[df["hidden_size_label"] == arch_label].copy()
        if df_arch.empty:
            print(f"Skipping architecture {arch_label}: no rows found.")
            continue
        arch_dir = ANALYSIS_DIR / arch_label
        save_species_count_reports(ANALYSIS_DIR, arch_label, df_arch)
        make_species_count_architecture_plots(arch_dir, arch_label, df_arch)

    print(f"Species-count analysis files saved to: {ANALYSIS_DIR}")
    return {"analysis_dir": ANALYSIS_DIR, "aggregated_csv": aggregated_csv}


# ======================================================================================
# FIGURE 2: ENSEMBLE VS SINGLE-MODEL COMPARISON
# ======================================================================================

SINGLE_RANKING_CSV = BASE_RESULTS_DIR / SCHEME / "Noise_Error_Rankings" / "fullrun_noise_aggregate_summary.csv"
ENSEMBLE_RANKING_CSV_CANDIDATES = [
    BASE_RESULTS_DIR / SCHEME / "Ensemble_Noise_Error_Rankings" / "fullrun_noise_aggregate_summary.csv",
    BASE_RESULTS_DIR / SCHEME / "Ensemble_(Noise_Error_Rankings)" / "fullrun_noise_aggregate_summary.csv",
]
ENSEMBLE_COMPARISON_OUTPUT_DIR = BASE_RESULTS_DIR / SCHEME / "PIC1_Ensemble_Comparison"
ENSEMBLE_COMPARISON_BASENAME = "ensemble_mse_and_improvement_errorbars__30_30_30"
PLOT_ERROR_BARS_ENSEMBLE_LEFT = True
PLOT_ERROR_BARS_ENSEMBLE_RIGHT = False


def build_ensemble_comparison(single_df: pd.DataFrame, ensemble_df: pd.DataFrame) -> pd.DataFrame:
    single = single_df[["species_key", "noise_percent", "species_label", MEAN_COL, STD_COL]].rename(
        columns={MEAN_COL: "single_mse_mean", STD_COL: "single_mse_std"}
    )
    ensemble = ensemble_df[["species_key", "noise_percent", MEAN_COL, STD_COL]].rename(
        columns={MEAN_COL: "ensemble_mse", STD_COL: "ensemble_mse_std"}
    )
    comp = single.merge(ensemble, on=["species_key", "noise_percent"], how="inner")
    if comp.empty:
        raise ValueError("No common species/noise rows found between single-model and ensemble CSVs.")

    order = {species_key(combo): idx for idx, combo in enumerate(REPRESENTATIVE_COMBINATIONS)}
    labels = {species_key(combo): species_label(combo) for combo in REPRESENTATIVE_COMBINATIONS}
    comp["combination_order"] = comp["species_key"].apply(lambda key: order[key])
    comp["species_label"] = comp["species_key"].apply(lambda key: labels[key])
    comp["improvement_percent"] = 100.0 * (comp["single_mse_mean"] - comp["ensemble_mse"]) / comp["single_mse_mean"]

    S = comp["single_mse_mean"].to_numpy(float)
    E = comp["ensemble_mse"].to_numpy(float)
    sigma_S = comp["single_mse_std"].fillna(0.0).to_numpy(float)
    sigma_E = comp["ensemble_mse_std"].fillna(0.0).to_numpy(float)
    sigma_I = np.full_like(S, np.nan, dtype=float)
    mask = S > 0.0
    sigma_I[mask] = np.sqrt(
        ((100.0 * E[mask] / (S[mask] ** 2)) * sigma_S[mask]) ** 2
        + ((100.0 / S[mask]) * sigma_E[mask]) ** 2
    )
    comp["improvement_percent_std"] = sigma_I
    return comp.sort_values(["combination_order", "noise_percent"]).reset_index(drop=True)


def warn_about_missing_representatives(single_df: pd.DataFrame, ensemble_df: pd.DataFrame, comp: pd.DataFrame) -> None:
    wanted = {species_key(combo): species_label(combo) for combo in REPRESENTATIVE_COMBINATIONS}
    checks = [
        ("Missing from single-model CSV", set(wanted) - set(single_df["species_key"])),
        ("Missing from ensemble CSV", set(wanted) - set(ensemble_df["species_key"])),
        ("Missing from final common comparison", set(wanted) - set(comp["species_key"])),
    ]
    for title, missing in checks:
        if missing:
            print(f"WARNING: {title}:")
            for key in sorted(missing, key=lambda k: wanted[k]):
                print(f"  - {wanted[key]}")


def run_ensemble_comparison_figure() -> dict[str, Path]:
    ensemble_path = find_first_existing(ENSEMBLE_RANKING_CSV_CANDIDATES, "ensemble ranking CSV")
    single_df = load_representative_noise_df(SINGLE_RANKING_CSV, "single_model_mean")
    ensemble_df = load_representative_noise_df(ensemble_path, "ensemble")
    comp = build_ensemble_comparison(single_df, ensemble_df)
    warn_about_missing_representatives(single_df, ensemble_df, comp)

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_TWO_PANEL, constrained_layout=False)
    left_df = ensemble_df.rename(columns={MEAN_COL: "ensemble_mse", STD_COL: "ensemble_mse_std"})
    plot_combo_lines(
        axes[0],
        left_df,
        "ensemble_mse",
        "ensemble_mse_std",
        errorbars=PLOT_ERROR_BARS_ENSEMBLE_LEFT,
        log_y=True,
        ylabel="Scaled MSE",
    )
    plot_combo_lines(
        axes[1],
        comp,
        "improvement_percent",
        "improvement_percent_std",
        errorbars=PLOT_ERROR_BARS_ENSEMBLE_RIGHT,
        log_y=False,
        ylabel="Improvement (%)",
    )
    axes[1].axhline(0.0, color="gray", linestyle="--", linewidth=1.2, alpha=0.9)

    fig.legend(
        handles=combo_legend_handles(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.04),
        ncol=4,
        frameon=False,
        fontsize=LEGEND_FONTSIZE,
        handlelength=1.8,
        columnspacing=1.2,
        handletextpad=0.5,
    )
    fig.subplots_adjust(**SUBPLOTS_ADJUST_TWO_PANEL)
    paths = save_standard_figure(fig, ENSEMBLE_COMPARISON_OUTPUT_DIR, ENSEMBLE_COMPARISON_BASENAME)

    if SAVE_DATA_CSV:
        data_path = ENSEMBLE_COMPARISON_OUTPUT_DIR / f"{ENSEMBLE_COMPARISON_BASENAME}__data.csv"
        export = comp.copy()
        export["species_key"] = export["species_key"].apply(lambda key: " | ".join(key))
        export.to_csv(data_path, index=False)
        paths["comparison_data_csv"] = data_path

        left_data_path = ENSEMBLE_COMPARISON_OUTPUT_DIR / f"{ENSEMBLE_COMPARISON_BASENAME}__left_panel_ensemble_data.csv"
        ensemble_export = ensemble_df.copy()
        ensemble_export["species_key"] = ensemble_export["species_key"].apply(lambda key: " | ".join(key))
        ensemble_export.to_csv(left_data_path, index=False)
        paths["left_panel_ensemble_data_csv"] = left_data_path

    if SAVE_MANIFEST:
        manifest_path = ENSEMBLE_COMPARISON_OUTPUT_DIR / f"{ENSEMBLE_COMPARISON_BASENAME}__manifest.json"
        write_json(
            manifest_path,
            {
                "single_csv": str(SINGLE_RANKING_CSV),
                "ensemble_csv": str(ensemble_path),
                "output_dir": str(ENSEMBLE_COMPARISON_OUTPUT_DIR),
                "architecture": ARCHITECTURE,
                "mean_column": MEAN_COL,
                "std_column": STD_COL,
                "representative_combinations": REPRESENTATIVE_COMBINATIONS,
                "plot_error_bars_left": PLOT_ERROR_BARS_ENSEMBLE_LEFT,
                "plot_error_bars_right": PLOT_ERROR_BARS_ENSEMBLE_RIGHT,
                "use_log_y_left": True,
                "figsize": list(FIGSIZE_TWO_PANEL),
                "subplots_adjust": SUBPLOTS_ADJUST_TWO_PANEL,
                "improvement_percent_formula": "100 * (single_mse_mean - ensemble_mse) / single_mse_mean",
                "improvement_errorbar_formula": "first-order propagation from single_mse_std and ensemble_mse_std",
                "saved_paths": {name: str(path) for name, path in paths.items()},
            },
        )
        paths["manifest_json"] = manifest_path

    print("Ensemble comparison outputs saved:")
    for name, path in paths.items():
        print(f"  {name}: {path}")
    return paths


# ======================================================================================
# FIGURE 3: ENSEMBLE INDIVIDUAL-NOISE TWO-PANEL PLOT
# ======================================================================================

ENSEMBLE_INDIVIDUAL_CSV_CANDIDATES = [
    BASE_RESULTS_DIR / SCHEME / "Ensemble_Individual_Noise" / "fullrun_ensemble_individual_noise_aggregate_summary.csv",
    BASE_RESULTS_DIR / SCHEME / "Ensemble_(Individual_Noise)" / "fullrun_ensemble_individual_noise_aggregate_summary.csv",
]
ENSEMBLE_INDIVIDUAL_OUTPUT_DIR = BASE_RESULTS_DIR / SCHEME / "PIC1_Ensemble_Individual_Noise_TwoPanel"
ENSEMBLE_INDIVIDUAL_BASENAME = "ensemble_individual_noise_two_panel__30_30_30"
LEFT_COMBINATION = ["O2(a)", "O2(b)", "O2(Hz)"]
RIGHT_COMBINATION = ["O2(a)", "O2(b)"]
CLEAN_BASELINE_ID = "clean_baseline"
PLOT_ERROR_BARS_INDIVIDUAL = True


def find_ensemble_individual_noise_csv() -> Path:
    return find_first_existing(ENSEMBLE_INDIVIDUAL_CSV_CANDIDATES, "ensemble individual-noise aggregate CSV")


def check_individual_noise_columns(df: pd.DataFrame, path: Path) -> None:
    required = ["hidden_size", "kept_species", "noisy_subset_id", "noisy_subset_label", "noise_percent", MEAN_COL, STD_COL]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")


def load_individual_noise_data() -> tuple[pd.DataFrame, Path]:
    path = find_ensemble_individual_noise_csv()
    df = pd.read_csv(path)
    check_individual_noise_columns(df, path)
    df = df.copy()
    df["hidden_size"] = df["hidden_size"].apply(normalize_architecture_text)
    df["noise_percent"] = pd.to_numeric(df["noise_percent"], errors="coerce")
    df[MEAN_COL] = pd.to_numeric(df[MEAN_COL], errors="coerce")
    df[STD_COL] = pd.to_numeric(df[STD_COL], errors="coerce")
    df = df[(df["hidden_size"] == ARCHITECTURE)].dropna(subset=["noise_percent", MEAN_COL]).copy()
    if df.empty:
        raise ValueError(f"No rows found in {path} for architecture {ARCHITECTURE!r}.")
    df["kept_species_key"] = df["kept_species"].apply(species_key)
    # Prefer noisy_species_names if present because it is usually comma-separated and unambiguous.
    source_col = "noisy_species_names" if "noisy_species_names" in df.columns else "noisy_subset_label"
    df["noisy_subset_key"] = df[source_col].apply(species_key)
    empty_subset_mask = df["noisy_subset_key"].apply(lambda key: len(key) == 0)
    if empty_subset_mask.any():
        df.loc[empty_subset_mask, "noisy_subset_key"] = df.loc[empty_subset_mask, "noisy_subset_label"].apply(species_key)
    return df, path


def get_individual_panel_data(df: pd.DataFrame, kept_species: list[str]):
    kept_key = species_key(kept_species)
    panel_df = df[df["kept_species_key"] == kept_key].copy()
    if panel_df.empty:
        available = df[["kept_species"]].drop_duplicates().sort_values("kept_species")["kept_species"].tolist()
        available_text = "\n".join(f"  - {item}" for item in available)
        raise ValueError(
            f"No ensemble individual-noise rows found for kept_species = {subset_label(kept_species)!r}.\n"
            "Available kept_species values in the aggregate CSV are:\n"
            f"{available_text}"
        )

    baseline_df = panel_df[
        (panel_df["noisy_subset_id"].astype(str) == CLEAN_BASELINE_ID)
        & np.isclose(panel_df["noise_percent"].astype(float), 0.0)
    ].copy()

    if baseline_df.empty:
        baseline_x = np.array([], dtype=float)
        baseline_y = np.array([], dtype=float)
        baseline_yerr = np.array([], dtype=float)
    else:
        baseline_x = np.array([0.0], dtype=float)
        baseline_y = np.array([baseline_df[MEAN_COL].mean()], dtype=float)
        baseline_yerr = np.array([baseline_df[STD_COL].fillna(0.0).mean()], dtype=float)

    nonbaseline_df = panel_df[panel_df["noisy_subset_id"].astype(str) != CLEAN_BASELINE_ID].copy()
    nonbaseline_df = nonbaseline_df[nonbaseline_df["noise_percent"].astype(float) > 0.0].copy()
    wanted_subset_keys = [species_key(combo) for combo in INNER_SUBSET_ORDER]
    nonbaseline_df = nonbaseline_df[nonbaseline_df["noisy_subset_key"].isin(wanted_subset_keys)].copy()
    return panel_df, nonbaseline_df, baseline_x, baseline_y, baseline_yerr


def plot_individual_panel(ax, df: pd.DataFrame, kept_species: list[str]) -> None:
    full_panel_df, nonbaseline_df, baseline_x, baseline_y, baseline_yerr = get_individual_panel_data(df, kept_species)

    for subset in INNER_SUBSET_ORDER:
        subset_key = species_key(subset)
        if not set(subset_key).issubset(set(species_key(kept_species))):
            continue
        subset_df = nonbaseline_df[nonbaseline_df["noisy_subset_key"] == subset_key].copy()
        if subset_df.empty:
            continue
        subset_df = (
            subset_df.groupby("noise_percent", as_index=False)
            .agg(**{MEAN_COL: (MEAN_COL, "mean"), STD_COL: (STD_COL, "mean")})
            .sort_values("noise_percent")
        )
        x = np.concatenate([baseline_x, subset_df["noise_percent"].to_numpy(float)])
        y = np.concatenate([baseline_y, subset_df[MEAN_COL].to_numpy(float)])
        yerr = np.concatenate([baseline_yerr, subset_df[STD_COL].fillna(0.0).to_numpy(float)])
        subset_tuple = ordered_subset_key(subset)

        if PLOT_ERROR_BARS_INDIVIDUAL:
            ax.errorbar(
                x,
                y,
                yerr=yerr_for_log_axis(y, yerr),
                color=SUBSET_COLORS[subset_tuple],
                marker=SUBSET_MARKERS[subset_tuple],
                linestyle="-",
                linewidth=LINEWIDTH,
                markersize=MARKERSIZE,
                capsize=CAPSIZE,
                elinewidth=ERRORBAR_LINEWIDTH,
            )
        else:
            ax.plot(
                x,
                y,
                color=SUBSET_COLORS[subset_tuple],
                marker=SUBSET_MARKERS[subset_tuple],
                linestyle="-",
                linewidth=LINEWIDTH,
                markersize=MARKERSIZE,
            )

    xticks = sorted(float(x) for x in full_panel_df["noise_percent"].dropna().unique())
    if 0.0 not in xticks:
        xticks = [0.0] + xticks
    ax.set_xticks(xticks)
    ax.set_xticklabels([format_noise(x) for x in xticks])
    ax.set_xlabel("Noise level (%)")
    ax.set_ylabel("Scaled MSE", labelpad=2)
    yvals = full_panel_df[MEAN_COL].dropna().to_numpy(dtype=float)
    if np.all(yvals > 0.0):
        ax.set_yscale("log")
    else:
        print(f"WARNING: non-positive Scaled MSE found for {subset_label(kept_species)}; using linear y-axis.")
    ax.grid(True, which="both", alpha=GRID_ALPHA)


def individual_legend_handles(df: pd.DataFrame) -> list[Line2D]:
    present_subset_keys = set(df.loc[df["noisy_subset_id"].astype(str) != CLEAN_BASELINE_ID, "noisy_subset_key"])
    handles = []
    for subset in INNER_SUBSET_ORDER:
        key = species_key(subset)
        if key not in present_subset_keys:
            continue
        subset_tuple = ordered_subset_key(subset)
        handles.append(
            Line2D(
                [0],
                [0],
                color=SUBSET_COLORS[subset_tuple],
                marker=SUBSET_MARKERS[subset_tuple],
                linestyle="-",
                linewidth=LINEWIDTH,
                markersize=MARKERSIZE,
                label=subset_label(subset_tuple),
            )
        )
    return handles


def run_ensemble_individual_noise_figure() -> dict[str, Path]:
    df, aggregate_csv = load_individual_noise_data()
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_TWO_PANEL, constrained_layout=False)
    plot_individual_panel(axes[0], df, LEFT_COMBINATION)
    plot_individual_panel(axes[1], df, RIGHT_COMBINATION)
    fig.legend(
        handles=individual_legend_handles(df),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.04),
        ncol=4,
        frameon=False,
        fontsize=LEGEND_FONTSIZE,
        handlelength=1.8,
        columnspacing=1.2,
        handletextpad=0.5,
    )
    fig.subplots_adjust(**SUBPLOTS_ADJUST_TWO_PANEL)
    paths = save_standard_figure(fig, ENSEMBLE_INDIVIDUAL_OUTPUT_DIR, ENSEMBLE_INDIVIDUAL_BASENAME)

    if SAVE_DATA_CSV:
        wanted_kept = {species_key(LEFT_COMBINATION), species_key(RIGHT_COMBINATION)}
        export = df[df["kept_species_key"].isin(wanted_kept)].copy()
        export["kept_species_key"] = export["kept_species_key"].apply(lambda x: " | ".join(x))
        export["noisy_subset_key"] = export["noisy_subset_key"].apply(lambda x: " | ".join(x))
        data_path = ENSEMBLE_INDIVIDUAL_OUTPUT_DIR / f"{ENSEMBLE_INDIVIDUAL_BASENAME}__data.csv"
        export.to_csv(data_path, index=False)
        paths["data_csv"] = data_path

    if SAVE_MANIFEST:
        manifest_path = ENSEMBLE_INDIVIDUAL_OUTPUT_DIR / f"{ENSEMBLE_INDIVIDUAL_BASENAME}__manifest.json"
        write_json(
            manifest_path,
            {
                "aggregate_csv": str(aggregate_csv),
                "output_dir": str(ENSEMBLE_INDIVIDUAL_OUTPUT_DIR),
                "architecture": ARCHITECTURE,
                "mean_column": MEAN_COL,
                "std_column": STD_COL,
                "left_combination": LEFT_COMBINATION,
                "right_combination": RIGHT_COMBINATION,
                "inner_subset_order": INNER_SUBSET_ORDER,
                "plot_error_bars": PLOT_ERROR_BARS_INDIVIDUAL,
                "use_log_y": True,
                "clean_baseline_policy": "For each panel, the panel's clean 0% ensemble baseline is prepended to every noisy-subset line.",
                "error_bar_definition": "test_mse_scaled_std from the aggregate CSV.",
                "figsize": list(FIGSIZE_TWO_PANEL),
                "subplots_adjust": SUBPLOTS_ADJUST_TWO_PANEL,
                "saved_paths": {name: str(path) for name, path in paths.items()},
            },
        )
        paths["manifest_json"] = manifest_path

    print("Ensemble individual-noise outputs saved:")
    for name, path in paths.items():
        print(f"  {name}: {path}")
    return paths


# ======================================================================================
# FIGURE 4: NOISY-TRAINING ENSEMBLE COMPARISON
# ======================================================================================

NOISY_TRAINED_ENSEMBLE_CSV = (
    BASE_RESULTS_DIR
    / OUTPUT_SCHEME
    / NOISY_TRAINING_EXPERIMENT_NAME
    / "Ensemble"
    / "Noise_MSE_Results"
    / "fullrun_noise_aggregate_summary.csv"
)
PRECOMPUTED_NOISY_TRAINING_COMPARISON_CSV = (
    BASE_RESULTS_DIR
    / OUTPUT_SCHEME
    / NOISY_TRAINING_EXPERIMENT_NAME
    / "Comparison_Against_Clean"
    / "ensemble"
    / "comparison_ensemble.csv"
)
ORIGINAL_ENSEMBLE_CSV_CANDIDATES = ENSEMBLE_RANKING_CSV_CANDIDATES
NOISY_TRAINING_OUTPUT_DIR = (
    BASE_RESULTS_DIR
    / OUTPUT_SCHEME
    / NOISY_TRAINING_EXPERIMENT_NAME
    / "PIC1_NoisyTraining_Ensemble_Comparison"
)
NOISY_TRAINING_BASENAME = "noisy_training_ensemble_mse_and_improvement__30_30_30"
PLOT_ERROR_BARS_NOISY_TRAINING_LEFT = True
PLOT_ERROR_BARS_NOISY_TRAINING_RIGHT = True
EXCLUDE_ZERO_NOISE_FROM_NOISY_TRAINING_RIGHT = True


def load_noisy_trained_aggregate() -> pd.DataFrame:
    if not NOISY_TRAINED_ENSEMBLE_CSV.exists():
        raise FileNotFoundError(
            f"Missing file: {NOISY_TRAINED_ENSEMBLE_CSV}\n"
            "Run NeuralNet_3Ks_NoisyTraining.py first."
        )
    return load_representative_noise_df(NOISY_TRAINED_ENSEMBLE_CSV, "noisy_trained_ensemble")


def finish_noisy_training_comparison_columns(comp: pd.DataFrame) -> pd.DataFrame:
    comp = comp.copy()
    original_mse = comp["original_ensemble_mse"].to_numpy(float)
    noisy_mse = comp["noisy_trained_ensemble_mse"].to_numpy(float)
    comp["improvement_percent"] = np.nan
    positive_original = original_mse > 0.0
    comp.loc[positive_original, "improvement_percent"] = 100.0 * (
        original_mse[positive_original] - noisy_mse[positive_original]
    ) / original_mse[positive_original]

    sigma_original = comp["original_ensemble_mse_std"].fillna(0.0).to_numpy(float)
    sigma_noisy = comp["noisy_trained_ensemble_mse_std"].fillna(0.0).to_numpy(float)
    sigma_improvement = np.full_like(original_mse, np.nan, dtype=float)
    sigma_improvement[positive_original] = np.sqrt(
        ((100.0 * noisy_mse[positive_original] / (original_mse[positive_original] ** 2)) * sigma_original[positive_original]) ** 2
        + ((100.0 / original_mse[positive_original]) * sigma_noisy[positive_original]) ** 2
    )
    comp["improvement_percent_std"] = sigma_improvement
    comp["mse_ratio_noisy_trained_over_original"] = noisy_mse / original_mse
    comp["noisy_training_helped"] = noisy_mse < original_mse
    return comp.sort_values(["combination_order", "noise_percent"]).reset_index(drop=True)


def load_noisy_training_comparison_from_precomputed() -> tuple[Optional[pd.DataFrame], Optional[Path]]:
    path = PRECOMPUTED_NOISY_TRAINING_COMPARISON_CSV
    if not path.exists():
        return None, None

    df = pd.read_csv(path)
    required = [
        "hidden_size",
        "kept_species",
        "noise_percent",
        "clean_trained_test_mse_scaled_mean",
        "noisy_trained_test_mse_scaled_mean",
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        print(
            f"WARNING: precomputed comparison CSV exists but is missing columns: {', '.join(missing)}. "
            "Falling back to direct original/noisy aggregate comparison."
        )
        return None, None

    df = df.copy()
    df["hidden_size"] = df["hidden_size"].apply(normalize_architecture_text)
    df["noise_percent"] = pd.to_numeric(df["noise_percent"], errors="coerce")
    df["original_ensemble_mse"] = pd.to_numeric(df["clean_trained_test_mse_scaled_mean"], errors="coerce")
    df["noisy_trained_ensemble_mse"] = pd.to_numeric(df["noisy_trained_test_mse_scaled_mean"], errors="coerce")
    df["original_ensemble_mse_std"] = pd.to_numeric(
        df.get("clean_trained_test_mse_scaled_std", pd.Series(np.nan, index=df.index)),
        errors="coerce",
    )
    df["noisy_trained_ensemble_mse_std"] = pd.to_numeric(
        df.get("noisy_trained_test_mse_scaled_std", pd.Series(np.nan, index=df.index)),
        errors="coerce",
    )
    df = df[(df["hidden_size"] == ARCHITECTURE) & noise_is_requested(df["noise_percent"])].dropna(
        subset=["noise_percent", "original_ensemble_mse", "noisy_trained_ensemble_mse"]
    ).copy()
    if df.empty:
        return None, None

    df["species_key"] = df["kept_species"].apply(species_key)
    wanted_keys = {species_key(combo) for combo in REPRESENTATIVE_COMBINATIONS}
    df = df[df["species_key"].isin(wanted_keys)].copy()
    if df.empty:
        return None, None

    order = {species_key(combo): idx for idx, combo in enumerate(REPRESENTATIVE_COMBINATIONS)}
    labels = {species_key(combo): species_label(combo) for combo in REPRESENTATIVE_COMBINATIONS}
    df["combination_order"] = df["species_key"].apply(lambda key: order[key])
    df["species_label"] = df["species_key"].apply(lambda key: labels[key])
    return finish_noisy_training_comparison_columns(df), path


def build_noisy_training_comparison_from_aggregates(noisy_df: pd.DataFrame) -> tuple[pd.DataFrame, Path]:
    original_path = find_first_existing(ORIGINAL_ENSEMBLE_CSV_CANDIDATES, "original clean-trained ensemble aggregate CSV")
    original_df = load_representative_noise_df(original_path, "original_clean_trained_ensemble")
    original = original_df[["species_key", "noise_percent", MEAN_COL, STD_COL]].rename(
        columns={MEAN_COL: "original_ensemble_mse", STD_COL: "original_ensemble_mse_std"}
    )
    noisy = noisy_df[["species_key", "noise_percent", MEAN_COL, STD_COL]].rename(
        columns={MEAN_COL: "noisy_trained_ensemble_mse", STD_COL: "noisy_trained_ensemble_mse_std"}
    )
    comp = original.merge(noisy, on=["species_key", "noise_percent"], how="inner")
    if comp.empty:
        raise ValueError("No common rows between original and noisy-trained ensemble CSVs.")

    order = {species_key(combo): idx for idx, combo in enumerate(REPRESENTATIVE_COMBINATIONS)}
    labels = {species_key(combo): species_label(combo) for combo in REPRESENTATIVE_COMBINATIONS}
    comp["combination_order"] = comp["species_key"].apply(lambda key: order[key])
    comp["species_label"] = comp["species_key"].apply(lambda key: labels[key])
    return finish_noisy_training_comparison_columns(comp), original_path


def noisy_training_right_panel_data(comp: pd.DataFrame) -> pd.DataFrame:
    if not EXCLUDE_ZERO_NOISE_FROM_NOISY_TRAINING_RIGHT:
        return comp.copy()
    return comp[~np.isclose(comp["noise_percent"].astype(float), 0.0, rtol=0.0, atol=1e-12)].copy()


def set_noisy_training_right_axis_limits(ax, plot_df: pd.DataFrame) -> None:
    y = plot_df["improvement_percent"].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float)
    if y.size == 0:
        return
    ymin = float(np.nanmin(y))
    ymax = float(np.nanmax(y))
    lower = min(-5.0, np.floor(ymin / 10.0) * 10.0 - 5.0)
    upper = max(105.0, np.ceil(ymax / 10.0) * 10.0 + 5.0)
    if EXCLUDE_ZERO_NOISE_FROM_NOISY_TRAINING_RIGHT:
        ax.set_ylim(lower, upper)
    formatter = ScalarFormatter(useOffset=False)
    formatter.set_scientific(False)
    ax.yaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))


def print_noisy_training_improvement_summary(comp: pd.DataFrame, comparison_source_path: Path) -> None:
    print(f"Comparison source used: {comparison_source_path}")
    useful = comp[~np.isclose(comp["noise_percent"].astype(float), 0.0, rtol=0.0, atol=1e-12)].copy()
    if useful.empty:
        return
    summary = (
        useful.groupby("noise_percent", as_index=False)
        .agg(
            mean_improvement_percent=("improvement_percent", "mean"),
            min_improvement_percent=("improvement_percent", "min"),
            max_improvement_percent=("improvement_percent", "max"),
            fraction_improved=("noisy_training_helped", "mean"),
        )
        .sort_values("noise_percent")
    )
    print("\nNonzero-noise improvement summary:")
    print(summary.to_string(index=False))
    zero = comp[np.isclose(comp["noise_percent"].astype(float), 0.0, rtol=0.0, atol=1e-12)]
    if not zero.empty:
        print(
            "\nNote: 0% improvement values are saved to the data CSV, but are excluded "
            "from the right panel by default because they can be extremely negative and "
            "compress the useful 0.5%--10% range."
        )


def run_noisy_training_figure() -> dict[str, Path]:
    noisy_df = load_noisy_trained_aggregate()
    comp, comparison_source_path = load_noisy_training_comparison_from_precomputed()
    if comp is None:
        comp, comparison_source_path = build_noisy_training_comparison_from_aggregates(noisy_df)

    assert comparison_source_path is not None
    print_noisy_training_improvement_summary(comp, comparison_source_path)

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_TWO_PANEL, constrained_layout=False)
    left_df = noisy_df.rename(columns={MEAN_COL: "noisy_trained_ensemble_mse", STD_COL: "noisy_trained_ensemble_mse_std"})
    plot_combo_lines(
        axes[0],
        left_df,
        "noisy_trained_ensemble_mse",
        "noisy_trained_ensemble_mse_std",
        errorbars=PLOT_ERROR_BARS_NOISY_TRAINING_LEFT,
        log_y=True,
        ylabel="Scaled MSE",
    )
    right_df = noisy_training_right_panel_data(comp)
    plot_combo_lines(
        axes[1],
        right_df,
        "improvement_percent",
        "improvement_percent_std",
        errorbars=PLOT_ERROR_BARS_NOISY_TRAINING_RIGHT,
        log_y=False,
        ylabel="Improvement (%)",
    )
    axes[1].axhline(0.0, color="gray", linestyle="--", linewidth=1.2, alpha=0.9)
    set_noisy_training_right_axis_limits(axes[1], right_df)

    fig.legend(
        handles=combo_legend_handles(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.04),
        ncol=4,
        frameon=False,
        fontsize=LEGEND_FONTSIZE,
        handlelength=1.8,
        columnspacing=1.2,
        handletextpad=0.5,
    )
    fig.subplots_adjust(**SUBPLOTS_ADJUST_TWO_PANEL)
    paths = save_standard_figure(fig, NOISY_TRAINING_OUTPUT_DIR, NOISY_TRAINING_BASENAME)

    if SAVE_DATA_CSV:
        comparison_data_path = NOISY_TRAINING_OUTPUT_DIR / f"{NOISY_TRAINING_BASENAME}__data_all_noise_levels.csv"
        export = comp.copy()
        export["species_key"] = export["species_key"].apply(lambda key: " | ".join(key))
        export.to_csv(comparison_data_path, index=False)
        paths["comparison_data_csv"] = comparison_data_path

        right_panel_data_path = NOISY_TRAINING_OUTPUT_DIR / f"{NOISY_TRAINING_BASENAME}__right_panel_plotted_data.csv"
        right_export = noisy_training_right_panel_data(comp).copy()
        right_export["species_key"] = right_export["species_key"].apply(lambda key: " | ".join(key))
        right_export.to_csv(right_panel_data_path, index=False)
        paths["right_panel_plotted_data_csv"] = right_panel_data_path

        left_panel_data_path = NOISY_TRAINING_OUTPUT_DIR / f"{NOISY_TRAINING_BASENAME}__left_panel_noisy_trained_data.csv"
        noisy_export = noisy_df.copy()
        noisy_export["species_key"] = noisy_export["species_key"].apply(lambda key: " | ".join(key))
        noisy_export.to_csv(left_panel_data_path, index=False)
        paths["left_panel_noisy_trained_data_csv"] = left_panel_data_path

    if SAVE_MANIFEST:
        manifest_path = NOISY_TRAINING_OUTPUT_DIR / f"{NOISY_TRAINING_BASENAME}__manifest.json"
        write_json(
            manifest_path,
            {
                "noisy_trained_ensemble_csv": str(NOISY_TRAINED_ENSEMBLE_CSV),
                "precomputed_comparison_csv": str(PRECOMPUTED_NOISY_TRAINING_COMPARISON_CSV),
                "comparison_source_used": str(comparison_source_path),
                "output_dir": str(NOISY_TRAINING_OUTPUT_DIR),
                "architecture": ARCHITECTURE,
                "noise_levels_percent": NOISE_LEVELS_PERCENT,
                "mean_column": MEAN_COL,
                "std_column": STD_COL,
                "representative_combinations": REPRESENTATIVE_COMBINATIONS,
                "plot_error_bars_left": PLOT_ERROR_BARS_NOISY_TRAINING_LEFT,
                "plot_error_bars_right": PLOT_ERROR_BARS_NOISY_TRAINING_RIGHT,
                "exclude_zero_noise_from_right_panel": EXCLUDE_ZERO_NOISE_FROM_NOISY_TRAINING_RIGHT,
                "use_log_y_left": True,
                "figsize": list(FIGSIZE_TWO_PANEL),
                "subplots_adjust": SUBPLOTS_ADJUST_TWO_PANEL,
                "improvement_percent_formula": "100 * (original_ensemble_mse - noisy_trained_ensemble_mse) / original_ensemble_mse",
                "positive_improvement_meaning": "noisy training improves the prediction",
                "negative_improvement_meaning": "noisy training makes the prediction worse",
                "zero_noise_handling": "0% is kept in saved data but excluded from the right panel by default.",
                "saved_paths": {name: str(path) for name, path in paths.items()},
            },
        )
        paths["manifest_json"] = manifest_path

    print("Noisy-training comparison outputs saved:")
    for name, path in paths.items():
        print(f"  {name}: {path}")
    return paths


# ======================================================================================
# CLI
# ======================================================================================

FIGURE_RUNNERS = {
    "species-count": run_species_count_figure,
    "ensemble-comparison": run_ensemble_comparison_figure,
    "ensemble-individual-noise": run_ensemble_individual_noise_figure,
    "noisy-training": run_noisy_training_figure,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate PIC1 result plots from already-computed CSV files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--figure",
        choices=["all", *FIGURE_RUNNERS.keys()],
        default="all",
        help="Which figure family to generate.",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="When --figure all is used, continue if one figure's source CSVs are missing or invalid.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    selected = list(FIGURE_RUNNERS.items()) if args.figure == "all" else [(args.figure, FIGURE_RUNNERS[args.figure])]
    all_outputs: dict[str, dict[str, str]] = {}
    failures: dict[str, str] = {}

    for figure_name, runner in selected:
        print(f"\n=== Generating {figure_name} ===")
        try:
            paths = runner()
            all_outputs[figure_name] = {name: str(path) for name, path in paths.items()}
        except Exception as exc:
            failures[figure_name] = f"{type(exc).__name__}: {exc}"
            if args.figure != "all" or not args.skip_missing:
                raise
            print(f"SKIPPED {figure_name}: {exc}")

    summary = {
        "selected_figure": args.figure,
        "architecture": ARCHITECTURE,
        "outputs": all_outputs,
        "failures": failures,
    }
    summary_path = BASE_RESULTS_DIR / SCHEME / "PIC1_Result_Plots" / "plot_pic1_results_summary.json"
    write_json(summary_path, summary)

    print("\nPlot generation summary:")
    print(f"  summary_json: {summary_path}")
    for figure_name, paths in all_outputs.items():
        print(f"\n{figure_name}:")
        for name, path in paths.items():
            print(f"  {name}: {path}")
    if failures:
        print("\nFailures:")
        for figure_name, message in failures.items():
            print(f"  {figure_name}: {message}")


if __name__ == "__main__":
    main()
