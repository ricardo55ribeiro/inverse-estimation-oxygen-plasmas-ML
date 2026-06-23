from __future__ import annotations

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


# ======================================================================================
# MAIN SETTINGS
# ======================================================================================

SCHEME = "O2_novib"
OUTPUT_SCHEME = "O2_novib_noisy"
ARCHITECTURE = "30, 30, 30"

BASE_RESULTS_DIR = Path("Results_NN")
OUTPUT_ROOT = BASE_RESULTS_DIR / "PIC1_PowerPoint_Figures"

MEAN_COL = "test_mse_scaled_mean"
STD_COL = "test_mse_scaled_std"
RAW_METRIC_COL = "test_mse_scaled"

# The seven combinations used in the current paper/presentation figures.
REPRESENTATIVE_COMBINATIONS = [
    ["O2(a)", "O2(b)", "O2(Hz)"],
    ["O2(a)", "O3(X)"],
    ["O2(a)", "O2(b)", "O3(X)"],
    ["O2(a)", "O(3P)", "O3(X)"],
    ["O2(a)", "O2(b)"],
    ["O2(X)", "O2(b)", "O(3P)"],
    ["O2(X)", "O2(a)", "O(3P)"],
]

NOISE_LEVELS_PERCENT = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]

# Same color/marker scheme as your paper plots.
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

# Inner-subset styles for the individual-noise figure.
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

# --------------------------------------------------------------------------------------
# PowerPoint visibility settings.
# These are the main values to edit if you want the plots even heavier/lighter.
# --------------------------------------------------------------------------------------

DPI = 300
SAVE_PDF = True
SAVE_PNG = True
SAVE_SVG = True  # Useful for PowerPoint because it stays vector-scalable.

# 16:9 slide-friendly dimensions.
# 10% shorter than the previous PowerPoint versions.
FIGSIZE_SINGLE = (13.33, 6.30)
FIGSIZE_TWO_PANEL = (13.33, 6.21)
FIGSIZE_HORIZONTAL = (13.33, 6.30)

# Special paper-height colored version of the unperturbed representative-combination plot.
# This one is intentionally NOT shortened with the PowerPoint figures above.
FIGSIZE_FIGURE1_PAPER = (10.62, 3.72)

LINEWIDTH = 3.0
MARKERSIZE = 8.0
ERRORBAR_LINEWIDTH = 2.0
CAPSIZE = 4.5
CAPTHICK = 1.8
AXIS_LINEWIDTH = 1.8
GRID_LINEWIDTH_MAJOR = 0.85
GRID_LINEWIDTH_MINOR = 0.55
ZERO_LINEWIDTH = 2.0

AXIS_LABEL_SIZE = 19
TICK_LABEL_SIZE = 16
LEGEND_SIZE = 14
SMALL_LEGEND_SIZE = 13

# Keep missing plots from stopping the full script. Set to False if you prefer hard failures.
SKIP_MISSING_SOURCES = True

# Plot content toggles. These keep the same content as your current paper plots by default.
PLOT_ERROR_BARS_ARCHITECTURE = False
PLOT_ERROR_BARS_UNPERTURBED = True
PLOT_ERROR_BARS_ENSEMBLE_LEFT = True
PLOT_ERROR_BARS_ENSEMBLE_RIGHT = False
PLOT_ERROR_BARS_INDIVIDUAL = True
PLOT_ERROR_BARS_NOISY_TRAINING_LEFT = True
PLOT_ERROR_BARS_NOISY_TRAINING_RIGHT = True
EXCLUDE_ZERO_NOISE_FROM_NOISY_TRAINING_RIGHT = True

CLEAN_BASELINE_ID = "clean_baseline"


# ======================================================================================
# PATHS
# ======================================================================================

NOISE_PIC1_CSV_CANDIDATES = [
    BASE_RESULTS_DIR / SCHEME / "Noise_PIC1_Paper" / "Noise_MSE_Results" / "fullrun_noise_aggregate_summary.csv",
    BASE_RESULTS_DIR / SCHEME / "Ensemble_Noise_Error_Rankings" / "fullrun_noise_aggregate_summary.csv",
    BASE_RESULTS_DIR / SCHEME / "Ensemble_(Noise_Error_Rankings)" / "fullrun_noise_aggregate_summary.csv",
    BASE_RESULTS_DIR / SCHEME / "Noise_Error_Rankings" / "fullrun_noise_aggregate_summary.csv",
]

SINGLE_RANKING_CSV = (
    BASE_RESULTS_DIR / SCHEME / "Noise_Error_Rankings" / "fullrun_noise_aggregate_summary.csv"
)

ENSEMBLE_RANKING_CSV_CANDIDATES = [
    BASE_RESULTS_DIR / SCHEME / "Ensemble_Noise_Error_Rankings" / "fullrun_noise_aggregate_summary.csv",
    BASE_RESULTS_DIR / SCHEME / "Ensemble_(Noise_Error_Rankings)" / "fullrun_noise_aggregate_summary.csv",
]

ENSEMBLE_INDIVIDUAL_CSV_CANDIDATES = [
    BASE_RESULTS_DIR / SCHEME / "Ensemble_Individual_Noise" / "fullrun_ensemble_individual_noise_aggregate_summary.csv",
    BASE_RESULTS_DIR / SCHEME / "Ensemble_(Individual_Noise)" / "fullrun_ensemble_individual_noise_aggregate_summary.csv",
]

NOISY_TRAINING_EXPERIMENT_NAME = "NoisyTraining_PIC1_Robustness"
NOISY_TRAINED_ENSEMBLE_CSV = (
    BASE_RESULTS_DIR
    / OUTPUT_SCHEME
    / NOISY_TRAINING_EXPERIMENT_NAME
    / "Ensemble"
    / "Noise_MSE_Results"
    / "fullrun_noise_aggregate_summary.csv"
)

NOISY_TRAINING_PRECOMPUTED_COMPARISON_CSV = (
    BASE_RESULTS_DIR
    / OUTPUT_SCHEME
    / NOISY_TRAINING_EXPERIMENT_NAME
    / "Comparison_Against_Clean"
    / "ensemble"
    / "comparison_ensemble.csv"
)

ORIGINAL_ENSEMBLE_CSV_CANDIDATES = ENSEMBLE_RANKING_CSV_CANDIDATES


# ======================================================================================
# GENERAL HELPERS
# ======================================================================================


def print_skip(plot_name: str, exc: Exception) -> None:
    if SKIP_MISSING_SOURCES:
        print(f"SKIPPED {plot_name}: {exc}")
    else:
        raise exc


def normalize_architecture_text(value) -> str:
    """Normalize strings like '(30, 30, 30)' and '[30, 30, 30]' to '30, 30, 30'."""
    text = str(value).strip()
    text = text.strip("[]()")
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if parts and all(re.fullmatch(r"\d+", p) for p in parts):
        return ", ".join(parts)
    return str(value).strip()


def split_species_text(text) -> tuple[str, ...]:
    """Parse CSV species strings while not splitting the plus sign in O2+(X)."""
    if pd.isna(text):
        return tuple()
    s = str(text).strip()
    if not s or s.lower() in {"clean_baseline", "clean baseline"}:
        return tuple()

    s = s.replace("[", "").replace("]", "").replace("'", "").replace('"', "")
    # Split on commas or on plus signs surrounded by spaces. Do not split O2+(X).
    parts = re.split(r"\s*,\s*|\s+\+\s+", s)
    return tuple(part.strip() for part in parts if part.strip())


def species_key(species) -> tuple[str, ...]:
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
    return tuple(split_species_text(species))


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


def find_noise_pic1_csv() -> Path:
    for path in NOISE_PIC1_CSV_CANDIDATES:
        if path.exists():
            return path

    # Last fallback: if Noise_PIC1_Paper used a timestamped subfolder, use newest aggregate.
    search_root = BASE_RESULTS_DIR / SCHEME / "Noise_PIC1_Paper"
    if search_root.exists():
        matches = list(search_root.rglob("fullrun_noise_aggregate_summary.csv"))
        if matches:
            return max(matches, key=lambda p: p.stat().st_mtime)

    return find_first_existing(NOISE_PIC1_CSV_CANDIDATES, "representative noise aggregate CSV")


def safe_numeric(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def ensure_mean_std_columns(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    """
    Make sure a results table has test_mse_scaled_mean/std columns.
    If it is a raw table with test_mse_scaled, aggregate it.
    """
    df = df.copy()
    if "hidden_size" not in df.columns or "kept_species" not in df.columns or "noise_percent" not in df.columns:
        raise ValueError(f"{path} must contain hidden_size, kept_species, and noise_percent columns.")

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
    optional_first_cols = [
        "species_config_name",
        "num_species_kept",
        "architecture",
    ]
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


def load_representative_noise_df(path: Path, source_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = ensure_mean_std_columns(df, path)
    df = safe_numeric(df, [MEAN_COL, STD_COL, "noise_percent"])

    df = df[
        (df["hidden_size"].astype(str) == ARCHITECTURE)
        & noise_is_requested(df["noise_percent"])
    ].dropna(subset=["noise_percent", MEAN_COL]).copy()

    if df.empty:
        raise ValueError(f"No rows found in {path} for architecture {ARCHITECTURE!r}.")

    df["species_key"] = df["kept_species"].apply(species_key)
    wanted_keys = {species_key(combo) for combo in REPRESENTATIVE_COMBINATIONS}
    df = df[df["species_key"].isin(wanted_keys)].copy()
    if df.empty:
        raise ValueError(f"No representative combinations found in {path}.")

    # Average duplicates if present.
    if df.duplicated(["species_key", "noise_percent"]).any():
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


def make_yerr_for_log_axis(y, yerr, cap_fraction: float = 0.95):
    y = np.asarray(y, dtype=float)
    yerr = np.asarray(yerr, dtype=float)
    yerr = np.where(np.isnan(yerr), 0.0, yerr)
    yerr = np.where(yerr < 0.0, 0.0, yerr)
    lower = np.minimum(yerr, np.maximum(0.0, y * cap_fraction))
    upper = yerr
    return np.vstack([lower, upper])


def style_axis(ax, xlabel: str, ylabel: str, log_y: bool = False, labelpad_y: float = 6.0) -> None:
    ax.set_xlabel(xlabel, fontsize=AXIS_LABEL_SIZE, labelpad=8)
    ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_SIZE, labelpad=labelpad_y)

    if log_y:
        yvals = np.asarray([v for line in ax.lines for v in line.get_ydata()], dtype=float)
        if yvals.size == 0 or np.all(yvals > 0):
            ax.set_yscale("log")

    ax.tick_params(axis="both", which="major", labelsize=TICK_LABEL_SIZE, width=AXIS_LINEWIDTH, length=7)
    ax.tick_params(axis="both", which="minor", width=AXIS_LINEWIDTH * 0.8, length=4)
    for spine in ax.spines.values():
        spine.set_linewidth(AXIS_LINEWIDTH)

    ax.grid(True, which="major", alpha=0.34, linewidth=GRID_LINEWIDTH_MAJOR)
    if log_y:
        ax.grid(True, which="minor", axis="y", alpha=0.18, linewidth=GRID_LINEWIDTH_MINOR)


def save_figure(fig, output_dir: Path, basename: str) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    if SAVE_PDF:
        paths["pdf"] = output_dir / f"{basename}.pdf"
        fig.savefig(paths["pdf"], bbox_inches="tight")
    if SAVE_PNG:
        paths["png"] = output_dir / f"{basename}.png"
        fig.savefig(paths["png"], dpi=DPI, bbox_inches="tight")
    if SAVE_SVG:
        paths["svg"] = output_dir / f"{basename}.svg"
        fig.savefig(paths["svg"], bbox_inches="tight")
    plt.close(fig)
    return paths


def save_manifest(output_dir: Path, basename: str, payload: dict) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{basename}__manifest.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)
    return path


def line_or_errorbar(ax, x, y, yerr, *, color, marker, label=None, errorbars=True, log_y=False):
    if errorbars:
        yerr_plot = make_yerr_for_log_axis(y, yerr) if log_y else yerr
        ax.errorbar(
            x,
            y,
            yerr=yerr_plot,
            color=color,
            marker=marker,
            linestyle="-",
            linewidth=LINEWIDTH,
            markersize=MARKERSIZE,
            capsize=CAPSIZE,
            capthick=CAPTHICK,
            elinewidth=ERRORBAR_LINEWIDTH,
            label=label,
        )
    else:
        ax.plot(
            x,
            y,
            color=color,
            marker=marker,
            linestyle="-",
            linewidth=LINEWIDTH,
            markersize=MARKERSIZE,
            label=label,
        )


def legend_handles_for_combos() -> list[Line2D]:
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


# ======================================================================================
# 1) SINGLE-PANEL NOISE ROBUSTNESS PLOT
# ======================================================================================


def make_architecture_noise_plot_ppt() -> dict[str, Path]:
    plot_name = "single-panel architecture noise robustness"
    try:
        source_path = find_noise_pic1_csv()
        df = load_representative_noise_df(source_path, "representative_noise")

        output_dir = OUTPUT_ROOT / "01_architecture_noise_robustness"
        basename = "architecture_30_30_30__species_PIC1__ppt"

        fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)

        for combo in REPRESENTATIVE_COMBINATIONS:
            key = species_key(combo)
            group = df[df["species_key"] == key].sort_values("noise_percent")
            if group.empty:
                continue

            combo_tuple = tuple(combo)
            x = group["noise_percent"].to_numpy(float)
            y = group[MEAN_COL].to_numpy(float)
            yerr = group[STD_COL].fillna(0.0).to_numpy(float)

            line_or_errorbar(
                ax,
                x,
                y,
                yerr,
                color=COMBO_COLORS[combo_tuple],
                marker=COMBO_MARKERS[combo_tuple],
                label=species_label(combo),
                errorbars=PLOT_ERROR_BARS_ARCHITECTURE,
                log_y=True,
            )

        xticks = sorted(df["noise_percent"].unique())
        ax.set_xticks(xticks)
        ax.set_xticklabels([format_noise(x) for x in xticks])
        style_axis(ax, "Noise level (%)", "Scaled MSE", log_y=True)

        fig.legend(
            handles=legend_handles_for_combos(),
            loc="lower center",
            bbox_to_anchor=(0.5, 0.025),
            ncol=4,
            frameon=False,
            fontsize=SMALL_LEGEND_SIZE,
            handlelength=2.0,
            columnspacing=1.1,
            handletextpad=0.5,
            labelspacing=0.55,
        )
        fig.subplots_adjust(left=0.085, right=0.985, top=0.965, bottom=0.235)

        paths = save_figure(fig, output_dir, basename)
        data_path = output_dir / f"{basename}__data.csv"
        df.to_csv(data_path, index=False)
        paths["data_csv"] = data_path
        paths["manifest_json"] = save_manifest(
            output_dir,
            basename,
            {
                "source_csv": str(source_path),
                "architecture": ARCHITECTURE,
                "output_dir": str(output_dir),
                "style": "PowerPoint-ready thicker lines and larger labels",
                "saved_paths": {k: str(v) for k, v in paths.items()},
            },
        )
        return paths
    except Exception as exc:
        print_skip(plot_name, exc)
        return {}


# ======================================================================================
# 2) CLEAN / UNPERTURBED REPRESENTATIVE-COMBINATION RANKING PLOT
# ======================================================================================


def make_unperturbed_representative_plot_ppt() -> dict[str, Path]:
    plot_name = "unperturbed representative-combination plot"
    try:
        source_path = find_noise_pic1_csv()
        df = load_representative_noise_df(source_path, "representative_noise")
        df = df[np.isclose(df["noise_percent"].astype(float), 0.0)].copy()
        if df.empty:
            raise ValueError(f"No 0% noise rows found in {source_path}.")

        # Keep the same top-to-bottom order as the representative list.
        df = df.sort_values("combination_order", ascending=True).reset_index(drop=True)
        y_pos = np.arange(len(df))

        output_dir = OUTPUT_ROOT / "02_unperturbed_representative_combinations"
        basename = "representative_combinations_unperturbed_30_30_30_v2__ppt"

        fig, ax = plt.subplots(figsize=FIGSIZE_HORIZONTAL)

        for idx, row in df.iterrows():
            combo = ordered_species_tuple(row["species_key"])
            combo_tuple = tuple(combo)
            x = float(row[MEAN_COL])
            xerr = row.get(STD_COL, np.nan)
            xerr = 0.0 if pd.isna(xerr) else float(xerr)

            if PLOT_ERROR_BARS_UNPERTURBED:
                ax.errorbar(
                    [x],
                    [idx],
                    xerr=[[min(xerr, 0.95 * x)], [xerr]],
                    color=COMBO_COLORS[combo_tuple],
                    marker="o",
                    linestyle="none",
                    markersize=MARKERSIZE + 1.0,
                    capsize=CAPSIZE + 0.5,
                    capthick=CAPTHICK,
                    elinewidth=ERRORBAR_LINEWIDTH,
                )
            else:
                ax.plot(
                    [x],
                    [idx],
                    color=COMBO_COLORS[combo_tuple],
                    marker="o",
                    linestyle="none",
                    markersize=MARKERSIZE + 1.0,
                )

        ax.set_yticks(y_pos)
        ax.set_yticklabels([species_label(key, joiner=" + ") for key in df["species_key"]])
        ax.invert_yaxis()
        ax.set_xscale("log")
        style_axis(ax, "Scaled MSE", "", log_y=False)
        ax.tick_params(axis="y", labelsize=15)
        ax.grid(True, which="minor", axis="x", alpha=0.18, linewidth=GRID_LINEWIDTH_MINOR)

        # Extra space on the left for long species labels.
        fig.subplots_adjust(left=0.315, right=0.985, top=0.965, bottom=0.155)

        paths = save_figure(fig, output_dir, basename)
        data_path = output_dir / f"{basename}__data.csv"
        df.to_csv(data_path, index=False)
        paths["data_csv"] = data_path
        paths["manifest_json"] = save_manifest(
            output_dir,
            basename,
            {
                "source_csv": str(source_path),
                "architecture": ARCHITECTURE,
                "noise_percent": 0.0,
                "output_dir": str(output_dir),
                "saved_paths": {k: str(v) for k, v in paths.items()},
            },
        )
        return paths
    except Exception as exc:
        print_skip(plot_name, exc)
        return {}


def make_figure1_paper_colored_plot() -> dict[str, Path]:
    """
    Extra paper-height colored version of the unperturbed representative-combination plot.

    Output folder requested:
        Results_NN/PIC1_PowerPoint_Figures/Figure1_Paper/

    This reproduces the 0% noise representative-combinations figure, keeps the compact
    paper-height layout, and uses the same color mapping as the presentation figures.
    """
    plot_name = "Figure1_Paper colored representative-combination plot"
    output_dir = OUTPUT_ROOT / "Figure1_Paper"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        source_path = find_noise_pic1_csv()
        df = load_representative_noise_df(source_path, "representative_noise")
        df = df[np.isclose(df["noise_percent"].astype(float), 0.0)].copy()
        if df.empty:
            raise ValueError(f"No 0% noise rows found in {source_path}.")

        # Same top-to-bottom order as the representative list.
        df = df.sort_values("combination_order", ascending=True).reset_index(drop=True)
        y_pos = np.arange(len(df))

        basename = "representative_combinations_unperturbed_30_30_30_v2_colored"

        fig, ax = plt.subplots(figsize=FIGSIZE_FIGURE1_PAPER)

        # Paper-scale styling: compact figure height, but colored markers/error bars.
        paper_marker_size = 5.2
        paper_errorbar_lw = 1.15
        paper_capsize = 3.0
        paper_capthick = 1.05
        paper_axis_label_size = 12.5
        paper_tick_label_size = 10.8
        paper_axis_lw = 1.0

        for idx, row in df.iterrows():
            combo = ordered_species_tuple(row["species_key"])
            combo_tuple = tuple(combo)
            x = float(row[MEAN_COL])
            xerr = row.get(STD_COL, np.nan)
            xerr = 0.0 if pd.isna(xerr) else float(xerr)

            ax.errorbar(
                [x],
                [idx],
                xerr=[[min(xerr, 0.95 * x)], [xerr]],
                color=COMBO_COLORS[combo_tuple],
                marker="o",
                linestyle="none",
                markersize=paper_marker_size,
                capsize=paper_capsize,
                capthick=paper_capthick,
                elinewidth=paper_errorbar_lw,
            )

        ax.set_yticks(y_pos)
        ax.set_yticklabels([species_label(key, joiner=" + ") for key in df["species_key"]])
        ax.invert_yaxis()
        ax.set_xscale("log")

        ax.set_xlabel("Scaled MSE", fontsize=paper_axis_label_size, labelpad=5)
        ax.set_ylabel("")
        ax.tick_params(
            axis="both",
            which="major",
            labelsize=paper_tick_label_size,
            width=paper_axis_lw,
            length=5,
        )
        ax.tick_params(
            axis="both",
            which="minor",
            width=paper_axis_lw * 0.8,
            length=3,
        )
        for spine in ax.spines.values():
            spine.set_linewidth(paper_axis_lw)

        ax.grid(True, which="major", alpha=0.30, linewidth=0.55)
        ax.grid(True, which="minor", axis="x", alpha=0.16, linewidth=0.40)

        # Compact paper layout with enough left margin for the long species labels.
        fig.subplots_adjust(left=0.285, right=0.985, top=0.955, bottom=0.175)

        paths = save_figure(fig, output_dir, basename)
        data_path = output_dir / f"{basename}__data.csv"
        df.to_csv(data_path, index=False)
        paths["data_csv"] = data_path
        paths["manifest_json"] = save_manifest(
            output_dir,
            basename,
            {
                "source_csv": str(source_path),
                "architecture": ARCHITECTURE,
                "noise_percent": 0.0,
                "output_dir": str(output_dir),
                "figure_size": FIGSIZE_FIGURE1_PAPER,
                "style": "paper-height colored version",
                "saved_paths": {k: str(v) for k, v in paths.items()},
            },
        )
        return paths
    except Exception as exc:
        print_skip(plot_name, exc)
        return {}



# ======================================================================================
# 3) ENSEMBLE VS SINGLE-MODEL COMPARISON PLOT
# ======================================================================================


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

    comp["improvement_percent"] = 100.0 * (
        comp["single_mse_mean"] - comp["ensemble_mse"]
    ) / comp["single_mse_mean"]

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


def plot_combo_lines(ax, df: pd.DataFrame, y_col: str, yerr_col: str, *, errorbars: bool, log_y: bool, ylabel: str) -> None:
    for combo in REPRESENTATIVE_COMBINATIONS:
        key = species_key(combo)
        group = df[df["species_key"] == key].sort_values("noise_percent")
        if group.empty:
            continue
        combo_tuple = tuple(combo)
        x = group["noise_percent"].to_numpy(float)
        y = group[y_col].to_numpy(float)
        yerr = group[yerr_col].fillna(0.0).to_numpy(float) if yerr_col in group.columns else np.zeros_like(y)
        line_or_errorbar(
            ax,
            x,
            y,
            yerr,
            color=COMBO_COLORS[combo_tuple],
            marker=COMBO_MARKERS[combo_tuple],
            errorbars=errorbars,
            log_y=log_y,
        )

    xticks = sorted(df["noise_percent"].unique())
    ax.set_xticks(xticks)
    ax.set_xticklabels([format_noise(x) for x in xticks])
    style_axis(ax, "Noise level (%)", ylabel, log_y=log_y)


def make_ensemble_comparison_plot_ppt() -> dict[str, Path]:
    plot_name = "ensemble comparison plot"
    try:
        ensemble_path = find_first_existing(ENSEMBLE_RANKING_CSV_CANDIDATES, "ensemble ranking CSV")
        single_df = load_representative_noise_df(SINGLE_RANKING_CSV, "single_model_mean")
        ensemble_df = load_representative_noise_df(ensemble_path, "ensemble")
        comp = build_ensemble_comparison(single_df, ensemble_df)

        output_dir = OUTPUT_ROOT / "03_ensemble_comparison"
        basename = "ensemble_mse_and_improvement_errorbars__30_30_30__ppt"

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
        axes[1].axhline(0.0, color="gray", linestyle="--", linewidth=ZERO_LINEWIDTH, alpha=0.9)

        fig.legend(
            handles=legend_handles_for_combos(),
            loc="lower center",
            bbox_to_anchor=(0.5, 0.025),
            ncol=4,
            frameon=False,
            fontsize=SMALL_LEGEND_SIZE,
            handlelength=2.0,
            columnspacing=1.1,
            handletextpad=0.5,
            labelspacing=0.55,
        )
        fig.subplots_adjust(left=0.085, right=0.985, top=0.965, bottom=0.245, wspace=0.22)

        paths = save_figure(fig, output_dir, basename)
        data_path = output_dir / f"{basename}__data.csv"
        comp.to_csv(data_path, index=False)
        paths["data_csv"] = data_path
        paths["manifest_json"] = save_manifest(
            output_dir,
            basename,
            {
                "single_csv": str(SINGLE_RANKING_CSV),
                "ensemble_csv": str(ensemble_path),
                "architecture": ARCHITECTURE,
                "improvement_percent_formula": "100 * (single_model_mean - ensemble) / single_model_mean",
                "output_dir": str(output_dir),
                "saved_paths": {k: str(v) for k, v in paths.items()},
            },
        )
        return paths
    except Exception as exc:
        print_skip(plot_name, exc)
        return {}


# ======================================================================================
# 4) ENSEMBLE INDIVIDUAL-NOISE TWO-PANEL PLOT
# ======================================================================================


def find_ensemble_individual_noise_global_csv() -> Optional[Path]:
    """Find the global aggregate CSV that is produced by NeuralNet_3Ks_Noise_Sensitivity_Ensemble.py."""
    for path in ENSEMBLE_INDIVIDUAL_CSV_CANDIDATES:
        if path.exists():
            return path

    # Robust fallback: search the O2_novib results tree. This catches timestamped or
    # slightly renamed folders while still preferring the ensemble individual-noise files.
    search_root = BASE_RESULTS_DIR / SCHEME
    if not search_root.exists():
        return None

    matches = list(search_root.rglob("fullrun_ensemble_individual_noise_aggregate_summary.csv"))
    if not matches:
        return None

    def score(path: Path):
        text = str(path).replace("\\", "/")
        preferred = (
            "Ensemble_Individual_Noise" in text
            or "Ensemble_(Individual_Noise)" in text
        )
        return (1 if preferred else 0, path.stat().st_mtime)

    return max(matches, key=score)


def combine_existing_individual_noise_aggregate_tables() -> tuple[pd.DataFrame, str]:
    """
    Fallback when the global fullrun aggregate is missing.

    The ensemble sensitivity code also writes species-level files named
    ensemble_individual_noise_aggregate_summary.csv and subset-level files named
    noise_aggregate_summary.csv. This function combines those existing aggregate
    files so the PowerPoint plot can still be regenerated without rerunning the NN.
    """
    roots = [
        BASE_RESULTS_DIR / SCHEME / "Ensemble_Individual_Noise",
        BASE_RESULTS_DIR / SCHEME / "Ensemble_(Individual_Noise)",
    ]

    species_level_paths: list[Path] = []
    for root in roots:
        if root.exists():
            species_level_paths.extend(root.rglob("ensemble_individual_noise_aggregate_summary.csv"))

    # Prefer species-level aggregate files because they are already combined cleanly per
    # input-species combination. Avoid including global files here.
    species_level_paths = [
        p for p in species_level_paths
        if p.name == "ensemble_individual_noise_aggregate_summary.csv"
    ]

    if species_level_paths:
        frames = [pd.read_csv(path) for path in sorted(set(species_level_paths))]
        source = "combined species-level aggregate files:\n" + "\n".join(str(p) for p in sorted(set(species_level_paths)))
        return pd.concat(frames, ignore_index=True, sort=False), source

    subset_level_paths: list[Path] = []
    for root in roots:
        if root.exists():
            subset_level_paths.extend(root.rglob("noise_aggregate_summary.csv"))

    if subset_level_paths:
        frames = [pd.read_csv(path) for path in sorted(set(subset_level_paths))]
        source = "combined subset-level aggregate files:\n" + "\n".join(str(p) for p in sorted(set(subset_level_paths)))
        return pd.concat(frames, ignore_index=True, sort=False), source

    searched = "\n".join(f"  - {root}" for root in roots)
    raise FileNotFoundError(
        "Could not find ensemble individual-noise aggregate data. Searched for:\n"
        "  - fullrun_ensemble_individual_noise_aggregate_summary.csv\n"
        "  - ensemble_individual_noise_aggregate_summary.csv\n"
        "  - noise_aggregate_summary.csv\n"
        "under these roots:\n"
        f"{searched}\n\n"
        "Run NeuralNet_3Ks_Noise_Sensitivity_Ensemble.py or plot_ensemble_individual_noise.py first."
    )


def normalize_individual_noise_aggregate(df: pd.DataFrame, source_label: str) -> pd.DataFrame:
    """Normalize all possible aggregate-table variants into the columns needed for plotting."""
    df = df.copy()

    required_base = ["hidden_size", "kept_species", "noisy_subset_id", "noise_percent"]
    missing_base = [col for col in required_base if col not in df.columns]
    if missing_base:
        raise ValueError(f"{source_label} is missing required columns: {', '.join(missing_base)}")

    if MEAN_COL not in df.columns:
        if RAW_METRIC_COL not in df.columns:
            raise ValueError(f"{source_label} contains neither {MEAN_COL!r} nor {RAW_METRIC_COL!r}.")

        # Raw per-repeat table fallback: aggregate it exactly as the original sensitivity
        # workflow does for this plot.
        first_cols = [
            "scheme",
            "experiment_name",
            "species_config_name",
            "kept_species",
            "num_species_kept",
            "input_size",
            "output_size",
            "hidden_size",
            "ensemble_seed_count",
            "ensemble_seeds",
            "noisy_subset_id",
            "noisy_subset_label",
            "noisy_species_positions",
            "noisy_species_names",
            "num_noisy_species",
            "noise_std",
            "noise_percent",
            "noise_label",
        ]
        group_cols = [col for col in first_cols if col in df.columns]
        df[RAW_METRIC_COL] = pd.to_numeric(df[RAW_METRIC_COL], errors="coerce")
        df = (
            df.dropna(subset=[RAW_METRIC_COL])
            .groupby(group_cols, as_index=False)
            .agg(**{MEAN_COL: (RAW_METRIC_COL, "mean"), STD_COL: (RAW_METRIC_COL, "std")})
        )

    if STD_COL not in df.columns:
        df[STD_COL] = np.nan

    # Some old files may not have the label/name helper columns. Reconstruct what is
    # needed from the available data.
    if "noisy_subset_label" not in df.columns:
        if "noisy_species_names" in df.columns:
            df["noisy_subset_label"] = df["noisy_species_names"].apply(
                lambda value: "Clean baseline"
                if str(value).strip().lower() in {"", "clean_baseline", "clean baseline", "nan"}
                else " + ".join(split_species_text(value))
            )
        else:
            df["noisy_subset_label"] = df["noisy_subset_id"].astype(str)

    if "noisy_species_names" not in df.columns:
        df["noisy_species_names"] = df["noisy_subset_label"]

    df["hidden_size"] = df["hidden_size"].apply(normalize_architecture_text)
    df["noise_percent"] = pd.to_numeric(df["noise_percent"], errors="coerce")
    df[MEAN_COL] = pd.to_numeric(df[MEAN_COL], errors="coerce")
    df[STD_COL] = pd.to_numeric(df[STD_COL], errors="coerce")

    df = df[(df["hidden_size"] == ARCHITECTURE) & noise_is_requested(df["noise_percent"])].copy()
    df = df.dropna(subset=["noise_percent", MEAN_COL])
    if df.empty:
        raise ValueError(f"No individual-noise rows found in {source_label} for architecture {ARCHITECTURE!r}.")

    df["kept_species_key"] = df["kept_species"].apply(species_key)

    # The official generator stores both noisy_subset_label (e.g. O2(a) + O2(b)) and
    # noisy_species_names (e.g. O2(a), O2(b)). Prefer noisy_species_names because it is
    # unambiguous; fall back to the label for older files.
    df["noisy_subset_key"] = df["noisy_species_names"].apply(species_key)
    empty_subset_mask = df["noisy_subset_key"].apply(lambda key: len(key) == 0)
    if empty_subset_mask.any():
        df.loc[empty_subset_mask, "noisy_subset_key"] = df.loc[empty_subset_mask, "noisy_subset_label"].apply(species_key)

    # If duplicate aggregate rows exist because we combined fallback files, average them.
    group_cols = [
        "kept_species_key",
        "noisy_subset_key",
        "hidden_size",
        "noise_percent",
        "noisy_subset_id",
    ]
    if df.duplicated(group_cols).any():
        df = (
            df.groupby(group_cols, as_index=False)
            .agg(
                kept_species=("kept_species", "first"),
                noisy_subset_label=("noisy_subset_label", "first"),
                noisy_species_names=("noisy_species_names", "first"),
                **{MEAN_COL: (MEAN_COL, "mean"), STD_COL: (STD_COL, "mean")},
            )
        )

    return df.reset_index(drop=True)


def load_individual_noise_aggregate() -> tuple[pd.DataFrame, str]:
    global_path = find_ensemble_individual_noise_global_csv()
    if global_path is not None:
        raw_df = pd.read_csv(global_path)
        source_label = str(global_path)
    else:
        raw_df, source_label = combine_existing_individual_noise_aggregate_tables()

    df = normalize_individual_noise_aggregate(raw_df, source_label)
    return df, source_label

def get_individual_panel_data(df: pd.DataFrame, kept_species: list[str]):
    kept_key = species_key(kept_species)
    panel_df = df[df["kept_species_key"] == kept_key].copy()
    if panel_df.empty:
        raise ValueError(f"No rows found for kept species: {subset_label(kept_species)}")

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
    _, nonbaseline_df, baseline_x, baseline_y, baseline_yerr = get_individual_panel_data(df, kept_species)

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
        line_or_errorbar(
            ax,
            x,
            y,
            yerr,
            color=SUBSET_COLORS[subset_tuple],
            marker=SUBSET_MARKERS[subset_tuple],
            errorbars=PLOT_ERROR_BARS_INDIVIDUAL,
            log_y=True,
        )

    xticks = sorted(df["noise_percent"].unique())
    ax.set_xticks(xticks)
    ax.set_xticklabels([format_noise(x) for x in xticks])
    style_axis(ax, "Noise level (%)", "Scaled MSE", log_y=True)


def make_individual_noise_plot_ppt() -> dict[str, Path]:
    plot_name = "ensemble individual-noise two-panel plot"
    try:
        df, source_path = load_individual_noise_aggregate()
        output_dir = OUTPUT_ROOT / "04_ensemble_individual_noise"
        basename = "ensemble_individual_noise_two_panel__30_30_30__ppt"

        left_combination = ["O2(a)", "O2(b)", "O2(Hz)"]
        right_combination = ["O2(a)", "O2(b)"]

        fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_TWO_PANEL, constrained_layout=False)
        plot_individual_panel(axes[0], df, left_combination)
        plot_individual_panel(axes[1], df, right_combination)

        legend_handles = [
            Line2D(
                [0],
                [0],
                color=SUBSET_COLORS[tuple(subset)],
                marker=SUBSET_MARKERS[tuple(subset)],
                linestyle="-",
                linewidth=LINEWIDTH,
                markersize=MARKERSIZE,
                label=subset_label(subset),
            )
            for subset in INNER_SUBSET_ORDER
        ]
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.025),
            ncol=4,
            frameon=False,
            fontsize=SMALL_LEGEND_SIZE,
            handlelength=2.0,
            columnspacing=1.1,
            handletextpad=0.5,
            labelspacing=0.55,
        )
        fig.subplots_adjust(left=0.085, right=0.985, top=0.965, bottom=0.245, wspace=0.22)

        paths = save_figure(fig, output_dir, basename)
        data_path = output_dir / f"{basename}__data.csv"
        df.to_csv(data_path, index=False)
        paths["data_csv"] = data_path
        paths["manifest_json"] = save_manifest(
            output_dir,
            basename,
            {
                "source_csv": str(source_path),
                "architecture": ARCHITECTURE,
                "left_combination": left_combination,
                "right_combination": right_combination,
                "output_dir": str(output_dir),
                "saved_paths": {k: str(v) for k, v in paths.items()},
            },
        )
        return paths
    except Exception as exc:
        print_skip(plot_name, exc)
        return {}



# ======================================================================================
# 5) NOISY-TRAINING ENSEMBLE COMPARISON PLOT
# ======================================================================================


def load_noisy_trained_aggregate() -> pd.DataFrame:
    path = NOISY_TRAINED_ENSEMBLE_CSV
    if not path.exists():
        raise FileNotFoundError(f"Missing noisy-trained ensemble CSV: {path}")
    df = load_representative_noise_df(path, "noisy_trained_ensemble")
    return df


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
    path = NOISY_TRAINING_PRECOMPUTED_COMPARISON_CSV
    if not path.exists():
        return None, None

    df = pd.read_csv(path)
    required = ["hidden_size", "kept_species", "noise_percent", "clean_trained_test_mse_scaled_mean", "noisy_trained_test_mse_scaled_mean"]
    if any(col not in df.columns for col in required):
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


def make_noisy_training_comparison_plot_ppt() -> dict[str, Path]:
    plot_name = "noisy-training ensemble comparison plot"
    try:
        noisy_df = load_noisy_trained_aggregate()
        comp, comparison_source_path = load_noisy_training_comparison_from_precomputed()
        if comp is None:
            comp, comparison_source_path = build_noisy_training_comparison_from_aggregates(noisy_df)

        output_dir = OUTPUT_ROOT / "05_noisy_training_ensemble_comparison"
        basename = "noisy_training_ensemble_mse_and_improvement__30_30_30__ppt"

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

        right_df = comp.copy()
        if EXCLUDE_ZERO_NOISE_FROM_NOISY_TRAINING_RIGHT:
            right_df = right_df[right_df["noise_percent"].astype(float) > 0.0].copy()
        plot_combo_lines(
            axes[1],
            right_df,
            "improvement_percent",
            "improvement_percent_std",
            errorbars=PLOT_ERROR_BARS_NOISY_TRAINING_RIGHT,
            log_y=False,
            ylabel="Improvement (%)",
        )
        axes[1].axhline(0.0, color="gray", linestyle="--", linewidth=ZERO_LINEWIDTH, alpha=0.9)

        fig.legend(
            handles=legend_handles_for_combos(),
            loc="lower center",
            bbox_to_anchor=(0.5, 0.025),
            ncol=4,
            frameon=False,
            fontsize=SMALL_LEGEND_SIZE,
            handlelength=2.0,
            columnspacing=1.1,
            handletextpad=0.5,
            labelspacing=0.55,
        )
        fig.subplots_adjust(left=0.085, right=0.985, top=0.965, bottom=0.245, wspace=0.22)

        paths = save_figure(fig, output_dir, basename)
        data_path = output_dir / f"{basename}__data.csv"
        comp.to_csv(data_path, index=False)
        paths["data_csv"] = data_path
        paths["manifest_json"] = save_manifest(
            output_dir,
            basename,
            {
                "noisy_trained_ensemble_csv": str(NOISY_TRAINED_ENSEMBLE_CSV),
                "comparison_source_used": str(comparison_source_path),
                "architecture": ARCHITECTURE,
                "improvement_percent_formula": "100 * (original_ensemble_mse - noisy_trained_ensemble_mse) / original_ensemble_mse",
                "exclude_zero_noise_from_right_panel": EXCLUDE_ZERO_NOISE_FROM_NOISY_TRAINING_RIGHT,
                "output_dir": str(output_dir),
                "saved_paths": {k: str(v) for k, v in paths.items()},
            },
        )
        return paths
    except Exception as exc:
        print_skip(plot_name, exc)
        return {}


# ======================================================================================
# MAIN
# ======================================================================================


def main() -> None:
    plt.rcParams.update(
        {
            "font.size": TICK_LABEL_SIZE,
            "axes.linewidth": AXIS_LINEWIDTH,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    all_paths: dict[str, dict[str, str]] = {}
    plot_jobs = [
        ("architecture_noise_robustness", make_architecture_noise_plot_ppt),
        ("unperturbed_representative_combinations", make_unperturbed_representative_plot_ppt),
        ("figure1_paper_colored", make_figure1_paper_colored_plot),
        ("ensemble_comparison", make_ensemble_comparison_plot_ppt),
        ("ensemble_individual_noise", make_individual_noise_plot_ppt),
        ("noisy_training_ensemble_comparison", make_noisy_training_comparison_plot_ppt),
    ]

    for name, func in plot_jobs:
        paths = func()
        if paths:
            all_paths[name] = {kind: str(path) for kind, path in paths.items()}

    summary_path = OUTPUT_ROOT / "powerpoint_plot_generation_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "architecture": ARCHITECTURE,
                "output_root": str(OUTPUT_ROOT),
                "style_summary": {
                    "linewidth": LINEWIDTH,
                    "markersize": MARKERSIZE,
                    "errorbar_linewidth": ERRORBAR_LINEWIDTH,
                    "axis_label_size": AXIS_LABEL_SIZE,
                    "tick_label_size": TICK_LABEL_SIZE,
                    "legend_size": LEGEND_SIZE,
                    "figsize_single": FIGSIZE_SINGLE,
                    "figsize_two_panel": FIGSIZE_TWO_PANEL,
                    "figsize_horizontal": FIGSIZE_HORIZONTAL,
                    "figsize_figure1_paper": FIGSIZE_FIGURE1_PAPER,
                    "save_pdf": SAVE_PDF,
                    "save_png": SAVE_PNG,
                    "save_svg": SAVE_SVG,
                },
                "saved_outputs": all_paths,
            },
            f,
            indent=4,
        )

    print("\nPowerPoint-ready outputs:")
    if not all_paths:
        print("  No plots were generated. Check the skipped messages above and the CSV paths in this script.")
    for plot_name, paths in all_paths.items():
        print(f"\n{plot_name}:")
        for kind, path in paths.items():
            print(f"  {kind}: {path}")
    print(f"\nSummary: {summary_path}")


if __name__ == "__main__":
    main()
