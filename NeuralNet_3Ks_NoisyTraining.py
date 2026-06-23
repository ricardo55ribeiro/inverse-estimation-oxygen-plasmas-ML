from tqdm import tqdm

import argparse
import copy
import json
import pickle
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from sklearn import preprocessing
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split

from src.NeuralNetworkModels import NeuralNet
from src.config import dict as dictionary


# ======================================================================================
# SETUP
# ======================================================================================

# Data are still read from the original O2_novib entry in src.config.
# The noisy-trained weights and results are intentionally stored under O2_novib_noisy.
DATA_SCHEME = "O2_novib"
OUTPUT_SCHEME = "O2_novib_noisy"
EXPERIMENT_NAME = "NoisyTraining_PIC1_Robustness"

BASE_RESULTS_DIR = Path("Results_NN")
SAVED_WEIGHTS_ROOT = Path("saved_weights")

RESULTS_ROOT = BASE_RESULTS_DIR / OUTPUT_SCHEME / EXPERIMENT_NAME
SINGLE_RESULTS_ROOT = RESULTS_ROOT / "Single_Models_Average" / "Noise_MSE_Results"
ENSEMBLE_RESULTS_ROOT = RESULTS_ROOT / "Ensemble" / "Noise_MSE_Results"
COMPARISON_RESULTS_ROOT = RESULTS_ROOT / "Comparison_Against_Clean"

# Main architecture used for the noisy-training robustness study.
ARCHITECTURES = [
    (30, 30, 30),
]

# O2(X) / O2(a) / O2(b) / O2(Hz) / O2+(X) / O(3P) 
# O(1D) / O+(gnd) / O-(gnd) / O3(X) / O3(exc)
SPECIES_CONFIGS = [
    ["O2(a)", "O2(b)", "O2(Hz)"],
    ["O2(a)", "O3(X)"],
    ["O2(a)", "O2(b)", "O3(X)"],
    ["O2(a)", "O(3P)", "O3(X)"],
    ["O2(a)", "O2(b)"],
    ["O2(X)", "O2(b)", "O(3P)"],
    ["O2(X)", "O2(a)", "O(3P)"],
]

# Test-time noise levels used in the final robustness plot.
EVAL_NOISE_STDS = [0.0, 0.005, 0.01, 0.02, 0.05, 0.10]
NOISE_REPEATS = 20
NOISE_BASE_SEED = 12345

# Train-time uncertainty/noise levels.
# Balanced distribution: keep clean data present, but force robustness across the noisy range.
TRAIN_NOISE_STDS = [0.0, 0.005, 0.01, 0.02, 0.05, 0.10]
TRAIN_NOISE_PROBS = [0.25, 0.15, 0.15, 0.15, 0.15, 0.15]
NOISY_TRAINING_BASE_SEED = 54321

# Mixed validation loss. Fixed noisy validation copies are generated once per model/seed.
# This keeps early stopping stable while selecting a model that is not only clean-optimized.
VAL_NOISE_STDS = [0.0, 0.01, 0.05, 0.10]
VAL_NOISE_WEIGHTS = [0.40, 0.20, 0.20, 0.20]
VAL_NOISE_BASE_SEED = 67890

# Seeds used for NN weights / train-validation split / training shuffle.
SEEDS = list(range(32, 52))
ENSEMBLE_SEEDS = SEEDS

# Model/training setup. The patience is intentionally larger than the clean-training scripts
# because noisy training gives a less smooth validation trajectory.
ACTIVATION = "tanh"
LEARNING_RATE = 0.0001
BATCH_SIZE = 16
MAX_EPOCHS = 5000
PATIENCE = 200
VAL_SPLIT = 0.1
VERBOSE_EPOCH_LOSSES = False

# Noise handling, matching the existing robustness/ranking workflow.
RESAMPLE_NEGATIVE_DENSITIES = True
MAX_NOISE_RESAMPLE_ATTEMPTS = 1000
CLIP_NEGATIVE_DENSITIES = False

# Append/reuse behavior for evaluation rows.
# If True, existing rows in the current output folders are reused and not recomputed.
SKIP_EXISTING_EVALUATIONS = True

# Plot/output options.
PLOT_MAIN_METRIC = "test_mse_scaled"
PLOT_MAIN_METRIC_LABEL = "Scaled MSE"
PLOT_USE_LOG_Y = True
PLOT_SAVE_PNG = True
PLOT_SAVE_PDF = True
PLOT_DPI = 300
PLOT_SHOW_ERRORBARS = True
PLOT_MARKER_SIZE = 3.0
PLOT_LINEWIDTH = 1.15
PLOT_MAX_LEGEND_COLUMNS = 4

PLOT_COLORS = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # green
    "#CC79A7",  # reddish purple
    "#D55E00",  # vermillion
    "#56B4E9",  # sky blue
    "#000000",  # black
]
PLOT_MARKERS = ["o", "s", "^", "v", "D", "P", "X"]
PLOT_SPECIES_ORDER = [", ".join(s) for s in SPECIES_CONFIGS]

# Automatic comparison with existing clean-trained results under Results_NN/O2_novib.
SAVE_COMPARISON_AGAINST_CLEAN = True
CLEAN_RESULTS_SEARCH_ROOT = BASE_RESULTS_DIR / DATA_SCHEME


# ======================================================================================
# SPECIES MAP AND DATASET
# ======================================================================================

SPECIES_MAP = {
    "O2(X)": [0, 11],
    "O2(a)": [1, 12],
    "O2(b)": [2, 13],
    "O2(Hz)": [3, 14],
    "O2+(X)": [4, 15],
    "O(3P)": [5, 16],
    "O(1D)": [6, 17],
    "O+(gnd)": [7, 18],
    "O-(gnd)": [8, 19],
    "O3(X)": [9, 20],
    "O3(exc)": [10, 21],
}

ALL_SPECIES = list(SPECIES_MAP.keys())


class LoadMultiPressureDatasetTorch(torch.utils.data.Dataset):
    def __init__(
        self,
        src_file,
        nspecies,
        num_pressure_conditions,
        react_idx=None,
        m_rows=None,
        columns=None,
        scaler_input=None,
        scaler_output=None,
    ):
        self.num_pressure_conditions = num_pressure_conditions

        all_data = np.loadtxt(
            src_file,
            max_rows=m_rows,
            usecols=columns,
            delimiter=None,
            comments="#",
            skiprows=0,
            dtype=np.float64,
        )
        all_data = np.atleast_2d(all_data)

        if len(all_data) % num_pressure_conditions != 0:
            raise ValueError(
                f"The number of rows in {src_file} ({len(all_data)}) is not divisible by "
                f"num_pressure_conditions ({num_pressure_conditions})."
            )

        ncolumns = all_data.shape[1]
        x_columns = np.arange(ncolumns - nspecies, ncolumns, 1)
        y_columns = react_idx
        if react_idx is None:
            y_columns = np.arange(0, ncolumns - nspecies, 1)

        raw_x_data = all_data[:, x_columns].copy()
        raw_y_data = all_data[:, y_columns].copy()

        x_data = raw_x_data.copy()
        y_data = raw_y_data * 1e30

        x_data = x_data.reshape(num_pressure_conditions, -1, x_data.shape[1])
        y_data = y_data.reshape(num_pressure_conditions, -1, y_data.shape[1])

        raw_x_data = raw_x_data.reshape(num_pressure_conditions, -1, raw_x_data.shape[1])
        raw_y_data = raw_y_data.reshape(num_pressure_conditions, -1, raw_y_data.shape[1])

        self.scaler_input = scaler_input or [
            preprocessing.MaxAbsScaler() for _ in range(num_pressure_conditions)
        ]
        self.scaler_output = scaler_output or [
            preprocessing.MaxAbsScaler() for _ in range(num_pressure_conditions)
        ]

        for i in range(num_pressure_conditions):
            if scaler_input is None:
                self.scaler_input[i].fit(x_data[i])
            if scaler_output is None:
                self.scaler_output[i].fit(y_data[i])

            x_data[i] = self.scaler_input[i].transform(x_data[i])
            y_data[i] = self.scaler_output[i].transform(y_data[i])

        x_data = np.transpose(x_data, (1, 0, 2)).reshape(
            -1,
            self.num_pressure_conditions * x_data.shape[-1],
        )
        y_data = y_data[0]

        raw_x_data = np.transpose(raw_x_data, (1, 0, 2)).reshape(
            -1,
            self.num_pressure_conditions * raw_x_data.shape[-1],
        )
        raw_y_data = raw_y_data[0]

        self.x_data = torch.from_numpy(x_data).float()
        self.y_data = torch.from_numpy(y_data).float()

        self.x_data_unscaled = torch.from_numpy(raw_x_data).float()
        self.y_data_unscaled = torch.from_numpy(raw_y_data).float()

    def get_unscaled_data(self):
        return self.x_data_unscaled, self.y_data_unscaled

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return len(self.x_data)

    def get_data(self):
        return self.x_data, self.y_data


# ======================================================================================
# GENERAL HELPERS
# ======================================================================================


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def safe_path_token(text):
    return (
        str(text)
        .replace("/", "-")
        .replace("\\", "-")
        .replace(":", "-")
        .replace("*", "")
        .replace("?", "")
        .replace('"', "")
        .replace("<", "")
        .replace(">", "")
        .replace("|", "")
        .replace(" ", "")
    )


def species_config_to_name(kept_species):
    return f"{len(kept_species)}__" + "__".join(safe_path_token(sp) for sp in kept_species)


def arch_to_folder_name(hidden_size):
    return ", ".join(map(str, hidden_size))


def arch_to_file_token(hidden_size_or_text):
    if isinstance(hidden_size_or_text, (tuple, list)):
        text = arch_to_folder_name(hidden_size_or_text)
    else:
        text = str(hidden_size_or_text)
    return text.replace(",", "_").replace(" ", "").replace("__", "_")


def noise_label(noise_std):
    return f"{100.0 * float(noise_std):g}%"


def normalize_noise_std_for_key(value):
    return int(round(float(value) * 1_000_000_000_000))


def normalize_probability_vector(values):
    values = np.asarray(values, dtype=np.float64)
    total = float(np.sum(values))
    if total <= 0.0:
        raise ValueError("Probability vector sum must be positive.")
    return (values / total).tolist()


def validate_species_config(kept_species):
    if not kept_species:
        raise ValueError("Each species configuration must contain at least one species.")

    unknown = [sp for sp in kept_species if sp not in SPECIES_MAP]
    if unknown:
        raise ValueError(
            "Unknown species name(s): "
            + ", ".join(unknown)
            + "\nValid names are: "
            + ", ".join(ALL_SPECIES)
        )

    if len(set(kept_species)) != len(kept_species):
        raise ValueError(f"Duplicate species found in configuration: {kept_species}")


def validate_all_species_configs(species_configs):
    for kept_species in species_configs:
        validate_species_config(kept_species)


def validate_architectures(architectures):
    if not architectures:
        raise ValueError("ARCHITECTURES cannot be empty.")
    for arch in architectures:
        if not isinstance(arch, (tuple, list)) or not arch:
            raise ValueError(f"Invalid architecture: {arch}")
        if any(int(n) <= 0 for n in arch):
            raise ValueError(f"Architecture values must be positive: {arch}")


def get_kept_columns(kept_species, num_pressure_conditions):
    kept_cols = []
    for p in range(num_pressure_conditions):
        for species in kept_species:
            if p >= len(SPECIES_MAP[species]):
                raise ValueError(
                    f"SPECIES_MAP for {species} does not contain pressure condition index {p}."
                )
            kept_cols.append(SPECIES_MAP[species][p])
    return kept_cols


def get_species_indices_within_condition(kept_species):
    return [SPECIES_MAP[species][0] for species in kept_species]


def build_feature_names(species_names, num_pressure_conditions):
    return [
        f"{species}_p{p + 1}"
        for p in range(num_pressure_conditions)
        for species in species_names
    ]


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def moving_average(x, window=25):
    x = np.asarray(x, dtype=float)
    if window <= 1 or len(x) == 0:
        return x
    kernel = np.ones(window) / window
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    x_padded = np.pad(x, (pad_left, pad_right), mode="edge")
    return np.convolve(x_padded, kernel, mode="valid")


def save_json(filepath, obj):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(obj, f, indent=4)


def load_json(filepath):
    with open(filepath, "r") as f:
        return json.load(f)


def save_pickle(filepath, obj):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(filepath):
    with open(filepath, "rb") as f:
        return pickle.load(f)


def save_loss_history_csv(output_dir, history):
    if history is None:
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame({"epoch": np.arange(1, len(history["train_loss"]) + 1)})
    for key, values in history.items():
        df[key] = values
        if key.endswith("loss"):
            df[f"{key}_smooth"] = moving_average(values, window=25)
    df.to_csv(output_dir / "loss_history.csv", index=False)


def load_loss_history_csv(model_dir):
    path = Path(model_dir) / "loss_history.csv"
    if not path.exists():
        return None

    df = pd.read_csv(path)
    if "train_loss" not in df.columns or "val_loss" not in df.columns:
        return None

    history = {}
    for col in df.columns:
        if col == "epoch" or col.endswith("_smooth"):
            continue
        history[col] = df[col].astype(float).tolist()
    return history


# ======================================================================================
# SAVED-WEIGHTS CACHE AND DATASET LOADING
# ======================================================================================


def saved_scheme_root(output_scheme=OUTPUT_SCHEME):
    return SAVED_WEIGHTS_ROOT / output_scheme


def saved_species_root(output_scheme, kept_species):
    return saved_scheme_root(output_scheme) / species_config_to_name(kept_species)


def saved_model_dir(output_scheme, kept_species, seed, hidden_size):
    return saved_species_root(output_scheme, kept_species) / f"seed_{seed:04d}" / arch_to_folder_name(hidden_size)


def saved_model_path(output_scheme, kept_species, seed, hidden_size):
    return saved_model_dir(output_scheme, kept_species, seed, hidden_size) / "model.pth"


def apply_species_subset(dataset, kept_species, num_pressure_conditions):
    kept_cols = get_kept_columns(kept_species, num_pressure_conditions)
    dataset.x_data = dataset.x_data[:, kept_cols]
    dataset.x_data_unscaled = dataset.x_data_unscaled[:, kept_cols]
    return dataset


def load_datasets_for_species(data_scheme, kept_species, scaler_input=None, scaler_output=None):
    validate_species_config(kept_species)

    src_file_train = dictionary[data_scheme]["main_dataset"]
    src_file_test = dictionary[data_scheme]["main_dataset_test"]
    nspecies = dictionary[data_scheme]["n_densities"]
    num_pressure_conditions = dictionary[data_scheme]["n_conditions"]

    dataset_train = LoadMultiPressureDatasetTorch(
        src_file_train,
        nspecies,
        num_pressure_conditions,
        react_idx=dictionary[data_scheme]["k_columns"],
        scaler_input=scaler_input,
        scaler_output=scaler_output,
    )

    dataset_test = LoadMultiPressureDatasetTorch(
        src_file_test,
        nspecies,
        num_pressure_conditions,
        react_idx=dictionary[data_scheme]["k_columns"],
        scaler_input=dataset_train.scaler_input,
        scaler_output=dataset_train.scaler_output,
    )

    apply_species_subset(dataset_train, kept_species, num_pressure_conditions)
    apply_species_subset(dataset_test, kept_species, num_pressure_conditions)

    return dataset_train, dataset_test


def save_species_level_metadata(data_scheme, output_scheme, kept_species, dataset_train, dataset_test):
    root = saved_species_root(output_scheme, kept_species)
    root.mkdir(parents=True, exist_ok=True)

    num_pressure_conditions = dictionary[data_scheme]["n_conditions"]
    feature_names = build_feature_names(kept_species, num_pressure_conditions)

    save_pickle(
        root / "scalers.pkl",
        {
            "scaler_input": dataset_train.scaler_input,
            "scaler_output": dataset_train.scaler_output,
        },
    )

    x_train, y_train = dataset_train.get_data()
    x_test, y_test = dataset_test.get_data()

    species_info = {
        "data_scheme": data_scheme,
        "output_scheme": output_scheme,
        "train_file": dictionary[data_scheme]["main_dataset"],
        "test_file": dictionary[data_scheme]["main_dataset_test"],
        "k_columns": list(dictionary[data_scheme]["k_columns"]),
        "num_pressure_conditions": int(num_pressure_conditions),
        "num_species_total": int(len(ALL_SPECIES)),
        "num_species_kept": int(len(kept_species)),
        "species_all": ALL_SPECIES,
        "kept_species": kept_species,
        "removed_species": [sp for sp in ALL_SPECIES if sp not in kept_species],
        "feature_names": feature_names,
        "x_train_shape": list(x_train.shape),
        "y_train_shape": list(y_train.shape),
        "x_test_shape": list(x_test.shape),
        "y_test_shape": list(y_test.shape),
        "training_mode": "noise_aware_input_augmentation",
    }
    save_json(root / "species_info.json", species_info)


def load_species_scalers(output_scheme, kept_species):
    path = saved_species_root(output_scheme, kept_species) / "scalers.pkl"
    if path.exists():
        return load_pickle(path)
    return None


def load_datasets_with_saved_scalers(data_scheme, output_scheme, kept_species):
    scalers = load_species_scalers(output_scheme, kept_species)

    if scalers is None:
        dataset_train, dataset_test = load_datasets_for_species(data_scheme, kept_species)
        save_species_level_metadata(data_scheme, output_scheme, kept_species, dataset_train, dataset_test)
        return dataset_train, dataset_test

    dataset_train, dataset_test = load_datasets_for_species(
        data_scheme,
        kept_species,
        scaler_input=scalers["scaler_input"],
        scaler_output=scalers["scaler_output"],
    )
    save_species_level_metadata(data_scheme, output_scheme, kept_species, dataset_train, dataset_test)
    return dataset_train, dataset_test


def expected_noisy_model_cache_metadata(
    data_scheme,
    output_scheme,
    kept_species,
    hidden_size,
    seed,
    activation,
    input_size,
    output_size,
):
    return {
        "data_scheme": data_scheme,
        "output_scheme": output_scheme,
        "kept_species": list(kept_species),
        "hidden_size": list(hidden_size),
        "seed": int(seed),
        "activation": activation,
        "input_size": int(input_size),
        "output_size": int(output_size),
        "k_columns": list(dictionary[data_scheme]["k_columns"]),
        "num_pressure_conditions": int(dictionary[data_scheme]["n_conditions"]),
        "training_mode": "noise_aware_input_augmentation",
        "train_noise_stds": [float(v) for v in TRAIN_NOISE_STDS],
        "train_noise_probs": [float(v) for v in normalize_probability_vector(TRAIN_NOISE_PROBS)],
        "val_noise_stds": [float(v) for v in VAL_NOISE_STDS],
        "val_noise_weights": [float(v) for v in normalize_probability_vector(VAL_NOISE_WEIGHTS)],
        "noisy_training_base_seed": int(NOISY_TRAINING_BASE_SEED),
        "val_noise_base_seed": int(VAL_NOISE_BASE_SEED),
        "learning_rate": float(LEARNING_RATE),
        "batch_size": int(BATCH_SIZE),
        "max_epochs": int(MAX_EPOCHS),
        "patience": int(PATIENCE),
        "val_split": float(VAL_SPLIT),
        "resample_negative_densities": bool(RESAMPLE_NEGATIVE_DENSITIES),
        "clip_negative_densities": bool(CLIP_NEGATIVE_DENSITIES),
        "noisy_training_version": 1,
    }


def cache_metadata_mismatches(info, expected):
    mismatches = []
    for key, expected_value in expected.items():
        current_value = info.get(key)
        if key in {"kept_species", "hidden_size", "k_columns"} and current_value is not None:
            current_value = list(current_value)
        if key in {"train_noise_stds", "train_noise_probs", "val_noise_stds", "val_noise_weights"} and current_value is not None:
            current_value = [float(v) for v in current_value]
            expected_value = [float(v) for v in expected_value]
            if len(current_value) == len(expected_value) and np.allclose(current_value, expected_value, rtol=0, atol=1e-15):
                continue
        if current_value != expected_value:
            mismatches.append((key, current_value, expected_value))
    return mismatches


# ======================================================================================
# NOISE AND SCALING HELPERS
# ======================================================================================


def make_noisy_inputs_unscaled(x_clean_unscaled, noise_std, rng):
    x_clean_unscaled = np.asarray(x_clean_unscaled, dtype=np.float64)
    noise_std = float(noise_std)

    if noise_std == 0.0:
        return x_clean_unscaled.copy()

    multiplicative_noise = rng.normal(
        loc=0.0,
        scale=noise_std,
        size=x_clean_unscaled.shape,
    )
    x_noisy = x_clean_unscaled * (1.0 + multiplicative_noise)

    if RESAMPLE_NEGATIVE_DENSITIES:
        negative_mask = x_noisy < 0.0
        attempts = 0

        while np.any(negative_mask):
            attempts += 1
            if attempts > MAX_NOISE_RESAMPLE_ATTEMPTS:
                raise RuntimeError(
                    f"Failed to generate non-negative noisy densities after "
                    f"{MAX_NOISE_RESAMPLE_ATTEMPTS} resampling attempts. "
                    f"noise_std={noise_std}"
                )

            new_noise = rng.normal(
                loc=0.0,
                scale=noise_std,
                size=int(np.sum(negative_mask)),
            )
            x_noisy[negative_mask] = x_clean_unscaled[negative_mask] * (1.0 + new_noise)
            negative_mask = x_noisy < 0.0

    elif CLIP_NEGATIVE_DENSITIES:
        x_noisy = np.clip(x_noisy, 0.0, None)

    return x_noisy


def transform_selected_unscaled_to_scaled(
    x_unscaled_selected,
    kept_species,
    scaler_input,
    num_pressure_conditions,
    nspecies,
):
    x_unscaled_selected = np.asarray(x_unscaled_selected, dtype=np.float64)
    n_samples = x_unscaled_selected.shape[0]
    n_kept = len(kept_species)
    species_indices = get_species_indices_within_condition(kept_species)

    expected_features = num_pressure_conditions * n_kept
    if x_unscaled_selected.shape[1] != expected_features:
        raise ValueError(
            f"Expected x_unscaled_selected to have {expected_features} columns "
            f"({num_pressure_conditions} conditions x {n_kept} species), "
            f"but got {x_unscaled_selected.shape[1]}."
        )

    x_scaled_selected = np.zeros_like(x_unscaled_selected, dtype=np.float64)

    for p in range(num_pressure_conditions):
        start = p * n_kept
        end = (p + 1) * n_kept

        selected_block = x_unscaled_selected[:, start:end]
        full_block = np.zeros((n_samples, nspecies), dtype=np.float64)
        full_block[:, species_indices] = selected_block

        transformed_full_block = scaler_input[p].transform(full_block)
        x_scaled_selected[:, start:end] = transformed_full_block[:, species_indices]

    return x_scaled_selected


def single_model_noise_rng_seed(seed, noise_std, noise_repeat):
    return (
        int(NOISE_BASE_SEED)
        + int(seed) * 1_000_000
        + int(noise_repeat) * 10_000
        + int(round(float(noise_std) * 1_000_000))
    )


def ensemble_noise_rng_seed(noise_std, noise_repeat):
    # No NN seed dependence: every model in the ensemble sees the same noisy realization.
    return (
        int(NOISE_BASE_SEED)
        + int(noise_repeat) * 10_000
        + int(round(float(noise_std) * 1_000_000))
    )


# ======================================================================================
# TRAINING / EVALUATION
# ======================================================================================


def predict_scaled(model, x_scaled_np):
    model.eval()
    x_tensor = torch.from_numpy(np.asarray(x_scaled_np, dtype=np.float32))
    with torch.no_grad():
        outputs = model(x_tensor).cpu().numpy()
    return outputs


def compute_scaled_metrics(targets_scaled, outputs_scaled):
    targets_scaled = np.asarray(targets_scaled, dtype=np.float64)
    outputs_scaled = np.asarray(outputs_scaled, dtype=np.float64)

    squared_errors = (outputs_scaled - targets_scaled) ** 2
    metrics = {
        "test_mse_scaled": float(np.mean(squared_errors)),
        "test_rmse_scaled": float(np.sqrt(np.mean(squared_errors))),
    }

    for i in range(targets_scaled.shape[1]):
        mse_i = float(np.mean(squared_errors[:, i]))
        metrics[f"k{i + 1}_mse_scaled"] = mse_i
        metrics[f"k{i + 1}_rmse_scaled"] = float(np.sqrt(mse_i))

    return metrics


def clean_evaluate_model(model, dataset_test):
    x_test, y_test = dataset_test.get_data()
    outputs_scaled = predict_scaled(model, x_test.numpy())
    return compute_scaled_metrics(y_test.numpy(), outputs_scaled)


def make_validation_tensors(
    dataset_train,
    val_indices,
    kept_species,
    seed,
):
    nspecies = dictionary[DATA_SCHEME]["n_densities"]
    num_pressure_conditions = dictionary[DATA_SCHEME]["n_conditions"]
    x_train_unscaled, _ = dataset_train.get_unscaled_data()
    x_val_unscaled = x_train_unscaled[val_indices].numpy()

    val_inputs_by_noise = []
    for noise_std in VAL_NOISE_STDS:
        rng_seed = (
            int(VAL_NOISE_BASE_SEED)
            + int(seed) * 1_000_000
            + int(round(float(noise_std) * 1_000_000))
        )
        rng = np.random.default_rng(rng_seed)
        x_noisy_unscaled = make_noisy_inputs_unscaled(x_val_unscaled, noise_std, rng)
        x_noisy_scaled = transform_selected_unscaled_to_scaled(
            x_noisy_unscaled,
            kept_species=kept_species,
            scaler_input=dataset_train.scaler_input,
            num_pressure_conditions=num_pressure_conditions,
            nspecies=nspecies,
        )
        val_inputs_by_noise.append(torch.from_numpy(x_noisy_scaled.astype(np.float32)))

    return val_inputs_by_noise


def train_model_noise_aware(
    model,
    criterion,
    optimizer,
    dataset_train,
    kept_species,
    seed,
    num_epochs=100,
    patience=5,
    batch_size=16,
    val_split=0.1,
    verbose_epoch_losses=False,
):
    train_len = int((1.0 - val_split) * len(dataset_train))
    val_len = len(dataset_train) - train_len
    if train_len <= 0 or val_len <= 0:
        raise ValueError("The train/validation split produced an empty subset.")

    split_generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(
        dataset_train,
        [train_len, val_len],
        generator=split_generator,
    )

    train_indices = np.asarray(train_subset.indices, dtype=np.int64)
    val_indices = np.asarray(val_subset.indices, dtype=np.int64)

    x_train_unscaled_all, _ = dataset_train.get_unscaled_data()
    _, y_train_scaled_all = dataset_train.get_data()
    y_val_scaled = y_train_scaled_all[val_indices]

    val_inputs_by_noise = make_validation_tensors(
        dataset_train=dataset_train,
        val_indices=val_indices,
        kept_species=kept_species,
        seed=seed,
    )
    val_weights = normalize_probability_vector(VAL_NOISE_WEIGHTS)
    train_noise_probs = normalize_probability_vector(TRAIN_NOISE_PROBS)

    nspecies = dictionary[DATA_SCHEME]["n_densities"]
    num_pressure_conditions = dictionary[DATA_SCHEME]["n_conditions"]

    best_model_wts = copy.deepcopy(model.state_dict())
    min_val_loss = np.inf
    epochs_no_improve = 0

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_loss_clean": [],
    }
    for noise_std in VAL_NOISE_STDS:
        if float(noise_std) != 0.0:
            token = f"val_loss_noise_{100.0 * float(noise_std):g}_percent".replace(".", "p")
            history[token] = []

    noise_rng = np.random.default_rng(int(NOISY_TRAINING_BASE_SEED) + int(seed) * 1_000_000)

    for epoch in range(num_epochs):
        model.train()
        train_loss_sum = 0.0
        num_train_seen = 0

        epoch_shuffle_rng = np.random.default_rng(int(seed) * 10_000_000 + epoch)
        shuffled_train_indices = train_indices.copy()
        epoch_shuffle_rng.shuffle(shuffled_train_indices)

        for start in range(0, len(shuffled_train_indices), batch_size):
            batch_indices = shuffled_train_indices[start:start + batch_size]
            x_batch_unscaled = x_train_unscaled_all[batch_indices].numpy()
            y_batch = y_train_scaled_all[batch_indices]

            noise_std = float(noise_rng.choice(TRAIN_NOISE_STDS, p=train_noise_probs))
            x_noisy_unscaled = make_noisy_inputs_unscaled(x_batch_unscaled, noise_std, noise_rng)
            x_noisy_scaled = transform_selected_unscaled_to_scaled(
                x_noisy_unscaled,
                kept_species=kept_species,
                scaler_input=dataset_train.scaler_input,
                num_pressure_conditions=num_pressure_conditions,
                nspecies=nspecies,
            )
            x_batch = torch.from_numpy(x_noisy_scaled.astype(np.float32))

            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            batch_size_actual = int(len(batch_indices))
            train_loss_sum += float(loss.item()) * batch_size_actual
            num_train_seen += batch_size_actual

        train_loss = train_loss_sum / max(num_train_seen, 1)

        model.eval()
        val_losses = []
        with torch.no_grad():
            for x_val in val_inputs_by_noise:
                outputs = model(x_val)
                val_losses.append(float(criterion(outputs, y_val_scaled).item()))

        val_loss = float(sum(w * loss for w, loss in zip(val_weights, val_losses)))
        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["val_loss_clean"].append(float(val_losses[0]))

        for idx, noise_std in enumerate(VAL_NOISE_STDS):
            if float(noise_std) != 0.0:
                token = f"val_loss_noise_{100.0 * float(noise_std):g}_percent".replace(".", "p")
                history[token].append(float(val_losses[idx]))

        if verbose_epoch_losses:
            print(
                f"Epoch {epoch + 1}, Training loss: {train_loss}, "
                f"Mixed validation loss: {val_loss}, Clean validation loss: {val_losses[0]}"
            )

        if val_loss < min_val_loss:
            epochs_no_improve = 0
            min_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                model.load_state_dict(best_model_wts)
                return model, history

    model.load_state_dict(best_model_wts)
    return model, history


def get_or_train_noisy_model(
    data_scheme,
    output_scheme,
    kept_species,
    hidden_size,
    seed,
    activation,
    learning_rate,
    batch_size,
    max_epochs,
    patience,
    val_split,
    dataset_train,
    dataset_test,
    verbose_epoch_losses=False,
):
    x_train, y_train = dataset_train.get_data()
    input_size = int(x_train.shape[1])
    output_size = int(y_train.shape[1])

    model_dir = saved_model_dir(output_scheme, kept_species, seed, hidden_size)
    model_path = model_dir / "model.pth"
    info_path = model_dir / "model_info.json"

    expected = expected_noisy_model_cache_metadata(
        data_scheme=data_scheme,
        output_scheme=output_scheme,
        kept_species=kept_species,
        hidden_size=hidden_size,
        seed=seed,
        activation=activation,
        input_size=input_size,
        output_size=output_size,
    )

    if model_path.exists() and info_path.exists():
        info = load_json(info_path)
        mismatches = cache_metadata_mismatches(info, expected)

        if not mismatches:
            model = NeuralNet(input_size, output_size, hidden_size, activ_f=activation)
            state_dict = torch.load(model_path, map_location="cpu")
            model.load_state_dict(state_dict)
            model.eval()

            loss_history = load_loss_history_csv(model_dir)

            record = {
                "reused_saved_weights": True,
                "saved_weights_path": str(model_path),
                "training_time_s": 0.0,
                "cached_training_time_s": float(info.get("training_time_s", 0.0)),
                "epochs_ran": int(info.get("epochs_ran", 0)),
                "best_epoch": int(info.get("best_epoch", 0)),
                "final_train_loss": float(info.get("final_train_loss", np.nan)),
                "final_val_loss": float(info.get("final_val_loss", np.nan)),
                "best_val_loss": float(info.get("best_val_loss", np.nan)),
            }
            return model, info, loss_history, record

        print("Saved noisy-trained weights found but metadata does not match the current run. Retraining:")
        for key, current_value, expected_value in mismatches:
            print(f"  {key}: cached={current_value} | expected={expected_value}")

    model_dir.mkdir(parents=True, exist_ok=True)
    set_global_seed(seed)

    model = NeuralNet(input_size, output_size, hidden_size, activ_f=activation)
    criterion = MSELoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    start = time.time()
    model, loss_history = train_model_noise_aware(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        dataset_train=dataset_train,
        kept_species=kept_species,
        seed=seed,
        num_epochs=max_epochs,
        patience=patience,
        batch_size=batch_size,
        val_split=val_split,
        verbose_epoch_losses=verbose_epoch_losses,
    )
    end = time.time()

    clean_metrics = clean_evaluate_model(model, dataset_test)
    training_time_s = float(end - start)

    torch.save(model.state_dict(), model_path)
    save_loss_history_csv(model_dir, loss_history)

    info = {
        **expected,
        "split_seed": int(seed),
        "shuffle_seed": int(seed),
        "weight_seed": int(seed),
        "depth": int(len(hidden_size)),
        "num_parameters": int(count_parameters(model)),
        "num_species_total": int(len(ALL_SPECIES)),
        "num_species_kept": int(len(kept_species)),
        "removed_species": [sp for sp in ALL_SPECIES if sp not in kept_species],
        "feature_names": build_feature_names(kept_species, dictionary[data_scheme]["n_conditions"]),
        "learning_rate": float(learning_rate),
        "batch_size": int(batch_size),
        "max_epochs": int(max_epochs),
        "patience": int(patience),
        "val_split": float(val_split),
        "epochs_ran": int(len(loss_history["train_loss"])),
        "best_epoch": int(np.argmin(loss_history["val_loss"]) + 1),
        "final_train_loss": float(loss_history["train_loss"][-1]),
        "final_val_loss": float(loss_history["val_loss"][-1]),
        "final_clean_val_loss": float(loss_history["val_loss_clean"][-1]),
        "best_val_loss": float(min(loss_history["val_loss"])),
        "training_time_s": training_time_s,
        **clean_metrics,
    }

    save_json(info_path, info)
    save_json(model_dir / "clean_metrics.json", clean_metrics)

    record = {
        "reused_saved_weights": False,
        "saved_weights_path": str(model_path),
        "training_time_s": training_time_s,
        "cached_training_time_s": training_time_s,
        "epochs_ran": int(len(loss_history["train_loss"])),
        "best_epoch": int(np.argmin(loss_history["val_loss"]) + 1),
        "final_train_loss": float(loss_history["train_loss"][-1]),
        "final_val_loss": float(loss_history["val_loss"][-1]),
        "best_val_loss": float(min(loss_history["val_loss"])),
    }

    return model, info, loss_history, record


# ======================================================================================
# ROBUSTNESS EVALUATION
# ======================================================================================


def run_noise_for_single_model(
    model,
    dataset_test,
    kept_species,
    hidden_size,
    seed,
    noise_std,
    noise_repeat,
    training_record,
):
    nspecies = dictionary[DATA_SCHEME]["n_densities"]
    num_pressure_conditions = dictionary[DATA_SCHEME]["n_conditions"]

    x_test_unscaled, _ = dataset_test.get_unscaled_data()
    _, y_test_scaled = dataset_test.get_data()

    rng_seed = single_model_noise_rng_seed(seed, noise_std, noise_repeat)
    rng = np.random.default_rng(rng_seed)

    x_noisy_unscaled = make_noisy_inputs_unscaled(
        x_test_unscaled.numpy(),
        noise_std=noise_std,
        rng=rng,
    )

    x_noisy_scaled = transform_selected_unscaled_to_scaled(
        x_noisy_unscaled,
        kept_species=kept_species,
        scaler_input=dataset_test.scaler_input,
        num_pressure_conditions=num_pressure_conditions,
        nspecies=nspecies,
    )

    outputs_scaled = predict_scaled(model, x_noisy_scaled)
    metrics = compute_scaled_metrics(y_test_scaled.numpy(), outputs_scaled)

    row = {
        "scheme": OUTPUT_SCHEME,
        "data_scheme": DATA_SCHEME,
        "experiment_name": EXPERIMENT_NAME,
        "evaluation_mode": "single_seed",
        "aggregation_definition": "mean/std later across neural-network seeds and noise repeats",
        "training_mode": "noise_aware_input_augmentation",
        "species_config_name": species_config_to_name(kept_species),
        "kept_species": ", ".join(kept_species),
        "num_species_kept": int(len(kept_species)),
        "input_size": int(x_noisy_scaled.shape[1]),
        "output_size": int(y_test_scaled.shape[1]),
        "hidden_size": arch_to_folder_name(hidden_size),
        "seed": int(seed),
        "noise_repeat": int(noise_repeat),
        "noise_std": float(noise_std),
        "noise_percent": float(100.0 * float(noise_std)),
        "noise_label": noise_label(noise_std),
        "noise_rng_seed": int(rng_seed),
        "reused_saved_weights": bool(training_record.get("reused_saved_weights", False)),
        "saved_weights_path": training_record.get("saved_weights_path", ""),
        "epochs_ran": int(training_record.get("epochs_ran", 0)),
        "best_epoch": int(training_record.get("best_epoch", 0)),
        "training_time_s": float(training_record.get("training_time_s", 0.0)),
        "cached_training_time_s": float(training_record.get("cached_training_time_s", 0.0)),
        **metrics,
    }
    return row


def predict_ensemble_scaled(models_by_seed, x_scaled_np):
    if not models_by_seed:
        raise ValueError("models_by_seed cannot be empty.")

    seeds = sorted(int(seed) for seed in models_by_seed.keys())
    outputs_by_seed = []
    for seed in seeds:
        outputs_by_seed.append(predict_scaled(models_by_seed[seed], x_scaled_np))

    outputs_by_seed = np.stack(outputs_by_seed, axis=0).astype(np.float64)
    outputs_mean = np.mean(outputs_by_seed, axis=0)
    outputs_std = np.std(outputs_by_seed, axis=0, ddof=1) if len(seeds) > 1 else np.zeros_like(outputs_mean)
    return outputs_mean, outputs_std, outputs_by_seed, seeds


def compute_individual_seed_metric_summary(targets_scaled, outputs_by_seed):
    targets_scaled = np.asarray(targets_scaled, dtype=np.float64)
    outputs_by_seed = np.asarray(outputs_by_seed, dtype=np.float64)

    per_seed_mse = []
    per_seed_rmse = []
    for outputs_scaled in outputs_by_seed:
        squared_errors = (outputs_scaled - targets_scaled) ** 2
        mse = float(np.mean(squared_errors))
        per_seed_mse.append(mse)
        per_seed_rmse.append(float(np.sqrt(mse)))

    per_seed_mse = np.asarray(per_seed_mse, dtype=np.float64)
    per_seed_rmse = np.asarray(per_seed_rmse, dtype=np.float64)

    return {
        "individual_seed_test_mse_scaled_mean": float(np.mean(per_seed_mse)),
        "individual_seed_test_mse_scaled_std": float(np.std(per_seed_mse, ddof=1)) if len(per_seed_mse) > 1 else 0.0,
        "individual_seed_test_mse_scaled_min": float(np.min(per_seed_mse)),
        "individual_seed_test_mse_scaled_max": float(np.max(per_seed_mse)),
        "individual_seed_test_rmse_scaled_mean": float(np.mean(per_seed_rmse)),
        "individual_seed_test_rmse_scaled_std": float(np.std(per_seed_rmse, ddof=1)) if len(per_seed_rmse) > 1 else 0.0,
        "individual_seed_test_rmse_scaled_min": float(np.min(per_seed_rmse)),
        "individual_seed_test_rmse_scaled_max": float(np.max(per_seed_rmse)),
    }


def compute_ensemble_uncertainty_summary(outputs_std):
    outputs_std = np.asarray(outputs_std, dtype=np.float64)

    metrics = {
        "ensemble_prediction_std_scaled_mean": float(np.mean(outputs_std)),
        "ensemble_prediction_std_scaled_max": float(np.max(outputs_std)),
    }
    for i in range(outputs_std.shape[1]):
        metrics[f"k{i + 1}_ensemble_prediction_std_scaled_mean"] = float(np.mean(outputs_std[:, i]))
        metrics[f"k{i + 1}_ensemble_prediction_std_scaled_max"] = float(np.max(outputs_std[:, i]))
    return metrics


def run_noise_for_seed_ensemble(
    models_by_seed,
    training_records_by_seed,
    dataset_test,
    kept_species,
    hidden_size,
    noise_std,
    noise_repeat,
):
    nspecies = dictionary[DATA_SCHEME]["n_densities"]
    num_pressure_conditions = dictionary[DATA_SCHEME]["n_conditions"]

    x_test_unscaled, _ = dataset_test.get_unscaled_data()
    _, y_test_scaled = dataset_test.get_data()

    rng_seed = ensemble_noise_rng_seed(noise_std, noise_repeat)
    rng = np.random.default_rng(rng_seed)

    x_noisy_unscaled = make_noisy_inputs_unscaled(
        x_test_unscaled.numpy(),
        noise_std=noise_std,
        rng=rng,
    )

    x_noisy_scaled = transform_selected_unscaled_to_scaled(
        x_noisy_unscaled,
        kept_species=kept_species,
        scaler_input=dataset_test.scaler_input,
        num_pressure_conditions=num_pressure_conditions,
        nspecies=nspecies,
    )

    outputs_mean_scaled, outputs_std_scaled, outputs_by_seed_scaled, ensemble_seeds = predict_ensemble_scaled(
        models_by_seed,
        x_noisy_scaled,
    )

    ensemble_metrics = compute_scaled_metrics(y_test_scaled.numpy(), outputs_mean_scaled)
    individual_summary = compute_individual_seed_metric_summary(y_test_scaled.numpy(), outputs_by_seed_scaled)
    uncertainty_summary = compute_ensemble_uncertainty_summary(outputs_std_scaled)

    reused_flags = [
        bool(training_records_by_seed.get(seed, {}).get("reused_saved_weights", False))
        for seed in ensemble_seeds
    ]
    saved_paths = [
        str(training_records_by_seed.get(seed, {}).get("saved_weights_path", ""))
        for seed in ensemble_seeds
    ]

    row = {
        "scheme": OUTPUT_SCHEME,
        "data_scheme": DATA_SCHEME,
        "experiment_name": EXPERIMENT_NAME,
        "evaluation_mode": "seed_ensemble",
        "ensemble_definition": "mean prediction over neural-network seeds before computing MSE; noise repeats kept separate",
        "training_mode": "noise_aware_input_augmentation",
        "species_config_name": species_config_to_name(kept_species),
        "kept_species": ", ".join(kept_species),
        "num_species_kept": int(len(kept_species)),
        "input_size": int(x_noisy_scaled.shape[1]),
        "output_size": int(y_test_scaled.shape[1]),
        "hidden_size": arch_to_folder_name(hidden_size),
        "ensemble_seed_values": ";".join(str(seed) for seed in ensemble_seeds),
        "num_ensemble_seeds": int(len(ensemble_seeds)),
        "noise_repeat": int(noise_repeat),
        "noise_std": float(noise_std),
        "noise_percent": float(100.0 * float(noise_std)),
        "noise_label": noise_label(noise_std),
        "noise_rng_seed": int(rng_seed),
        "num_reused_saved_weights": int(sum(reused_flags)),
        "fraction_reused_saved_weights": float(np.mean(reused_flags)) if reused_flags else np.nan,
        "all_saved_weights_reused": bool(all(reused_flags)) if reused_flags else False,
        "saved_weights_paths": "|".join(saved_paths),
        **ensemble_metrics,
        **individual_summary,
        **uncertainty_summary,
    }
    return row


# ======================================================================================
# RESULTS MANAGEMENT
# ======================================================================================


def result_key_from_values(hidden_size_text, seed, noise_std, noise_repeat, mode):
    if mode == "single_seed":
        return (
            str(hidden_size_text),
            int(seed),
            normalize_noise_std_for_key(noise_std),
            int(noise_repeat),
        )
    if mode == "seed_ensemble":
        return (
            str(hidden_size_text),
            normalize_noise_std_for_key(noise_std),
            int(noise_repeat),
        )
    raise ValueError(f"Unknown mode: {mode}")


def result_key_from_row(row, mode):
    if mode == "single_seed":
        return result_key_from_values(
            row["hidden_size"],
            row["seed"],
            row["noise_std"],
            row["noise_repeat"],
            mode,
        )
    return result_key_from_values(
        row["hidden_size"],
        None,
        row["noise_std"],
        row["noise_repeat"],
        mode,
    )


def species_results_folder(noise_results_root, kept_species):
    return Path(noise_results_root) / species_config_to_name(kept_species)


def species_noise_results_path(noise_results_root, kept_species):
    return species_results_folder(noise_results_root, kept_species) / "noise_results.csv"


def load_existing_species_results(noise_results_root, kept_species):
    path = species_noise_results_path(noise_results_root, kept_species)
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as exc:
        print(f"Warning: could not read existing results at {path}: {exc}")
        return pd.DataFrame()


def existing_keys_for_species(existing_df, mode):
    if existing_df.empty:
        return set()
    required = ["hidden_size", "noise_std", "noise_repeat"]
    if mode == "single_seed":
        required.append("seed")
    if any(col not in existing_df.columns for col in required):
        return set()
    return {result_key_from_row(row, mode) for _, row in existing_df.iterrows()}


def deduplicate_results(df, mode):
    if df.empty:
        return df.copy()

    df = df.copy()
    df["__noise_std_key"] = df["noise_std"].apply(normalize_noise_std_for_key)

    duplicate_key = [
        "scheme",
        "experiment_name",
        "species_config_name",
        "hidden_size",
        "noise_repeat",
        "__noise_std_key",
    ]
    if mode == "single_seed":
        duplicate_key.append("seed")

    required = [col for col in duplicate_key if col != "__noise_std_key"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        print("Warning: cannot fully de-duplicate because columns are missing: " + ", ".join(missing))
        return df.drop(columns=["__noise_std_key"])

    df = df.drop_duplicates(subset=duplicate_key, keep="last")
    return df.drop(columns=["__noise_std_key"])


def sort_noise_results(df, mode):
    if df.empty:
        return df.copy()

    preferred_cols = [
        "num_species_kept",
        "species_config_name",
        "hidden_size",
    ]
    if mode == "single_seed":
        preferred_cols.append("seed")
    preferred_cols.extend(["noise_std", "noise_repeat"])
    sort_cols = [col for col in preferred_cols if col in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)
    return df


def aggregate_noise_results(df, mode):
    if df.empty:
        return df.copy()

    group_cols = [
        "scheme",
        "data_scheme",
        "experiment_name",
        "evaluation_mode",
        "training_mode",
        "species_config_name",
        "kept_species",
        "num_species_kept",
        "input_size",
        "output_size",
        "hidden_size",
        "noise_std",
        "noise_percent",
        "noise_label",
    ]
    group_cols = [col for col in group_cols if col in df.columns]

    numeric_cols = set(df.select_dtypes(include=[np.number]).columns)
    metric_cols = [col for col in df.columns if "_scaled" in col and col in numeric_cols]
    if not metric_cols:
        raise ValueError("Cannot aggregate noise results. No numeric '*_scaled' metric columns were found.")

    agg_dict = {col: ["mean", "std", "min", "max"] for col in metric_cols}
    agg_dict["noise_repeat"] = ["nunique", "count"]

    if mode == "single_seed" and "seed" in df.columns:
        agg_dict["seed"] = ["nunique"]
        if "reused_saved_weights" in df.columns:
            work_df = df.copy()
            work_df["reused_saved_weights_int"] = work_df["reused_saved_weights"].astype(int)
            agg_dict["reused_saved_weights_int"] = ["mean", "sum"]
        else:
            work_df = df.copy()
    else:
        work_df = df.copy()
        if "num_ensemble_seeds" in df.columns:
            agg_dict["num_ensemble_seeds"] = ["first", "min", "max"]
        if "ensemble_seed_values" in df.columns:
            agg_dict["ensemble_seed_values"] = ["first"]
        if "fraction_reused_saved_weights" in df.columns:
            agg_dict["fraction_reused_saved_weights"] = ["mean", "min", "max"]
        if "num_reused_saved_weights" in df.columns:
            agg_dict["num_reused_saved_weights"] = ["mean", "min", "max"]
        if "all_saved_weights_reused" in df.columns:
            work_df["all_saved_weights_reused_int"] = work_df["all_saved_weights_reused"].astype(int)
            agg_dict["all_saved_weights_reused_int"] = ["mean", "sum"]

    agg = work_df.groupby(group_cols, as_index=False).agg(agg_dict)
    agg.columns = [
        col if isinstance(col, str) else "_".join([c for c in col if c])
        for col in agg.columns.to_flat_index()
    ]

    rename_map = {
        "noise_repeat_nunique": "num_noise_repeats",
        "noise_repeat_count": "num_evaluations",
        "seed_nunique": "num_seeds",
        "reused_saved_weights_int_mean": "fraction_reused_saved_weights",
        "reused_saved_weights_int_sum": "num_reused_saved_weights",
        "num_ensemble_seeds_first": "num_seeds",
        "ensemble_seed_values_first": "ensemble_seed_values",
        "all_saved_weights_reused_int_mean": "fraction_all_saved_weights_reused",
        "all_saved_weights_reused_int_sum": "num_all_saved_weights_reused",
    }
    agg.rename(columns=rename_map, inplace=True)

    if "num_seeds" in agg.columns and "num_ensemble_seeds" not in agg.columns and mode == "seed_ensemble":
        agg["num_ensemble_seeds"] = agg["num_seeds"]

    return sort_noise_results(agg, mode)


def load_direct_species_noise_results_if_valid(folder, mode):
    folder = Path(folder)
    path = folder / "noise_results.csv"
    if not folder.is_dir() or not path.exists():
        return None

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        print(f"Skipping {folder.name}: could not read noise_results.csv: {exc}")
        return None

    if df.empty:
        return None

    required_cols = ["species_config_name", "hidden_size", "noise_repeat", "noise_std", "test_mse_scaled"]
    if mode == "single_seed":
        required_cols.append("seed")
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"Skipping {folder.name}: noise_results.csv is missing columns: " + ", ".join(missing))
        return None

    species_config_names = df["species_config_name"].astype(str).dropna().unique()
    if len(species_config_names) != 1 or species_config_names[0] != folder.name:
        print(f"Skipping {folder.name}: folder name does not match unique species_config_name inside CSV.")
        return None

    return df


def rebuild_fullrun_noise_files_from_direct_species_folders(noise_results_root, mode):
    noise_results_root = Path(noise_results_root)
    noise_results_root.mkdir(parents=True, exist_ok=True)

    all_species_dfs = []
    included_folders = []
    for folder in sorted(noise_results_root.iterdir(), key=lambda p: p.name):
        species_df = load_direct_species_noise_results_if_valid(folder, mode)
        if species_df is None:
            continue
        all_species_dfs.append(species_df)
        included_folders.append(folder.name)

    if not all_species_dfs:
        print(f"No valid direct species-combination folders found in {noise_results_root}.")
        return pd.DataFrame(), pd.DataFrame()

    full_df = pd.concat(all_species_dfs, ignore_index=True)
    full_df = deduplicate_results(full_df, mode)
    full_df = sort_noise_results(full_df, mode)
    full_df.to_csv(noise_results_root / "fullrun_noise_results.csv", index=False)

    full_agg = aggregate_noise_results(full_df, mode)
    full_agg.to_csv(noise_results_root / "fullrun_noise_aggregate_summary.csv", index=False)

    save_json(
        noise_results_root / "fullrun_rebuild_info.json",
        {
            "mode": mode,
            "included_species_folders": included_folders,
            "num_included_species_folders": int(len(included_folders)),
            "num_fullrun_rows": int(len(full_df)),
            "num_fullrun_aggregate_rows": int(len(full_agg)),
        },
    )
    return full_df, full_agg


def append_current_results_to_species_folders(noise_results_root, current_results, mode):
    noise_results_root = Path(noise_results_root)
    noise_results_root.mkdir(parents=True, exist_ok=True)

    current_df = pd.DataFrame(current_results)
    if current_df.empty:
        print(f"No new {mode} rows to append. Rebuilding global files from existing species folders only.")
        return rebuild_fullrun_noise_files_from_direct_species_folders(noise_results_root, mode)

    for species_name, new_species_df in current_df.groupby("species_config_name"):
        species_root = noise_results_root / str(species_name)
        species_root.mkdir(parents=True, exist_ok=True)
        path = species_root / "noise_results.csv"

        if path.exists():
            try:
                old_species_df = pd.read_csv(path)
                combined = pd.concat([old_species_df, new_species_df], ignore_index=True)
            except Exception as exc:
                print(f"Warning: could not read existing {path}; overwriting with current rows. Error: {exc}")
                combined = new_species_df.copy()
        else:
            combined = new_species_df.copy()

        combined = deduplicate_results(combined, mode)
        combined = sort_noise_results(combined, mode)
        combined.to_csv(path, index=False)

        species_agg = aggregate_noise_results(combined, mode)
        species_agg.to_csv(species_root / "noise_aggregate_summary.csv", index=False)

    return rebuild_fullrun_noise_files_from_direct_species_folders(noise_results_root, mode)


# ======================================================================================
# PLOTS AND TABLES
# ======================================================================================


def plot_safe_path_token(text):
    return safe_path_token(text).replace(",", "_")


def get_metric_mean_std_cols(metric):
    return f"{metric}_mean", f"{metric}_std"


def get_plot_metric_values(df, metric):
    mean_col, std_col = get_metric_mean_std_cols(metric)
    if mean_col in df.columns:
        y = pd.to_numeric(df[mean_col], errors="coerce")
        yerr = pd.to_numeric(df[std_col], errors="coerce") if std_col in df.columns else None
    elif metric in df.columns:
        y = pd.to_numeric(df[metric], errors="coerce")
        yerr = None
    else:
        raise ValueError(f"Metric {metric} not found in columns.")
    return y, yerr


def short_species_label(row_or_df):
    if isinstance(row_or_df, pd.DataFrame):
        if "kept_species" in row_or_df.columns and len(row_or_df) > 0:
            return str(row_or_df["kept_species"].iloc[0])
        if "species_config_name" in row_or_df.columns and len(row_or_df) > 0:
            return str(row_or_df["species_config_name"].iloc[0])
        return "unknown"
    if "kept_species" in row_or_df and pd.notna(row_or_df["kept_species"]):
        return str(row_or_df["kept_species"])
    return str(row_or_df.get("species_config_name", "unknown"))


def plot_style_for(index):
    return {
        "color": PLOT_COLORS[index % len(PLOT_COLORS)],
        "marker": PLOT_MARKERS[index % len(PLOT_MARKERS)],
    }


def format_noise_axes(ax):
    ax.set_xlabel("Noise level (%)")
    ax.set_ylabel(PLOT_MAIN_METRIC_LABEL)
    if PLOT_USE_LOG_Y:
        ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.25)


def save_current_figure(output_base):
    output_base = Path(output_base)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    if PLOT_SAVE_PNG:
        plt.savefig(output_base.with_suffix(".png"), dpi=PLOT_DPI, bbox_inches="tight")
    if PLOT_SAVE_PDF:
        plt.savefig(output_base.with_suffix(".pdf"), dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()


def save_noise_plot(noise_results_root, aggregate_df, mode, title_prefix):
    if aggregate_df.empty:
        return

    plots_root = Path(noise_results_root) / "Plots"
    plots_root.mkdir(parents=True, exist_ok=True)

    metric_mean_col, _ = get_metric_mean_std_cols(PLOT_MAIN_METRIC)
    if metric_mean_col not in aggregate_df.columns and PLOT_MAIN_METRIC not in aggregate_df.columns:
        raise ValueError(f"Cannot plot. Metric not found: {PLOT_MAIN_METRIC}")

    df = aggregate_df.copy()
    df["noise_percent"] = pd.to_numeric(df["noise_percent"], errors="coerce")
    df = df[df["hidden_size"].isin([arch_to_folder_name(a) for a in ARCHITECTURES])].copy()

    for hidden_size, arch_df in df.groupby("hidden_size"):
        plt.figure(figsize=(10.5, 4.2))

        ordered_species = [s for s in PLOT_SPECIES_ORDER if s in set(arch_df["kept_species"].astype(str))]
        extra_species = [s for s in sorted(arch_df["kept_species"].astype(str).unique()) if s not in ordered_species]
        ordered_species.extend(extra_species)

        for idx, kept_species_text in enumerate(ordered_species):
            species_df = arch_df[arch_df["kept_species"].astype(str) == kept_species_text].copy()
            if species_df.empty:
                continue
            species_df = species_df.sort_values("noise_percent")
            y, yerr = get_plot_metric_values(species_df, PLOT_MAIN_METRIC)
            style = plot_style_for(idx)

            if PLOT_SHOW_ERRORBARS and yerr is not None:
                plt.errorbar(
                    species_df["noise_percent"],
                    y,
                    yerr=yerr,
                    label=kept_species_text,
                    linewidth=PLOT_LINEWIDTH,
                    markersize=PLOT_MARKER_SIZE,
                    capsize=2,
                    **style,
                )
            else:
                plt.plot(
                    species_df["noise_percent"],
                    y,
                    label=kept_species_text,
                    linewidth=PLOT_LINEWIDTH,
                    markersize=PLOT_MARKER_SIZE,
                    **style,
                )

        ax = plt.gca()
        format_noise_axes(ax)
        ax.set_title(f"{title_prefix} | Architecture {hidden_size}")
        plt.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.22),
            ncol=PLOT_MAX_LEGEND_COLUMNS,
            fontsize=8,
            frameon=False,
        )
        plt.tight_layout()
        token = arch_to_file_token(hidden_size)
        save_current_figure(plots_root / f"noise_mse_vs_noise_{mode}_{token}")


def save_summary_txt(noise_results_root, aggregate_df, mode):
    noise_results_root = Path(noise_results_root)
    summary_path = noise_results_root / "summary.txt"
    metric_col = f"{PLOT_MAIN_METRIC}_mean"

    lines = []
    lines.append(f"Noisy-training robustness summary ({mode})")
    lines.append("=" * 80)
    lines.append(f"Data scheme: {DATA_SCHEME}")
    lines.append(f"Output scheme: {OUTPUT_SCHEME}")
    lines.append(f"Experiment: {EXPERIMENT_NAME}")
    lines.append(f"Architectures: {[arch_to_folder_name(a) for a in ARCHITECTURES]}")
    lines.append(f"Seeds: {SEEDS[0]} to {SEEDS[-1]} ({len(SEEDS)} seeds)")
    lines.append(f"Train noise stds: {TRAIN_NOISE_STDS}")
    lines.append(f"Train noise probabilities: {normalize_probability_vector(TRAIN_NOISE_PROBS)}")
    lines.append(f"Validation noise stds: {VAL_NOISE_STDS}")
    lines.append(f"Validation noise weights: {normalize_probability_vector(VAL_NOISE_WEIGHTS)}")
    lines.append(f"Evaluation noise stds: {EVAL_NOISE_STDS}")
    lines.append(f"Noise repeats: {NOISE_REPEATS}")
    lines.append("")

    if aggregate_df.empty:
        lines.append("No aggregate rows found.")
    elif metric_col in aggregate_df.columns:
        table_cols = [
            "hidden_size",
            "kept_species",
            "noise_percent",
            metric_col,
        ]
        std_col = f"{PLOT_MAIN_METRIC}_std"
        if std_col in aggregate_df.columns:
            table_cols.append(std_col)
        for extra_col in ["num_seeds", "num_noise_repeats", "num_evaluations"]:
            if extra_col in aggregate_df.columns:
                table_cols.append(extra_col)
        lines.append(aggregate_df[table_cols].sort_values(["hidden_size", "kept_species", "noise_percent"]).to_string(index=False))

    with open(summary_path, "w") as f:
        f.write("\n".join(lines))


def save_run_info(noise_results_root, mode, total_requested, total_existing, total_to_run):
    save_json(
        Path(noise_results_root) / "noise_run_info.json",
        {
            "data_scheme": DATA_SCHEME,
            "output_scheme": OUTPUT_SCHEME,
            "experiment_name": EXPERIMENT_NAME,
            "evaluation_mode": mode,
            "results_root": str(noise_results_root),
            "saved_weights_root": str(SAVED_WEIGHTS_ROOT / OUTPUT_SCHEME),
            "species_configs": SPECIES_CONFIGS,
            "architectures": [list(a) for a in ARCHITECTURES],
            "seeds": SEEDS,
            "ensemble_seeds": ENSEMBLE_SEEDS if mode == "seed_ensemble" else None,
            "eval_noise_stds": EVAL_NOISE_STDS,
            "eval_noise_labels": [noise_label(s) for s in EVAL_NOISE_STDS],
            "noise_repeats": NOISE_REPEATS,
            "noise_base_seed": NOISE_BASE_SEED,
            "train_noise_stds": TRAIN_NOISE_STDS,
            "train_noise_probs": normalize_probability_vector(TRAIN_NOISE_PROBS),
            "val_noise_stds": VAL_NOISE_STDS,
            "val_noise_weights": normalize_probability_vector(VAL_NOISE_WEIGHTS),
            "main_metric": PLOT_MAIN_METRIC,
            "training_mode": "noise-aware dynamic multiplicative Gaussian input augmentation",
            "noise_type": "multiplicative Gaussian noise applied to all selected unscaled input-density features",
            "resample_negative_densities": RESAMPLE_NEGATIVE_DENSITIES,
            "max_noise_resample_attempts": MAX_NOISE_RESAMPLE_ATTEMPTS,
            "clip_negative_densities": CLIP_NEGATIVE_DENSITIES,
            "cache_policy": "load compatible saved_weights/O2_novib_noisy model; otherwise train with noisy inputs and save before evaluation",
            "skip_existing_evaluations": SKIP_EXISTING_EVALUATIONS,
            "total_requested_evaluations": int(total_requested),
            "total_existing_requested_evaluations_skipped": int(total_existing),
            "total_new_evaluations_to_run": int(total_to_run),
        },
    )


# ======================================================================================
# EXECUTION PLANNING AND RUNNERS
# ======================================================================================


def plan_single_tasks(noise_results_root):
    task_plan = []
    total_requested = 0
    total_existing = 0

    for kept_species in SPECIES_CONFIGS:
        existing_df = load_existing_species_results(noise_results_root, kept_species)
        existing_keys = existing_keys_for_species(existing_df, "single_seed") if SKIP_EXISTING_EVALUATIONS else set()

        species_tasks = []
        for hidden_size in ARCHITECTURES:
            hidden_text = arch_to_folder_name(hidden_size)
            for seed in SEEDS:
                for noise_std in EVAL_NOISE_STDS:
                    for noise_repeat in range(NOISE_REPEATS):
                        total_requested += 1
                        key = result_key_from_values(hidden_text, seed, noise_std, noise_repeat, "single_seed")
                        if key in existing_keys:
                            total_existing += 1
                            continue
                        species_tasks.append((hidden_size, seed, float(noise_std), int(noise_repeat)))
        task_plan.append((kept_species, species_tasks))

    return task_plan, total_requested, total_existing, total_requested - total_existing


def plan_ensemble_tasks(noise_results_root):
    task_plan = []
    total_requested = 0
    total_existing = 0

    for kept_species in SPECIES_CONFIGS:
        existing_df = load_existing_species_results(noise_results_root, kept_species)
        existing_keys = existing_keys_for_species(existing_df, "seed_ensemble") if SKIP_EXISTING_EVALUATIONS else set()

        species_tasks = []
        for hidden_size in ARCHITECTURES:
            hidden_text = arch_to_folder_name(hidden_size)
            for noise_std in EVAL_NOISE_STDS:
                for noise_repeat in range(NOISE_REPEATS):
                    total_requested += 1
                    key = result_key_from_values(hidden_text, None, noise_std, noise_repeat, "seed_ensemble")
                    if key in existing_keys:
                        total_existing += 1
                        continue
                    species_tasks.append((hidden_size, float(noise_std), int(noise_repeat)))
        task_plan.append((kept_species, species_tasks))

    return task_plan, total_requested, total_existing, total_requested - total_existing


def load_or_train_models_for_species(dataset_train, dataset_test, kept_species, hidden_size, seeds):
    models_by_seed = {}
    training_records_by_seed = {}

    for seed in seeds:
        model, _, _, training_record = get_or_train_noisy_model(
            data_scheme=DATA_SCHEME,
            output_scheme=OUTPUT_SCHEME,
            kept_species=kept_species,
            hidden_size=hidden_size,
            seed=seed,
            activation=ACTIVATION,
            learning_rate=LEARNING_RATE,
            batch_size=BATCH_SIZE,
            max_epochs=MAX_EPOCHS,
            patience=PATIENCE,
            val_split=VAL_SPLIT,
            dataset_train=dataset_train,
            dataset_test=dataset_test,
            verbose_epoch_losses=VERBOSE_EPOCH_LOSSES,
        )
        models_by_seed[int(seed)] = model
        training_records_by_seed[int(seed)] = training_record

    return models_by_seed, training_records_by_seed


def run_single_model_average_evaluation():
    validate_all_species_configs(SPECIES_CONFIGS)
    validate_architectures(ARCHITECTURES)
    SINGLE_RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    task_plan, total_requested, total_existing, total_to_run = plan_single_tasks(SINGLE_RESULTS_ROOT)
    save_run_info(SINGLE_RESULTS_ROOT, "single_seed", total_requested, total_existing, total_to_run)

    print("Single-model average evaluation")
    print(f"Requested evaluations: {total_requested}")
    print(f"Existing evaluations skipped: {total_existing}")
    print(f"New evaluations to run: {total_to_run}")

    current_results = []
    with tqdm(total=total_to_run, desc="Single noisy-trained robustness") as pbar:
        for kept_species, species_tasks in task_plan:
            if not species_tasks:
                continue

            dataset_train, dataset_test = load_datasets_with_saved_scalers(DATA_SCHEME, OUTPUT_SCHEME, kept_species)
            models_records_cache = {}

            for hidden_size, seed, noise_std, noise_repeat in species_tasks:
                cache_key = (tuple(hidden_size), int(seed))
                if cache_key not in models_records_cache:
                    model, _, _, training_record = get_or_train_noisy_model(
                        data_scheme=DATA_SCHEME,
                        output_scheme=OUTPUT_SCHEME,
                        kept_species=kept_species,
                        hidden_size=hidden_size,
                        seed=seed,
                        activation=ACTIVATION,
                        learning_rate=LEARNING_RATE,
                        batch_size=BATCH_SIZE,
                        max_epochs=MAX_EPOCHS,
                        patience=PATIENCE,
                        val_split=VAL_SPLIT,
                        dataset_train=dataset_train,
                        dataset_test=dataset_test,
                        verbose_epoch_losses=VERBOSE_EPOCH_LOSSES,
                    )
                    models_records_cache[cache_key] = (model, training_record)

                model, training_record = models_records_cache[cache_key]
                row = run_noise_for_single_model(
                    model=model,
                    dataset_test=dataset_test,
                    kept_species=kept_species,
                    hidden_size=hidden_size,
                    seed=seed,
                    noise_std=noise_std,
                    noise_repeat=noise_repeat,
                    training_record=training_record,
                )
                current_results.append(row)
                pbar.set_postfix(
                    species=species_config_to_name(kept_species),
                    seed=seed,
                    noise=row["noise_label"],
                    mse=f"{row['test_mse_scaled']:.3e}",
                )
                pbar.update(1)

    _, aggregate_df = append_current_results_to_species_folders(SINGLE_RESULTS_ROOT, current_results, "single_seed")
    save_summary_txt(SINGLE_RESULTS_ROOT, aggregate_df, "single_seed")
    save_noise_plot(
        noise_results_root=SINGLE_RESULTS_ROOT,
        aggregate_df=aggregate_df,
        mode="single_seed",
        title_prefix="Noisy-trained single models averaged over seeds/repeats",
    )
    print(f"Single-model results saved to: {SINGLE_RESULTS_ROOT}")
    return aggregate_df


def run_ensemble_evaluation():
    validate_all_species_configs(SPECIES_CONFIGS)
    validate_architectures(ARCHITECTURES)
    ENSEMBLE_RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    task_plan, total_requested, total_existing, total_to_run = plan_ensemble_tasks(ENSEMBLE_RESULTS_ROOT)
    save_run_info(ENSEMBLE_RESULTS_ROOT, "seed_ensemble", total_requested, total_existing, total_to_run)

    print("Seed-ensemble evaluation")
    print(f"Requested evaluations: {total_requested}")
    print(f"Existing evaluations skipped: {total_existing}")
    print(f"New evaluations to run: {total_to_run}")

    current_results = []
    with tqdm(total=total_to_run, desc="Ensemble noisy-trained robustness") as pbar:
        for kept_species, species_tasks in task_plan:
            if not species_tasks:
                continue

            dataset_train, dataset_test = load_datasets_with_saved_scalers(DATA_SCHEME, OUTPUT_SCHEME, kept_species)
            ensemble_cache = {}

            for hidden_size, noise_std, noise_repeat in species_tasks:
                cache_key = tuple(hidden_size)
                if cache_key not in ensemble_cache:
                    models_by_seed, training_records_by_seed = load_or_train_models_for_species(
                        dataset_train=dataset_train,
                        dataset_test=dataset_test,
                        kept_species=kept_species,
                        hidden_size=hidden_size,
                        seeds=ENSEMBLE_SEEDS,
                    )
                    ensemble_cache[cache_key] = (models_by_seed, training_records_by_seed)

                models_by_seed, training_records_by_seed = ensemble_cache[cache_key]
                row = run_noise_for_seed_ensemble(
                    models_by_seed=models_by_seed,
                    training_records_by_seed=training_records_by_seed,
                    dataset_test=dataset_test,
                    kept_species=kept_species,
                    hidden_size=hidden_size,
                    noise_std=noise_std,
                    noise_repeat=noise_repeat,
                )
                current_results.append(row)
                pbar.set_postfix(
                    species=species_config_to_name(kept_species),
                    noise=row["noise_label"],
                    mse=f"{row['test_mse_scaled']:.3e}",
                )
                pbar.update(1)

    _, aggregate_df = append_current_results_to_species_folders(ENSEMBLE_RESULTS_ROOT, current_results, "seed_ensemble")
    save_summary_txt(ENSEMBLE_RESULTS_ROOT, aggregate_df, "seed_ensemble")
    save_noise_plot(
        noise_results_root=ENSEMBLE_RESULTS_ROOT,
        aggregate_df=aggregate_df,
        mode="seed_ensemble",
        title_prefix="Noisy-trained seed ensemble",
    )
    print(f"Ensemble results saved to: {ENSEMBLE_RESULTS_ROOT}")
    return aggregate_df


# ======================================================================================
# CLEAN-TRAINED COMPARISON
# ======================================================================================


def clean_candidate_paths():
    if not CLEAN_RESULTS_SEARCH_ROOT.exists():
        return []
    return sorted(CLEAN_RESULTS_SEARCH_ROOT.rglob("fullrun_noise_aggregate_summary.csv"))


def read_clean_candidate(path):
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if df.empty:
        return None
    required = {"kept_species", "hidden_size", "noise_percent"}
    if not required.issubset(df.columns):
        return None
    metric_col = f"{PLOT_MAIN_METRIC}_mean"
    if metric_col not in df.columns and PLOT_MAIN_METRIC not in df.columns:
        return None
    return df


def score_clean_candidate(df, path, prefer_ensemble):
    path_text = str(path).lower()
    is_ensemble_path = "ensemble" in path_text or (
        "evaluation_mode" in df.columns and df["evaluation_mode"].astype(str).str.contains("ensemble", case=False, na=False).any()
    ) or "num_ensemble_seeds" in df.columns

    if prefer_ensemble and not is_ensemble_path:
        mode_score = 0
    elif (not prefer_ensemble) and is_ensemble_path:
        mode_score = 0
    else:
        mode_score = 10_000

    required_species = set(PLOT_SPECIES_ORDER)
    required_archs = {arch_to_folder_name(a) for a in ARCHITECTURES}
    required_noises = {float(100.0 * s) for s in EVAL_NOISE_STDS}

    work = df.copy()
    work["noise_percent"] = pd.to_numeric(work["noise_percent"], errors="coerce")

    species_score = len(required_species.intersection(set(work["kept_species"].astype(str))))
    arch_score = len(required_archs.intersection(set(work["hidden_size"].astype(str))))
    noise_score = len(required_noises.intersection(set(np.round(work["noise_percent"].dropna().astype(float), 12))))
    row_score = len(work)
    mtime_score = path.stat().st_mtime / 1e12

    return mode_score + species_score * 1000 + arch_score * 100 + noise_score * 10 + min(row_score, 999) / 1000 + mtime_score


def find_best_clean_aggregate(prefer_ensemble):
    best = None
    best_score = -np.inf
    inspected = []

    for path in clean_candidate_paths():
        df = read_clean_candidate(path)
        if df is None:
            continue
        score = score_clean_candidate(df, path, prefer_ensemble=prefer_ensemble)
        inspected.append((str(path), float(score), int(len(df))))
        if score > best_score:
            best = (path, df)
            best_score = score

    return best, inspected


def standardize_aggregate_for_comparison(df, prefix):
    df = df.copy()
    metric_mean_col = f"{PLOT_MAIN_METRIC}_mean"
    metric_std_col = f"{PLOT_MAIN_METRIC}_std"

    if metric_mean_col not in df.columns and PLOT_MAIN_METRIC in df.columns:
        df[metric_mean_col] = df[PLOT_MAIN_METRIC]
    if metric_std_col not in df.columns:
        df[metric_std_col] = np.nan

    keep_cols = [
        "kept_species",
        "species_config_name",
        "hidden_size",
        "noise_percent",
        metric_mean_col,
        metric_std_col,
    ]
    keep_cols = [col for col in keep_cols if col in df.columns]
    out = df[keep_cols].copy()
    out["noise_percent"] = pd.to_numeric(out["noise_percent"], errors="coerce")

    rename = {
        metric_mean_col: f"{prefix}_{PLOT_MAIN_METRIC}_mean",
        metric_std_col: f"{prefix}_{PLOT_MAIN_METRIC}_std",
    }
    if "species_config_name" in out.columns:
        rename["species_config_name"] = f"{prefix}_species_config_name"
    out.rename(columns=rename, inplace=True)
    return out


def build_comparison_table(noisy_agg, clean_agg):
    noisy_std = standardize_aggregate_for_comparison(noisy_agg, "noisy_trained")
    clean_std = standardize_aggregate_for_comparison(clean_agg, "clean_trained")

    merged = noisy_std.merge(
        clean_std,
        on=["kept_species", "hidden_size", "noise_percent"],
        how="inner",
    )

    noisy_col = f"noisy_trained_{PLOT_MAIN_METRIC}_mean"
    clean_col = f"clean_trained_{PLOT_MAIN_METRIC}_mean"
    if merged.empty or noisy_col not in merged.columns or clean_col not in merged.columns:
        return merged

    merged["mse_ratio_noisy_trained_over_clean_trained"] = merged[noisy_col] / merged[clean_col]
    merged["improvement_factor_clean_over_noisy_trained"] = merged[clean_col] / merged[noisy_col]
    merged["percent_change_vs_clean"] = 100.0 * (merged[noisy_col] - merged[clean_col]) / merged[clean_col]
    merged["noisy_training_helped"] = merged[noisy_col] < merged[clean_col]

    return merged.sort_values(["hidden_size", "kept_species", "noise_percent"]).reset_index(drop=True)


def save_comparison_plot(comparison_df, output_root, mode):
    if comparison_df.empty:
        return

    output_root = Path(output_root)
    plots_root = output_root / "Plots"
    plots_root.mkdir(parents=True, exist_ok=True)

    ratio_col = "mse_ratio_noisy_trained_over_clean_trained"
    if ratio_col not in comparison_df.columns:
        return

    for hidden_size, arch_df in comparison_df.groupby("hidden_size"):
        plt.figure(figsize=(10.5, 4.2))

        ordered_species = [s for s in PLOT_SPECIES_ORDER if s in set(arch_df["kept_species"].astype(str))]
        extra_species = [s for s in sorted(arch_df["kept_species"].astype(str).unique()) if s not in ordered_species]
        ordered_species.extend(extra_species)

        for idx, kept_species_text in enumerate(ordered_species):
            species_df = arch_df[arch_df["kept_species"].astype(str) == kept_species_text].copy()
            species_df = species_df.sort_values("noise_percent")
            style = plot_style_for(idx)
            plt.plot(
                species_df["noise_percent"],
                species_df[ratio_col],
                label=kept_species_text,
                linewidth=PLOT_LINEWIDTH,
                markersize=PLOT_MARKER_SIZE,
                **style,
            )

        ax = plt.gca()
        ax.axhline(1.0, color="black", linewidth=1.0, linestyle="--", alpha=0.7)
        ax.set_xlabel("Noise level (%)")
        ax.set_ylabel("MSE ratio: noisy-trained / clean-trained")
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.25)
        ax.set_title(f"Noisy training vs clean training | {mode} | Architecture {hidden_size}")
        plt.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.22),
            ncol=PLOT_MAX_LEGEND_COLUMNS,
            fontsize=8,
            frameon=False,
        )
        plt.tight_layout()
        token = arch_to_file_token(hidden_size)
        save_current_figure(plots_root / f"comparison_ratio_{mode}_{token}")


def save_comparison_summary_txt(comparison_df, output_root, mode, clean_path, inspected):
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / f"comparison_summary_{mode}.txt"

    lines = []
    lines.append(f"Clean-trained vs noisy-trained comparison ({mode})")
    lines.append("=" * 80)
    lines.append(f"Selected clean aggregate source: {clean_path}")
    lines.append("")

    if comparison_df.empty:
        lines.append("No matching rows found between noisy-trained aggregate and clean-trained aggregate.")
    else:
        cols = [
            "hidden_size",
            "kept_species",
            "noise_percent",
            f"clean_trained_{PLOT_MAIN_METRIC}_mean",
            f"noisy_trained_{PLOT_MAIN_METRIC}_mean",
            "mse_ratio_noisy_trained_over_clean_trained",
            "improvement_factor_clean_over_noisy_trained",
            "percent_change_vs_clean",
            "noisy_training_helped",
        ]
        cols = [col for col in cols if col in comparison_df.columns]
        lines.append(comparison_df[cols].to_string(index=False))
        lines.append("")

        grouped = comparison_df.groupby("noise_percent", as_index=False).agg(
            mse_ratio_noisy_trained_over_clean_trained_mean=("mse_ratio_noisy_trained_over_clean_trained", "mean"),
            improvement_factor_clean_over_noisy_trained_mean=("improvement_factor_clean_over_noisy_trained", "mean"),
            fraction_rows_helped=("noisy_training_helped", "mean"),
        )
        lines.append("Grouped by noise_percent:")
        lines.append(grouped.to_string(index=False))

    lines.append("")
    lines.append("Inspected clean candidate files:")
    for candidate_path, score, nrows in inspected:
        lines.append(f"  score={score:.3f} rows={nrows:6d} path={candidate_path}")

    with open(path, "w") as f:
        f.write("\n".join(lines))


def save_clean_comparison_for_mode(noisy_agg, mode, prefer_ensemble):
    selected, inspected = find_best_clean_aggregate(prefer_ensemble=prefer_ensemble)
    mode_root = COMPARISON_RESULTS_ROOT / mode
    mode_root.mkdir(parents=True, exist_ok=True)

    save_json(
        mode_root / "clean_source_search_manifest.json",
        {
            "prefer_ensemble": bool(prefer_ensemble),
            "search_root": str(CLEAN_RESULTS_SEARCH_ROOT),
            "inspected_candidates": [
                {"path": p, "score": s, "num_rows": n} for p, s, n in inspected
            ],
        },
    )

    if selected is None:
        with open(mode_root / f"comparison_summary_{mode}.txt", "w") as f:
            f.write(
                "No usable clean-trained aggregate source was found.\n"
                f"Searched under: {CLEAN_RESULTS_SEARCH_ROOT}\n"
            )
        print(f"WARNING: No clean-trained aggregate found for comparison mode {mode}.")
        return pd.DataFrame()

    clean_path, clean_agg = selected
    comparison_df = build_comparison_table(noisy_agg, clean_agg)
    comparison_df.to_csv(mode_root / f"comparison_{mode}.csv", index=False)
    save_comparison_plot(comparison_df, mode_root, mode)
    save_comparison_summary_txt(comparison_df, mode_root, mode, clean_path, inspected)

    print(f"Comparison for {mode} saved to: {mode_root}")
    print(f"Clean source used for {mode}: {clean_path}")
    return comparison_df


def save_clean_comparisons(single_agg=None, ensemble_agg=None):
    if not SAVE_COMPARISON_AGAINST_CLEAN:
        return
    COMPARISON_RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    if single_agg is not None and not single_agg.empty:
        single_comparison = save_clean_comparison_for_mode(
            noisy_agg=single_agg,
            mode="single_models_average",
            prefer_ensemble=False,
        )
        if not single_comparison.empty:
            single_comparison.to_csv(COMPARISON_RESULTS_ROOT / "comparison_single_models_average.csv", index=False)

    if ensemble_agg is not None and not ensemble_agg.empty:
        ensemble_comparison = save_clean_comparison_for_mode(
            noisy_agg=ensemble_agg,
            mode="ensemble",
            prefer_ensemble=True,
        )
        if not ensemble_comparison.empty:
            ensemble_comparison.to_csv(COMPARISON_RESULTS_ROOT / "comparison_ensemble.csv", index=False)


# ======================================================================================
# MAIN
# ======================================================================================


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Train 3K neural networks with noisy input-density augmentation, evaluate "
            "their test-time noise robustness, and compare them with clean-trained models."
        )
    )
    parser.add_argument(
        "--mode",
        choices=["all", "single", "ensemble", "comparison"],
        default="all",
        help=(
            "Workflow to run. 'all' runs single-model evaluation, ensemble evaluation, "
            "and clean-trained comparison. 'comparison' only rebuilds the comparison from "
            "existing noisy-training aggregate CSV files."
        ),
    )
    parser.add_argument(
        "--rerun-existing",
        action="store_true",
        help="Recompute evaluations even when matching rows already exist in the output CSV files.",
    )
    parser.add_argument(
        "--no-comparison",
        action="store_true",
        help="Skip the automatic comparison against clean-trained results.",
    )
    return parser.parse_args(argv)


def load_existing_noisy_aggregate(noise_results_root):
    aggregate_path = Path(noise_results_root) / "fullrun_noise_aggregate_summary.csv"
    if not aggregate_path.exists():
        raise FileNotFoundError(
            f"Could not find existing noisy-training aggregate file:\n{aggregate_path}\n"
            "Run the corresponding noisy-training evaluation first."
        )
    return pd.read_csv(aggregate_path)


def save_experiment_info(mode):
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    save_json(
        RESULTS_ROOT / "experiment_info.json",
        {
            "data_scheme": DATA_SCHEME,
            "output_scheme": OUTPUT_SCHEME,
            "experiment_name": EXPERIMENT_NAME,
            "mode_latest_invocation": mode,
            "architecture_policy": "Only the architectures listed in ARCHITECTURES are run.",
            "species_configs": SPECIES_CONFIGS,
            "train_noise_stds": TRAIN_NOISE_STDS,
            "train_noise_probs": normalize_probability_vector(TRAIN_NOISE_PROBS),
            "val_noise_stds": VAL_NOISE_STDS,
            "val_noise_weights": normalize_probability_vector(VAL_NOISE_WEIGHTS),
            "eval_noise_stds": EVAL_NOISE_STDS,
            "seeds": SEEDS,
            "noise_repeats": NOISE_REPEATS,
            "skip_existing_evaluations": SKIP_EXISTING_EVALUATIONS,
            "save_comparison_against_clean": SAVE_COMPARISON_AGAINST_CLEAN,
            "single_results_root": str(SINGLE_RESULTS_ROOT),
            "ensemble_results_root": str(ENSEMBLE_RESULTS_ROOT),
            "comparison_results_root": str(COMPARISON_RESULTS_ROOT),
        },
    )


def main(argv=None):
    global SKIP_EXISTING_EVALUATIONS, SAVE_COMPARISON_AGAINST_CLEAN

    args = parse_args(argv)
    if args.rerun_existing:
        SKIP_EXISTING_EVALUATIONS = False
    if args.no_comparison:
        SAVE_COMPARISON_AGAINST_CLEAN = False

    validate_all_species_configs(SPECIES_CONFIGS)
    validate_architectures(ARCHITECTURES)
    save_experiment_info(args.mode)

    single_agg = None
    ensemble_agg = None

    if args.mode in {"all", "single"}:
        single_agg = run_single_model_average_evaluation()

    if args.mode in {"all", "ensemble"}:
        ensemble_agg = run_ensemble_evaluation()

    if args.mode == "comparison":
        single_agg = load_existing_noisy_aggregate(SINGLE_RESULTS_ROOT)
        ensemble_agg = load_existing_noisy_aggregate(ENSEMBLE_RESULTS_ROOT)

    if SAVE_COMPARISON_AGAINST_CLEAN:
        save_clean_comparisons(single_agg=single_agg, ensemble_agg=ensemble_agg)

    print("Done.")
    print(f"Saved noisy-trained weights under: {SAVED_WEIGHTS_ROOT / OUTPUT_SCHEME}")
    print(f"Saved all noisy-training results under: {RESULTS_ROOT}")


if __name__ == "__main__":
    main()
