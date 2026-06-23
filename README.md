# Inverse Estimation of Reaction-Rate Coefficients in Oxygen Plasmas using Machine Learning

This repository contains the code and data used to study an inverse modelling problem in oxygen plasma kinetics, which is estimating selected reaction-rate coefficients from final plasma species densities using neural networks.

The usual direction in plasma kinetic modelling is the forward problem, where a set of reaction-rate coefficients is chosen, a kinetic model is run and the resulting species densities are obtained. This project studies the inverse direction. Given the final densities of selected heavy species at different pressure conditions, the goal is to recover the reaction-rate coefficients that generated them.

The work focuses mainly on an oxygen plasma kinetic scheme without vibrational resolution, using LoKI-generated simulation data and fully connected neural networks trained to estimate three reaction-rate coefficients, denoted here as `K1`, `K2`, and `K3`.

The full 10-page report is available in [`docs/Report.pdf`](docs/Report.pdf).

---

## Project overview

### Problem

Reaction-rate coefficients are central parameters in plasma kinetic models. They determine how strongly individual reactions create or destroy plasma species. In practice, these coefficients may be uncertain, estimated, taken from databases or obtained from experiments and calculations with varying confidence levels. Errors in these coefficients can propagate to the predicted plasma composition.

This repository investigates whether machine learning can be used to solve the inverse problem:

```text
final species densities  →  reaction-rate coefficients
```

For the main 3-coefficient problem, the neural networks receive final heavy-species densities at two pressure conditions and predict the corresponding values of `K1`, `K2`, and `K3`.

### Main research questions

The code in this repository is organized around the following questions:

1. **Inverse estimation:** Can a neural network recover reaction-rate coefficients from final species densities?
2. **Input-species selection:** Which species densities contain the most information about the unknown coefficients?
3. **Noise robustness:** How strongly does measurement-like noise in the input densities affect the inverse estimate?
4. **Ensembling:** Does averaging predictions over multiple independently trained neural networks improve robustness?
5. **Noisy training:** Can training with noisy input densities improve performance when the test inputs are noisy?

---

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── docs/
│   └── Report.pdf
│
├── src/
│   ├── NeuralNetworkModels.py
│   ├── config.py
│   └── ...
│
├── data/
│   ├── datapoints_O2_novib_mainNet_2surrog.txt
│   ├── datapoints_O2_novib_mainNet_2surrog_test.txt
│   └── ...
│
├── 9Ks_Dataset/
│   └── ...
│
├── NeuralNet_3Ks_Noiseless_Run.py
├── NeuralNet_3Ks_Noise_Ranking.py
├── NeuralNet_3Ks_Noise_Sensitivity.py
├── NeuralNet_3Ks_NoisyTraining.py
├── NeuralNet_9Ks_Test.py
│
├── plot_pic1_results.py
└── plot_pic1_powerpoint_figures.py
```

### Important generated folders

The following folders are generated when running the scripts and are intentionally not present in this repository:

```text
Results_NN/
saved_weights/
```

`Results_NN/` contains experiment outputs, aggregate CSV files, plots, and manifests.

`saved_weights/` contains trained model weights and scaler metadata. This folder can become large, so it is not tracked in Git.

Pretrained weights are available separately here:

[Google Drive folder with saved weights](https://drive.google.com/drive/folders/1Aa5-M8woTDJkob1Ew7725sULk9lKCkq5?usp=sharing)

If `saved_weights/` is not present, the scripts will train the required models from scratch.

---

## Data

The main dataset used by the 3-coefficient oxygen plasma inverse problem is defined in `src/config.py` under the key `O2_novib`.

The main files are:

```text
data/datapoints_O2_novib_mainNet_2surrog.txt
data/datapoints_O2_novib_mainNet_2surrog_test.txt
```

Each sample corresponds to a LoKI-generated oxygen plasma simulation. The inputs used by the neural networks are selected final heavy-species densities at two pressure conditions. The targets are the reaction-rate coefficients `K1`, `K2`, and `K3`.

The species considered in the main oxygen scheme are:

```text
O2(X), O2(a), O2(b), O2(Hz), O2+(X), O(3P), O(1D), O+(gnd), O-(gnd), O3(X), O3(exc)
```

For each selected species, the model receives its final density at both pressure conditions. Therefore, using `Ns` species gives `2 × Ns` input features.

---

## Machine-learning setup

The main models are fully connected feed-forward neural networks implemented in PyTorch.

Configuration used:

- activation function: hyperbolic tangent (`tanh`)
- optimizer: Adam
- loss function: mean squared error on scaled coefficients (feel free to use RMSE, which might be better since it preserves units and gives similar results to MSE)
- train/validation split: 90% / 10%
- multiple random seeds are used to estimate sensitivity to initialization and data splitting
- selected architectures include `(30, 30)`, `(30, 30, 30)`, and `(50, 50)` depending on the experiment

The target coefficients are scaled during training. Reported errors in the main robustness plots use scaled MSE unless otherwise stated.

---

## Installation

Python 3.10 or newer is recommended.

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Using pretrained weights

The repository is designed to run with or without pretrained weights.

To reuse cached models:

1. Download the `saved_weights/` folder from the Google Drive link above.
2. Place it in the repository root, next to the Python scripts.
3. Run the scripts normally.

Expected structure:

```text
.
├── saved_weights/
│   ├── O2_novib/
│   └── O2_novib_noisy/
└── NeuralNet_3Ks_Noise_Ranking.py
```

If compatible weights are found, the scripts reuse them. If not, the relevant models are trained and new weights are saved automatically.

---

## Running the main experiments

Run all commands from the repository root.

### 1. Noise ranking: single models

This evaluates clean-trained single neural networks under noisy test inputs.

```bash
python NeuralNet_3Ks_Noise_Ranking.py --mode single
```

Main output:

```text
Results_NN/O2_novib/Noise_Error_Rankings/
```

### 2. Noise ranking: seed ensemble

This evaluates an ensemble formed by averaging predictions from multiple neural-network seeds.

```bash
python NeuralNet_3Ks_Noise_Ranking.py --mode ensemble
```

Main output:

```text
Results_NN/O2_novib/Ensemble_Noise_Error_Rankings/
```

### 3. Individual-noise sensitivity

This experiment tests how performance changes when noise is applied only to specific input species or subsets of input species.

For the final ensemble individual-noise results:

```bash
python NeuralNet_3Ks_Noise_Sensitivity.py --mode ensemble
```

Main output:

```text
Results_NN/O2_novib/Ensemble_Individual_Noise/
```

A single-model version is also available:

```bash
python NeuralNet_3Ks_Noise_Sensitivity.py --mode single
```

### 4. Noisy-training robustness

This experiment trains models with noisy input augmentation and evaluates whether this improves robustness under noisy test inputs.

```bash
python NeuralNet_3Ks_NoisyTraining.py --mode all
```

Main output:

```text
Results_NN/O2_novib_noisy/NoisyTraining_PIC1_Robustness/
```

Other available modes:

```bash
python NeuralNet_3Ks_NoisyTraining.py --mode single
python NeuralNet_3Ks_NoisyTraining.py --mode ensemble
python NeuralNet_3Ks_NoisyTraining.py --mode comparison
```

---


## Plotting scripts

### `plot_pic1_results.py`

This script merges the normal result-plotting workflows. It reads aggregate CSV files generated by the experiment scripts and produces final analysis figures.

Available modes:

```bash
python plot_pic1_results.py --figure all
python plot_pic1_results.py --figure species-count
python plot_pic1_results.py --figure ensemble-comparison
python plot_pic1_results.py --figure ensemble-individual-noise
python plot_pic1_results.py --figure noisy-training
```

Recommended robust command:

```bash
python plot_pic1_results.py --figure all --skip-missing
```

The main generated plot folders include:

```text
Results_NN/O2_novib/PIC1_Ensemble_Comparison/
Results_NN/O2_novib/PIC1_Ensemble_Individual_Noise_TwoPanel/
Results_NN/O2_novib_noisy/NoisyTraining_PIC1_Robustness/PIC1_NoisyTraining_Ensemble_Comparison/
```

### `plot_pic1_powerpoint_figures.py`

This script generates presentation-ready versions of the figures with larger labels, thicker lines, heavier axes, and vector-friendly outputs.

```bash
python plot_pic1_powerpoint_figures.py
```

Main output:

```text
Results_NN/PIC1_PowerPoint_Figures/
```

---

## Project report

The complete written report is available here:

[`docs/Report.pdf`](docs/Report.pdf)

It contains the detailed motivation, methodology, dataset description, model setup, experiments, and discussion of the results.
