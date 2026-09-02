# Scalable Multicollinearity Recovery

Reference implementation and experiments for the paper **"Scalable Multicollinearity Recovery via Mixed-Integer Optimization"** (Chih-Hua Hsu and Ting-Yu Liao, Department of Industrial and Systems Engineering, Chung Yuan Christian University).

Scalable Multicollinearity Recovery (SMR) is a framework for detecting multicollinear relationships. It builds on the mixed-integer quadratic optimization of Bertsimas and Li (2020) but differs in formulation, scalability, and theoretical grounding. SMR adds a correlation-based screen and a parameter-free eigenvector screen, recasts the minimum-support program with a Special Ordered Set (SOS-1) constraint and a closed-form verification step, and introduces an irreducibility test and a residual-guided fast-path completion. Together, these reduce false positives and allow detection to scale from 1,000 to 10,000 predictors.

## Method overview

Given a normalized design matrix `X`, SMR searches for minimal groups of columns that are (near-)linearly dependent. The pipeline has four stages:

1. **Dimensionality-reduction screen** — narrows the search to features likely to be involved in a multicollinear relationship, using either a correlation screen (`Corr_Dimensionality_Reduction`, z-score threshold on each column's strongest off-diagonal correlation) or a parameter-free eigenvector screen (`Eigvec_Dimensionality_Reduction`, features that load significantly on small-eigenvalue eigenvectors).
2. **Minimum-support detection** — a mixed-integer program (`Minimum_Support`) finds the smallest set of columns spanned by the small-eigenvalue subspace, using an SOS-1 constraint to link the binary support to the coefficient vector. The original Bertsimas big-M formulation is retained as `Bertsimas_Minimum_Support` for comparison.
3. **Verification** — each candidate support is confirmed by a closed-form `_inequality_inspection` (smallest eigenvalue below a norm threshold) and an `_irreducibility_inspection` (no proper subset is already collinear).
4. **Fast-path completion** — when a candidate fails the inequality check, a residual-guided greedy step (`_fast_path`) attempts to recover a genuine relationship instead of discarding it.

At 10,000 predictors, SMR maintains 100% detection accuracy while reducing the false-positive rate from about 33% to 9% and runtime from roughly 6,000 seconds to under 200 seconds.

## Repository structure

```
.
├── Scalable_Multicollinearity_Recovery.py     # Core methodology (the Multicollinear class + data generation)
├── Simulation.py                              # Experiment drivers used in the manuscript
├── main.py                                    # Entry point / configuration for reproducing experiments
├── example.py                                 # Example code of using the SMR method
├── Data/                                      # Real-world datasets (_Raw and _Processed CSVs)
└── Results/                                   # Output directory (will be created automatically after running the simulation)
```

### `Scalable_Multicollinearity_Recovery.py`

Core implementation. Key components:

- `Simulation_Data(...)` — generates synthetic design matrices with planted multicollinear relationships of sizes 2, 3, 4, 8, 10, 15, and 20 features.
- `Data_Preprocessing(...)`, `Normalize(...)`, `Drop_Perfect(...)` — helpers for preparing real-world data (one-hot encoding, column normalization, removal of perfectly collinear columns).
- `Multicollinear` — the main class. It bundles the two screens (`Corr_Dimensionality_Reduction`, `Eigvec_Dimensionality_Reduction`), the minimum-support solvers (`Minimum_Support`, `Bertsimas_Minimum_Support`), the verification steps (`Inspection`, `_inequality_inspection`, `_irreducibility_inspection`), the fast-path (`_fast_path`), and the top-level detectors `Enhanced_Detection` / `Ablation_Detection` and `Bertsimas_Detection`. The scoring utilities `Multicollinear_score` and `Reduction_score` report the accuracy and false-positive rate.

### `Simulation.py`

Experiment drivers that call the core methods and write results:

- `run_ablation_simulation(...)` — synthetic ablation study. Runs Methods A–E, SMR (Corr and Eigen variants), and the Bertsimas/Original baseline on generated datasets, then runs paired Wilcoxon / t-tests. The seven configurations isolate one component at a time (screen, inequality inspection, irreducibility inspection, fast-path recovery).
- `run_reduction_simulation(...)` — compares the eigenvector screen against the correlation screen at several z thresholds.
- `run_SMR_comparison` — runs SMR-Corr and SMR-Eigvec across minimum-coefficient levels 0.0–1.0.
- `run_realworld_detection(...)` — runs SMR and/or Bertsimas detection on the pre-processed real-world datasets.

#### Ablation Study

| Method      | Screen  | Inequality Inspection | Irreducibility Inspection | Fast-path recovery |
|-------------|---------|:---------------------:|:-------------------------:|:-------------------:|
| A           | Corr    | –                     | –                         | –                   |
| B           | Eigvec  | –                     | –                         | –                   |
| C           | –       | ✓                     | –                         | –                   |
| D           | –       | ✓                     | ✓                         | –                   |
| E           | Corr    | ✓                     | ✓                         | –                   |
| SMR-Corr    | Corr    | ✓                     | ✓                         | ✓                   |
| SMR-Eigvec  | Eigvec  | ✓                     | ✓                         | ✓                   |

### `main.py`

Configuration and entry point. Set `BASE_DIR` for output, choose which experiments to run via the `RUN_*` flags, and select data scales. The two scales follow Table 2 of the manuscript:

| Scale | n      | p      | Relationship counts (MR2..MR20) |
|-------|--------|--------|---------------------------------|
| small | 2,000  | 1,000  | `[0, 5, 3, 1, 1, 0, 0]`         |
| large | 20,000 | 10,000 | `[0, 3, 3, 1, 1, 1, 1]`         |

The `MR` list is ordered `[MR2, MR3, MR4, MR8, MR10, MR15, MR20]`, where `MRk` is the number of planted relationships involving `k` features.

### `Data/`

Real-world datasets used in `run_realworld_detection`. Each dataset is stored as a `_Raw.csv` (original) and a `_Processed.csv` (cleaned, ready for detection) pair. Detection reads the `_Processed.csv` files.

| Dataset (base name)                        | Source | Sample size | Raw Features | Processed Features  |
|--------------------------------------------|--------|-------------|--------------|---------------------|
| mtp_Ver1                                   | OpenML | 4,450       | 202          | 194                 |
| Image Segmentation                         | UCI    | 210         | 19           | 18                  |
| Statlog (Image Segmentation)               | UCI    | 2,310       | 19           | 18                  |
| Vertebral Column                           | UCI    | 310         | 6            | 6                   |
| Energy Efficiency                          | UCI    | 768         | 8            | 8                   |
| Electrical Grid Stability Simulated Data   | UCI    | 10,000      | 12           | 12                  |

## Requirements

- Python 3.11+
- [Gurobi](https://www.gurobi.com/) with `gurobipy` and a valid license (required for the mixed-integer optimization; free academic licenses are available)
- `numpy`, `pandas`, `scipy`, `openpyxl`, `rich`

```bash
conda create -n SMR python=3.11
conda activate SMR
pip install -r requirements.txt
```

## Usage

Configure and run the experiments from `main.py`:

```bash
python main.py
```

Open `main.py` and set the flags for the experiments you want:

- `RUN_SCALES` — which data scales (`'small'`, `'large'`) to run.
- `RUN_ABLATION` — synthetic ablation study plus significance tests.
- `RUN_REDUCTION` — eigenvector vs. correlation screen comparison.
- `RUN_SMR_COMPARISON` — SMR-Corr vs. SMR-Eigvec across minimum-coefficient levels 0.0–1.0.
- `RUN_REALWORLD` — detection on the datasets in `Data/`.

Shared settings such as `NOISE_SCALE`, `SIMULATIONS`, and `SEED` are also defined at the top of `main.py`. Results are written under `Results/` (created automatically), organized into `Ablation/`, `Reduction/`, `SMR Comparison/`, and `RealWorld/` subfolders with performance tables (`Performance.xlsx`), detected/ground-truth workbooks, and, when enabled, per-simulation trace CSVs.

### Using the detector directly (`example.py`)

```python
import numpy as np
import Scalable_Multicollinearity_Recovery as SMR

# Synthetic example: 2000 x 1000 design with planted relationships
indices, coef, X, feature_col = SMR.Simulation_Data(
    n=2000, p=1000, MR=[0, 5, 3, 1, 1, 0, 0], noise_scale=0.01, rand_seed=911122
)

detector = SMR.Multicollinear(
    reduction=True, reduction_method='eigvec',
    Inequality_Inspection=True, Irreducibility_Inspection=True, fastpath=True
)
relationships = detector.SMR_Main(X=X, col_names=feature_col)
acc, fpr = detector.Multicollinear_score(relationships, indices)
print(f'Accuracy: {acc:.1f}%, False-positive rate: {fpr:.1f}%')
```

## Citation

If you use this code, please cite:

> Chih-Hua Hsu and Ting-Yu Liao. *Scalable Multicollinearity Recovery via Mixed-Integer Optimization.*

The method builds on:

> Bertsimas, D., & Li, M. L. 2020. “Scalable Holistic Linear Regression.” Operations Research Letters 48 (3): 203-208. https://doi.org/10.1016/j.orl.2020.02.008
