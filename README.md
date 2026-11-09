# levy-sde-electricity

Code for the ICLR 2027 submission *When Do Heavy Tails Help? Identifiability Limits and Predictive
Uncertainty in Lévy-Driven Neural SDEs for Electricity Markets*.

We compare a neural SDE driven by α-stable Lévy noise (LévySDE-Net, α = 1.5) with an otherwise
identical Gaussian-driven model (α = 2) and a noise-free neural ODE, on hourly day-ahead electricity
prices for three Nordic price areas (DK1, SE4, NO4). Neural models use five seeds and four horizons,
and are compared with linear AR baselines that have Gaussian, empirical and GARCH(1,1)-t predictive
distributions. Main findings:

- **Point forecasts:** the forcing distribution does not change them. The neural ODE matches the
  Gaussian SDE to four decimal places, and a direct AR(8) is as accurate or more so.
- **Probabilistic forecasts:** after validation calibration, Lévy forcing improves CRPS by widening
  under-dispersed ensembles, not through better tail forecasts. AR(8) + GARCH(1,1)-t is better than
  both SDEs.
- **Forecaster's dilemma:** scoring only the hours with large price moves favours Lévy forcing in
  every market. Threshold-weighted CRPS over all hours, a proper tail score, reverses this.

## Repository contents

| File | Purpose |
|---|---|
| `DK1_revision.ipynb` | Full pipeline for DK1: data, splits, training (77 runs), calibration, all evaluation tables; also the AMI calculation |
| `SE4_revision.ipynb` | Same pipeline for SE4 |
| `NO4_revision.ipynb` | Same pipeline for NO4 |
| `sensititvity.ipynb` | Synthetic recovery of α for a Lévy-driven Ornstein–Uhlenbeck process (hourly and monthly-equivalent sampling) |
| `run_ident.py` | Sample-size study: sd(α̂) for n = 300 to 19,200 (reads the simulator from `sensititvity.ipynb`) |
| `fastlik.py` | Fast α-stable log-likelihood (bicubic interpolation of a SciPy table), used by `run_ident.py` |
| `identifiability_vs_n.csv` | Saved output of `run_ident.py` |
| `cao_fast.py` | Cao's E₁/E₂ embedding statistics for DK1 price differences |
| `original_submission/` | Notebooks from the original submission, kept unchanged for reference |

## Where each result comes from

Each area notebook writes its tables to `revision_runs_<AREA>/tables/`.

| Paper result | Source |
|---|---|
| Point forecasts, all horizons (MSE, MAE) | Area notebooks, cell 5 |
| CRPS, coverage, interval width and spread–skill, before and after calibration | Area notebooks, cells 5 and 8 |
| Diebold–Mariano tests (MSE, CRPS, jump-hour CRPS) | Area notebooks, cell 5 |
| Tail Diebold–Mariano tests (twCRPS), AR(8) + GARCH(1,1)-t baseline, validation CRPS against κ | Area notebooks, cell 6 |
| Stability sweep and comparison with GARCH (α ∈ {1.2, …, 2.0}) | Area notebooks, cells 5 and 7 |
| Threshold-weighted CRPS (×10³), all horizons; forecaster's-dilemma comparison | Area notebooks, cell 8 |
| Ensemble-size convergence (N = 25 to 1,000) | Area notebooks, cell 5 |
| Selected noise multipliers κ\* | Area notebooks, cell 5 |
| Synthetic recovery of α, error correlations | `sensititvity.ipynb` |
| sd(α̂) against sample length | `run_ident.py` → `identifiability_vs_n.csv` |
| Cao E₁/E₂ (DK1) | `cao_fast.py` |
| Average mutual information (DK1) | `DK1_revision.ipynb`, cell after cell 2 |

## Data

The data are not redistributed here. Download them and place the files in the paths set in cell 1
of each notebook (`SOURCES`).

- **DK1:** hourly day-ahead prices (EUR/MWh) from Nord Pool, 30 Aug 2015 – 30 Apr 2021.
- **SE4 and NO4:** `time_series_60min_singleindex.csv` from the
  [Open Power System Data](https://data.open-power-system-data.org/time_series/) time-series package
  (sourced from the ENTSO-E Transparency Platform), Jan 2015 – Sep 2020.

Cell 2 of each notebook sorts the series, removes duplicate timestamps, reports gaps and forms price
differences. Splits are chronological: the last 20% of each series is the test period, and the last
10% of the remaining training period is a validation block, used only for calibrating κ and for
the AR baselines' residual distributions.

## Running

```bash
pip install numpy pandas scipy scikit-learn torch arch
```

1. Open an area notebook and set `AREA` and the data paths in cell 1.
2. Set `FAST_CHECK = True` for a quick end-to-end test (a few minutes), or `False` for the full results.
3. Run all cells. Training (cell 4) is resumable: completed runs are saved and skipped on restart.

The full grid for one area (77 runs: two SDEs at four horizons with five seeds, the neural ODE, the
α sweep and the LSTM) takes 4–9 hours on a single CPU. No GPU is needed.

For the sample-size study, put `run_ident.py`, `fastlik.py` and `sensititvity.ipynb` in the same
folder and run `python run_ident.py` (about 5 minutes).

## Reproducibility

Seeds are fixed for Python, NumPy and PyTorch, and evaluation ensembles use a separate fixed stream
(seed 10,000 + s for training seed s). Deterministic algorithms are not enforced, so results may
differ in the final digits across hardware.
