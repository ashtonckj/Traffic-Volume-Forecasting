# Traffic Volume Forecasting

Hourly forecasting of the **Metro Interstate Traffic Volume** dataset, comparing four
models on the same data, the same 70/15/15 chronological split, and the same weekly
seasonal differencing (`D = 1`, `s = 168`):

| Model | Script |
| --- | --- |
| Seasonal Naive (benchmark) | `src/models/seasonal_naive.R` |
| SARIMA / SARIMAX | `src/models/sarima.R` |
| XGBoost | `src/models/xg_boost.R` |
| LSTM | `src/models/lstm.R` |

---

## 1. Requirements

- **R 4.2 or newer** (RStudio recommended)
- For the LSTM only: **Python 3** with TensorFlow/Keras (see note below)

### Install the R packages

Run this once in the R console:

```r
install.packages(c(
  "dplyr", "tidyverse", "lubridate", "zoo", "reshape2", "tibble",
  "ggplot2", "gridExtra", "forecast", "tseries", "xgboost",
  "jsonlite", "keras3"
))
```

### Extra step for the LSTM script

`keras3` needs a Python backend. Run this once, after installing the packages above:

```r
keras3::install_keras()
```

If you would rather skip the deep-learning part, comment out the `"src/models/lstm.R"`
line in `main.R` — every other model runs on R packages alone.

---

## 2. How to run

**`main.R` is the only file you need to run.** It sources every script in the correct
order (preprocessing -> figures -> models) and prints the timing of each step.

In RStudio: open `Traffic-Volume-Forecasting.Rproj`, open `main.R`, and click **Source**.

Or from a terminal in the project folder:

```bash
Rscript main.R
```

All paths in the project are relative to the project root, so make sure the working
directory is this folder (opening the `.Rproj` file does that automatically).

Everything the scripts produce is written into `data/processed/` and `output/`. The
results of a previous run are already committed, so the folders can be inspected without
running anything.

> Note: `src/tuning/tuning_lstm.R` is intentionally commented out in `main.R`. It is the
> hyperparameter search for the LSTM and takes roughly 8 hours. Its winning configuration
> is already hard-coded in `src/models/lstm.R`, so it does not need to be re-run.

---

## 3. File directory

```
Traffic-Volume-Forecasting/
│
├── main.R                     <-- RUN THIS. Sources every script below in order.
├── Traffic-Volume-Forecasting.Rproj
├── README.md
│
├── data/
│   ├── raw/
│   │   └── Metro_Interstate_Traffic_Volume.csv   Original dataset (unmodified)
│   └── processed/
│       ├── traffic_volume_processed.csv          THE shared dataset. Written by preprocessing.R:
│       │                                         cleaned hourly grid + train/validation/test
│       │                                         labels. All four models read only this file.
│       └── *.rds, scale_params.csv, ...          Leftover intermediates from earlier versions
│                                                 of the pipeline; not used by the current scripts.
│
├── src/
│   ├── preprocessing/
│   │   └── preprocessing.R                       Deduplicates, fills gaps to a strict hourly
│   │                                             grid, builds calendar features, writes the split
│   ├── figures/
│   │   ├── figures_exploratory_data_analysis.R   -> output/exploratory/
│   │   └── figures_time_series_diagnostics.R     -> output/diagnostics/ (ACF/PACF, STL, ADF/KPSS)
│   ├── models/
│   │   ├── seasonal_naive.R                      Benchmark model
│   │   ├── sarima.R                              SARIMA / SARIMAX
│   │   ├── xg_boost.R                            XGBoost
│   │   └── lstm.R                                LSTM
│   └── tuning/
│       └── tuning_lstm.R                         LSTM hyperparameter search (~8 h, not run by main.R)
│
└── output/                                       All generated results
    ├── exploratory/                              Histograms, boxplots, hourly profiles, correlations
    ├── diagnostics/                              Stationarity and autocorrelation diagnostics
    └── models/
        ├── seasonal_naive/
        ├── sarima/            SARIMA without exogenous variables
        ├── sarimax_holiday/   + holiday dummy
        ├── sarimax_full/      + holiday and weather variables
        ├── xgboost/
        └── lstm/
```
