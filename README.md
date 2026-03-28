# FPL Optimization — IIM Blasters

![Python](https://img.shields.io/badge/Python-3.12%2B-blue?logo=python)
![R](https://img.shields.io/badge/R-4.0%2B-276DC3?logo=r)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.1%2B-F7931E?logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7%2B-red)
![PuLP](https://img.shields.io/badge/PuLP-2.7%2B-orange)
![License](https://img.shields.io/badge/License-Academic-lightgrey)

A research project combining machine learning prediction and portfolio optimization to select optimal Fantasy Premier League (FPL) squads. Developed by the **IIM Blasters** team as an academic study at the Indian Institute of Management (2023).

The system predicts per-player expected points (xP) using Random Forest and XGBoost models trained on 8+ seasons of historical data (2016–2024), then applies linear programming and portfolio theory to select the highest-value 15-player squad under official FPL constraints.

---

## Weekly Pipeline (GW32–38, 2025-26)

A new automated weekly pipeline built for the live 2025-26 season (GW32–38). Fully Python — replaces the R lpSolve optimizer with PuLP, and replaces notebook-by-notebook execution with a single CLI.

### How it works

Four scheduled phases per gameweek:

```
  PRE-DEADLINE              POST-DEADLINE             POST-GW
  ─────────────             ─────────────             ──────────
  Fetch bootstrap           (no action)               Fetch live
  Capture ep_next xP                                  player data
  Save bootstrap                                      Save gw{N}_live.csv
  snapshot
         │                                                  │
         ▼                                                  ▼
  PREDICT (run after deadline)              RETRAIN (manual, on-demand)
  ───────────────────────────               ────────────────────────────
  Build merged dataset                      Rebuild features on full
  Engineer features                         historical + live dataset
  Generate xP predictions                   Retrain Random Forest
  Filter player availability                Save rf_model_gw{N}.sav
  Optimize team (PuLP ILP)                  Print MAE vs old model
  Save xi_gw{N}.csv + squad_gw{N}.csv
```

**Data layers:**

| Layer | Source | Coverage |
|-------|--------|----------|
| Historical base | vaastav `merged_gw.csv` | 2016-17 through 2024-25 (static) |
| Current season base | vaastav `merged_gw.csv` | 2025-26 GW1–29 (updated ~3x/season) |
| Current season live | FPL API `element-summary/{id}/` | GW30+ (fetched after each GW) |

Live rows are only used for GWs not already covered in vaastav (dedup prefers vaastav).

**Player availability filtering** uses a decision table (first-match-wins):

| Status | Chance of playing | Action |
|--------|-------------------|--------|
| `i`, `u`, `s`, `n` | any | Hard exclude |
| any | 0% or 25% | Hard exclude |
| any | 50% | xP × 0.50 |
| `d` | null | xP × 0.50 |
| any | 75% | xP × 0.75 |
| `a` | 100% or null | No change |

### Setup

```bash
# Python dependencies (includes pytest, PuLP)
pip install -r requirements.txt

# Clone vaastav dataset (if not already present)
git clone https://github.com/vaastav/Fantasy-Premier-League.git data/Fantasy-Premier-League
```

### Running the weekly pipeline

```bash
# Phase 1: Before each GW deadline — fetch bootstrap and capture xP snapshot
python -m src.pipeline.run pre-deadline

# Phase 2: After deadline — generate predictions and optimal team
python -m src.pipeline.run predict --gw 32

# Phase 3: After GW completes — collect actual results and live player data
python -m src.pipeline.run post-gw

# Phase 4: Retrain model (manual, run when you have enough new data)
python -m src.pipeline.run retrain --gw 32

# Or run phases 1+2 together
python -m src.pipeline.run full
```

The `predict` phase prints the optimal XI directly and saves `results/xi_gw{N}.csv` and `results/squad_gw{N}.csv`.

**Using the cron wrapper (Linux/macOS/WSL):**

```bash
chmod +x scripts/weekly_run.sh
./scripts/weekly_run.sh pre-deadline   # logs to logs/pre-deadline_<timestamp>.log
./scripts/weekly_run.sh predict --gw 32
```

### GW32–38 Calendar

| GW | Deadline (UTC) |
|----|----------------|
| 32 | Apr 10, 2026 17:30 |
| 33 | Apr 18, 2026 10:00 |
| 34 | Apr 24, 2026 17:30 |
| 35 | May 2, 2026 12:30 |
| 36 | May 9, 2026 12:30 |
| 37 | May 17, 2026 12:30 |
| 38 | May 24, 2026 13:30 |

**Pre-GW checklist:**
1. `python -m src.pipeline.run pre-deadline` — captures xP and bootstrap snapshot
2. `python -m src.pipeline.run predict --gw <N>` — generates optimal team
3. Review the XI output — check excluded/scaled players make sense
4. Make transfers in the FPL app
5. After GW: `python -m src.pipeline.run post-gw` to collect live results

### Model management

The active model is configured in `src/config.py`:

```python
ACTIVE_MODEL = MODELS_DIR / "rf_model.sav"
```

To promote a retrained model:
1. Run `python -m src.pipeline.run retrain --gw <N>` — saves `models/rf_model_gw<N>.sav` and prints MAE comparison
2. If the new model is better, update `ACTIVE_MODEL` in `src/config.py`

Without a trained model the pipeline falls back to FPL API `ep_next` values (captured during `pre-deadline`).

### Running tests

```bash
pytest tests/ -v

# Unit tests only (faster)
pytest tests/ -v --ignore=tests/test_integration.py

# Integration test (requires vaastav dataset)
pytest tests/test_integration.py -v -s
```

---

## Pipeline Overview (Original Notebook-Based System)

```
┌─────────────────────────────────────────────────────────────────────┐
│                        FPL OPTIMIZATION PIPELINE                    │
└─────────────────────────────────────────────────────────────────────┘

  ┌──────────────┐    ┌──────────────────┐    ┌──────────────────────┐
  │  DATA        │    │  FEATURE         │    │  ML PREDICTION       │
  │  COLLECTION  │───▶│  ENGINEERING     │───▶│                      │
  │              │    │                  │    │  Random Forest       │
  │  FPL API     │    │  Rolling avgs    │    │  XGBoost             │
  │  Understat   │    │  Momentum        │    │  (positional models) │
  │  FBref       │    │  xG / xA         │    │                      │
  │  vaastav/FPL │    │  Pressure stats  │    │  Output: xP per      │
  │  (2016-2024) │    │  Player clusters │    │  player per GW       │
  └──────────────┘    └──────────────────┘    └──────────┬───────────┘
                                                         │
                                              ┌──────────▼───────────┐
                                              │  PORTFOLIO           │
                                              │  OPTIMIZATION (R)    │
                                              │                      │
                                              │  lpSolve (ILP)       │
                                              │  xP maximization     │
                                              │  Sharpe ratio        │
                                              │  Min-risk            │
                                              │                      │
                                              │  Output: 15-player   │
                                              │  squad + 11-man XI   │
                                              └──────────────────────┘
```

---

## Directory Structure

```
fpl-optimization/
├── data/
│   └── README.md               # Data sourcing instructions
├── docs/
│   ├── PaperWIP.pdf            # Research paper
│   ├── Methodology.pptx        # Project methodology slides
│   └── literature/             # 10 academic reference papers
├── notebooks/
│   ├── 01_eda.ipynb                      # Exploratory data analysis
│   ├── 02_feature_engineering.ipynb      # Time series feature creation
│   ├── 03_player_clustering.ipynb        # New-player cold-start clustering
│   ├── 04_model_training.ipynb           # Global RF & XGBoost training
│   ├── 05_model_training_positional.ipynb# Position-specific model variants
│   ├── 06_team_optimization.ipynb        # Team selection analysis
│   └── 07_team_key_mapping.ipynb         # Team ID ↔ name mapping utility
├── src/
│   ├── data_collection/        # Python: API collector, cleaners, parsers,
│   │                           #         mergers, Understat, FBref scrapers
│   └── optimization/           # R: optimization scripts (see below)
│       ├── FPL_xPMin.R         # xP * xMin maximization (primary model)
│       ├── FPL.R               # Base LP optimizer with position weights
│       ├── xMin.R              # Expected-minutes calculator
│       ├── Covariance_TotalPoints.R  # Covariance matrix for Sharpe model
│       └── lpSolveSample.R     # lpSolve demonstration / sandbox
├── models/                     # Trained models (git-ignored)
│   ├── rf_model.sav            # Serialized Random Forest
│   └── xgb_model.sav           # Serialized XGBoost
├── results/
│   ├── results_summary_xP.csv  # GW-by-GW xP-max results
│   ├── results_summary_xPMin.csv # GW-by-GW xPMin results
│   ├── results_summary_best.csv  # Retrospective best-possible teams
│   ├── predictions.csv         # Raw model predictions
│   └── results_comparison.xlsx # Cross-strategy comparison
├── plots/                      # 15 visualization PNGs
├── .gitignore
├── requirements.txt
└── CLAUDE.md
```

---

## Setup and Installation

### Prerequisites

- Python 3.8+
- R 4.0+
- Jupyter Notebook or JupyterLab

### 1. Clone the repository

```bash
git clone <repo-url>
cd fpl-optimization
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

Key Python packages:
| Package | Purpose |
|---|---|
| pandas, numpy | Data handling |
| scikit-learn | Random Forest, preprocessing |
| xgboost | Gradient boosting |
| matplotlib, seaborn | Visualization |
| beautifulsoup4, requests | Web scraping (Understat, FBref) |
| scipy | Statistical utilities |

### 3. Install R dependencies

```r
install.packages(c("lpSolve", "tidyverse", "ggplot2", "caret", "readr"))
```

### 4. Download the external dataset

The project uses the [Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League) dataset (vaastav, 9 seasons, 2016–2024):

```bash
git clone https://github.com/vaastav/Fantasy-Premier-League.git data/Fantasy-Premier-League
```

See `data/README.md` for full data setup instructions.

---

## Usage

Run the pipeline in notebook order. Each notebook builds on outputs from the previous stage.

### Stage 1 — Exploratory Data Analysis

```bash
jupyter notebook notebooks/01_eda.ipynb
```

Explores historical player data, points distributions, autocorrelation, and performance by value bracket.

### Stage 2 — Feature Engineering

```bash
jupyter notebook notebooks/02_feature_engineering.ipynb
```

Constructs the time-series feature set from raw gameweek data. Outputs `data/timeseries_dataset.csv`.

Key features created:
- Rolling averages (3-GW, 5-GW windows) for points, goals, assists, minutes
- Momentum indicators (recent-form deltas)
- Advanced stats: xG, xA, expected goal involvements (from Understat/FBref)
- Under-pressure statistics, ICT index components
- Position-weighted expected minutes (xMin)

### Stage 3 — Player Clustering

```bash
jupyter notebook notebooks/03_player_clustering.ipynb
```

Clusters new/unfamiliar players by playing style to assign cold-start xP estimates. Uses silhouette analysis to determine optimal cluster count.

### Stage 4 — Model Training (Global)

```bash
jupyter notebook notebooks/04_model_training.ipynb
```

Trains Random Forest and XGBoost regressors on the full player pool. Includes hyperparameter tuning (max depth, n_estimators, max features) and feature importance analysis. Saves models to `models/`.

### Stage 5 — Positional Model Training

```bash
jupyter notebook notebooks/05_model_training_positional.ipynb
```

Trains separate models for GK, DEF, MID, FWD positions to capture position-specific performance patterns.

### Stage 6 — Team Optimization

```bash
jupyter notebook notebooks/06_team_optimization.ipynb
```

Loads model predictions and runs the R optimization scripts to select optimal squads. Produces `results/` output files.

### Stage 7 — Team Key Mapping

```bash
jupyter notebook notebooks/07_team_key_mapping.ipynb
```

Utility notebook for resolving FPL team IDs to human-readable names across seasons.

### Running the R Optimization Scripts Directly

```r
# From R or RStudio — update setwd() paths first
source("src/optimization/FPL_xPMin.R")   # Primary optimizer
source("src/optimization/FPL.R")         # Base LP optimizer
```

---

## Optimization Strategies

All optimization is formulated as **Integer Linear Programming (ILP)** using the `lpSolve` package in R, selecting a 15-player squad and a starting 11-player XI.

### Squad Constraints (all strategies)

| Constraint | Value |
|---|---|
| Total budget | £100M |
| Squad size | 15 players |
| Starting XI | 11 players |
| Goalkeepers | 2 (squad), 1 (XI) |
| Defenders | 5 (squad), 3–5 (XI) |
| Midfielders | 5 (squad), 2–5 (XI) |
| Forwards | 3 (squad), 1–3 (XI) |
| Max players from same club | 3 |

### Strategy Descriptions

| Script | Strategy | Objective |
|---|---|---|
| `FPL_xPMin.R` | **xPMin Maximization** | Maximize `xP × (xMin/90) × pos_weight` — accounts for injury and rotation risk by weighting predicted points by expected playing time |
| `FPL_xP.R` | **Expected Points Max** | Maximize raw `xP` — pure prediction maximization without minutes adjustment |
| `Covariance_TotalPoints.R` + `FPL_Sharpe.R` | **Sharpe Ratio** | Portfolio theory approach: maximize risk-adjusted returns using the covariance matrix of historical player points. Penalizes correlated players from the same team |
| `FPL_best.R` | **Retrospective Best** | Oracle benchmark: selects the optimal team using actual points (used to evaluate prediction quality) |
| `xMin.R` | **Min-Risk** | Minimizes variance / downside risk subject to a minimum expected-points threshold |

The **xPMin model** (`FPL_xPMin.R`) is the primary strategy, combining ML predictions with a playing-time adjustment:

```
xPMin = xP × (avg_minutes_last_4_GW / 90) × position_weight
```

---

## Model Performance

Results are evaluated gameweek-by-gameweek on the 2022–23 Premier League season (GW1–GW38). Key metrics from `results/results_summary_xP.csv`:

| Metric | Value |
|---|---|
| Average MAE (xP vs actual, per player) | ~0.85 |
| Average predicted team xP per GW | ~96 points |
| Average actual points scored per GW | ~92 points |
| Best single gameweek (predicted/actual) | GW9: 135 xP / 160 actual |

Notable result: GW17 produced a 152-point prediction with 148 actual points — a near-perfect team selection.

Feature importance analysis (see `plots/feature_importance.png`) identifies recent rolling averages and xG/xA-based metrics as the strongest predictors.

---

## Data Sources

| Source | Data | Access |
|---|---|---|
| [FPL Official API](https://fantasy.premierleague.com/api/) | Player prices, points, fixtures, ownership | Public REST API |
| [vaastav/Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League) | Historical GW data, 2016–2024 (9 seasons) | GitHub dataset |
| [Understat](https://understat.com/) | xG, xA, shot maps, expected goal involvements | Web scraping (`src/data_collection/understat.py`) |
| [FBref](https://fbref.com/) | Advanced defensive/pressing stats | Web scraping (`src/data_collection/fbref.py`) |

Raw data files are not tracked in git. Follow instructions in `data/README.md` to reconstruct the dataset locally.

---

## Visualizations

Generated plots are stored in `plots/`. Key outputs include:

| Plot | Description |
|---|---|
| `feature_importance.png` | RF/XGBoost feature importance rankings |
| `PointsDistribution.png` | Distribution of player points across seasons |
| `autocorrel.png` | Autocorrelation of player performance across GWs |
| `boxplot_byvalue.png` | Points distribution by price bracket |
| `silhouette.png` | Silhouette scores for player clustering |
| `rf_nestimators.png` | RF hyperparameter tuning — n_estimators |
| `rf_maxdepth.png` | RF hyperparameter tuning — max depth |
| `PointsvxMin.png` | Correlation: actual points vs expected minutes |
| `hetero_train/test.png` | Residual heteroscedasticity checks |
| `Lastmatchperformance.png` | Per-player last-match performance overview |

---

## Project Structure Notes

- **Language split:** Data collection and ML are in Python; optimization is in R.
- **Models are git-ignored.** Re-run notebooks 04–05 to regenerate `rf_model.sav` and `xgb_model.sav`.
- **Hardcoded paths in R scripts** (`setwd(...)`) were set for the original development environment. Update these paths before running.
- The `_original/` directory contains all original source files exactly as submitted for the academic project; it is git-ignored.

---

## References

Academic papers referenced in this project are archived in `docs/literature/`. The research paper is available at `docs/PaperWIP.pdf`.

Key methodological references:

1. Markowitz, H. (1952). Portfolio Selection. *Journal of Finance*.
2. vaastav. (2019–2024). *Fantasy-Premier-League dataset*. GitHub.
3. Pappalardo et al. (2019). A public data set of spatio-temporal match events in soccer leagues. *Nature Scientific Data*.
4. Breiman, L. (2001). Random Forests. *Machine Learning*.
5. Chen & Guestrin (2016). XGBoost: A Scalable Tree Boosting System. *KDD*.

---

## License

This project was produced as an academic study at the Indian Institute of Management (IIM Blasters team, 2023). It is intended for educational and research purposes.
