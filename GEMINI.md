# GEMINI.md — FPL Optimization Project

Academic study project (2023, "IIM Blasters" team) for Fantasy Premier League prediction and team selection using ML-based point prediction and constrained portfolio optimization.

There are two distinct systems in this repo: the **original notebook pipeline** (research/historical) and the **weekly production pipeline** (`src/pipeline/`, live GW use).

---

## Tech Stack

- **Python**: pandas, scikit-learn, xgboost, pulp, beautifulsoup4
- **R**: lpSolve — legacy notebooks only. Do not extend. PuLP has replaced it for live optimization.

---

## Directory Structure

```
fpl-assistant/
├── _original/          # Archived original files — do not modify (git-ignored)
├── data/
│   └── Fantasy-Premier-League/  # vaastav dataset — clone separately (see Data Setup)
├── docs/               # Research paper, methodology notes, plans
├── notebooks/          # 7 Jupyter notebooks (original research pipeline)
├── src/
│   ├── config.py           # Central config — ACTIVE_MODEL path, seasons, API URLs
│   ├── data_collection/    # 9 Python scripts — FPL API, Understat, FBref scrapers
│   ├── optimization/       # R scripts (legacy, do not extend)
│   └── pipeline/           # Weekly production pipeline (Python/PuLP)
│       ├── prepare.py      # Build multi-season dataset; attaches persistent player code
│       ├── features.py     # Vectorized rolling/momentum feature engineering
│       ├── predict.py      # Load model, generate xP predictions
│       ├── availability.py # Hybrid availability filter (hard exclude / soft scale)
│       ├── optimize.py     # PuLP ILP — squad + XI + captain selection
│       ├── fetch.py        # FPL API calls with exponential backoff
│       └── run.py          # CLI entry point (4 phases)
├── tests/              # pytest unit + integration tests (64 tests)
├── scripts/
│   └── fetch_bootstrap_snapshots.py  # Called by GitHub Actions daily_bootstrap.yml
├── models/             # Trained .sav files — git-ignored; regenerate via retrain
├── results/            # Output CSVs (xi_gwN.csv, squad_gwN.csv, snapshots/)
├── logs/               # Local run output — git-ignored
├── requirements.txt
└── .gitignore
```

---

## Weekly Pipeline — Quick Start

```bash
pip install -r requirements.txt
git clone https://github.com/vaastav/Fantasy-Premier-League.git data/Fantasy-Premier-League
```

### CLI Phases

```bash
# Phase 1 — before GW deadline: fetch bootstrap, save xP snapshot
python -m src.pipeline.run pre-deadline

# Phase 2 — generate team selection
python -m src.pipeline.run predict --gw <N>

# Phase 3 — after GW ends: collect live results
python -m src.pipeline.run post-gw

# Phase 4 — retrain model on expanded data
python -m src.pipeline.run retrain --gw <N>

# Or run phases 1+2 together
python -m src.pipeline.run full
```

### Model Management

`ACTIVE_MODEL` in `src/config.py` controls which model is used. To promote a newly trained model:
```python
# src/config.py
ACTIVE_MODEL = MODELS_DIR / "rf_model_gw<N>.sav"
```
If the model file is missing or has mismatched feature names, the pipeline automatically falls back to FPL API `ep_next` values.

### Running Tests

```bash
python -m pytest tests/ -q                          # unit tests only (fast)
python -m pytest tests/test_integration.py -v       # requires vaastav clone
```

---

## Original Notebook Pipeline

The notebooks are numbered in execution order and represent the original research pipeline (not used for live GW picks):

| Step | Notebook | Description |
|------|----------|-------------|
| 1 | `01_eda.ipynb` | Exploratory data analysis |
| 2 | `02_feature_engineering.ipynb` | Rolling averages, momentum, xG/xA features |
| 3 | `03_player_clustering.ipynb` | Cluster players for cold-start handling |
| 4 | `04_model_training.ipynb` | Train Random Forest + XGBoost (general) |
| 5 | `05_model_training_positional.ipynb` | Positional models (GK/DEF/MID/FWD) |
| 6 | `06_team_optimization.ipynb` | Prepare predicted points for optimizer |
| 7 | `07_team_key_mapping.ipynb` | Player key reconciliation |

---

## Data Setup

Both pipelines depend on the **vaastav/Fantasy-Premier-League** dataset:

```bash
git clone https://github.com/vaastav/Fantasy-Premier-League.git data/Fantasy-Premier-League
```

**Note:** vaastav stopped weekly updates after 2024-25. The weekly pipeline patches current-season data directly from the live FPL API (`pre-deadline` and `post-gw` phases).

---

## FPL Constraints (optimizer)

- Budget: £100M (1000 in 0.1M units)
- Squad: 2 GK + 5 DEF + 5 MID + 3 FWD (15 players)
- XI: 1 GK, 3–5 DEF, 2–5 MID, 1–3 FWD (11 players)
- Max 3 players from any single club

---

## Gotchas

- **Element ID recycling**: FPL recycles `element` IDs each season. The pipeline uses the `code` field from `players_raw.csv` as the persistent cross-season player identifier. Never group historical GW data by `element` alone — always use `code`.
- **Stale models**: `rf_model.sav` (original notebooks) has 117 features with old naming (`1_assists` etc). Current pipeline uses 18 features (`assists_roll_4` etc). After cloning, run `retrain` before `predict` or the pipeline falls back to API xP.
- **models/ is git-ignored**: `.sav` files must be regenerated. After cloning, run `python -m src.pipeline.run retrain --gw <latest>`.
- **Bootstrap cache**: `results/snapshots/bootstrap_gw<N>.json` is used if < 48h old. Delete it to force a fresh API fetch.
- **Windows junction for data**: If using git worktrees, `data/Fantasy-Premier-League` is not copied. Create a directory junction: `mklink /J .worktrees/<branch>/data/Fantasy-Premier-League data/Fantasy-Premier-League`.

---

## Known Issues

- **Dependency versions**: Original notebooks used 2019 packages. `requirements.txt` is updated but some notebooks may need minor adjustments for deprecated pandas/sklearn APIs.
- **R layer is legacy**: `src/optimization/` R scripts remain for historical reproducibility only. Do not extend them.
- **vaastav data gaps**: 2025-26 season data is not in vaastav. The weekly pipeline bridges this via live FPL API patches, but Understat/FBref features are unavailable for the current season.