# CLAUDE.md — FPL Optimization Project

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
├── _original/                # Archived research code (git-ignored, NOT FOR EXTENSION)
│   ├── notebooks/            #   7 Jupyter notebooks (research/historical only)
│   ├── optimization/         #   10 R scripts (lpSolve; DO NOT USE — PuLP replaced)
│   └── data_collection/      #   10 Python scripts (legacy FPL/Understat/FBref)
├── data/
│   └── Fantasy-Premier-League/  # vaastav dataset — clone separately (see Data Setup)
├── docs/
│   ├── fpl-rules.md             # Full FPL constraint reference (scoring, transfers, chips)
│   ├── glossary.md              # All pipeline variables: identity, FDR, features, optimizer, config
│   ├── improvements-roadmap.md  # P1-P5 roadmap with research insights
│   └── superpowers/specs/       # Implementation specs for improvements
├── src/
│   ├── config.py                # Central config — ACTIVE_MODEL path, seasons, API URLs, load_user_config()
│   └── pipeline/                # Active weekly production pipeline (Python/PuLP)
│       ├── prepare.py           # Build multi-season dataset; attaches persistent player code
│       ├── features.py          # Vectorized rolling/momentum feature engineering
│       ├── predict.py           # Load model, generate xP predictions; saves predictions_gw{N}.csv
│       ├── availability.py      # Hybrid availability filter (hard exclude / soft scale)
│       ├── optimize.py          # PuLP ILP — squad + XI + captain selection
│       ├── fetch.py             # FPL API calls with exponential backoff
│       ├── user.py              # UserTeamState dataclass; fetch squad/bank/FTs from FPL API
│       ├── recommend.py         # Multi-GW ILP transfer planner with FDR weighting
│       ├── analysis.py          # Post-match analysis: prediction misses, dream team, accuracy log
│       └── run.py               # CLI entry point (6 phases)
├── tests/                   # pytest unit + integration tests (115 tests)
├── scripts/
│   └── fetch_bootstrap_snapshots.py  # Called by GitHub Actions daily_bootstrap.yml
├── models/                  # Trained .sav files — git-ignored; regenerate via retrain
├── results/                 # Output CSVs (xi_gwN.csv, squad_gwN.csv, predictions_gwN.csv,
│                            #              recommend_gwN.csv, accuracy_log.csv, snapshots/)
├── logs/                    # Local run output — git-ignored
├── user_config.example.yaml # Template — copy to user_config.yaml and fill in entry_id
├── user_config.yaml         # User team IDs & preferences (git-ignored, required for recommend/post-gw analysis)
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

# Phase 2 — generate team selection (also saves results/predictions_gw{N}.csv)
python -m src.pipeline.run predict --gw <N>

# Phase 2b — transfer recommendations (requires user_config.yaml + predictions_gw{N}.csv)
python -m src.pipeline.run recommend --gw <N>
python -m src.pipeline.run recommend --gw <N> --horizon 3          # plan 3 GWs ahead
python -m src.pipeline.run recommend --gw <N> --wildcard           # unconstrained rebuild
python -m src.pipeline.run recommend --gw <N> --team alt           # use alt team from config

# Phase 3 — after GW ends: collect live results + post-match analysis
python -m src.pipeline.run post-gw

# Phase 4 — retrain model on expanded data
python -m src.pipeline.run retrain --gw <N>

# Or run phases 1+2 together
python -m src.pipeline.run full
```

**`recommend` flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--gw N` | — | Target gameweek (must match a saved `predictions_gw{N}.csv`) |
| `--horizon N` | from `user_config.yaml` | GWs to plan ahead (1–5) |
| `--wildcard` | false | Treat as wildcard/free-hit — full unconstrained squad rebuild |
| `--team KEY` | `default` | Team key from `user_config.yaml` (`default` or `alt`) |

### Model Management

`ACTIVE_MODEL` in `src/config.py` controls which model is used. To promote a newly trained model:
```python
# src/config.py
ACTIVE_MODEL = MODELS_DIR / "rf_model_gw<N>.sav"
```
If the model file is missing or has mismatched feature names, the pipeline automatically falls back to FPL API `ep_next` values.

### Model Promotion via GitHub Releases

The daily GitHub Actions workflow downloads the model from the latest GitHub Release before running predict. To promote a newly retrained model:

```bash
# 1. Retrain locally
python -m src.pipeline.run retrain --gw <N>

# 2. Create a GitHub Release with the model as an asset
gh release create "gw<N>" models/rf_model_gw<N>.sav \
  --title "Model GW<N>" \
  --notes "Retrained after GW<N> with <M> seasons of data."

# 3. Update ACTIVE_MODEL in src/config.py for local runs
# ACTIVE_MODEL = MODELS_DIR / "rf_model_gw<N>.sav"
```

The workflow downloads all `*.sav` assets from the latest release tagged `gw*`. The release tag must start with `gw` (e.g., `gw34`, `gw35`). `gh release list` returns releases in reverse chronological order — the first result is used.

**Secrets required (set in GitHub repo → Settings → Secrets):**
- `DISCORD_PRICE_CHANGE_WEBHOOK_URL` — daily price-change notifications (rename from `DISCORD_WEBHOOK_URL`)
- `DISCORD_DEADLINE_WEBHOOK_URL` — deadline approaching alert
- `DISCORD_PREDICT_RECOMMEND_WEBHOOK_URL` — predict + recommend results summary
- `USER_CONFIG_YAML` — full contents of `user_config.yaml` (required for predict/recommend auto-trigger)

**Migration note:** The existing `DISCORD_WEBHOOK_URL` secret must be renamed to `DISCORD_PRICE_CHANGE_WEBHOOK_URL` in GitHub repo settings, or the price-change step will silently skip.

### Running Tests

```bash
python -m pytest tests/ -q                          # unit tests only (fast)
python -m pytest tests/test_integration.py -v       # requires vaastav clone
```

---

## Archived Research Code

**DO NOT EXTEND OR USE `_original/` CODE**

The original research notebooks, R optimization layer, and legacy data collection scripts are archived in `_original/` for historical reference only:

- `_original/notebooks/`: 7 Jupyter notebooks with early ML experiments and optimization ideas
- `_original/optimization/`: 10 R scripts using lpSolve (superseded by Python/PuLP)
- `_original/data_collection/`: legacy scrapers for FPL, Understat, FBref (not maintained)

**Why archived?** These were proof-of-concept research. The active production system is in `src/pipeline/` (Python, PuLP-based). Any improvements should extend the active pipeline, not the archived code.

**Learn from archives:** See `docs/improvements-roadmap.md` for key learnings and improvement ideas (P1-P5) derived from this research.

---

## Data Setup

Both pipelines depend on the **vaastav/Fantasy-Premier-League** dataset:

```bash
git clone https://github.com/vaastav/Fantasy-Premier-League.git data/Fantasy-Premier-League
```

**Note:** vaastav stopped weekly updates after 2024-25. The weekly pipeline patches current-season data directly from the live FPL API (`pre-deadline` and `post-gw` phases).

---

## FPL Constraints (optimizer)

- Budget: £100M
- Squad: 2 GK + 5 DEF + 5 MID + 3 FWD (15 players)
- XI: 1 GK, 3–5 DEF, 2–5 MID, 1–3 FWD (11 players)
- Max 3 players from any single club
- Transfers: Up to 5 free transfers per gameweek (banking allowed, cap 5), -4 points per extra transfer
- Chips: 2× Wildcard, 2× Freehit, 2× Bench Boost, 2× Triple Captain (split across season halves)

**Full rules:** See `docs/fpl-rules.md` for complete scoring, BPS, transfer, and chip mechanics.

---

## Gotchas

- **Element ID recycling**: FPL recycles `element` IDs each season. The pipeline uses the `code` field from `players_raw.csv` as the persistent cross-season player identifier. Never group historical GW data by `element` alone — always use `code`.
- **Stale models**: `rf_model.sav` (original notebooks) has 117 features with old naming (`1_assists` etc). Current pipeline uses 18 features (`assists_roll_4` etc). After cloning, run `retrain` before `predict` or the pipeline falls back to API xP.
- **models/ is git-ignored**: `.sav` files must be regenerated. After cloning, run `python -m src.pipeline.run retrain --gw <latest>`.
- **Bootstrap cache**: `results/snapshots/bootstrap_gw<N>.json` is used if < 48h old. Delete it to force a fresh API fetch.
- **FDR column naming**: Despite sounding symmetric, `fdr_team` and `fdr_opp` are NOT interchangeable for xP weighting. `fdr_team` = how hard the fixture is FOR the player's team (spans 1–5, use this). `fdr_opp` = how hard the fixture is for the opponent (reflects the player's own team quality — near-constant 4–5 for elite teams like Arsenal all season, giving no signal). Always use `fdr_team` in `fdr_weight = 1.0 - fdr_sensitivity × (fdr_team − 3) / 2`. See `docs/glossary.md`.
- **Windows junction for data**: If using git worktrees, `data/Fantasy-Premier-League` is not copied. Create a directory junction: `mklink /J .worktrees/<branch>/data/Fantasy-Premier-League data/Fantasy-Premier-League`.

---

## Known Issues

- **R layer archived**: `src/optimization/` R scripts moved to `_original/optimization/`. Do not restore them.
- **vaastav data gaps**: 2025-26 season data is not in vaastav. The weekly pipeline bridges this via live FPL API patches, but Understat/FBref features are unavailable for the current season.
