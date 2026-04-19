# CLAUDE.md — FPL Optimization Project

Fantasy Premier League prediction and team selection using ML-based point prediction (Random Forest) and constrained portfolio optimization (PuLP ILP). Built for live weekly use in the 2025-26 season.

The active production system is `src/pipeline/`. Everything in `_original/` is archived research — do not extend it.

---

## Tech Stack

- **Python**: pandas, scikit-learn, xgboost, pulp, beautifulsoup4, matplotlib

---

## Directory Structure

```
fpl-assistant/
├── _original/                # Archived research code (git-ignored, NOT FOR EXTENSION)
├── data/
│   ├── Fantasy-Premier-League/  # vaastav dataset — clone separately (see Data Setup)
│   └── snapshots/
│       └── 2025-26/
│           └── gw{N}/
│               └── bootstrap.json    # Per-GW bootstrap snapshot
├── docs/
│   ├── fpl-rules.md             # Full FPL constraint reference
│   ├── glossary.md              # All pipeline variables: FDR, features, optimizer, config
│   ├── improvements-roadmap.md  # P1-P5 roadmap
│   └── superpowers/specs/       # Implementation specs
├── src/
│   ├── config.py                # Central config — ACTIVE_MODEL, seasons, API URLs, gw_dir(), snapshot_dir()
│   └── pipeline/
│       ├── prepare.py           # Build multi-season dataset; attaches persistent player code
│       ├── features.py          # Vectorized rolling/momentum feature engineering
│       ├── predict.py           # Load model, generate xP predictions
│       ├── availability.py      # Hybrid availability filter (hard exclude / soft scale)
│       ├── optimize.py          # PuLP ILP — squad + XI + captain selection
│       ├── fetch.py             # FPL API calls with exponential backoff
│       ├── user.py              # UserTeamState dataclass; fetch squad/bank/FTs from FPL API
│       ├── recommend.py         # Multi-GW ILP transfer planner with FDR weighting
│       ├── analysis.py          # Post-match analysis: prediction misses, dream team, accuracy log
│       └── run.py               # CLI entry point (6 phases)
├── tests/                       # pytest unit + integration tests (~350 tests)
├── scripts/
│   ├── fetch_bootstrap_snapshots.py  # Called by GitHub Actions daily_bootstrap.yml
│   └── generate_reports.py           # Generates rank comparison PNGs from accuracy_log.csv
├── models/                      # Trained .sav files — git-ignored; regenerate via retrain
├── results/
│   ├── accuracy_log.csv              # GW-by-GW performance: your pts, recommended pts, xP, rank
│   ├── 2025-26/
│   │   ├── actual_transfers.csv      # All actual transfers made across GWs (appended each post-gw)
│   │   └── gw{N}/
│   │       ├── predictions.csv       # Model xP predictions for all players
│   │       ├── optimal_squad.csv     # 15-player squad; captain = highest xP starter
│   │       ├── recommended_squad.csv # Post-transfer squad (written by recommend phase)
│   │       ├── recommend.csv         # Transfer plan output
│   │       └── actual_squad.csv      # Actual squad played (written by post-gw; GW finished only)
│   └── reports/
│       ├── rank_comparison_gw.png    # Per-GW bar chart: your pts vs optimal vs recommended
│       └── rank_comparison_season.png # Cumulative season line chart
├── user_config.example.yaml    # Template — copy to user_config.yaml and fill in entry_id
├── user_config.yaml            # User team IDs & preferences (git-ignored)
└── requirements.txt
```

---

## Weekly Pipeline

```bash
pip install -r requirements.txt
git clone https://github.com/vaastav/Fantasy-Premier-League.git data/Fantasy-Premier-League
```

### CLI Phases

```bash
# Phase 1 — before GW deadline
#   Writes: data/snapshots/2025-26/gw{N}/bootstrap.json
python -m src.pipeline.run pre-deadline

# Phase 2 — after deadline: generate predictions and optimal team
#   Writes: results/2025-26/gw{N}/predictions.csv
#           results/2025-26/gw{N}/optimal_squad.csv  (captain = highest xP starter)
python -m src.pipeline.run predict --gw <N>

# Phase 2b — transfer recommendations (requires user_config.yaml)
#   Writes: results/2025-26/gw{N}/recommend.csv
#           results/2025-26/gw{N}/recommended_squad.csv
python -m src.pipeline.run recommend --gw <N>
python -m src.pipeline.run recommend --gw <N> --horizon 3     # plan 3 GWs ahead
python -m src.pipeline.run recommend --gw <N> --wildcard      # unconstrained rebuild
python -m src.pipeline.run recommend --gw <N> --team alt      # use alt team from config

# Phase 3 — after GW finishes: collect actual results and update reports
#   Writes: results/2025-26/gw{N}/actual_squad.csv   (only if bootstrap finished=True)
#           results/2025-26/actual_transfers.csv      (appended)
#           results/accuracy_log.csv                  (appended)
#           results/reports/*.png                     (regenerated)
python -m src.pipeline.run post-gw

# Phase 4 — retrain model
python -m src.pipeline.run retrain --gw <N>

# Regenerate performance charts at any time
python scripts/generate_reports.py --from-gw 31
```

**`recommend` flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--gw N` | — | Target gameweek (must match a saved `predictions.csv` under `gw{N}/`) |
| `--horizon N` | from `user_config.yaml` | GWs to plan ahead (1–5) |
| `--wildcard` | false | Full unconstrained squad rebuild |
| `--team KEY` | `default` | Team key from `user_config.yaml` (`default` or `alt`) |

### Model Management

`ACTIVE_MODEL` in `src/config.py` controls which model is used. Promotion is automated by `src/pipeline/promote.py` — running `retrain` triggers benchmarking, GitHub Release, and manifest update. If the model file is missing, the pipeline falls back to FPL API `ep_next` values automatically.

### Running Tests

```bash
python -m pytest tests/ -q                       # all tests
python -m pytest tests/test_integration.py -v    # requires vaastav clone
```

---

## Data Setup

The pipeline uses the **vaastav/Fantasy-Premier-League** dataset for historical data (2016–2024). vaastav stopped weekly updates after 2024-25; the pipeline bridges 2025-26 via the live FPL API.

```bash
git clone https://github.com/vaastav/Fantasy-Premier-League.git data/Fantasy-Premier-League
```

---

## FPL Constraints (optimizer)

- Budget: £100M; Squad: 2 GK + 5 DEF + 5 MID + 3 FWD; XI: 1 GK, 3–5 DEF, 2–5 MID, 1–3 FWD
- Max 3 players from any single club
- Transfers: up to 5 free transfers per GW (banking allowed, cap 5), -4 pts per extra

**Full rules:** `docs/fpl-rules.md`

---

## Gotchas

- **Element ID recycling**: FPL recycles `element` IDs each season. Always use the `code` field as the persistent cross-season player identifier — never group by `element` alone.
- **models/ is git-ignored**: `.sav` files must be regenerated. After cloning, run `retrain` before `predict`.
- **Bootstrap cache**: `data/snapshots/2025-26/gw<N>/bootstrap.json` is used if < 48h old. Delete to force a fresh API fetch.
- **actual_squad.csv guard**: `post-gw` only writes `actual_squad.csv` when `bootstrap["finished"] == True`. Re-run after the GW completes.
- **recommended_pts in accuracy_log**: Scores XI starters only with captain 2×.
- **FDR column naming**: `fdr_team` = difficulty FOR the player's team (use this for xP weighting). `fdr_opp` = difficulty for the opponent (near-constant for elite teams, gives no signal). See `docs/glossary.md`.
- **Windows junction for data**: In git worktrees, `data/Fantasy-Premier-League` is not copied. Create a junction: `mklink /J .worktrees/<branch>/data/Fantasy-Premier-League data/Fantasy-Premier-League`.

---

## CI / Bootstrap Pipeline Notes

The bootstrap snapshot (`data/snapshots/2025-26/gw{N}/bootstrap.json`) is the data input for CI runs. `phase_predict` uses the ML model (downloaded from GitHub Releases) when available; otherwise falls back to `ep_next` from the snapshot. Do not clone vaastav in CI — the ep_next fallback is the intended path.

Key code path (`src/pipeline/run.py::phase_predict`):
- If `build_merged_dataset` returns empty (no vaastav), skips feature engineering
- Seeds predictions directly from bootstrap `ep_next` values
