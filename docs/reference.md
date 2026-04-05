# FPL Assistant — Reference

Supporting data for the improvements roadmap: historical baselines, model benchmarks, feature importance, optimizer constraints, and API endpoints.

---

## Baseline Performance

The R-based `FPL_xPMin` script ran over 33/38 GWs in 2022-23 and is the closest historical benchmark:

| Metric | Value |
|--------|-------|
| GWs covered | 33 / 38 (GW 5–38, skipping blank GW 7) |
| Avg predicted xP | 98.9 pts/GW |
| Avg actual pts | 95.0 pts/GW |
| Prediction accuracy | 96% |
| Best GW | GW 29 — 148.1 xP predicted, 146 actual (Watkins captain) |
| Most-captained | Haaland (15/33 GWs) |

**Insight:** The optimizer already captures ~96% of achievable value. Further gains must come from better *point prediction*, not solver tuning.

---

## ML Model Baselines (2022-23 data)

| Model | MAE | RMSE | R² | Notes |
|-------|-----|------|----|-------|
| Mean baseline | 1.556 | 2.372 | — | Predict every player scores the mean |
| Linear regression | 1.075 | 1.967 | — | Simple but competitive |
| Random Forest (global) | 1.035 | 1.948 | 0.313 | **Current production model** |
| XGBoost (global) | 1.026 | 1.952 | — | ~1% edge over RF; Track C/P1a |
| RF positional — GK | 0.770 | — | 0.438 | Most predictable position |
| RF positional — DEF | 0.910 | — | — | |
| RF positional — MID | 1.048 | — | — | |
| RF positional — FWD | 1.249 | — | 0.321 | Least predictable |

**Key insight:** Global model R² ~0.31 means we explain ~31% of per-GW variance. Positional models do not consistently beat the global model; position is already encoded as a feature.

---

## Top Predictive Features (NB04 feature importance)

Ranked by RF mean decrease in impurity:

1. `minutes` / `minutes_roll_4` — 16–30% importance. **Playing time is king.** Two-stage model (Track C/P1b) targets this.
2. `ict_index_roll_4` — influence + creativity + threat composite. In pipeline.
3. `total_points_roll_4` — form momentum. In pipeline.
4. `bps_roll_4` — bonus point system predicts bonus allocation. In pipeline.
5. `xG` / `xA` — **commented out in NB04** because Understat scraper was broken. Track B/P4 revives these.

---

## Optimizer Constraints

Hard constraints enforced in `src/pipeline/optimize.py`. Do not regress these when modifying the optimizer.

```
Squad (15 players):
  - Budget ≤ £100.0M (now_cost in tenths: ≤ 1000)
  - Exactly 2 GK, 5 DEF, 5 MID, 3 FWD
  - Max 3 players from any single club

XI (11 starters):
  - Exactly 1 GK
  - At least 3 DEF, 2 MID, 1 FWD
  - Total starters = 11

Captain / Vice-Captain:
  - 1 captain (xP × 2), 1 vice-captain (fallback)
```

---

## API Endpoints

```python
BASE_URL = "https://fantasy.premierleague.com/api/"

# Core (src/pipeline/fetch.py)
bootstrap  = BASE_URL + "bootstrap-static/"          # player + team metadata
element    = BASE_URL + "element-summary/{id}/"      # per-player GW history
fixtures   = BASE_URL + "fixtures/"                  # fixtures with FDR
live       = BASE_URL + "event/{gw}/live/"           # live GW points

# Track A additions (src/pipeline/user.py via src/config.py FPL_*_URL constants)
entry      = BASE_URL + "entry/{id}/"                # bank, league membership
picks      = BASE_URL + "entry/{id}/event/{gw}/picks/"  # squad picks + selling prices
history    = BASE_URL + "entry/{id}/history/"        # GW-by-GW history + transfer log
standings  = BASE_URL + "leagues-classic/{id}/standings/?page_standings={p}&event={gw}"
```

No authentication required. 3–5 s sleep between player fetches is safe (700 players ≈ 35 min full collection).

---

## Notebook Observations

| Notebook | What it does | Runnable? | Key blocker |
|----------|-------------|-----------|-------------|
| 01_eda.ipynb | FPL API data collector. 35+ min runtime. | No | Hardcoded Windows path; superseded |
| 02_feature_engineering.ipynb | Rolling/momentum features. Vectorised cell `80c05094` is 100× faster than iterrows. | Partial | Needs `cleaned_merged_seasons1.csv` |
| 03_player_clustering.ipynb | KMeans (K=3) cold-start prior. MAE 1.556 → 1.327. | Partial | Needs `team_key.csv` from NB07 |
| 04_model_training.ipynb | Global RF + XGBoost. XGBoost wins (MAE 1.026). Top feature: minutes (16–30%). | Yes (path fix) | Hardcoded season path |
| 05_model_training_positional.ipynb | Positional models. GK best (0.770), FWD worst (1.249). | Yes (path fix) | Hardcoded season path |
| 06_team_optimization.ipynb | MINLP/GEKKO exploration. Incomplete. | No | `gekko` unmaintained |
| 07_team_key_mapping.ipynb | Builds `team_key.csv` for NB03. | Yes (path fix) | Hardcoded path |
