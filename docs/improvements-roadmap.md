# FPL Assistant — Improvements Roadmap

> **How to use this file:** Each _Track_ is one initiative — a self-contained batch of work you can ship in a day or a week. Tracks link to detailed plan files. Start a track by opening its plan and following task-by-task instructions. Tracks are ordered by impact/effort ratio.

Last updated: 2026-03-29.

---

## Objective

Build the most accurate, automated Fantasy Premier League assistant possible — one that fetches live data, predicts player points with an ML model, recommends optimal transfers, and learns from its own mistakes each week.

## Goals (Season 2025-26)

1. Reduce prediction MAE below 1.0 (current baseline: RF global MAE 1.035)
2. Automate the full weekly cycle: pre-deadline → predict → recommend → post-gw → retrain
3. Make transfer recommendations that beat a "do-nothing" baseline by ≥ 5 pts/GW on average
4. Maintain 100% test coverage on all pipeline modules

## Success Metrics

| Metric | Baseline | Target | How measured |
|--------|----------|--------|--------------|
| Prediction MAE | 1.035 (RF global) | < 1.0 | `retrain` phase RMSE log |
| Prediction R² | 0.313 | > 0.35 | `retrain` phase |
| Weekly pts (your team) | — | ≥ avg score | `results/accuracy_log.csv` |
| Recommend vs do-nothing | — | +5 pts/GW avg | `results/accuracy_log.csv` |
| Test count | 116 | Growing | `pytest tests/ -q` |

---

## Initiative Tracker

### ✅ Track A — User Team Sync, Recommend & Post-Match Analysis
**Status:** COMPLETE (2026-03-29) · **Review:** PENDING (see below)
**Plan:** [`docs/superpowers/plans/2026-03-29-track-a-team-sync-and-analysis.md`](superpowers/plans/2026-03-29-track-a-team-sync-and-analysis.md)

**What was built:**
- `src/pipeline/user.py` — fetches real squad, bank, free transfers from FPL API
- `src/pipeline/recommend.py` — multi-GW ILP transfer planner with FT banking + hit cost
- `src/pipeline/analysis.py` — post-match prediction misses, dream team, accuracy log
- `src/pipeline/run.py` extended — `recommend` phase + `--horizon/--wildcard/--team` flags
- `results/accuracy_log.csv` — per-GW benchmark log (best/avg/percentile)
- 116 tests passing

**CLI now available:**
```bash
python -m src.pipeline.run recommend --gw <N> --horizon 3
python -m src.pipeline.run post-gw
```

#### 🔍 Pending Review Items (Track A)

These items shipped in Track A but need real-GW validation before being marked fully trusted:

| # | Item | Location | What to check |
|---|------|----------|---------------|
| A-R1 | Multi-GW FDR fallback | `recommend.py:_recommend_multi_gw()` | When `team_id_map` is unavailable, future-GW xP falls back to raw predictions with no FDR discount. Verify fallback triggers correctly and doesn't silently over-value difficult-fixture players. |
| A-R2 | Captain relaxation | `recommend.py:_recommend_single_gw()` | Captain is allowed to be in squad but not strictly enforced to be in XI. Check that the ILP always selects the captain in the starting XI in practice. |
| A-R3 | FT banking end-of-season | `user.py:_compute_free_transfers()` | Behaviour at GW1 (fresh start) and GW38 (season end) edge cases. Simulate with a mock history. |
| A-R4 | Selling price haircut | `user.py:compute_selling_price()` | FPL rules: no haircut on price drops. Confirm with a real GW where prices have dropped and compare against FPL app selling price. |
| A-R5 | Accuracy log first run | `analysis.py:append_accuracy_log()` | File is created on first `post-gw` run. Verify headers match schema; test append idempotency when `post-gw` is re-run for same GW. |

**Suggested next action:** Run `recommend --gw 31` before the next deadline, then `post-gw` after GW31 results. Compare recommendations vs actuals to validate all five items above.

---

### 📋 Track B — Model Quality: Understat, Positional Models, Fallback Benchmarking
**Status:** NOT STARTED · **Effort:** ~2–3 days
**Plan:** [`docs/superpowers/plans/2026-03-29-track-b-model-quality.md`](superpowers/plans/2026-03-29-track-b-model-quality.md)

**Objective:** Improve prediction accuracy by testing better fallback strategies, reviving xG/xA features, and researching per-position models.

**Success gate:** Each sub-item is only adopted if it lowers MAE on held-out historical GWs. No-regression rule: overall MAE must not worsen.

**Tasks (in recommended order):**

| Task | ID | Description | Effort | Decision gate |
|------|----|-------------|--------|---------------|
| 1 | P5 | Fallback strategy benchmarking — test `ep_this` vs `ep_next` vs rolling avg | 2 h | Adopt winner |
| 2 | P4a | Revive Understat scraper — `src/pipeline/understat.py` | 4 h | Only if DOM works |
| 3 | P4b | FPL↔Understat player name matching | 2 h | Needed for P4a |
| 4 | P4c | Integrate xG/xA features into `prepare.py` + `features.py` | 3 h | Only if MAE improves |
| 5 | P3 | Positional model research notebook | 3 h | Only if MAE improves |
| — | Final | Test suite + mark roadmap | 1 h | All prior tasks done |

**Notes:**
- Start with Task 1 (P5) — fully independent, uses only existing code
- P4 (Tasks 2–4) requires Understat scraper not to have broken its DOM; may need `playwright`
- P3 (Task 5) benefits from P4's xG/xA features but can run on existing features first

---

### 💡 Track C — Quick ML Wins (High Impact, Low Effort)
**Status:** BACKLOG · **Effort:** ~1 day each
**Plan:** Not yet written — write plan before starting

| # | ID | What | Why | Effort |
|---|----|------|-----|--------|
| 1 | P1a | Swap RF for XGBoost | NB04: XGBoost MAE 1.026 vs RF 1.035 — 1% consistent improvement | 2 h |
| 2 | P1b | Two-stage playing-time model | `minutes` is top feature (16–30% importance); predicting P(plays ≥60min) as Stage 1 reduces residual error | 1 day |

**P1a (XGBoost swap) — implementation sketch:**
```python
# In src/pipeline/run.py:phase_retrain()
from xgboost import XGBRegressor
model = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6,
                     subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)
```
File: `src/pipeline/run.py:phase_retrain()`. Risk: low — drop-in replacement.

**P1b (Two-stage model) — outline:**
- Stage 1: `LogisticRegression` on `minutes_roll_4`, `selected_by_percent`, `availability_status` → P(plays ≥ 60 min)
- Stage 2: current RF/XGB only for players where Stage 1 > 0.5
- New file: `src/pipeline/predict.py:predict_playing_time()`
- Risk: medium — needs binary minutes ≥ 60 target column in training data

---

### 💡 Track D — Optimizer Enhancements
**Status:** BACKLOG · **Effort:** ~2–4 days
**Plan:** Not yet written — write plan before starting

| # | ID | What | Why | Effort |
|---|----|------|-----|--------|
| 1 | P2a | Sharpe ratio / risk-adjusted optimization | Pure xP over-weights volatile attackers; risk-adjusted is more robust to blanks | 3 days |
| 2 | P2b | Cold-start player clustering | Players with < 8 GW history get NaN features; KMeans prior (NB03) gives MAE 1.327 vs 1.556 | 1 day |

**P2a notes:** True QP requires `cvxpy` or `scipy`; PuLP only handles LP/ILP. CLI flag `--strategy sharpe`.

**P2b notes:** Cluster by `[position, selected_by_percent, now_cost]`. New file: `src/pipeline/clustering.py`.

---

### 💡 Track E — Automation & Data Quality
**Status:** BACKLOG · **Effort:** varies
**Plan:** Not yet written

| # | ID | What | Why | Effort |
|---|----|------|-----|--------|
| 1 | P3a | Pipeline scheduling (cron / GitHub Actions) | Automate weekly cycle — pre-deadline 48h before, post-gw 2h after final whistle | 1 day |
| 2 | P3b | FBref defensive stats | Tackles, interceptions, blocks improve DEF/MID prediction; currently only `clean_sheets` | 2 days |
| 3 | P3c | Ensemble predictions (RF + XGBoost) | Averaging models reduces variance; all models already serialised | 0.5 day |

**P3b warning:** FBref aggressively rate-limits scrapers and changes table structure. Consider `understat-client` or commercial data instead.

---

### 💡 Track F — Web App (Dashboard + API)
**Status:** BACKLOG · **Effort:** ~1 week
**Plan:** Not yet written — run brainstorming skill before starting

**Objective:** Expose pipeline outputs as a mobile-friendly web dashboard accessible anywhere, with a FastAPI backend that is designed from day one to be consumed by the Chrome extension (Track G+).

**Stack decisions (researched 2026-03-30):**
- **Backend:** FastAPI (async, `lifespan` pattern) + SQLAlchemy 2.x async
- **Database:** SQLite locally (`aiosqlite`) → Render free Postgres in prod (`asyncpg`) — swapped via `DATABASE_URL` env var. Alembic for migrations with `render_as_batch=True` for SQLite compatibility.
- **Frontend:** Vite + React (not Next.js — incompatible with Chrome extension MV3 content scripts)
- **Monorepo:** pnpm workspaces — `packages/ui` shared component library consumed by both web app and future extension
- **Deployment:** Render free tier (Dockerfile preferred over Procfile). Alternatives to evaluate later: Railway, Fly.io, Koyeb.
- **CORS:** `allow_origin_regex=r"chrome-extension://.*"` so the extension can call the same API

**Dashboard layout (single scrollable page):**
1. **Current squad** — 15 player cards with xP, price, availability badge, FPL news indicator
2. **Transfer recommendations** — suggested in/out with xP delta and reasoning (FDR, form, fixture)
3. **Transfer trends** — ownership changes, price rise/fall alerts
4. **Player news feed** — FPL API news (most reliable) + press conference signals + social signals (Track G)
5. **Gameweek history** — past GW results, your score vs avg/best
6. **Accuracy log** — prediction vs actual chart, MAE trend over season

**API contract (must be stable for Chrome extension reuse):**
```
GET  /api/squad          → current squad with xP + availability
GET  /api/recommend      → transfer recommendations with reasoning
GET  /api/predictions    → full player xP rankings
GET  /api/accuracy-log   → per-GW benchmark history
GET  /api/news           → player news feed (FPL + social signals)
POST /api/pipeline/run   → trigger pipeline phase (predict/recommend)
```

**Key constraint:** The API shape defined here becomes the Chrome extension's data contract. Design it before building the extension (Track G+).

---

### 💡 Track G — Social Media Signals (Display → Auto-Adjust)
**Status:** BACKLOG · **Effort:** ~1 week (Phase 1: display) + ~1 week (Phase 2: xP adjustment)
**Plan:** Not yet written — run brainstorming skill before starting. **Depends on Track F** (needs the news feed API endpoint).

**Objective:** Incorporate pre-deadline intelligence — injury news, rotation risk, lineup leaks — into the dashboard and eventually into the xP model itself.

**Signal source hierarchy (most → least reliable):**

| Tier | Source | Method | Reliability |
|------|--------|---------|-------------|
| 1 | **FPL API `news` field** | Already in pipeline (`availability.py`) | Highest — official |
| 2 | **Club press conferences** | Scrape official club sites / BBC Sport match previews | High — direct manager quotes |
| 3 | **Minutes tracker** | Compute from existing `element-summary` data — flag players with 90min in UCL/EL within 72h of PL deadline | Medium — heuristic |
| 4 | **X posts (Nitter RSS)** — FPL Focal, Ben Crellin (BGW/DGW updates) | Nitter RSS feed polling; manual paste fallback if Nitter is down | Medium — source-dependent |

**Phase 1 — Display only (Track G1):**
- Aggregate all four signal tiers into a unified `PlayerSignal` model: `{player_code, source, signal_type, text, confidence, timestamp}`
- Display in dashboard news feed panel (Track F)
- No impact on xP — human reads and decides

**Phase 2 — xP auto-adjustment (Track G2):**
- Parse signals into structured flags: `rotation_risk`, `doubt`, `confirmed_starter`, `blank_gw`, `double_gw`
- Apply discount/boost multipliers to xP before `recommend` runs
- Only adopt after Phase 1 feedback log validates source accuracy ≥ threshold

**Feedback logging (both phases):**
- When team sheets arrive (~1h before kickoff), log each signal against actual lineup: `{signal_id, predicted_status, actual_started, source, gw}`
- Build per-source accuracy scores over the season
- Use scores to weight signal confidence in Phase 2

**X/Twitter access strategy:**
- Primary: Nitter RSS (self-hosted or public instance) — free, no API key
- Fallback: Manual paste input in dashboard (`POST /api/news/manual`)
- Both routes log identically to the feedback system
- Twitter paid API ($100/mo) explicitly out of scope for personal use

---

## Recommended Weekly Rhythm

| When | Action |
|------|--------|
| **Mon/Tue** | Review `accuracy_log.csv` from last GW. Note prediction misses. |
| **Thu (deadline -48h)** | `python -m src.pipeline.run pre-deadline` then `predict --gw N` |
| **Thu–Fri** | `python -m src.pipeline.run recommend --gw N --horizon 3` — make transfers |
| **Sat night** | `python -m src.pipeline.run post-gw` — collect results, update accuracy log |
| **Monthly** | `python -m src.pipeline.run retrain --gw N` — re-train model on new data |
| **Off-season** | Work one Track per week: B → C → D → E → F → G |

---

## Reference: Baseline Performance

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

## Reference: ML Model Baselines (2022-23 data)

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

## Reference: Top Predictive Features (NB04 feature importance)

Ranked by RF mean decrease in impurity:

1. `minutes` / `minutes_roll_4` — 16–30% importance. **Playing time is king.** Two-stage model (Track C/P1b) targets this.
2. `ict_index_roll_4` — influence + creativity + threat composite. In pipeline.
3. `total_points_roll_4` — form momentum. In pipeline.
4. `bps_roll_4` — bonus point system predicts bonus allocation. In pipeline.
5. `xG` / `xA` — **commented out in NB04** because Understat scraper was broken. Track B/P4 revives these.

---

## Reference: Optimizer Constraints

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

## Reference: API Endpoints

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

## Archive: Notebook Observations

| Notebook | What it does | Runnable? | Key blocker |
|----------|-------------|-----------|-------------|
| 01_eda.ipynb | FPL API data collector. 35+ min runtime. | No | Hardcoded Windows path; superseded |
| 02_feature_engineering.ipynb | Rolling/momentum features. Vectorised cell `80c05094` is 100× faster than iterrows. | Partial | Needs `cleaned_merged_seasons1.csv` |
| 03_player_clustering.ipynb | KMeans (K=3) cold-start prior. MAE 1.556 → 1.327. | Partial | Needs `team_key.csv` from NB07 |
| 04_model_training.ipynb | Global RF + XGBoost. XGBoost wins (MAE 1.026). Top feature: minutes (16–30%). | Yes (path fix) | Hardcoded season path |
| 05_model_training_positional.ipynb | Positional models. GK best (0.770), FWD worst (1.249). | Yes (path fix) | Hardcoded season path |
| 06_team_optimization.ipynb | MINLP/GEKKO exploration. Incomplete. | No | `gekko` unmaintained |
| 07_team_key_mapping.ipynb | Builds `team_key.csv` for NB03. | Yes (path fix) | Hardcoded path |
