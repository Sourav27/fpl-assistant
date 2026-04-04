# FPL Assistant — Improvements Roadmap

> **How to use this file:** Each _Track_ is one initiative — a self-contained batch of work you can ship in a day or a week. Tracks link to detailed plan files. Start a track by opening its plan and following task-by-task instructions. Tracks are ordered by impact/effort ratio.

Last updated: 2026-04-03.

---

## Objective

Never miss a deadline. Always have a recommendation ready. Make better decisions than doing nothing — and better than following the crowd.

---

## Goals (Season 2025-26)

Four goals that matter for a real FPL season. Not technical targets — outcomes.

### 1. Beat the template, not just the average

The **template** is the consensus squad held by top managers — players with the highest effective ownership (EO). EO = ownership% − benched% + captain%. When a template player hauls, everyone holding them gains equally, so rank doesn't move. Rank gains come from differentials: low-EO players who score while template managers miss out.

Sticking to the template protects rank but makes climbing impossible. Beating it requires identifying when to deviate — going against a popular captain choice, owning a player before mass adoption, or holding a differential through a blank.

**Target:** End ≥ 50% of gameweeks with a rank better than the EO-weighted template team for that week.

### 2. Outperform doing nothing

Research is clear: active transfers do not automatically beat holding your squad. One study found that using an XGBoost transfer algorithm was actually outperformed by the original drafted team held unchanged all season — because the model's prediction errors made transfers net-negative. The average FPL transfer loses value.

The assistant's recommendations only justify themselves if they net positive over the season. Beating do-nothing is the minimum bar.

**Target:** Cumulative points from recommended transfers net positive vs holding GW1 squad unchanged over the full season. Algorithmic strategies achieving top 10% overall rank are considered excellent (research shows top 12% via linear regression, top 4% via combinatorial optimisation, top 1% via Bayesian RL).

### 3. Never miss a deadline — every chip decided with a scenario

Missed deadlines are unforced errors. Chip timing is the highest-leverage decision of a season: Bench Boost on DGWs (maximises 30-player fixture potential), Triple Captain on a premium DGW player, Free Hit to survive blanks, Wildcard before a run of good fixtures — planned 4–6 gameweeks ahead by tracking fixture congestion.

Every chip should be backed by a what-if comparison: "BB in GW32 DGW vs GW35 — projected difference X pts."

**Targets:**
- Zero missed deadlines: `pre-deadline` + `predict` + `recommend` run before every GW kickoff
- All 8 chips (2× each) deployed with a scenario comparison from the assistant

### 4. Predictions grounded in the best available benchmarks

The best open-source FPL model (OpenFPL) achieves RMSE 5.14 / MAE 4.32 for haulers (5+ pt players) and is comparable to FPL Review's commercial model. For classifying "will this player haul?" (6+ pts), top models achieve 89–91% accuracy. For starting prediction, logistic regression achieves 90.7% for forwards, 87.1% for defenders.

Our current RF global model (MAE 1.035 across all players) is a reasonable starting point. The gap to close is on hauler prediction accuracy — the high-variance tail that matters most for captain and transfer decisions.

**Target:** Match or beat the best community benchmark available at time of measurement. Minimum: model achieves higher directional accuracy than `ep_next` (FPL's own forecast) on held-out GWs. Every recommendation shows the reasoning: fixture, form, EO risk of not owning.

---

## Success Metrics

| Goal | Metric | Baseline | Target | Measured by |
|------|--------|----------|--------|-------------|
| Beat template | GWs finishing above EO-template score | — | ≥ 50% of GWs | `accuracy_log.csv` vs template xP |
| Beat do-nothing | Season pts delta vs holding GW1 squad | — | Net positive | `accuracy_log.csv` cumulative |
| Overall rank | End-of-season overall rank percentile | — | Top 20% (good), Top 10% (target) | FPL website |
| Never miss deadline | Missed deadlines per season | — | 0 | Cron / run logs |
| Chip confidence | Chips played with scenario comparison | 0 / 8 | 8 / 8 | Manual review |
| Hauler prediction | RMSE for players scoring 5+ pts | — | ≤ 5.14 (OpenFPL benchmark) | `retrain` evaluation |
| Overall MAE | Mean absolute error per player per GW | 1.035 | ≤ best public benchmark | `retrain` evaluation |
| Spearman ρ | Rank correlation (predicted vs actual pts) | — | ≥ 0.65 (good), ≥ 0.70 (target) | `accuracy_log.csv` |
| Recommendation reasoning | Transfers include fixture/form/EO rationale | None | Every recommendation | `recommend_gw{N}.csv` |

### Benchmark context

- **Do-nothing threshold:** A transfer needs to gain ≥ 6 pts over the held player just to break even against a -4 hit. Without a hit, any gain is positive — but the average active manager still loses to do-nothing due to prediction error and emotional decisions.
- **Top rank tiers:** Top 100k (~1.5%) = good. Top 10k (~0.15%) = very good. Top 1k (~0.015%) = elite. Research-backed algorithmic systems range from top 20% (baseline) to top 0.5% (multi-stream commercial grade).
- **Chip timing window:** Research recommends mapping chip strategy 4–6 GWs ahead using fixture congestion forecasts. Ben Crellin (BGW/DGW updates) is cited as a reliable signal source for this — already planned for Track G.
- **EO reference:** A player with 90% ownership captained by 70% has EO = 159%. Any week you captain differently from 70% of managers, you're taking a differential captain bet — high risk, high rank-change potential.

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

**Critique & Validation (2026-03-30):**
Comprehensive review of GW31 logs (`accuracy_log.csv`, `recommend_gw31.csv`) and comparison with live FPL API data (entry 1681779) confirms several critical bugs in the `post-gw` and `recommend` phases:

1. **Bench Points Leakage:** `your_pts` incorrectly sums all 15 players in your squad. In FPL, only the starting 11 (plus auto-subs) contribute to your score. The current log is "noisy" and doesn't reflect your actual game score.
2. **Captain Bonus Missing:** The 13 points from your captain (B.Fernandes) were not doubled in the `your_pts` calculation, leading to a significant 13-point underestimate of your actual score (44 vs 57).
3. **Invalid "Recommended" Baseline:** `recommended_pts` only sums players transferred *in*, while `your_pts` sums 15 players. This makes the comparison mathematically invalid (comparing a 3-man sub-team to a 15-man squad).
4. **Multi-GW Leakage:** The analysis logic includes players from the *entire* recommendation plan (e.g., Walker for GW32) in the current GW31 stats.
5. **Fixture-Blind Recommender:** The optimizer ignored the `fixtures` list entirely, recommending Semenyo for GW31 despite Man City having a blank gameweek.

#### 🛠️ Fixes Required (Immediate Priority — execute before Track B)

**Fix A-F1 — Score calculation in `run.py:phase_post_gw()`**
- `your_pts`: fetch directly from FPL API entry history (already accounts for auto-subs, VC, captain, bench-boost). Do not reconstruct from player sums.
- Player-level reconstruction (XI + auto-subs) is kept separately for **decision quality analysis**: a non-starting player who was auto-subbed out counts as a selection miss, not a points gain.

**Fix A-F2 — Accuracy log columns redesigned**

Replace the single `recommended_pts` with four distinct score columns and matching xP columns:

| Column | Definition |
|--------|-----------|
| `your_pts` | Actual FPL score from API (captain, auto-subs, VC all correct) |
| `your_xp` | Adjusted xP for your actual XI + captain (×2 boost applied) |
| `recommended_pts` | Your pre-GW squad with GW N transfers applied → best XI scored with live auto-sub/VC/captain adjustments. If no transfer recommended, equals `your_pts`. |
| `recommended_xp` | Adjusted xP for recommended XI + captain (×2 boost applied) |
| `wildcard_pts` | Pre-deadline optimal 15-man squad (£100M budget, optimizer run on deadline prices) → best XI scored with live adjustments |
| `wildcard_xp` | Adjusted xP for wildcard XI + captain (×2 boost applied) |
| `dream_team_pts` | Post-match best 11 individual players from actuals (existing, no budget constraint) |

`your_pts == recommended_pts` when you followed all recommendations exactly.

**Fix A-F3 — Multi-GW leakage in `run.py:phase_post_gw()`**
- Filter `rec_df` to `gw == current_gw` before extracting `player_in` names.
- Only GW N transfers are applied when computing `recommended_pts`. Future-horizon transfers are excluded.

**Fix A-F4 — xP correction layer (architectural fix)**

Root cause: the ML model predicts `xP` blind to fixture reality. A player on a blanking team gets the same `xP` as if they had a fixture, corrupting every downstream consumer (optimizer, analysis, accuracy log).

**Design decision:** Bake adjustments into `predictions_gw{N}.csv` with two columns:
- `raw_xp` — pure ML model output (used only for model evaluation and improvement tracking)
- `xp` — corrected xP used by all consumers: optimizer, `recommend.py`, `analysis.py`, accuracy log

**Correction pass** runs in `predict.py` after ML inference, applying (in order):
1. **Blank GW zeroing** — `xp = 0` for any player whose team has no fixture in GW N (from `fixtures` API)
2. **FDR weighting** — `xp *= compute_fdr_weight(fdr_team, fdr_sensitivity)` per player
3. **Availability scaling** — already done in `availability.py`; confirm it writes to the same corrected `xp` column

With this fix, `_recommend_single_gw()` no longer needs a special `build_xp_matrix()` call — it simply reads corrected `xp` from the predictions dataframe. All consumers uniformly correct by default.

**Files touched:** `predict.py` (correction pass), `save_full_predictions()` (add `raw_xp` column), `recommend.py` (use `xp` not `raw_xp`), `analysis.py`, `run.py`, `tests/test_predict.py`

**Fix A-F5 — Historical GW replay test infrastructure**

The bugs found in GW31 were not caught by the existing test suite because unit tests use mock data. Mock data cannot replicate the full chain: real fixtures → real blanks → real xP errors → real recommendation mistakes. We need tests grounded in real historical data.

**Design:**
- `tests/test_integration_replay.py` — new module; each test replays a complete past GW end-to-end
- Test fixture data sourced from:
  - `data/Fantasy-Premier-League/` (vaastav dataset — historical GW stats, player data)
  - FPL API historical endpoints (entry picks, live GW points, benchmarks) — fetched once and cached under `tests/fixtures/gw{N}/`
  - Saved pipeline outputs (`results/predictions_gw{N}.csv`, `results/recommend_gw{N}.csv`, `results/accuracy_log.csv`) as ground-truth snapshots
- **GW31 replay test** (priority — run before GW32 deadline): confirm corrected `xp` zeros Semenyo/blanking players, `recommended_pts` matches expected score, `your_pts` matches FPL API entry score
- **GW30 replay test**: smoke test for a "normal" non-blank GW
- **GW1 replay test**: edge case — first GW, no rolling history, no prior transfers, FT banking from zero

**Replay framework API:**
```python
# tests/test_integration_replay.py
@pytest.fixture
def gw31_fixtures():
    return load_cached_gw_fixtures(gw=31)  # reads from tests/fixtures/gw31/

def test_gw31_blank_xp_zeroed(gw31_fixtures):
    """Semenyo (Man City blank) must have corrected xp == 0."""
    predictions = run_predict_phase(gw31_fixtures)
    semenyo = predictions[predictions["name"] == "Semenyo"]
    assert semenyo["xp"].iloc[0] == 0.0
    assert semenyo["raw_xp"].iloc[0] > 0  # model was blind to blank

def test_gw31_your_pts_matches_api(gw31_fixtures):
    """your_pts must equal actual FPL score from entry history."""
    result = run_post_gw_phase(gw31_fixtures, entry_id=1681779)
    assert result["your_pts"] == gw31_fixtures["entry_history"]["points"]

def test_gw31_recommended_pts_no_future_leakage(gw31_fixtures):
    """recommended_pts must not include Walker (GW32 transfer)."""
    result = run_post_gw_phase(gw31_fixtures, entry_id=1681779)
    assert "Walker" not in result["recommended_squad"]
```

**Philosophy:** Every time a real-GW bug is found, a replay test for that GW is added immediately — before the fix. This ensures the fix is verified against real data and the bug cannot regress. Mock tests remain for unit-level logic; replay tests cover integration correctness.

**Files:** `tests/test_integration_replay.py`, `tests/fixtures/gw31/` (cached API responses), `tests/conftest.py` (add `load_cached_gw_fixtures()` helper)

**Suggested next action:** Execute Fixes A-F1 through A-F5 as Track A.1 before moving to Track B. A-F5 (replay tests) must be written before A-F1 through A-F4 are implemented — TDD on real data.

---

### 📋 Track B — Fixture-Aware Per-Position Models
**Status:** SPEC READY · **Effort:** ~3–4 days
**Plan:** Write detailed plan before starting (use `writing-plans` skill)

**Objective:** Replace the single global RF model with 4 per-position RF models (GK/DEF/MID/FWD), each trained with fixture-aware features. Switch primary evaluation metric from MAE to Spearman rank correlation (ρ). Handle DGW/BGW with per-fixture prediction and aggregation.

**Success gate:** Spearman ρ ≥ 0.65 on held-out GWs (top-200k quality). Secondary: overall MAE must not worsen vs global RF baseline (1.035). Each position model must beat or match the global model for that position's players.

#### Design: Approach C — Per-Position Models

**Why per-position?** Different positions score points through fundamentally different mechanisms (GKs: saves/CS, FWDs: goals). A global model learns averaged feature weights. Per-position models let `xGC_rolling_4` dominate for GK/DEF while `opponent_form_rolling_6` dominates for MID/FWD. Historical baselines: GK 0.770, DEF 0.910, MID 1.048, FWD 1.249 — the global model (1.035) underperforms GK/DEF positional models.

**Sample sizes (4 seasons × 38 GWs):** GK ~9k rows, DEF ~18k, MID ~22k, FWD ~14k — all sufficient for RF.

#### New fixture features (all 4 models)

| Feature | Source | What it captures |
|---------|--------|------------------|
| `xGC_rolling_4` | Opponent's xGoals Conceded, 4-GW rolling avg | Opponent defensive quality — highest ROI (+0.08–0.15 MAE) |
| `opponent_form_rolling_6` | Opponent's avg pts allowed to position, 6-GW rolling | Mid-season form shifts |
| `is_home` | FPL fixture data | Binary venue factor (+0.03–0.06 MAE) |
| `fixture_count` | Count of fixtures in GW | 0 = BGW, 1 = normal, 2 = DGW |
| `rest_days` | Days between fixture 1 and fixture 2 (DGW only) | Rotation/fatigue risk for DGW |
| `is_fixture_2` | Binary — marks second game in DGW | Fatigue decay: model learns lower xMin for second fixture |

#### DGW/BGW handling

**DGW:** Predict xP **per fixture** separately, then sum. For fixture 2, include `rest_days` and `is_fixture_2=1` as features — the model learns that tight turnarounds reduce expected minutes and per-minute output. Replaces the flat 1.8× multiplier.

**BGW:** `fixture_count=0` → `xP = 0`. Handled by the A-F4 xP correction layer (already in pipeline).

#### Config changes

`ACTIVE_MODEL` (single path) → `ACTIVE_MODELS` (dict of 4):
```python
ACTIVE_MODELS = {
    "GK":  MODELS_DIR / "rf_gk_gw{N}.sav",
    "DEF": MODELS_DIR / "rf_def_gw{N}.sav",
    "MID": MODELS_DIR / "rf_mid_gw{N}.sav",
    "FWD": MODELS_DIR / "rf_fwd_gw{N}.sav",
}
```
Fallback per position: if model missing or feature mismatch → `ep_next` for that position's players only. Other positions unaffected.

#### Metric changes

| Priority | Metric | What | Target |
|----------|--------|------|--------|
| Primary | **Spearman ρ** | Rank correlation between predicted and actual points | ≥ 0.65 (top 200k); stretch ≥ 0.70 (top 50k) |
| Secondary | MAE | Mean absolute error per player per GW | ≤ 1.035 (no regression) |
| Tertiary | Hauler MAE | MAE for players scoring 5+ pts | ≤ 5.14 (OpenFPL benchmark) |

**Why Spearman ρ over MAE:** FPL is a ranking game — you pick 11 from 15, captain from 11, transfer from 500+. Getting the relative order right matters more than the exact number. A model that predicts Salah 8.0 and Saka 7.9 (low MAE) but ranks Saka above Salah when Salah scores 15 is worse than a model with higher MAE that ranks correctly.

#### Tasks

| Task | ID | Description | Files touched | Effort |
|------|----|-------------|---------------|--------|
| 1 | B-F1 | Compute `xGC_rolling_4`, `opponent_form_rolling_6` in feature engineering | `features.py` | 3 h |
| 2 | B-F2 | Join opponent-side data in `prepare.py`: map fixture → opponent team → opponent rolling stats | `prepare.py` | 3 h |
| 3 | B-F3 | Add `is_home`, `fixture_count`, `rest_days`, `is_fixture_2`; per-fixture prediction + sum for DGW | `predict.py`, `features.py` | 4 h |
| 4 | B-F4 | Route players by `element_type` to correct position model at prediction time | `predict.py` | 2 h |
| 5 | B-F5 | `ACTIVE_MODELS` dict in config; per-position fallback to `ep_next` | `config.py`, `predict.py` | 2 h |
| 6 | B-F6 | `retrain` phase: train 4 models, save with position suffix, print per-position MAE + ρ | `run.py` | 3 h |
| 7 | B-F7 | Add Spearman ρ to `accuracy_log.csv` (new column); compute in `analysis.py` | `analysis.py`, `run.py` | 2 h |
| 8 | B-F8 | TDD tests: fixture features, DGW aggregation, position routing, fallback, ρ computation | `tests/` | 4 h |

**Recommended order:** B-F8 (test shells) → B-F1 → B-F2 → B-F5 → B-F4 → B-F3 → B-F6 → B-F7

#### Dependencies & risks

- **B-F1/B-F2 depend on opponent xGC data availability.** vaastav's `teams/` directory has `strength_overall_home/away` but not per-GW xGC. Options: (a) compute from player-level `goals_conceded` aggregated per team per GW (available in vaastav `gws/`), or (b) scrape from Understat. Prefer (a) — no external dependency.
- **A-F4 xP correction layer must land first** — BGW zeroing and FDR weighting are prerequisites for Track B's fixture features to work correctly end-to-end.
- **`rest_days` for DGW** requires fixture dates (kickoff times) from the FPL fixtures API. Already fetched in `fetch.py` but not currently parsed for date math.

#### Deferred from old Track B spec

The following items from the original Track B spec are **not included** in this design — they remain valuable but are independent:

| Item | Where it moved | Rationale |
|------|---------------|-----------|
| P5: Fallback benchmarking (`ep_this` vs `ep_next` vs rolling avg) | Track C (quick wins) | Independent of model architecture; 2h standalone task |
| P4a-c: Understat scraper + xG/xA features | Track E (data quality) | High-value features but blocked on Understat DOM stability; can layer onto per-position models later |

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
