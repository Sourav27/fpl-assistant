# FPL Assistant — Improvements Roadmap

> **How to use this file:** Each _Track_ is one initiative — a self-contained batch of work you can ship in a day or a week. Tracks link to detailed plan files. Start a track by opening its plan and following task-by-task instructions. Tracks are ordered by impact/effort ratio.

Last updated: 2026-04-12.

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
**Status:** COMPLETE (2026-04-04) · **Tests:** 135 passing
**Plan:** [`docs/superpowers/plans/2026-03-29-track-a-team-sync-and-analysis.md`](superpowers/plans/2026-03-29-track-a-team-sync-and-analysis.md)

**What was built:**
- `src/pipeline/user.py` — fetches real squad, bank, free transfers from FPL API
- `src/pipeline/recommend.py` — multi-GW ILP transfer planner with FT banking + hit cost
- `src/pipeline/analysis.py` — post-match prediction misses, dream team, accuracy log
- `src/pipeline/run.py` extended — `recommend` phase + `--horizon/--wildcard/--team` flags
- `results/accuracy_log.csv` — per-GW benchmark log (best/avg/percentile)
- Bootstrap snapshot caching (`results/snapshots/bootstrap_gw{N}.json`) + daily GitHub Action

**Post-ship fixes (A-F1 through A-F5), completed 2026-04-04:**
- **A-F1** — `your_pts` now sourced directly from FPL API entry history (auto-subs, captain, VC all correct)
- **A-F2** — Accuracy log redesigned: `your_pts/xp`, `recommended_pts/xp`, `wildcard_pts/xp`, `dream_team_pts` columns
- **A-F3** — Multi-GW leakage fixed: `rec_df` filtered to `gw == current_gw` before `recommended_pts` computation
- **A-F4** — xP correction layer in `predict.py`: blank GW zeroing → FDR weighting → availability scaling; `raw_xp` preserved for model evaluation
- **A-F5** — Replay test infrastructure: `tests/test_integration_replay.py` + `tests/fixtures/gw{N}/` cached API snapshots; GW30 and GW31 covered

**Refactoring (same branch):**
- `availability.py`: replaced `iterrows` with vectorized `np.select` + boolean masking (10× faster); logging at INFO level
- `features.py`: fixed cross-season rolling window data leakage (`groupby([player_id, "season"])`)
- `config.py`: added `SNAPSHOTS_DIR` constant (eliminated 4 inline path constructions)
- `recommend.py`: fixed `bank_after` calculation (post-transfer bank in £M); unified single/multi-GW return format; module-level imports
- `run.py`: consolidated `SNAPSHOTS_DIR`, `ELEMENT_TYPE_MAP` imports; removed dead variable

**CLI:**
```bash
python -m src.pipeline.run recommend --gw <N> --horizon 3
python -m src.pipeline.run post-gw
```

---

### ✅ Track B — Fixture-Aware Per-Position Models
**Status:** COMPLETE (2026-04-12) · **Tests:** 259 passing
**Plan:** [`docs/superpowers/plans/2026-04-12-track-b-fixture-aware-per-position-models.md`](superpowers/plans/2026-04-12-track-b-fixture-aware-per-position-models.md)

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

### ✅ Track E — Scheduling: Deadline Detection & Model Promotion
**Status:** COMPLETE (2026-04-05) · **Tests:** 159 passing
**Plan:** [`docs/superpowers/plans/2026-04-04-track-e-scheduling.md`](superpowers/plans/2026-04-04-track-e-scheduling.md)

**What was built:**
- `tests/test_fetch_bootstrap_snapshots.py` — E-F1 coverage: `_price_change_summary` + `price_changes_latest.txt` (22 tests total)
- `scripts/check_deadline.py` — reads bootstrap JSON, returns hours until next GW deadline, writes GitHub Actions output vars
- `tests/test_check_deadline.py` — 6 unit tests for deadline detection
- `.github/workflows/daily_bootstrap.yml` — 8 new steps: deadline check, deadline Discord alert, model download from GitHub Releases, user_config write, predict+recommend trigger, results commit, Discord results notification
- `CLAUDE.md` — "Model Promotion via GitHub Releases" subsection with `gh release create` workflow and secrets docs
- `src/pipeline/run.py` — post-transfer squad save: `squad_recommend_gw{N}.csv` (15 players) and `xi_recommend_gw{N}.csv` (11 starters)
- `scripts/format_discord_results.py` — `format_wildcard_xi_block()` and `format_my_team_block()`: pure formatting functions for Discord messages
- `tests/test_format_discord_results.py` — 10 tests (captain identity by element ID, bench header, bank/transfers display, future-GW exclusion)
- `tests/test_run_recommend_saves_squad.py` — 2 tests for squad/XI CSV persistence

#### Scheduling tasks completed

| ID | What | Status |
|----|------|--------|
| E-F1 | Bootstrap action tests: `_price_change_summary` + `price_changes_latest.txt` | ✅ done |
| E-F2a | `scripts/check_deadline.py` helper — parse bootstrap → hours until deadline | ✅ done |
| E-F2b | GitHub Actions: deadline proximity trigger for predict+recommend + model download | ✅ done |
| E-F3 | Document GitHub Releases model promotion workflow in `CLAUDE.md` | ✅ done |
| E-F4a | Save `squad_recommend_gw{N}.csv` + `xi_recommend_gw{N}.csv` from `phase_recommend()` | ✅ done |
| E-F4b | Discord notification: Wildcard XI + My Team After Transfers (15 players, bench, bank, FTs) | ✅ done |



---

### ✅ Track H — Data Sources Integration
**Status:** COMPLETE (2026-04-05) · **Tests:** 208 passing
**Plan:** [`docs/superpowers/plans/2026-04-05-track-h-data-sources.md`](superpowers/plans/2026-04-05-track-h-data-sources.md)

**What was built:**
- `src/pipeline/datasources/` — new package with one module per external source
- `src/pipeline/datasources/signals.py` — `PlayerSignal` dataclass, `resolve_player_name`, `log_unresolved_name`; unresolved player names logged to `results/signal_unresolved.csv`
- `src/pipeline/datasources/understat.py` — async understatAPI client: `fetch_understat_player_gw_stats` + `compute_team_xgc_per_gw` (EPL only)
- `src/pipeline/datasources/soccerdata_client.py` — FotMob wrapper: `fetch_fotmob_player_minutes` + `cross_validate_with_fpl` (European/international only); reliability gate: MAE ≤ 5 min AND correlation ≥ 0.95
- `src/pipeline/datasources/ffs.py` — Fantasy Football Scout RSS parser → `PlayerSignal` list; rule-based keyword classification (injured > doubt > available > general_news)
- `src/pipeline/datasources/reddit.py` — Reddit r/FantasyPL JSON API client → `PlayerSignal` list; display-only Phase 1; confidence = 0.5
- `src/pipeline/datasources/premierinjuries.py` — HTML scraper → `PlayerSignal` list; `cross_verify_against_fpl` flags contradictions (signal must NOT adjust xP if contradicted)
- `src/pipeline/source_validation.py` — Spearman ρ gate: `run_xg_validation_gate(understat_rho, fpl_opta_rho, tolerance=0.05)`; logs to `results/source_validation.csv`
- `src/pipeline/signal_feedback.py` — per-source accuracy logger: `append_signal_feedback` + `compute_source_accuracy`; feeds Track G Phase 2 activation gate (≥ 80% accuracy over ≥ 15 obs per source-type pair)
- `src/config.py` — added `SOURCE_VALIDATION_CSV`, `SIGNAL_ACCURACY_CSV`, `SIGNAL_UNRESOLVED_CSV`
- `tests/datasources/` — 43 new tests across unit + integration suites (all mocked, no live HTTP in CI)

**Commits:** `2295d0c` → `4e4662e` (13 commits on `feature/track-h-data-sources`) + H-C follow-up (7 commits, merged via worktree)

**What each source unblocks:**
- **H-F1/H-F2** → Track B `xGC_rolling_4` feature (use understat if gate passes, else vaastav `goals_conceded`)
- **H-F3** → Track B DGW rotation (`rest_days` feature), Track G Tier 3 minutes tracker
- **H-F4/H-F5/H-F6** → Track F `GET /api/news` endpoint (PlayerSignal list ready to serve)
- **H-F7** → Track G Phase 2 xP auto-adjustment (accuracy log accumulating from day 1)

**H-C consolidation (post-ship, same branch):**
- **H-C1** — `SOURCE_COLUMN_MAP` added to `datasources/__init__.py`; normalises column names across sources
- **H-C2** — `understat.py` rewritten to use `soccerdata` sync API; slimmed to `xg_chain` + `xg_buildup` only (xg_buildup excluded from model — kept for future use)
- **H-C3** — `soccerdata_client.py` and `test_soccerdata.py` deleted; FotMob wrapper removed (superseded by H-C2 rewrite)
- **H-C4** — `data/espn_player_id_map.csv` seeded with confirmed FPL→ESPN ID mappings
- **H-C5** — `espn_client.py` added — fetches non-PL stats (UCL/EL) via ESPN eventlog API; used for Track B `rest_days` feature
- **H-C6** — `availability_features.py` added — unified availability feature assembly (combines FPL API, FFS, premierinjuries signals)
- **H-C7** — `source_validation.py` docstring updated to reflect post-H-C2 reality (soccerdata sync path)

#### Deferred

- **P3b (FBref):** Dropped — FBref aggressively rate-limits and changes table structure. understatAPI covers the same territory more reliably.
- **P3c (Ensemble predictions):** Deferred until Track B ships. After Track B: 4 RF + 4 XGBoost = 8 models; ensemble averages within each position bucket. Effort: 0.5 day.

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
| **Off-season** | Work one Track per week: E → H → B → C → D → F |

---

> **Reference data** (baselines, model benchmarks, feature importance, optimizer constraints, API endpoints) has been moved to [`docs/reference.md`](reference.md).
