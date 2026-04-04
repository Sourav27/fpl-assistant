# Roadmap Review & Updates Design
**Date:** 2026-04-04  
**Session:** Multi-agent critique (data-scientist, ml-engineer, data-engineer) + owner interview  
**Scope:** All tracks except Track A (complete). Tracks B–G reviewed, benchmarks validated against NotebookLM research sources.

---

## Summary of Changes

This document captures decisions from a structured critique + interview session. It amends `docs/improvements-roadmap.md` with:
1. Track reordering
2. Benchmark corrections (sourced vs unsourced)
3. Per-track design changes
4. New signal source hierarchy for Track G

---

## 1. Track Reordering

**Old order:** B → C → D → E → F → G  
**New order: E → B → C → D → F → G**

**Rationale:** Track E data source work (understatAPI, FotMob, xG validation) directly feeds Track B's feature engineering (`xGC_rolling_4`, European minutes for rotation features). Track B should not start until the xG source validation gate passes.

---

## 2. Benchmark Corrections

### Validated (confirmed against research sources)

| Claim | Status | Notes |
|-------|--------|-------|
| OpenFPL RMSE 5.142 / MAE 4.317 for haulers (5+ pts) | Confirmed | OpenFPL beats FPL Review (RMSE 5.172) for haulers |
| 89–91% haul classification accuracy | Confirmed with nuance | 91% = GBM on pure stats; 89% = multi-stream GBM (stats + betting + NLP). Two different models |
| 90.7% FWD / 87.1% DEF starting prediction accuracy | Confirmed | Logistic regression, starting lineup prediction paper |
| Top 12% linear regression / top 4% combinatorial / top 1% Bayesian RL | Confirmed | All sourced |
| Top 0.5% multi-stream GBM | Confirmed | **Missing from roadmap — add to benchmark context** |
| XGBoost transfer algorithm beaten by holding GW1 squad | Confirmed | Pokharel et al. — attributed to model accuracy limits, not "never transfer" |

### Not sourced — remove or replace

| Claim | Action |
|-------|--------|
| **Spearman ρ ≥ 0.65 tied to top 200k manager rank** | **Remove.** No source found. Replace with empirically grounded target (see Track B section). |
| 96% optimizer efficiency from R/GEKKO baseline | **Scrap.** R-based MILP was not thoroughly tested. Replace with perfect squad gap metric. |

---

## 3. Success Metrics Table — Changes

### Remove
- `Spearman ρ ≥ 0.65 (good), ≥ 0.70 (target)` — unsourced threshold, no rank percentile link in literature.

### Add
| Metric | What | Target | Measured by |
|--------|------|--------|-------------|
| **Within-squad Spearman ρ** | `scipy.stats.spearmanr(predicted_xP, actual_points)` over the 15-player squad selected by the optimizer each GW. Source: `your_picks` dataframe in `phase_post_gw` (already has `xP` and `actual_points` columns after A-F1/A-F2). Logged as `squad_spearman_rho` column in `accuracy_log.csv` via `append_accuracy_log`. Not computable until B-F7 lands. | ≥ 0.60 first season; improve each retrain | `accuracy_log.csv` post-GW (B-F7) |
| **Perfect squad gap** | Post-GW: re-run `optimize.py` with actual GW points substituted as `xP` inputs; compare against the pre-GW selected squad. Log `perfect_squad_pts / selected_squad_pts` ratio. Since PuLP/CBC is a proven-optimal MILP solver, any gap is entirely prediction error. **When `--strategy robust` was used:** re-run the robust objective with `d_j = 0` (actual points as `c_j`) so the gap measures prediction error only, not the robust discount. | Track as baseline; no hard target until 5+ GWs logged | Post-GW replay in `analysis.py` |

### Update
- **Top rank tiers:** Add top 0.5% (multi-stream GBM) to benchmark context.
- **Hauler prediction target:** ≤ 5.14 RMSE (OpenFPL benchmark) — keep, confirmed sourced.

---

## 4. Track E — Data Collection Pipeline (NOW FIRST PRIORITY)

### P3a: Event-Driven Scheduling

**Replace fixed cron with event-driven deadline detection.**

- Daily bootstrap action fires at **02:00 UTC (7:30am IST)** — immediately after FPL prices update at 01:30 UTC (7am IST).
  - **Pending verification:** Confirm FPL bootstrap API reflects price changes by 7:30am IST. Test GW32 day.
- Action reads `events[*].is_next.deadline_time` from fetched bootstrap JSON.
- If `deadline_time - now < 48h`: trigger predict + recommend phases automatically.
- Model artifacts served via **GitHub Releases** — promote a new model by uploading `.sav` as a release asset tagged `gw{N}` (e.g., `gw34`). Action downloads via `gh release download` before running predict.
- `user_config.yaml` injected via GitHub Actions secret (not committed).

**Failure alerting:** GitHub Actions natively notifies on workflow failure. No additional tooling needed for MVP.

**Effort estimate (revised):** 2–3 days (not 1 day — model artifact management + deadline detection logic + secret injection add complexity).

**Tasks:**
| ID | Description | Files touched |
|----|-------------|---------------|
| E-F1 | Update bootstrap action: shift to 02:00 UTC, add deadline detection logic | `.github/workflows/daily_bootstrap.yml` |
| E-F2 | Add predict + recommend trigger when deadline < 48h; download model from GitHub Releases | `.github/workflows/daily_bootstrap.yml`, `scripts/` |
| E-F3 | Document GitHub Releases model promotion workflow | `CLAUDE.md` |

### P3b: Data Sources (Split into validated sub-tasks)

#### E-F4: understatAPI — PL xG/xA/xGC
- Use `understatAPI` library (`EPL` league, current season) for per-player xG, xA per GW.
- Covers: EPL only (6 supported leagues: EPL, La_Liga, Bundesliga, Serie_A, Ligue_1, RFPL).
- **Does NOT cover:** Champions League, Europa League, international matches.
- Feeds: `xGC_rolling_4` in Track B (team-level xGC aggregated from player rows).
- **Complexity note:** `understatAPI` returns shot-level data per player per match, not team-level xGC directly. Deriving team-level xGC requires: (1) fetch all player shot rows per match; (2) for each fixture, sum opponent player `xG` shots to get `xGC` for the defending team; (3) join by `team_id` and `match_id`. Estimated additional effort: 2–4h beyond basic API call. Verify whether `understatAPI.get_team_stats()` exposes team-level xGA directly before implementing the shot-level aggregation route.

#### E-F5: xG Source Validation Gate (MUST pass before Track B uses external xG)
- Compute Spearman ρ between `understat_xG` and `actual_goals` for EPL players over last 2 seasons.
- Compare against Spearman ρ between `FPL_Opta_xG` (`expected_goals` column in vaastav) and `actual_goals` on same sample.
- **Gate:** understat xG must achieve ρ ≥ FPL Opta xG ρ (or within 0.05) to be used in model features.
- If gate fails: fall back to aggregated `goals_conceded` from vaastav player rows for `xGC_rolling_4`.
- Log both ρ values to a new `results/source_validation.csv`.

#### E-F6: soccerdata + FotMob — European/International Minutes
- Use `soccerdata` library with FotMob source for UCL, UEL, international match minutes.
- Feeds: Track G Tier 3 minutes tracker (flag players with 90 min in UEFA competition within 72h of PL deadline).
- Also feeds: future Track B rotation features for DGW fatigue modelling.
- **Reliability check:** Cross-validate FotMob minutes against FPL `element-summary` for PL matches (where both sources exist). Log MAE between sources.
- GW32 (post-international break) is the first natural test window.

#### E-F7: Fantasy Football Scout RSS Parser
- Standard RSS 2.0 feed at `https://www.fantasyfootballscout.co.uk/feed`, updates hourly.
- Covers: injuries, DGW/BGW confirmations, international duty minutes, team news.
- Parse into `PlayerSignal` structure: `{player_name, source, signal_type, text, timestamp}`.
- **Player name → player_code resolution:** multi-field join strategy required — FFS RSS uses full names (e.g., "Mohamed Salah") while FPL bootstrap uses `web_name` abbreviations (e.g., "Salah"). Resolution order: (1) exact match on `web_name`; (2) if ambiguous (multiple players match), fall back to matching against `first_name + ' ' + second_name` from bootstrap; (3) if still unresolved, log to `results/signal_unresolved.csv` for manual review. Treat ambiguous matches as unresolved rather than picking arbitrarily. Common-surname collisions (e.g., "Wilson" = Callum Wilson or Ben Wilson) must never resolve silently.
- Feeds: Track G Phase 1 news feed display.

#### E-F8: Reddit r/FantasyPL API Client
- Reddit JSON API (no auth required for public subreddits): `https://www.reddit.com/r/FantasyPL/new.json`
- Collect top posts 24–48h before deadline. Filter for: injury mentions, rotation alerts, differential suggestions.
- Research finding: effective for identifying differentials with under 10% ownership generating buzz before price/ownership moves.
- Phase 1: display only. Phase 2: community sentiment signal with per-source accuracy tracking.

#### E-F9: premierinjuries.com Scraper
- Structured website for @BenDinnery injury readiness content. No X/Twitter dependency.
- Scrape player injury status table. Parse into `PlayerSignal` with `signal_type: doubt | available | injured`.
- Cross-verify every signal against FPL API `status` field before any xP adjustment.

### P3c: Ensemble Predictions
- **Deferred until after Track B ships.**
- After Track B: becomes 4 RF models + 4 XGBoost models (8 total). Ensemble averages within each position bucket.
- Effort remains 0.5 day but must be re-scoped against the per-position architecture.

---

## 5. Track B — Fixture-Aware Per-Position Models

### Breaking Changes from Current Spec

#### Fallback: per-position → all-or-nothing
**Old:** if a position model is missing, fall back to `ep_next` for that position only (other positions use the model).  
**New:** all 4 models must be present and feature-compatible, or the entire squad selection falls back to `ep_next`.  
**Why:** mixing model xP and `ep_next` xP in the same optimizer objective corrupts the ILP — they are on different calibration scales. All-or-nothing gives clean failure attribution.

**Action required on roadmap:** Update `docs/improvements-roadmap.md` Track B "Config changes" section — replace the per-position fallback description with all-or-nothing. The roadmap currently reads "Fallback per position: if model missing or feature mismatch → `ep_next` for that position's players only. Other positions unaffected." This is now superseded and must be corrected to avoid a live contradiction with `predict.py` implementation.

```python
# config.py
ACTIVE_MODELS = {
    "GK":  MODELS_DIR / "rf_gk_gw{N}.sav",
    "DEF": MODELS_DIR / "rf_def_gw{N}.sav",
    "MID": MODELS_DIR / "rf_mid_gw{N}.sav",
    "FWD": MODELS_DIR / "rf_fwd_gw{N}.sav",
}
# predict.py: if any model missing → log warning → use ep_next for ALL players
```

#### Model lockstep constraint
All 4 models must share the same GW label to be promoted together. The retrain phase must train and save all 4 atomically, or fail entirely.

#### Walk-forward CV (mandatory pre-condition)
- **Before any B-F code:** implement walk-forward CV harness.
- Train: seasons 2021-22 through 2024-25 (all 4 historical seasons; `src/config.py:SEASONS` already includes all of these).
- Test: 2025-26 (31 GWs available as of 2026-04-04). **Note:** 2025-26 data availability depends on `post-gw` having been run for each GW. Verify `data/Fantasy-Premier-League/data/2025-26/gws/` contains `gw{N}_live.csv` files before running CV. If fewer than ~10 GWs are available, defer walk-forward CV validation until end of season and use 2024-25 as temporary held-out test set.
- All reported ρ and MAE metrics must use this temporal split.
- The `train_test_split(random_state=42)` in `phase_retrain` causes **temporal leakage** — future GW rows from 2025-26 can appear in the training fold, making metrics unreliable. Fix as part of B-F6: replace with a temporal split — rows from seasons up to and including 2024-25 go to train; 2025-26 rows go to test. Do not use row-level random sampling.

#### Pre-conditions before starting (audits required)
1. **Team ID audit:** verify FPL team IDs are stable across vaastav seasons for promoted/relegated clubs. Run before B-F2 (opponent join). If IDs change, build a team ID normalisation map.
2. **Kickoff time availability:** verify `kickoff_time` exists in historical `fixtures.csv` for all 4 training seasons. Required for `rest_days` feature. If absent for any season, `rest_days` cannot be used as a training feature (only available at inference) — must be documented and handled.

### Success Gate — Replace Spearman ρ ≥ 0.65

**Old gate:** Spearman ρ ≥ 0.65 (no source).  
**New gates (all must pass on 2025-26 walk-forward held-out test):**
1. Hauler RMSE ≤ 5.14 (OpenFPL benchmark, sourced) — primary
2. Per-position MAE ≤ global model MAE for that position's players — no regression
3. Within-squad Spearman ρ tracked post-GW in `accuracy_log.csv` — logged, no hard target first season

### Track C P1b: Soft Multiplier (design constraint, not a Track B task)
This is a **design constraint on Track C P1b's implementation**, not a task within Track B scope. Track B tasks (B-F1 through B-F8) do not depend on Track C P1b being shipped first. Update roadmap Track C P1b description to include the soft multiplier formulation.

The two-stage model spec must use a **soft multiplier**, not a hard 0.5 threshold:
```python
# Correct: expected value formulation
xP_adjusted = xP_stage2 * P_plays_60min  # soft weight

# Wrong (current spec): cliff edge
if P_plays_60min > 0.5:
    use xP_stage2
else:
    xP = 0  # binary exclusion
```
Note: `availability_status` is not available in vaastav training data. Use `minutes > 0` in the previous GW as the historical proxy for Stage 1 training.

### Tasks — additional pre-conditions

| ID | Description | Pre-condition |
|----|-------------|--------------|
| B-PRE-1 | Walk-forward CV harness | Before any B-F task |
| B-PRE-2 | Team ID audit across vaastav seasons | Before B-F2 |
| B-PRE-3 | Kickoff time availability check in historical fixtures | Before B-F3 |
| B-PRE-4 | E-F5 xG validation gate | Before using understat xGC in B-F1b (see below) |

**B-F1 split into two sub-tasks (B-F1 is now unconditionally startable after B-PRE-1 through B-PRE-3):**

| ID | Description | Dependency |
|----|-------------|------------|
| B-F1a | Implement `xGC_rolling_4` using vaastav `goals_conceded` aggregated per team per GW — always available, no external dependency. This is both the permanent fallback and the starting baseline. | B-PRE-1, B-PRE-2 |
| B-F1b | If E-F5 passes: swap `xGC_rolling_4` source to `understat_xGC`. If E-F5 fails: keep vaastav fallback from B-F1a. | E-F5 gate result |

---

## 6. Track D — Optimizer Enhancements

### Replace Sharpe with Max-Min Robust ILP

**Old:** Sharpe ratio / QP optimization (requires cvxpy, not LP-compatible).  
**New:** Max-min robust ILP — maximize worst-case squad score using a box uncertainty set. LP-compatible, implementable in PuLP today. Research-backed (cited in sources as robust integer programming approach).

**Formulation:**
- Each player has predicted xP (`c_j`) ± uncertainty margin (`d_j`).
- `d_j = std_dev_roll_6` — rolling standard deviation of player's actual `total_points` over last 6 GWs. Computed via pandas `.std()` over a 6-row rolling window using the same `groupby([player_id, 'season'])` pattern as existing rolling means in `features.py`. Source: vaastav merged GW rows (same as all other rolling features) — not `element-summary`. Note: `DEFAULT_WINDOWS = [4, 8]` in `features.py` does not include window 6; D-F1 adds a separate `.std()` rolling call, not an extension of the existing `.mean()` loop.
- Objective: maximize `sum((c_j - d_j) * x_j)` — worst-case total.
- Same ILP constraints as current `optimize.py` (budget, positional limits, 3-per-club).

**Implementation target:** Extend `recommend.py` (Option A — lower risk). The multi-GW transfer planner already has horizon loop and hit cost model. Add `d_j` per player per GW as a new input column in `predictions_gw{N}.csv`.

**Sharpe:** Deferred to Track D Phase 2, after robust ILP is validated and prediction quality (Track B/C) is solid.

### Discrete Task Structure (single culprit per E2E test)

| ID | Description | Files touched | Test assertion |
|----|-------------|---------------|----------------|
| D-F1 | Add `std_dev_roll_6` computation to `features.py` | `features.py` | Test: `std_dev_roll_6` is non-null for players with ≥ 6 GW history; null for < 6 |
| D-F2 | Persist `std_dev_roll_6` in `predictions_gw{N}.csv` | `predict.py` | Test: column present in output CSV; no NaN for qualified players |
| D-F3 | Add `--strategy robust` flag to `recommend` CLI | `run.py` | Test: flag accepted; default strategy unchanged |
| D-F4 | Implement worst-case objective in `recommend.py` | `recommend.py` | Test: robust squad xP ≤ deterministic squad xP (worst-case ≤ expected) |
| D-F5 | Add horizon + transfer cost to robust objective | `recommend.py` | Test: 2-GW robust plan penalises hit correctly vs 1-GW |
| D-F6 | Log `robust_strategy` flag in `accuracy_log.csv` | `analysis.py` | Test: column present; value matches strategy used |
| D-F7 | TDD tests for all above | `tests/` | — |

### Optimizer Metric: Perfect Squad Gap
- Post-GW: re-run `optimize.py` with actual GW points as `xP` inputs.
- Compare against what was selected pre-GW.
- Log `perfect_squad_pts / selected_squad_pts` ratio per GW to `accuracy_log.csv`.
- Since PuLP/CBC is a proven-optimal MILP solver, any gap is prediction error, not solver error.

---

## 7. Track F — Web App

### Key Design Decisions (interview outcomes)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Hosting | **Deferred to deployment evaluation** | Render free tier cold start is Critical risk on deadline day; Railway Hobby ($5/mo) and Fly.io free tier are alternatives. Evaluate at deployment. |
| API versioning | `/api/v1/` prefix from day one | Track B changes predictions schema (DGW per-fixture columns); Chrome extension must pin to a version |
| `POST /api/pipeline/run` security | Static token in `X-API-Key` header + 202 Accepted + async background task | Prevents compute spam and FPL API rate-limit abuse; no full auth system needed for personal use |
| Data freshness | Add `generated_at` (CSV file mtime) to every API response | Dashboard can surface "predictions last updated: 3 days ago" on deadline day |
| Database dev environment | **Postgres from day one** (local Docker container) | SQLite loose typing (accepts strings in float columns) causes silent migration failures; boolean 0/1 vs `true` asymmetry breaks Pydantic responses |

---

## 8. Track G — Social Media Signals

### Updated Signal Source Hierarchy

| Tier | Source | Method | Signal types | Reliability |
|------|--------|---------|--------------|-------------|
| 1 | **FPL API `news` field** | Already in pipeline | Official availability status | Highest — ground truth |
| 2 | **Fantasy Football Scout RSS** | Standard RSS 2.0, hourly | Injuries, DGW/BGW confirmations, international duty minutes | High — structured, FPL-specific |
| 3 | **understatAPI + FotMob (soccerdata)** | Python libraries | European/international xG, xA, minutes played | Medium — cross-validated against FPL |
| 4 | **Reddit r/FantasyPL API** | Free JSON API, no auth | Differential buzz, community wisdom 24–48h ahead of price moves | Medium — research-backed |
| 4b | **premierinjuries.com** | Structured scrape | @BenDinnery injury readiness, physical status | Medium — structured website |
| 5 | **X: @FPL_Rockstar** | Nitter RSS / manual paste | Pre-deadline lineup leaks | Medium — account-specific, fragile |

**Dropped:** General Nitter RSS of @FPLFocal — replaced by Fantasy Football Scout RSS which covers the same territory with better reliability and structure.

**Added from research:** @FPL_Rockstar (lineup leaks), @BenDinnery (injuries via premierinjuries.com), Reddit r/FantasyPL (community wisdom / differential signals).

### Cross-Verification Rule (hard constraint, both phases)
Any Tier 2–5 signal that contradicts the FPL API `status` field is:
- **Downgraded to display-only** — shown in dashboard but never fed into xP adjustment.
- **Logged with contradiction flag** for per-source accuracy tracking.

FPL API is the final word on availability. External signals are only acted on in the window before FPL API has updated.

### Phase 2 Hard Gate — No Code Until Spec Exists
Phase 2 (xP auto-adjustment) must NOT be implemented until the following spec document is written and approved:

1. **Activation threshold per signal type** — source accuracy ≥ 80% over ≥ 15 confirmed observations per source-type pair (e.g., `premierinjuries.com × doubt`). For signal types that cannot reach 15 observations within a season (e.g., `confirmed_starter` for a specific player), the Phase 2 spec must define either a pooled fallback threshold (e.g., ≥ 10 observations across all `confirmed_starter` signals from that source) or an explicit deferral policy for that signal type. The ≥ 15 threshold does not apply pooled across all signal types — each source-type pair is evaluated independently.
2. **Error cost matrix** — false positive on `confirmed_starter` costs captain multiplier (12–20 pts); false negative costs 2–4 pts. Thresholds must reflect asymmetry.
3. **Dry-run mode** — mandatory. Phase 2 must be activatable in "what would it have done?" mode before any live xP adjustment.
4. **Feature flag** — Phase 2 defaults OFF. Activated per signal type individually, not globally.
5. **Per-source accuracy log** — `results/signal_accuracy.csv` tracking `{signal_id, source, signal_type, predicted_status, actual_started, gw}`.

### NLP Parsing Guidance (from research)
Research found standard NLP classifiers fail on FPL-specific language:
- FPL community uses emojis as signals (🚑 = injured player) — not parseable by standard sentiment models.
- "Confident he'll be fine" and "doubtful" require context, not keyword matching.
- Recommendation: use rule-based parsing for structured sources (premierinjuries.com, FFS RSS). Reserve NLP for Reddit/X only, and only for Phase 2 after Phase 1 feedback validates source quality.

---

## 9. Pending Verifications (action items before implementation)

| Item | How to verify | Gates |
|------|--------------|-------|
| FPL bootstrap API reflects price changes by 7:30am IST (02:00 UTC) | Check bootstrap response tomorrow morning at 02:00 UTC; compare `now_cost` values before/after 01:30 UTC | E-F1 bootstrap action timing |
| `kickoff_time` in historical vaastav `fixtures.csv` for all 4 training seasons | `head data/Fantasy-Premier-League/data/2021-22/fixtures.csv` | B-F3 `rest_days` feature |
| FPL team IDs stable across vaastav seasons for promoted/relegated clubs | Compare `teams/` directory across 4 seasons | B-F2 opponent join |
| understat xG ρ vs FPL Opta xG ρ on actual goals | E-F5 validation task | Track B xGC feature source |
| FotMob minutes vs FPL element-summary minutes for PL matches | E-F6 reliability check | Track G Tier 3 |

---

## 10. Decisions Explicitly Deferred

| Item | Status |
|------|--------|
| Sharpe ratio / QP optimization | Track D Phase 2 — after robust ILP validated and predictions solid |
| Track F hosting (Render vs Railway vs Fly.io) | Evaluate at deployment time |
| Reddit NLP / X general sentiment | Phase 2 only, after Phase 1 accuracy data collected |
| Track G Phase 2 xP multiplier values | Defined in Phase 2 threshold spec document, not this spec |
| Hyperparameter tuning per position model | Track B stretch goal — after basic per-position training validated |
