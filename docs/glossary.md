# FPL Pipeline — Variable Glossary

All columns, constants, and derived variables used across the active pipeline (`src/pipeline/`).
Variables are grouped by origin: vaastav/API input → engineered features → optimizer outputs.

---

## Player Identity

| Variable | Source | Description |
|----------|--------|-------------|
| `element` | vaastav / FPL API | **Seasonal** FPL player ID. Recycled each season — do NOT use for cross-season joins. |
| `code` | `players_raw.csv` → `prepare.py` | **Persistent** internal FPL player code. Stable across seasons. Primary key for all historical joins. |
| `name` / `web_name` | FPL API `bootstrap-static` | Display name (e.g. `"Saka"`). For presentation only, not joining. |
| `position` | `element_type` in API, normalised in `fetch.py` | Position string: `GK`, `DEF`, `MID`, `FWD`. Vaastav uses same schema. |
| `team` | FPL API / vaastav | Team name string (e.g. `"Arsenal"`). Not an ID — safe for display and grouping. |
| `opponent_team` | vaastav / API `element-summary` | Integer team ID of the opposing team for that GW fixture. |

---

## Fixture & FDR

> **Source:** `add_fixture_difficulty()` in `prepare.py:53-65`, joined from `fixtures.csv`.

| Variable | Source Column | Description |
|----------|--------------|-------------|
| `fixture` | vaastav GW data | FPL fixture ID. Used to join with `fixtures.csv`. |
| `was_home` | vaastav GW data | `True` if the player's team played at home in that GW. |
| `fdr_team` | `team_h_difficulty` (if home) or `team_a_difficulty` (if away) | **How hard this fixture is FOR the player's own team.** High = hard for the player. FDR 1=very easy opponent, 5=very hard opponent. Spans 1–5 across a season. **Use this for xP weighting.** |
| `fdr_opp` | `team_a_difficulty` (if home) or `team_h_difficulty` (if away) | **How hard this fixture is FOR the opponent.** Reflects the player's own team's strength as rated by FPL. For elite teams (e.g. Arsenal), this is near-constant at 4–5 all season — no useful signal for xP adjustment. |

> **For xP weighting use `fdr_team`** (verified against 2025-26 Arsenal fixtures):
> `fdr_weight = 1.0 - fdr_sensitivity × (fdr_team − 3) / 2`
>
> | `fdr_team` | Interpretation | `fdr_weight` (sensitivity=0.15) |
> |-----------|---------------|--------------------------------|
> | 1 | Very easy opponent | 1.15 |
> | 2 | Easy opponent | 1.075 |
> | 3 | Average opponent | 1.0 |
> | 4 | Hard opponent | 0.925 |
> | 5 | Very hard opponent | 0.85 |
>
> **Why not `fdr_opp`?** For Arsenal, `fdr_opp` = 4 or 5 every GW (opponents always find Arsenal difficult).
> Using `fdr_opp` in the formula discounts Saka whenever Arsenal are strong — exactly backwards.
> `fdr_team` spans 1–5 for Arsenal (e.g. GW25 vs Sunderland=2, GW33 at Man City=5) and gives correct direction.

---

## Base Performance Stats (per GW row)

These come directly from vaastav `merged_gw.csv` or live API (`fetch.py`).

| Variable | Description |
|----------|-------------|
| `total_points` | FPL points scored that GW (includes all bonuses). Target variable for model training. |
| `minutes` | Minutes played that GW. |
| `goals_scored` | Goals scored. |
| `assists` | Assists. |
| `clean_sheets` | 1 if the player's team kept a clean sheet, 0 otherwise. |
| `goals_conceded` | Goals conceded by the player's team. |
| `bonus` | Bonus points awarded (0–3, top 3 BPS earners). |
| `bps` | Bonus Points System score — raw BPS before bonus allocation. Useful leading indicator. |
| `starts` | 1 if the player started, 0 if sub. |

---

## ICT Index Components

Proprietary FPL ratings, each 0–100+:

| Variable | Description |
|----------|-------------|
| `influence` | Impact on the match (tackles, interceptions, key passes, goals etc.). |
| `creativity` | Chance creation (key passes, crossing, dribbling). |
| `threat` | Goal threat (shots, shots on target, touches in box). |
| `ict_index` | Composite of influence + creativity + threat. General player quality proxy. |

---

## Expected Stats (xStats)

Available from 2022-23 onwards via FPL API. Not in older vaastav seasons.

| Variable | Description |
|----------|-------------|
| `expected_goals` | xG — expected goals from shot positions. Leading indicator for forwards/midfielders. |
| `expected_assists` | xA — expected assists from key passes. |
| `expected_goal_involvements` | xG + xA combined. |
| `expected_goals_conceded` | xGC — expected goals conceded (relevant for defenders/GKs). |

---

## Transfer & Ownership

| Variable | Description |
|----------|-------------|
| `transfers_in` | Number of FPL managers who transferred the player IN that GW. |
| `transfers_out` | Number of FPL managers who transferred the player OUT that GW. |
| `transfers_net` | `transfers_in - transfers_out` — engineered in `features.py:add_form_features()`. Positive = net demand. |
| `selected` | Number of FPL managers who owned the player going into that GW. |
| `value` | Player price in vaastav data (0.1M units, e.g., 80 = £8.0M). Synonym for `now_cost` in API context. |
| `now_cost` | Player's current price from the FPL API (0.1M units). Normalised to `value` in `fetch.py:normalize_player_gw_to_vaastav()`. |

---

## Engineered Features

Generated by `features.py`. All use `shift(1)` so GW N's features are derived from GWs 1…N-1 only (no data leakage).

### Rolling Averages

Pattern: `{stat}_roll_{window}` — rolling mean over last `window` gameweeks.

| Variable | Window | Description |
|----------|--------|-------------|
| `total_points_roll_4` | 4 GWs | Short-term form (points). **Primary model feature.** |
| `total_points_roll_8` | 8 GWs | Long-term form (points). |
| `minutes_roll_4` | 4 GWs | Minutes played recently — proxy for rotation risk. |
| `minutes_roll_8` | 8 GWs | Season-long minutes trend. |
| `ict_index_roll_4` | 4 GWs | Recent ICT form. |
| `ict_index_roll_8` | 8 GWs | Season-long ICT trend. |
| `bps_roll_4` | 4 GWs | Recent BPS form. |
| `bps_roll_8` | 8 GWs | Season-long BPS trend. |
| `goals_scored_roll_4` | 4 GWs | Recent goal scoring rate. |
| `assists_roll_4` | 4 GWs | Recent assist rate. |
| `clean_sheets_roll_4` | 4 GWs | Recent clean sheet rate (relevant for GKs/DEFs). |
| `influence_roll_4` | 4 GWs | Recent influence. |
| `creativity_roll_4` | 4 GWs | Recent creativity. |
| `threat_roll_4` | 4 GWs | Recent threat. |

### Momentum Features

Pattern: `{stat}_momentum` = `{stat}_roll_4 - {stat}_roll_8`. Positive = improving form.

| Variable | Description |
|----------|-------------|
| `total_points_momentum` | Points trending up (positive) or down (negative) vs season average. |
| `minutes_momentum` | Playing time trending up (positive = more starts recently). |
| `ict_index_momentum` | ICT trending up vs season average. |

---

## Prediction Output

| Variable | Source | Description |
|----------|--------|-------------|
| `xP` | `predict.py` output | **Expected Points** — model's predicted FPL points for the next GW. Primary optimizer input. |
| `ep_next` | FPL API `bootstrap-static` | FPL's own expected points for the next GW. Used as **fallback** when no trained model is available (or model features are mismatched). |
| `ep_this` | FPL API `bootstrap-static` | FPL's expected points for the current (live) GW. Captured pre-deadline via `fetch.py:extract_xp_snapshot()`. |

---

## Availability Filtering

| Variable | Source | Description |
|----------|--------|-------------|
| `status` | FPL API `elements[].status` | Player availability code: `a`=available, `d`=doubtful, `i`=injured, `u`=unavailable, `s`=suspended, `n`=not in squad. |
| `chance` | FPL API `elements[].chance_of_playing_next_round` | % chance of playing: `null`, `25`, `50`, `75`, `100`. |
| `news` | FPL API `elements[].news` | Free-text injury/suspension note. |

**Availability decision table** (from `availability.py`, first match wins):

| Condition | Action |
|-----------|--------|
| `status` in `{i, u, s, n}` | Hard exclude |
| `chance` in `{0, 25}` | Hard exclude |
| `chance == 50` | `xP × 0.50` |
| `status == 'd'` and `chance is None` | `xP × 0.50` |
| `chance == 75` | `xP × 0.75` |
| Otherwise | No adjustment |

---

## Optimizer Variables

| Variable | Description |
|----------|-------------|
| `squad` | 15-player optimal squad (output of `select_squad()`). |
| `xi` | Best 11 from the squad (output of `select_xi()`). Formation: 1 GK, 3–5 DEF, 2–5 MID, 1–3 FWD. |
| `bench` | 4 players in squad but not in XI. |
| `captain` | Player with highest `xP` in the XI. Points doubled. |
| `vice_captain` | Player with second-highest `xP` in the XI. |
| `total_xp` | `xi["xP"].sum() + captain["xP"]` — XI total with captain doubling. |

---

## Config Constants (`src/config.py`)

| Constant | Value | Description |
|----------|-------|-------------|
| `SQUAD_RULES["budget"]` | 1000 | Budget in 0.1M units (= £100M). |
| `SQUAD_RULES["squad_size"]` | 15 | Total squad size. |
| `SQUAD_RULES["xi_size"]` | 11 | Starting XI size. |
| `SQUAD_RULES["max_per_team"]` | 3 | Max players from any single club. |
| `SQUAD_RULES["positions"]` | `{GK:2, DEF:5, MID:5, FWD:3}` | Squad composition. |
| `AVAILABILITY_HARD_EXCLUDE_STATUS` | `{i, u, s, n}` | Status codes causing hard exclusion. |
| `AVAILABILITY_HARD_EXCLUDE_CHANCE` | `{0, 25}` | Chance values causing hard exclusion. |
| `AVAILABILITY_SOFT_SCALE` | `{50: 0.50, 75: 0.75}` | Chance → xP scale factor. |
| `BOOTSTRAP_MAX_AGE_HOURS` | 48 | Bootstrap cache lifetime before forced refresh. |
| `ACTIVE_MODEL` | `models/rf_model_gw<N>.sav` | Path to the active trained model. Update after each retrain. |

---

## Planned Variables (P1 Improvements Spec)

From `docs/superpowers/specs/2026-03-28-improvements-design.md`:

| Variable | Source | Description |
|----------|--------|-------------|
| `entry_id` | `user_config.yaml` | User's FPL team ID for the current season (changes each season). |
| `selling_price` | `/api/entry/{id}/transfers/` | What the user will receive if they sell a player (= buy price + 50% of profit, rounded down, in 0.1M units). Always ≤ `now_cost`. |
| `bank` | `/api/entry/{id}/` | Remaining unspent budget in 0.1M units. |
| `free_transfers` | `/api/entry/{id}/history/` | Banked free transfers available (range 1–5). Unused FTs bank up to 5, then are lost. |
| `active_chip` | `/api/entry/{id}/event/{gw}/picks/` | Current active chip: `wildcard`, `freehit`, `bboost`, `3xc`, or `None`. |
| `total_value` | Derived | `sum(selling_prices) + bank` — total squad value including bank. |
| `fdr_weight` | Derived in optimizer | `1.0 - fdr_sensitivity × (fdr_team − 3) / 2`. Multiplier applied to `xP` per player per GW in multi-GW planning. |
| `fdr_sensitivity` | `user_config.yaml` | Controls how aggressively FDR adjusts xP (default 0.15; range 0–0.30). |
| `horizon_gws` | `user_config.yaml` | Number of future GWs to plan transfers for (default 5). |
| `hit_cost` | FPL rules | Points penalty per extra transfer beyond free transfers (-4 per hit). |
| `avg_score` | FPL API `/api/event/{gw}/live/` | Average FPL manager score for the GW. Benchmark for season log. |
| `median_score` | FPL API league standings | 50th-percentile score for the GW. |
| `best_rank` | FPL API | Best rank achieved in the season (overall). |

---

## Vaastav Dataset Columns (not used in current pipeline)

These exist in `merged_gw.csv` but are not currently consumed by the active pipeline:

| Column | Notes |
|--------|-------|
| `clearances_blocks_interceptions` | Defensive stat; unavailable from FPL API — NaN-filled for API-sourced rows. |
| `recoveries` | Unavailable from API. |
| `tackles` | Unavailable from API. |
| `defensive_contribution` | Unavailable from API. |
| `round` | Alias for `GW` in API-sourced rows. |
| `kickoff_time` | ISO timestamp of fixture kickoff. |
