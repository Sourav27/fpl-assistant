# Track H — Data Source Consolidation Design

**Date:** 2026-04-05  
**Status:** Approved for planning  
**Based on:** `docs/superpowers/plans/2026-04-05-track-h-data-sources.md` (Track H complete, 208 tests)  
**Next step:** Implementation plan via writing-plans skill

---

## Problem Statement

Track H built 9 data source modules. Live testing against real APIs revealed two broken sources (understatapi async bug, FotMob session server down), one source with no player-level method (soccerdata FotMob), and one blocked by anti-scraping (FBref). Simultaneously, live pulls confirmed significant competition coverage gaps: FPL API misses 180+ non-PL minutes per player per UCL round. This spec defines the corrected source map, replaces broken clients, and adds ESPN as the non-PL source.

---

## Source Decisions

| Source | Decision | Reason |
|--------|----------|--------|
| FPL API | **Keep — primary** | PL post-GW actuals + pre-GW bootstrap. Full KPI set confirmed live. |
| soccerdata Understat | **Keep — unique KPIs only** | `xg_chain` + `xg_buildup` have no FPL equivalent. Drop all overlapping columns (xg, xa, goals, assists, shots, key_passes, yellow_cards, red_cards). Fix: replace `understatapi` async client with `soccerdata.Understat` (synchronous). |
| ESPN direct API | **Add — replaces FotMob** | Covers UCL, UEL, FA Cup (`eng.fa`), Carabao Cup (`eng.league_cup`), FIFA internationals (`fifa.friendly`). Confirmed via live eventlog pulls (Cole Palmer: FA Cup, Carabao, UCL, England internationals all returned). No xG/xA available. |
| FBref via soccerdata | **Drop** | Anti-scraping 403 in live test. No UCL/cup coverage via soccerdata wrapper (Big 5 + WC/Euros only). |
| FotMob via soccerdata | **Drop** | Session server at `46.101.91.154:6006` down. `read_player_match_stats` method does not exist in soccerdata FotMob class. |
| FFS RSS | **Keep — availability corroboration** | Confirmed live: 2 Enzo mentions day-of (suspension signal). |
| Reddit r/FantasyPL | **Keep — availability corroboration** | Confirmed live: score-688 post flagged Enzo drop same day. |
| premierinjuries.com | **Keep — availability fallback** | Structured injury status. Not listed = not injured (confirmed for Enzo suspension vs injury distinction). |

---

## SOURCE_COLUMN_MAP

Declared in `src/pipeline/datasources/__init__.py`. All source modules must emit exactly the columns listed for their source key.

```python
SOURCE_COLUMN_MAP = {
    # --- Per-match actuals (post-GW, shift(1) for rolling features) ---
    # Source: element-summary/{id}/history — one row per PL match played
    "fpl_post_gw": {
        "role": "primary",
        "competitions": ["PL"],
        "timing": "after_gw",
        "endpoint": "element-summary/{id}/history",
        "columns": [
            # Join / fixture context
            "element",           # FPL player id (current-season, not persistent — use code for joins)
            "fixture",           # FPL fixture id
            "round",             # GW number
            "kickoff_time",      # ISO datetime
            "opponent_team",     # opposing team id
            "was_home",
            "team_h_score", "team_a_score",
            "modified",          # last modified timestamp
            # Performance
            "minutes", "starts",
            "goals_scored", "assists", "own_goals",
            "clean_sheets", "goals_conceded",
            "saves", "penalties_saved", "penalties_missed",
            "yellow_cards", "red_cards",
            # Opta xG / ICT
            "expected_goals", "expected_assists",
            "expected_goal_involvements", "expected_goals_conceded",
            "influence", "creativity", "threat", "ict_index",
            # Defensive
            "clearances_blocks_interceptions", "recoveries",
            "tackles", "defensive_contribution",
            # Bonus / points
            "bonus", "bps", "total_points",
            # Transfer market / ownership
            "transfers_in", "transfers_out", "transfers_balance", "selected",
            # Price
            "value",             # price at time of match (tenths of £M)
        ],
    },

    # --- Pre-GW snapshot (available at prediction time) ---
    # Source: bootstrap-static/elements — one row per player, current state
    # NOTE: also contains season-to-date cumulative totals (same stat names as
    # fpl_post_gw). These are season aggregates, not per-match rows. Feature
    # selection in Track C will decide which pre-GW cumulative stats to use.
    "fpl_pre_gw": {
        "role": "pre_gw_snapshot",
        "competitions": ["PL"],
        "timing": "before_gw",
        "endpoint": "bootstrap-static/elements",
        "columns": [
            # --- Identity / metadata ---
            "id",                # FPL season-specific id
            "code",              # persistent cross-season player code
            "element_type",      # 1=GKP 2=DEF 3=MID 4=FWD
            "team",              # team id
            "team_code",         # persistent team code
            "first_name", "second_name", "web_name", "known_name",
            "opta_code",         # Opta player code
            "squad_number",
            "birth_date",
            "region",            # nationality region code
            "team_join_date",
            "removed",           # True if delisted mid-season
            "special",           # True for special (e.g. combined cards)
            "has_temporary_code",
            "photo",             # photo filename
            # --- Availability (feeds availability_features.py) ---
            "status",            # a/d/i/u/s/n
            "news", "news_added",
            "chance_of_playing_this_round", "chance_of_playing_next_round",
            "can_transact",      # transfer eligibility flag
            "can_select",        # squad selection eligibility flag
            "scout_risks",       # FPL scout risk flags
            "scout_news_link",   # link to scout article
            # --- Set piece role ---
            "corners_and_indirect_freekicks_order",
            "corners_and_indirect_freekicks_text",
            "direct_freekicks_order", "direct_freekicks_text",
            "penalties_order", "penalties_text",
            # --- Season-to-date cumulative stats (same fields as fpl_post_gw) ---
            "minutes", "starts",
            "goals_scored", "assists", "own_goals",
            "clean_sheets", "goals_conceded",
            "saves", "penalties_saved", "penalties_missed",
            "yellow_cards", "red_cards",
            "expected_goals", "expected_assists",
            "expected_goal_involvements", "expected_goals_conceded",
            "influence", "creativity", "threat", "ict_index",
            "clearances_blocks_interceptions", "recoveries",
            "tackles", "defensive_contribution",
            "bonus", "bps",
            # --- Per-90 season aggregates ---
            "expected_goals_per_90", "expected_assists_per_90",
            "expected_goal_involvements_per_90", "expected_goals_conceded_per_90",
            "clean_sheets_per_90", "saves_per_90",
            "goals_conceded_per_90", "starts_per_90",
            "defensive_contribution_per_90",
            # --- FPL form / prediction signals ---
            "form", "points_per_game", "ep_next", "ep_this",
            "event_points",      # points in the most recent GW
            "total_points",      # season total
            "dreamteam_count", "in_dreamteam",
            # --- Rank signals ---
            "ict_index_rank", "ict_index_rank_type",
            "creativity_rank", "creativity_rank_type",
            "threat_rank", "threat_rank_type",
            "influence_rank", "influence_rank_type",
            "now_cost_rank", "now_cost_rank_type",
            "form_rank", "form_rank_type",
            "points_per_game_rank", "points_per_game_rank_type",
            "selected_rank", "selected_rank_type",
            # --- Price / value ---
            "now_cost",
            "cost_change_event", "cost_change_event_fall",
            "cost_change_start", "cost_change_start_fall",
            "price_change_percent",
            "value_form", "value_season",
            # --- Transfer momentum ---
            "transfers_in", "transfers_out",
            "transfers_in_event", "transfers_out_event",
            # --- Ownership ---
            "selected_by_percent",
        ],
    },

    # --- Understat: unique creative chain metrics (PL only) ---
    "understat": {
        "role": "unique",
        "competitions": ["PL"],
        "timing": "after_gw",
        "client": "soccerdata.Understat (synchronous)",
        "columns": ["xg_chain", "xg_buildup"],
        "join_key": "(player_code, gw_date)",  # requires date→GW mapping (see risks)
    },

    # --- ESPN: all non-PL competitions ---
    "espn": {
        "role": "primary",
        "competitions": ["UCL", "UEL", "UECL", "FA_Cup", "Carabao", "FIFA_Friendly", "INT"],
        "espn_league_slugs": [
            "uefa.champions",    # confirmed via Palmer/Enzo live test
            "uefa.europa",       # unverified slug — probe before use
            "uefa.europa.conf",  # unverified slug — probe before use
            "eng.fa",            # confirmed via Palmer FA Cup live test
            "eng.league_cup",    # confirmed via Palmer Carabao Cup live test
            "fifa.friendly",     # confirmed via Palmer England internationals live test
        ],
        "timing": "after_match",
        "seasons": "2021-present",
        "endpoint": "sports.core.api.espn.com/v2/sports/soccer/athletes/{id}/eventlog",
        "columns": [
            "minutes", "goals", "assists",
            "shots", "shots_on_target",
            "yellow_cards", "red_cards",
            "fouls_committed", "fouls_suffered",
            "offsides",
        ],
        "note": "No xG/xA available from ESPN in any competition (confirmed live).",
    },

    # --- Availability signals ---
    "fpl_news": {
        "role": "availability_primary",
        "timing": "before_gw",
        "columns": ["is_injured", "is_doubt", "is_suspended", "availability_raw_text"],
    },
    "premierinjuries": {
        "role": "availability_fallback",
        "timing": "before_gw",
        "columns": ["is_injured", "is_doubt"],
    },
    "ffs": {
        "role": "availability_corroboration",
        "timing": "rolling",
        "columns": ["signal_type", "signal_confidence"],
    },
    "reddit": {
        "role": "availability_corroboration",
        "timing": "rolling",
        "columns": ["signal_type", "signal_confidence"],
    },
}
```

---

## Module Changes

### 1. `understat.py` — fix client + slim to unique columns

**Problem:** Uses `understatapi.UnderstatClient` async context manager (broken in v0.7.1 — raises `TypeError`).  
**Fix:** Replace with `soccerdata.Understat` (synchronous). Emit only `xg_chain` + `xg_buildup`. Drop all other columns.

```python
# Before (broken)
async with UnderstatClient() as client:
    data = await client.league(league="EPL").get_player_data(season=season)

# After (correct)
import soccerdata as sd
u = sd.Understat(leagues="ENG-Premier League", seasons="2324")  # format: "YYYY" e.g. "2324" for 2023-24, "2425" for 2024-25
df = u.read_player_match_stats()
return df[["xg_chain", "xg_buildup"]]
```

**Date→GW join requirement:** Understat returns calendar dates. A `date → GW` mapping must be applied before joining to the FPL GW dataset. Use the FPL fixtures API (`/api/fixtures/`) to build this map per season. Blank/double GWs must be handled (multiple fixtures on same date → same GW).

### 2. `soccerdata_client.py` → `espn_client.py` (rename + full rewrite)

**Replaces:** FotMob wrapper (entire file deleted).  
**New responsibilities:**

- `resolve_espn_player_id(player_code, web_name, second_name) -> int | None`  
  Looks up from a seeded `data/espn_player_id_map.csv` first. Falls back to roster fuzzy-match with confidence threshold ≥ 0.85. Unresolved players logged to `results/espn_unresolved.csv` for manual review.

- `fetch_espn_player_season(espn_id, season_year) -> pd.DataFrame`  
  Calls the eventlog endpoint, filters out `eng.1` (PL — covered by FPL API), fetches match summary for each non-PL event. Returns one row per non-PL match with all ESPN columns.  
  Caches per `results/espn_cache/player_{espn_id}_season_{year}.csv`. Idempotent re-runs skip already-cached seasons.

- `fetch_espn_recent(espn_id, days=30) -> pd.DataFrame`  
  Weekly prediction run: last 30 days of non-PL matches for fatigue signal.

**Historical depth probe (must run before backfill):**  
Before building the full 2021–present backfill, verify that the eventlog API returns 2021-22 data for a known UCL participant (e.g. Thiago Silva, Chelsea, UCL 2021-22). If the endpoint returns empty items for seasons before 2023-24, the non-PL features are unavailable for older training seasons and the design must fall back to match-count-only features for those years.

**Rate limiting:** Add 1.0s sleep between player fetches. On `429` response: exponential backoff (2s, 4s, 8s, max 3 retries). Checkpoint after each player: if interrupted, re-run resumes from last completed player.

### 3. `availability_features.py` — new module (wraps + extends `availability.py`)

**Replaces the dual-system risk.** Rather than running alongside `availability.py`, this module is the single availability entry point. `availability.py`'s `HybridAvailabilityFilter` is called internally.

**Produces four feature columns per player:**

```python
# is_injured: FPL status in {i, u} → physically injured/unavailable due to injury
# status 's' (suspended) → is_suspended (separate column, not is_injured)
# status 'n' (not in squad / loaned out) → excluded from squad entirely, not flagged as injured
is_injured   = int(fpl_status in {"i", "u"})
is_suspended = int(fpl_status == "s")

# is_doubt: FPL status == 'd' (primary signal, NOT chance threshold)
# Edge case: status='d' with chance=100 means doubtful-tagged but expected to start
# → is_doubt=1 but availability.py does NOT scale xP (correct behaviour preserved)
is_doubt = int(fpl_status == "d")

# Fallback: if FPL news empty AND status='a', check premierinjuries
if fpl_status == "a" and not fpl_news:
    is_injured = int(premierinjuries_status == "injured")
    is_doubt   = int(premierinjuries_status == "doubt")

# signal_confidence: weighted average of agreeing sources
# FPL=1.0, premierinjuries=0.8, FFS=0.6, Reddit=0.5
signal_confidence = _compute_weighted_confidence(sources_agreeing)

# n_corroborating_sources: count of sources flagging injury or doubt
n_corroborating_sources = sum(1 for s in all_sources if s.agrees_with_primary)
```

**Note:** `is_doubt` is driven by `status == 'd'`, not by `chance_of_playing < 75`. The chance threshold is a separate, weaker signal that can be used as an additional feature column but must not define `is_doubt`.

### 4. `src/pipeline/source_validation.py` — docstring update only

File lives at the **pipeline root** (`src/pipeline/source_validation.py`), not inside `datasources/`. This is intentional — it is a pipeline-level gate, not a per-source module. Clarify in docstring that the Spearman ρ gate now validates `xg_chain`/`xg_buildup` correlation against actual goal-chain outcomes (not xG vs actual goals, since xG is now FPL-only).

### 5. `__init__.py` — add `SOURCE_COLUMN_MAP`

Export `SOURCE_COLUMN_MAP` alongside `PlayerSignal` as the package's top-level declaration of source ownership.

---

## ESPN Player ID Map

**File:** `data/espn_player_id_map.csv`  
**Columns:** `fpl_code, web_name, espn_id, espn_name, verified`  
**Scope:** ~50-80 players who regularly appear in non-PL competitions (UCL regulars, England/Argentina/etc internationals).  
**Seed:** Manual one-time build using confirmed IDs (Enzo=285450, Palmer=296395, etc).  
**Maintenance:** New signings added at start of each season. Unresolved fuzzy matches surface to `results/espn_unresolved.csv`.

---

## Data Flow

### Training build (2021 → present, run once + annual retrain)

```
For each season:
  1. fpl_post_gw    → element-summary history per player → master GW CSV
  2. understat      → soccerdata.Understat.read_player_match_stats()
                      → date→GW join using FPL fixtures API
                      → append xg_chain, xg_buildup to master GW CSV
  3. espn           → espn_client.fetch_espn_player_season(espn_id, year)
                      → cached per player/season
                      → stored as results/espn_cache/non_pl_{season}.csv
                      → joined on (fpl_code, match_date) → null for PL-only GWs
```

### Weekly prediction run (pre-deadline)

```
  1. fpl_pre_gw     → bootstrap-static snapshot
  2. availability   → availability_features.py:
                        FPL news+status (primary)
                        → premierinjuries (fallback if status='a', no news)
                        → FFS + Reddit (corroboration only)
                        → outputs: is_injured, is_doubt,
                                   signal_confidence, n_corroborating_sources
  3. espn recent    → espn_client.fetch_espn_recent(espn_id, days=30)
                        → non_pl_minutes_roll_4 fatigue signal
```

### Join keys

| Pair | Key |
|------|-----|
| FPL ↔ Understat | `(player_code, season, gw)` via date→GW map |
| FPL ↔ ESPN | `(fpl_code → espn_id lookup, match_date)` |
| Availability ↔ FPL | `player_code` via existing name→code resolver |

---

## Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| ESPN eventlog only covers recent seasons | Probe 2021-22 UCL data before backfill. If unavailable, non-PL features null for pre-2023 training rows — acceptable, model learns from PL features for those years. |
| ESPN rate limiting / 429s | 1s sleep between requests, exponential backoff, per-player-season cache, resumable. |
| Fuzzy name matching for ESPN ID | Seeded manual map for 50-80 non-PL regulars. Fuzzy fallback with ≥0.85 confidence threshold. Unresolved → human review queue. |
| Understat date→GW alignment | FPL fixtures API date→GW map per season. Blank/double GWs handled explicitly. |
| `status='d'` + `chance=100` edge case | `is_doubt` driven by `status == 'd'` only. Chance value is a separate feature column, not the `is_doubt` definition. |
| Dual availability systems | `availability_features.py` wraps `availability.py` internally. Single entry point for all availability logic. |

---

## Files Added / Modified

| File | Change |
|------|--------|
| `src/pipeline/datasources/__init__.py` | Add `SOURCE_COLUMN_MAP` export |
| `src/pipeline/datasources/understat.py` | Fix: replace `understatapi` async → `soccerdata.Understat` sync; slim to `xg_chain`+`xg_buildup`; add date→GW join |
| `src/pipeline/datasources/soccerdata_client.py` | **Delete** |
| `src/pipeline/datasources/espn_client.py` | **New**: eventlog fetch, historical backfill, caching, ID resolver |
| `src/pipeline/datasources/availability_features.py` | **New**: unified availability feature assembly, wraps `availability.py` |
| `src/pipeline/source_validation.py` | Docstring update only |
| `data/espn_player_id_map.csv` | **New**: seeded FPL code → ESPN ID lookup |
| `tests/datasources/test_espn_client.py` | **New**: unit tests for espn_client (HTTP mocked) |
| `tests/datasources/test_availability_features.py` | **New**: unit tests for availability feature assembly |
| `tests/datasources/test_soccerdata.py` | Update: fix understat tests to use soccerdata.Understat |

---

## Out of Scope (deferred to Track C / feature engineering)

- Selecting which columns become model input features (`x-features`) — deferred to feature engineering
- Unified `PlayerMatchRecord` merge layer — deferred to Track C after feature selection
- FA Cup via direct scraping (ESPN `eng.fa` eventlog covers it sufficiently)
- WC qualifiers (CONMEBOL) — ESPN returned 0 events for March 2026; not in scope until confirmed available
