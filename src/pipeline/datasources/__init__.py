from .signals import PlayerSignal, resolve_player_name, log_unresolved_name

SOURCE_COLUMN_MAP = {
    # --- Per-match actuals (post-GW). One row per PL match played. ---
    # Source: element-summary/{id}/history
    "fpl_post_gw": {
        "role": "primary",
        "competitions": ["PL"],
        "timing": "after_gw",
        "endpoint": "element-summary/{id}/history",
        "columns": [
            # Join / fixture context
            "element", "fixture", "round", "kickoff_time",
            "opponent_team", "was_home", "team_h_score", "team_a_score", "modified",
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
            # Price at time of match (tenths of £M)
            "value",
        ],
    },

    # --- Pre-GW snapshot (available at prediction time). One row per player. ---
    # Source: bootstrap-static/elements
    "fpl_pre_gw": {
        "role": "pre_gw_snapshot",
        "competitions": ["PL"],
        "timing": "before_gw",
        "endpoint": "bootstrap-static/elements",
        "columns": [
            # Identity / metadata
            "id", "code", "element_type", "team", "team_code",
            "first_name", "second_name", "web_name", "known_name",
            "opta_code", "squad_number", "birth_date", "region",
            "team_join_date", "removed", "special", "has_temporary_code", "photo",
            # Availability
            "status", "news", "news_added",
            "chance_of_playing_this_round", "chance_of_playing_next_round",
            "can_transact", "can_select", "scout_risks", "scout_news_link",
            # Set piece role
            "corners_and_indirect_freekicks_order", "corners_and_indirect_freekicks_text",
            "direct_freekicks_order", "direct_freekicks_text",
            "penalties_order", "penalties_text",
            # Season-to-date cumulative stats
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
            # Per-90 season aggregates
            "expected_goals_per_90", "expected_assists_per_90",
            "expected_goal_involvements_per_90", "expected_goals_conceded_per_90",
            "clean_sheets_per_90", "saves_per_90",
            "goals_conceded_per_90", "starts_per_90", "defensive_contribution_per_90",
            # FPL form / prediction signals
            "form", "points_per_game", "ep_next", "ep_this",
            "event_points", "total_points",
            "dreamteam_count", "in_dreamteam",
            # Rank signals
            "ict_index_rank", "ict_index_rank_type",
            "creativity_rank", "creativity_rank_type",
            "threat_rank", "threat_rank_type",
            "influence_rank", "influence_rank_type",
            "now_cost_rank", "now_cost_rank_type",
            "form_rank", "form_rank_type",
            "points_per_game_rank", "points_per_game_rank_type",
            "selected_rank", "selected_rank_type",
            # Price / value
            "now_cost",
            "cost_change_event", "cost_change_event_fall",
            "cost_change_start", "cost_change_start_fall",
            "price_change_percent", "value_form", "value_season",
            # Transfer momentum
            "transfers_in", "transfers_out",
            "transfers_in_event", "transfers_out_event",
            # Ownership
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
        "join_key": "(player_code, gw_date)",
    },

    # --- ESPN: all non-PL competitions ---
    "espn": {
        "role": "primary",
        "competitions": ["UCL", "UEL", "UECL", "FA_Cup", "Carabao", "FIFA_Friendly", "INT"],
        "espn_league_slugs": [
            "uefa.champions",
            "uefa.europa",
            "uefa.europa.conf",
            "eng.fa",
            "eng.league_cup",
            "fifa.friendly",
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

__all__ = ["PlayerSignal", "resolve_player_name", "log_unresolved_name", "SOURCE_COLUMN_MAP"]
