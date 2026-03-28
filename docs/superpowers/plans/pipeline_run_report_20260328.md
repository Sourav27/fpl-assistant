# Weekly Pipeline Bug Report — 2026-03-28

This report documents the observations and bugs identified during the end-to-end testing of the `src.pipeline.run` module for the 2025-26 season (GW31/32).

---

## Phase 1: Pre-deadline
**Status:** SUCCESS
- **Observation:** Correctly identified GW31 as current and GW32 as the next target.
- **Observation:** Successfully captured `ep_this` snapshots from the FPL API and saved them to `data/Fantasy-Premier-League/data/2025-26/gws/xP32.csv`.
- **Observation:** Cached the bootstrap snapshot for 48-hour reuse.

---

## Phase 2: Predict
**Status:** FAILED (Initially), SUCCESS (After Retrain)

### Bug P2-01: Feature Name Mismatch
- **Severity:** CRITICAL
- **Symptom:** `ValueError: The feature names should match those that were passed during fit.`
- **Root Cause:** The legacy model `models/rf_model.sav` was trained using a different feature engineering script (likely from the original notebooks) with names like `1_assists`, `1_bps`, etc. The new `src/pipeline/features.py` uses vectorized rolling names like `assists_roll_4`.
- **Fix:** Retrained the model using the `retrain` phase to align features.

### Bug P2-02: Metadata/ID Drift (The "Kane-at-Brentford" Bug)
- **Severity:** BLOCKER (for accuracy)
- **Symptom:** Prediction output shows nonsensical player/team/position mappings:
    - `Kevin_De Bruyne_215` listed as **DEF** for **Crystal Palace**.
    - `Harry_Kane_338` (not in FPL) listed as **DEF** for **Brentford**.
    - `Pierre-Emerick_Aubameyang_11` listed as **FWD** for **Crystal Palace**.
- **Root Cause:** FPL `element` IDs are seasonal. ID 215 in 2025-26 is a different player than ID 215 in 2022-23. The `build_merged_dataset` in `src/pipeline/prepare.py` joins multiple seasons of `merged_gw.csv` without mapping players to a global ID. 
- **Impact:** The model is effectively being trained on a "Frankenstein" player history where one ID represents multiple different human players across different years. This invalidates rolling averages and momentum features.

---

## Phase 3: Post-GW
**Status:** SUCCESS
- **Observation:** Successfully fetched player histories for GW31 (664 players).
- **Observation:** Saved live data to `gw31_live.csv`, allowing for incremental updates to the historical dataset without waiting for the `vaastav` repository to sync.

---

## Phase 4: Retrain
**Status:** SUCCESS (with workaround)
- **Observation:** The `phase_retrain` function fails if `ACTIVE_MODEL` exists but has a different feature set, as it attempts to perform a performance comparison.
- **Workaround:** Temporarily renamed `ACTIVE_MODEL` in `src/config.py` to allow the new model to be saved.
- **Result:** New model `models/rf_model_gw31.sav` achieved MAE 1.20 and R2 0.234.

---

## Recommendations

1. **Global ID Mapping:** Modify `src/pipeline/prepare.py` to use `data/player_ids.csv` to map seasonal `element` IDs to a consistent `code` (the FPL global player ID).
2. **Feature Alignment:** Deprecate the legacy `rf_model.sav` and ensure all future training uses the `engineer_features` function from `src/pipeline/features.py`.
3. **Environment Sync:** Update `requirements.txt` to match the `scikit-learn` version (1.7.2) used in the project to avoid `InconsistentVersionWarning`.
4. **Team/Position Validation:** Add a validation step in `predict.py` to ensure positions match the current `bootstrap` data rather than relying on historical row metadata.
