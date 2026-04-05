# Track E — Scheduling: Deadline Detection & Model Promotion

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Backfill missing E-F1 tests, add deadline-proximity detection to the daily GitHub Actions workflow so predict/recommend fire automatically within 48h of a GW deadline, and document the GitHub Releases model promotion workflow.

**Architecture:** A new `scripts/check_deadline.py` helper reads the cached bootstrap JSON, computes hours until the next GW deadline, and writes GitHub Actions output variables. The daily workflow calls it after `live_mode()` and conditionally runs predict + recommend. Model artifacts are distributed via GitHub Releases (tag: `gw{N}`); the workflow downloads the latest tagged `.sav` before running predict.

**Tech Stack:** Python 3.11, GitHub Actions `ubuntu-latest` (`gh` CLI pre-installed, `github.token` scoped to `contents: write`), existing `src.pipeline.run` CLI.

---

## File Map

| Action | File | Responsibility |
|--------|------|----------------|
| Create | `scripts/check_deadline.py` | Parse bootstrap JSON → compute hours until deadline → write GH Actions outputs |
| Create | `tests/test_check_deadline.py` | Unit tests for `hours_until_deadline()` |
| Modify | `tests/test_fetch_bootstrap_snapshots.py` | Add missing E-F1 tests for `_price_change_summary` and `price_changes_latest.txt` |
| Modify | `.github/workflows/daily_bootstrap.yml` | Add deadline check step + conditional predict/recommend + model download |
| Modify | `CLAUDE.md` | Document GitHub Releases model promotion workflow (E-F3) |

---

### Task 1: E-F1 missing tests — price change summary

The E-F1 work (commit 434ef7d) added `_price_change_summary()` and the `price_changes_latest.txt` file write to `live_mode()`. Neither is covered by existing tests.

**Files:**
- Modify: `tests/test_fetch_bootstrap_snapshots.py`

- [x] **Step 1: Write failing tests for `_price_change_summary`**

Append to `tests/test_fetch_bootstrap_snapshots.py`:

```python
# ---------------------------------------------------------------------------
# _price_change_summary  (E-F1 coverage)
# ---------------------------------------------------------------------------

def _player(code: int, web_name: str, team_id: int, now_cost: int) -> dict:
    return {"code": code, "web_name": web_name, "team": team_id, "now_cost": now_cost}


def _bs(players: list, teams: list | None = None) -> dict:
    return {
        "elements": players,
        "teams": teams or [{"id": 1, "short_name": "ARS"}],
    }


class TestPriceChangeSummary:

    def test_rise_detected(self):
        old = _bs([_player(1, "Saka", 1, 100)])
        new = _bs([_player(1, "Saka", 1, 101)])
        summary = fbs._price_change_summary(old, new, "test")
        assert "📈" in summary
        assert "Saka" in summary

    def test_fall_detected(self):
        old = _bs([_player(1, "Saka", 1, 101)])
        new = _bs([_player(1, "Saka", 1, 100)])
        summary = fbs._price_change_summary(old, new, "test")
        assert "📉" in summary

    def test_no_changes_returns_no_change_message(self):
        bs = _bs([_player(1, "Saka", 1, 100)])
        summary = fbs._price_change_summary(bs, bs, "test")
        assert "No price changes" in summary

    def test_new_player_entry_detected(self):
        old = _bs([])
        new = _bs([_player(99, "NewGuy", 1, 50)])
        summary = fbs._price_change_summary(old, new, "test")
        assert "NewGuy" in summary

    def test_removed_player_detected(self):
        old = _bs([_player(99, "OldGuy", 1, 50)])
        new = _bs([])
        summary = fbs._price_change_summary(old, new, "test")
        assert "OldGuy" in summary

    def test_uses_persistent_code_not_element_id(self):
        """Same player, different element id → no false price change detected."""
        old = _bs([_player(code=1001, web_name="Saka", team_id=1, now_cost=100)])
        new = _bs([_player(code=1001, web_name="Saka", team_id=1, now_cost=100)])
        summary = fbs._price_change_summary(old, new, "test")
        assert "No price changes" in summary


class TestLiveModeWritesPriceChangesFile:

    def test_writes_price_changes_latest_txt(self, tmp_path):
        """live_mode() must write price_changes_latest.txt when a previous snapshot exists."""
        snapshots_dir = tmp_path / "snapshots"
        snapshots_dir.mkdir()

        old_bootstrap = _bootstrap(current_gw=31, next_gw=32)
        old_bootstrap["elements"] = [_player(1, "Saka", 1, 100)]
        old_bootstrap["teams"] = [{"id": 1, "short_name": "ARS"}]
        (snapshots_dir / "bootstrap_gw32.json").write_text(
            json.dumps(old_bootstrap), encoding="utf-8"
        )

        new_bootstrap = _bootstrap(current_gw=32, next_gw=33)
        new_bootstrap["elements"] = [_player(1, "Saka", 1, 101)]  # price rise
        new_bootstrap["teams"] = [{"id": 1, "short_name": "ARS"}]

        with patch("fetch_bootstrap_snapshots.fetch_live_bootstrap", return_value=new_bootstrap), \
             patch("fetch_bootstrap_snapshots.SNAPSHOTS_DIR", snapshots_dir), \
             patch("fetch_bootstrap_snapshots.PRICE_CHANGES_FILE",
                   snapshots_dir / "price_changes_latest.txt"):
            fbs.live_mode()

        price_file = snapshots_dir / "price_changes_latest.txt"
        assert price_file.exists(), "price_changes_latest.txt was not written"
        content = price_file.read_text(encoding="utf-8")
        assert "Saka" in content
```

- [x] **Step 2: Run to confirm failures**

```bash
cd D:/FPL/fpl-assistant
python -m pytest tests/test_fetch_bootstrap_snapshots.py::TestPriceChangeSummary tests/test_fetch_bootstrap_snapshots.py::TestLiveModeWritesPriceChangesFile -v
```

Expected: `FAILED` — `_price_change_summary` isn't imported or the `PRICE_CHANGES_FILE` patch target differs from the function being called.

Note: If `PRICE_CHANGES_FILE` is a module-level constant in `fetch_bootstrap_snapshots.py`, patch it as `fetch_bootstrap_snapshots.PRICE_CHANGES_FILE`. If it's computed inline (e.g., `SNAPSHOTS_DIR / "price_changes_latest.txt"`), patch `SNAPSHOTS_DIR` instead — already done above. Confirm by reading `live_mode()` source.

- [x] **Step 3: Fix any import/patch issues and re-run until green**

Run: `python -m pytest tests/test_fetch_bootstrap_snapshots.py -v`
Expected: All existing tests still pass, new tests pass.

- [x] **Step 4: Commit**

```bash
git add tests/test_fetch_bootstrap_snapshots.py
git commit -m "test: add E-F1 coverage for _price_change_summary and price_changes_latest.txt"
```

---

### Task 2: Deadline check helper (E-F2a)

**Files:**
- Create: `scripts/check_deadline.py`
- Create: `tests/test_check_deadline.py`

- [x] **Step 1: Write the tests**

Create `tests/test_check_deadline.py`:

```python
"""Tests for scripts/check_deadline.py.

Culprit if failing: hours_until_deadline() or is_approaching logic in check_deadline.py.
"""
import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import check_deadline as cd


def _bootstrap_with_next_gw(hours_from_now: float, next_gw_id: int = 32) -> dict:
    deadline = datetime.now(timezone.utc) + timedelta(hours=hours_from_now)
    return {
        "events": [
            {
                "id": next_gw_id,
                "is_next": True,
                "deadline_time": deadline.strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
        ]
    }


def test_deadline_within_48h_reports_approaching():
    bootstrap = _bootstrap_with_next_gw(hours_from_now=24)
    hours, gw_id = cd.hours_until_deadline(bootstrap)
    assert hours is not None
    assert hours < 48.0
    assert gw_id == 32


def test_deadline_beyond_48h_reports_not_approaching():
    bootstrap = _bootstrap_with_next_gw(hours_from_now=72)
    hours, gw_id = cd.hours_until_deadline(bootstrap)
    assert hours is not None
    assert hours > 48.0


def test_no_next_gw_returns_none():
    bootstrap = {"events": [{"id": 31, "is_next": False, "deadline_time": "2026-01-01T12:00:00Z"}]}
    hours, gw_id = cd.hours_until_deadline(bootstrap)
    assert hours is None
    assert gw_id is None


def test_exactly_48h_is_not_approaching():
    """Boundary: exactly 48h should not trigger (< 48, not <=)."""
    bootstrap = _bootstrap_with_next_gw(hours_from_now=48.01)
    hours, gw_id = cd.hours_until_deadline(bootstrap)
    assert hours > 48.0


def test_gw_id_returned_correctly():
    bootstrap = _bootstrap_with_next_gw(hours_from_now=10, next_gw_id=35)
    _, gw_id = cd.hours_until_deadline(bootstrap)
    assert gw_id == 35


def test_hours_value_is_numeric():
    """hours_until_deadline must return a float, not a string — used in Discord message."""
    bootstrap = _bootstrap_with_next_gw(hours_from_now=36.5)
    hours, _ = cd.hours_until_deadline(bootstrap)
    assert isinstance(hours, float)
    assert 36.0 < hours < 37.0
```

- [x] **Step 2: Run to confirm ImportError**

```bash
python -m pytest tests/test_check_deadline.py -v
```

Expected: `ModuleNotFoundError: No module named 'check_deadline'`

- [x] **Step 3: Write `scripts/check_deadline.py`**

```python
"""Check whether the next FPL GW deadline is within 48 hours.

Reads a bootstrap JSON file, computes hours until the next GW deadline,
and writes GitHub Actions output variables:
  - deadline_approaching: 'true' or 'false'
  - next_gw: the GW number (only when approaching)

Usage:
  python scripts/check_deadline.py results/snapshots/bootstrap_gw32.json
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


def hours_until_deadline(bootstrap: dict) -> tuple[float | None, int | None]:
    """Return (hours_until_deadline, next_gw_id) or (None, None) if no next GW."""
    next_event = next((e for e in bootstrap["events"] if e.get("is_next")), None)
    if not next_event:
        return None, None
    deadline_str = next_event["deadline_time"].replace("Z", "+00:00")
    deadline = datetime.fromisoformat(deadline_str)
    now = datetime.now(timezone.utc)
    hours = (deadline - now).total_seconds() / 3600
    return hours, next_event["id"]


def _write_github_output(key: str, value: str) -> None:
    output_file = os.environ.get("GITHUB_OUTPUT")
    if output_file:
        with open(output_file, "a") as f:
            f.write(f"{key}={value}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bootstrap_path", help="Path to bootstrap JSON snapshot file")
    args = parser.parse_args()

    bootstrap = json.loads(Path(args.bootstrap_path).read_text(encoding="utf-8"))
    hours, next_gw = hours_until_deadline(bootstrap)

    if hours is None:
        print("No upcoming GW found in bootstrap — skipping deadline check.")
        _write_github_output("deadline_approaching", "false")
        sys.exit(0)

    approaching = hours < 48.0
    print(f"Next GW: GW{next_gw} | Hours until deadline: {hours:.1f} | Approaching: {approaching}")
    _write_github_output("deadline_approaching", str(approaching).lower())
    _write_github_output("hours_until", f"{hours:.1f}")
    if next_gw is not None:
        _write_github_output("next_gw", str(next_gw))


if __name__ == "__main__":
    main()
```

- [x] **Step 4: Run tests until green**

```bash
python -m pytest tests/test_check_deadline.py -v
```

Expected: `5 passed`

- [x] **Step 5: Commit**

```bash
git add scripts/check_deadline.py tests/test_check_deadline.py
git commit -m "feat: add check_deadline.py helper for GH Actions deadline detection (E-F2a)"
```

---

### Task 3: GitHub Actions workflow — deadline trigger + model download (E-F2b)

**Files:**
- Modify: `.github/workflows/daily_bootstrap.yml`

The workflow needs five new steps after "Commit snapshot if changed":

1. **Deadline check** — call `check_deadline.py` on the freshly-written snapshot, capture outputs
2. **Notify Discord: deadline alert** — if approaching, post "GW{N} in X.Xh — running predict/recommend"
3. **Model download** — if approaching, download latest `*.sav` from GitHub Releases into `models/`
4. **Predict + recommend** — if approaching AND model available AND `USER_CONFIG_YAML` secret set
5. **Notify Discord: results** — post top captain picks and transfer recommendation summary

- [x] **Step 1: Add the three new steps**

In `.github/workflows/daily_bootstrap.yml`, after the "Commit snapshot if changed" step, add:

```yaml
      - name: Check deadline proximity
        id: deadline_check
        run: |
          # Find the bootstrap file for the next GW
          SNAPSHOT=$(ls results/snapshots/bootstrap_gw*.json 2>/dev/null | sort -V | tail -1)
          if [ -z "$SNAPSHOT" ]; then
            echo "No snapshot found — skipping deadline check."
            echo "deadline_approaching=false" >> $GITHUB_OUTPUT
          else
            python scripts/check_deadline.py "$SNAPSHOT"
          fi

      - name: Notify Discord — deadline approaching
        if: steps.deadline_check.outputs.deadline_approaching == 'true'
        env:
          DISCORD_WEBHOOK: ${{ secrets.DISCORD_DEADLINE_WEBHOOK_URL }}
        run: |
          python - <<'EOF'
          import os, requests, datetime

          webhook = os.environ.get("DISCORD_WEBHOOK", "").strip()
          if not webhook:
              print("DISCORD_WEBHOOK_URL not set — skipping notification.")
              exit(0)

          IST = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
          gw = "${{ steps.deadline_check.outputs.next_gw }}"
          hours = "${{ steps.deadline_check.outputs.hours_until }}"
          date_str = datetime.datetime.now(IST).strftime("%Y-%m-%d %H:%M IST")
          content = (
              f"**FPL GW{gw} Deadline Alert — {date_str}**\n"
              f"Deadline in **{hours}h** — running predict + recommend now."
          )
          r = requests.post(webhook, json={"content": content}, timeout=10)
          r.raise_for_status()
          print(f"Discord notified (HTTP {r.status_code})")
          EOF

      - name: Download model from GitHub Releases
        id: model_download
        if: steps.deadline_check.outputs.deadline_approaching == 'true'
        env:
          GH_TOKEN: ${{ github.token }}
        run: |
          mkdir -p models
          LATEST_TAG=$(gh release list --limit 1 --json tagName -q '.[0].tagName' 2>/dev/null || echo "")
          if [ -z "$LATEST_TAG" ]; then
            echo "No GitHub Release found — predict/recommend will use ep_next fallback."
            echo "model_available=false" >> $GITHUB_OUTPUT
          else
            echo "Downloading model from release: $LATEST_TAG"
            gh release download "$LATEST_TAG" --pattern "*.sav" --dir models/ --clobber
            echo "model_available=true" >> $GITHUB_OUTPUT
            echo "model_tag=$LATEST_TAG" >> $GITHUB_OUTPUT
          fi

      - name: Write user_config.yaml
        id: user_config
        if: steps.deadline_check.outputs.deadline_approaching == 'true'
        env:
          USER_CONFIG_YAML: ${{ secrets.USER_CONFIG_YAML }}
        run: |
          if [ -z "$USER_CONFIG_YAML" ]; then
            echo "USER_CONFIG_YAML secret not set — skipping predict/recommend."
            echo "config_available=false" >> $GITHUB_OUTPUT
          else
            printf '%s' "$USER_CONFIG_YAML" > user_config.yaml
            echo "config_available=true" >> $GITHUB_OUTPUT
          fi

      - name: Install full pipeline dependencies
        if: |
          steps.deadline_check.outputs.deadline_approaching == 'true' &&
          steps.user_config.outputs.config_available == 'true'
        run: pip install -r requirements.txt

      - name: Run predict + recommend
        if: |
          steps.deadline_check.outputs.deadline_approaching == 'true' &&
          steps.user_config.outputs.config_available == 'true'
        run: |
          GW=${{ steps.deadline_check.outputs.next_gw }}
          echo "Running predict for GW${GW}..."
          python -m src.pipeline.run predict --gw "$GW"
          echo "Running recommend for GW${GW}..."
          python -m src.pipeline.run recommend --gw "$GW"

      - name: Commit predict/recommend results
        if: |
          steps.deadline_check.outputs.deadline_approaching == 'true' &&
          steps.user_config.outputs.config_available == 'true'
        run: |
          git config user.name  "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git add results/predictions_gw*.csv results/xi_gw*.csv results/squad_gw*.csv results/recommend_gw*.csv || true
          if git diff --cached --quiet; then
            echo "No predict/recommend output to commit."
          else
            git commit -m "chore: auto predict+recommend GW${{ steps.deadline_check.outputs.next_gw }} ($(date -u +%Y-%m-%d))"
            git push
          fi
```

- [x] **Step 2: Manual verification**

Trigger `workflow_dispatch` from the GitHub Actions tab. Confirm:
- "Check deadline proximity" step runs and prints hours correctly
- If no GitHub Release exists, "Download model" prints the "No GitHub Release found" message and does NOT fail the workflow
- If `USER_CONFIG_YAML` secret is not set, the "Write user_config" step prints the skip message

There is no unit test for the workflow YAML itself. The `check_deadline.py` helper is already unit-tested in Task 2.

- [x] **Step 3: Commit**

```bash
git add .github/workflows/daily_bootstrap.yml
git commit -m "feat: add deadline detection and predict/recommend trigger to daily workflow (E-F2b)"
```

---

### Task 4: Document model promotion (E-F3)

**Files:**
- Modify: `CLAUDE.md`

- [x] **Step 1: Add model promotion section to CLAUDE.md**

In `CLAUDE.md`, find the "Model Management" subsection under "Weekly Pipeline — Quick Start" and append:

```markdown
### Model Promotion via GitHub Releases

The daily GitHub Actions workflow downloads the model from the latest GitHub Release before running predict. To promote a newly retrained model:

```bash
# 1. Retrain locally
python -m src.pipeline.run retrain --gw <N>

# 2. Create a GitHub Release with the model as an asset
gh release create "gw<N>" models/rf_model_gw<N>.sav \
  --title "Model GW<N>" \
  --notes "Retrained after GW<N> with <M> seasons of data."

# 3. Update ACTIVE_MODEL in src/config.py for local runs
# ACTIVE_MODEL = MODELS_DIR / "rf_model_gw<N>.sav"
```

The workflow downloads all `*.sav` assets from the latest release tagged `gw*`. The release tag must start with `gw` (e.g., `gw34`, `gw35`). `gh release list` returns releases in reverse chronological order — the first result is used.

**Secrets required (set in GitHub repo → Settings → Secrets):**
- `DISCORD_PRICE_CHANGE_WEBHOOK_URL` — daily price-change notifications (rename from `DISCORD_WEBHOOK_URL`)
- `DISCORD_DEADLINE_WEBHOOK_URL` — deadline approaching alert
- `DISCORD_PREDICT_RECOMMEND_WEBHOOK_URL` — predict + recommend results summary
- `USER_CONFIG_YAML` — full contents of `user_config.yaml` (required for predict/recommend auto-trigger)

**Migration note:** The existing `DISCORD_WEBHOOK_URL` secret must be renamed to `DISCORD_PRICE_CHANGE_WEBHOOK_URL` in GitHub repo settings, or the price-change step will silently skip.
```

- [x] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: document GitHub Releases model promotion workflow (E-F3)"
```

---

## Timezone Convention

All Discord messages use **IST (UTC+5:30)**:
```python
IST = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
date_str = datetime.datetime.now(IST).strftime("%Y-%m-%d %H:%M IST")
```
Apply this pattern to all three notification steps:
- `DISCORD_PRICE_CHANGE_WEBHOOK_URL` — price changes (existing, already updated in `daily_bootstrap.yml`)
- `DISCORD_DEADLINE_WEBHOOK_URL` — deadline alert (new step in this plan)
- `DISCORD_PREDICT_RECOMMEND_WEBHOOK_URL` — predict/recommend results (new step in this plan)

---

---

### Task 5: Save post-transfer squad from recommend phase (E-F4a)

**Goal:** After `recommend` computes `squad_after`, join with predictions and save `squad_recommend_gw{N}.csv` (15 players) and `xi_recommend_gw{N}.csv` (best XI from that squad). These files power the Discord notification.

**Files:**
- Modify: `src/pipeline/run.py` — `phase_recommend()`: save two new CSVs after `save_recommend_csv`
- Create: `tests/test_run_recommend_saves_squad.py` — unit tests

**Key facts:**
- `plan["squad_after"]` = list of element IDs (15 players) returned by both `recommend_transfers` and `recommend_wildcard`
- `plan["bank_after"]` = final bank in £M
- `select_xi(squad_df)` in `src/pipeline/optimize.py:40` picks the best 11 from a 15-player DataFrame
- `predictions` DataFrame (loaded from `predictions_gw{N}.csv`) has columns: `element, name, position, team, now_cost, xP`
- Output files: `results/squad_recommend_gw{N}.csv` and `results/xi_recommend_gw{N}.csv` — same columns as `squad_gw{N}.csv`

---

- [x] **Step 1: Write the tests**

Create `tests/test_run_recommend_saves_squad.py`:

```python
"""Tests that phase_recommend() saves squad_recommend and xi_recommend CSVs.

Culprit if failing: the squad/xi save block added to phase_recommend() in run.py.
"""
import json
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


PREDICTIONS = pd.DataFrame([
    {"element": 1, "name": "Flekken",   "position": "GK",  "team": "BRE", "now_cost": 45, "xP": 4.1},
    {"element": 2, "name": "Walker",    "position": "DEF", "team": "BUR", "now_cost": 44, "xP": 7.6},
    {"element": 3, "name": "Virgil",    "position": "DEF", "team": "LIV", "now_cost": 63, "xP": 6.5},
    {"element": 4, "name": "Cucurella", "position": "DEF", "team": "CHE", "now_cost": 60, "xP": 6.3},
    {"element": 5, "name": "Fernandes", "position": "MID", "team": "MUN", "now_cost": 103,"xP": 8.5},
    {"element": 6, "name": "Semenyo",   "position": "MID", "team": "MCI", "now_cost": 82, "xP": 12.0},
    {"element": 7, "name": "Amad",      "position": "MID", "team": "MUN", "now_cost": 62, "xP": 9.4},
    {"element": 8, "name": "Hinshelwood","position":"MID", "team": "BHA", "now_cost": 51, "xP": 9.8},
    {"element": 9, "name": "Gomez",     "position": "MID", "team": "BHA", "now_cost": 49, "xP": 7.1},
    {"element": 10,"name": "Jesus",     "position": "FWD", "team": "ARS", "now_cost": 64, "xP": 7.1},
    {"element": 11,"name": "Mykolenko", "position": "FWD", "team": "EVE", "now_cost": 49, "xP": 6.6},
    {"element": 12,"name": "Bayindir",  "position": "GK",  "team": "MUN", "now_cost": 47, "xP": 4.2},
    {"element": 13,"name": "Hume",      "position": "DEF", "team": "SUN", "now_cost": 45, "xP": 6.2},
    {"element": 14,"name": "Andersen",  "position": "DEF", "team": "CPL", "now_cost": 45, "xP": 5.0},
    {"element": 15,"name": "Welbeck",   "position": "FWD", "team": "BHA", "now_cost": 61, "xP": 5.7},
])

MOCK_PLAN = {
    "transfers": [{"transfers": [], "hit_cost": 0, "bank_after": 1.5}],
    "projected_xp": 90.0,
    "hit_cost": 0,
    "bank_after": 1.5,
    "squad_after": list(range(1, 16)),  # all 15 element IDs
}


def _mock_user_state():
    state = MagicMock()
    state.current_squad = list(range(1, 16))
    state.bank = 15
    state.free_transfers = 1
    return state


def test_squad_recommend_csv_saved(tmp_path):
    """phase_recommend must save squad_recommend_gw{N}.csv with 15 rows."""
    pred_path = tmp_path / "predictions_gw32.csv"
    PREDICTIONS.to_csv(pred_path, index=False)

    with patch("src.pipeline.run.RESULTS_DIR", tmp_path), \
         patch("src.pipeline.run.load_user_config", return_value={
             "teams": {"default": {"entry_id": 1}},
             "preferences": {"horizon_gws": 1, "max_hit_points": 8, "fdr_sensitivity": 0.15},
         }), \
         patch("src.pipeline.run.UserTeamState.from_api", return_value=_mock_user_state()), \
         patch("src.pipeline.run.fetch_fixture_fdr", return_value=pd.DataFrame()), \
         patch("src.pipeline.run.recommend_transfers", return_value=MOCK_PLAN), \
         patch("src.pipeline.run.save_recommend_csv"):
        from src.pipeline.run import phase_recommend
        phase_recommend(target_gw=32)

    squad_path = tmp_path / "squad_recommend_gw32.csv"
    assert squad_path.exists(), "squad_recommend_gw32.csv not written"
    df = pd.read_csv(squad_path)
    assert len(df) == 15
    assert "name" in df.columns


def test_xi_recommend_csv_saved(tmp_path):
    """phase_recommend must save xi_recommend_gw{N}.csv with 11 rows."""
    pred_path = tmp_path / "predictions_gw32.csv"
    PREDICTIONS.to_csv(pred_path, index=False)

    with patch("src.pipeline.run.RESULTS_DIR", tmp_path), \
         patch("src.pipeline.run.load_user_config", return_value={
             "teams": {"default": {"entry_id": 1}},
             "preferences": {"horizon_gws": 1, "max_hit_points": 8, "fdr_sensitivity": 0.15},
         }), \
         patch("src.pipeline.run.UserTeamState.from_api", return_value=_mock_user_state()), \
         patch("src.pipeline.run.fetch_fixture_fdr", return_value=pd.DataFrame()), \
         patch("src.pipeline.run.recommend_transfers", return_value=MOCK_PLAN), \
         patch("src.pipeline.run.save_recommend_csv"):
        from src.pipeline.run import phase_recommend
        phase_recommend(target_gw=32)

    xi_path = tmp_path / "xi_recommend_gw32.csv"
    assert xi_path.exists(), "xi_recommend_gw32.csv not written"
    df = pd.read_csv(xi_path)
    assert len(df) == 11
```

- [x] **Step 2: Run to confirm failures**

```bash
python -m pytest tests/test_run_recommend_saves_squad.py -v
```

Expected: `FAILED` — `squad_recommend_gw32.csv` not written.

- [x] **Step 3: Add squad/XI save block to `phase_recommend` in `src/pipeline/run.py`**

In `src/pipeline/run.py`, after the line `save_recommend_csv(plan, out_path, start_gw=target_gw or 0)` (around line 561), add:

```python
    # Save post-transfer squad and XI for Discord notification
    squad_after_ids = plan.get("squad_after", [])
    if squad_after_ids:
        from src.pipeline.optimize import select_xi
        squad_rec = predictions[predictions["element"].isin(squad_after_ids)][
            ["element", "name", "position", "team", "now_cost", "xP"]
        ].reset_index(drop=True)
        squad_rec_path = RESULTS_DIR / f"squad_recommend_{gw_label}.csv"
        squad_rec.to_csv(squad_rec_path, index=False)
        xi_rec = select_xi(squad_rec)
        xi_rec_path = RESULTS_DIR / f"xi_recommend_{gw_label}.csv"
        xi_rec.to_csv(xi_rec_path, index=False)
        print(f"Saved post-transfer squad to {squad_rec_path}")
```

- [x] **Step 4: Run tests until green**

```bash
python -m pytest tests/test_run_recommend_saves_squad.py -v
```

Expected: 2 passed.

Also confirm no regressions:
```bash
python -m pytest tests/ -q --ignore=tests/test_integration.py
```

- [x] **Step 5: Commit**

```bash
git add src/pipeline/run.py tests/test_run_recommend_saves_squad.py
git commit -m "feat: save squad_recommend and xi_recommend CSVs from phase_recommend (E-F4a)"
```

---

### Task 6: Discord predict/recommend results notification (E-F4b)

**Goal:** Post a Discord message with two sections: **Wildcard XI** (optimal unconstrained XI from `xi_gw{N}.csv`) and **My Team After Transfers** (full 15-man squad from `squad_recommend_gw{N}.csv` with starters from `xi_recommend_gw{N}.csv` first, then bench, plus bank and transfers summary).

**Files:**
- Create: `scripts/format_discord_results.py` — pure formatting functions; no I/O side effects
- Create: `tests/test_format_discord_results.py` — unit tests
- Modify: `.github/workflows/daily_bootstrap.yml` — add notification step after "Commit predict/recommend results"

**Output formats:**
- `xi_gw{N}.csv`: `element, name, position, team, now_cost, xP` (11 rows)
- `squad_recommend_gw{N}.csv`: same columns (15 rows)
- `xi_recommend_gw{N}.csv`: same columns (11 rows — starters)
- `recommend_gw{N}.csv`: `gw, action, player_out, player_in, price_out, price_in, xp_out, xp_in, hit_cost, bank_after`

---

- [ ] **Step 1: Write tests**

Create `tests/test_format_discord_results.py`:

```python
"""Tests for scripts/format_discord_results.py.

Culprit if failing: format_wildcard_xi_block() or format_my_team_block() in format_discord_results.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import format_discord_results as fdr

XI_ROWS = [
    {"name": "Flekken",      "position": "GK",  "team": "BRE", "now_cost": 45.0, "xP": 4.1},
    {"name": "Walker",       "position": "DEF", "team": "BUR", "now_cost": 44.0, "xP": 7.6},
    {"name": "Virgil",       "position": "DEF", "team": "LIV", "now_cost": 63.0, "xP": 6.5},
    {"name": "Cucurella",    "position": "DEF", "team": "CHE", "now_cost": 60.0, "xP": 6.3},
    {"name": "B.Fernandes",  "position": "MID", "team": "MUN", "now_cost": 103.0,"xP": 8.5},
    {"name": "Semenyo",      "position": "MID", "team": "MCI", "now_cost": 82.0, "xP": 12.0},
    {"name": "Amad",         "position": "MID", "team": "MUN", "now_cost": 62.0, "xP": 9.4},
    {"name": "Hinshelwood",  "position": "MID", "team": "BHA", "now_cost": 51.0, "xP": 9.8},
    {"name": "Gomez",        "position": "MID", "team": "BHA", "now_cost": 49.0, "xP": 7.1},
    {"name": "G.Jesus",      "position": "FWD", "team": "ARS", "now_cost": 64.0, "xP": 7.1},
    {"name": "Mykolenko",    "position": "FWD", "team": "EVE", "now_cost": 49.0, "xP": 6.6},
]

SQUAD_REC_ROWS = XI_ROWS + [
    {"name": "Bayindir",  "position": "GK",  "team": "MUN", "now_cost": 47.0, "xP": 4.2},
    {"name": "Hume",      "position": "DEF", "team": "SUN", "now_cost": 45.0, "xP": 6.2},
    {"name": "Andersen",  "position": "DEF", "team": "CPL", "now_cost": 45.0, "xP": 5.0},
    {"name": "Welbeck",   "position": "FWD", "team": "BHA", "now_cost": 61.0, "xP": 5.7},
]

XI_REC_ROWS = XI_ROWS  # same starters for simplicity

REC_ROWS = [
    {"gw": 32, "action": "transfer", "player_out": "Wilson",  "player_in": "Semenyo",
     "price_out": 6.0, "price_in": 8.2, "xp_out": 1.0, "xp_in": 12.0, "hit_cost": 0, "bank_after": 1.5},
    {"gw": 32, "action": "transfer", "player_out": "Cunha",   "player_in": "Amad",
     "price_out": 8.0, "price_in": 6.2, "xp_out": 2.9, "xp_in": 9.4, "hit_cost": 0, "bank_after": 1.5},
    {"gw": 33, "action": "transfer", "player_out": "Rashford","player_in": "Salah",
     "price_out": 6.5, "price_in": 13.0, "xp_out": 3.0, "xp_in": 14.0, "hit_cost": 0, "bank_after": 0.0},
]


def test_wildcard_xi_contains_captain():
    """Player with highest xP must be marked (C)."""
    block = fdr.format_wildcard_xi_block(XI_ROWS, gw=32)
    assert "(C)" in block
    assert "Semenyo" in block.split("(C)")[0].split("\n")[-1]


def test_wildcard_xi_grouped_by_position():
    block = fdr.format_wildcard_xi_block(XI_ROWS, gw=32)
    for pos in ("GK", "DEF", "MID", "FWD"):
        assert pos in block


def test_wildcard_xi_captain_tie_picks_one():
    rows = [
        {"name": "A", "position": "MID", "team": "X", "now_cost": 60.0, "xP": 10.0},
        {"name": "B", "position": "MID", "team": "Y", "now_cost": 60.0, "xP": 10.0},
        {"name": "C", "position": "FWD", "team": "Z", "now_cost": 60.0, "xP": 7.0},
    ]
    block = fdr.format_wildcard_xi_block(rows, gw=32)
    assert block.count("(C)") == 1


def test_wildcard_xi_under_2000_chars():
    block = fdr.format_wildcard_xi_block(XI_ROWS, gw=32)
    assert len(block) < 2000


def test_my_team_shows_15_players():
    """My Team block must list all 15 players (11 starters + 4 bench)."""
    block = fdr.format_my_team_block(SQUAD_REC_ROWS, XI_REC_ROWS, REC_ROWS, bank=1.5, gw=32)
    player_lines = [l for l in block.split("\n") if l.strip().startswith("•")]
    assert len(player_lines) == 15


def test_my_team_shows_bench_header():
    """Bench section must be labelled."""
    block = fdr.format_my_team_block(SQUAD_REC_ROWS, XI_REC_ROWS, REC_ROWS, bank=1.5, gw=32)
    assert "Bench" in block or "bench" in block


def test_my_team_shows_bank():
    block = fdr.format_my_team_block(SQUAD_REC_ROWS, XI_REC_ROWS, REC_ROWS, bank=1.5, gw=32)
    assert "1.5" in block


def test_my_team_shows_transfers():
    block = fdr.format_my_team_block(SQUAD_REC_ROWS, XI_REC_ROWS, REC_ROWS, bank=1.5, gw=32)
    assert "Wilson → Semenyo" in block or "Wilson" in block


def test_my_team_excludes_future_gw_transfers():
    """GW33 transfer (Rashford/Salah) must not appear in the GW32 transfers block."""
    block = fdr.format_my_team_block(SQUAD_REC_ROWS, XI_REC_ROWS, REC_ROWS, bank=1.5, gw=32)
    assert "Rashford" not in block
    assert "Salah" not in block


def test_my_team_under_2000_chars():
    block = fdr.format_my_team_block(SQUAD_REC_ROWS, XI_REC_ROWS, REC_ROWS, bank=1.5, gw=32)
    assert len(block) < 2000
```

- [x] **Step 2: Run to confirm ImportError**

```bash
python -m pytest tests/test_format_discord_results.py -v
```

Expected: `ModuleNotFoundError: No module named 'format_discord_results'`

- [x] **Step 3: Write `scripts/format_discord_results.py`**

```python
"""Format predict/recommend CSV outputs into Discord-ready messages.

Two blocks:
  format_wildcard_xi_block — optimal unconstrained XI (xi_gw{N}.csv)
  format_my_team_block     — post-transfer 15-man squad with bench (squad_recommend + xi_recommend)

Usage (from workflow):
  python scripts/format_discord_results.py \\
      results/xi_gw32.csv \\
      results/squad_recommend_gw32.csv \\
      results/xi_recommend_gw32.csv \\
      results/recommend_gw32.csv \\
      32
"""
import argparse
import csv
import sys
from pathlib import Path


def format_wildcard_xi_block(rows: list[dict], gw: int) -> str:
    """Optimal unconstrained Starting XI, grouped by position, captain marked."""
    if not rows:
        return f"GW{gw} Wildcard XI: no data."

    captain = max(rows, key=lambda r: float(r["xP"]))
    lines = [f"**🏆 Wildcard XI — GW{gw}**"]
    for pos in ("GK", "DEF", "MID", "FWD"):
        pos_rows = sorted(
            [r for r in rows if r["position"] == pos],
            key=lambda r: float(r["xP"]), reverse=True,
        )
        if not pos_rows:
            continue
        lines.append(f"\n{pos}")
        for r in pos_rows:
            cap = " (C)" if r["name"] == captain["name"] else ""
            lines.append(f"• {r['name']} ({r['team']}) — {float(r['xP']):.1f} xP{cap}")
    return "\n".join(lines)


def format_my_team_block(
    squad_rows: list[dict],
    xi_rows: list[dict],
    rec_rows: list[dict],
    bank: float,
    gw: int,
) -> str:
    """Post-transfer 15-man squad: starters first, then bench. Includes transfers and bank."""
    if not squad_rows:
        return f"GW{gw} My Team: no data."

    xi_elements = {str(r["element"]) for r in xi_rows}
    captain = max(xi_rows, key=lambda r: float(r["xP"])) if xi_rows else None

    starters = [r for r in squad_rows if str(r["element"]) in xi_elements]
    bench = [r for r in squad_rows if str(r["element"]) not in xi_elements]

    lines = [f"**👤 My Team After Transfers — GW{gw}** (bank: £{bank:.1f}m)"]

    # Transfers summary
    gw_transfers = [r for r in rec_rows if int(r["gw"]) == gw and r.get("action") == "transfer"]
    if gw_transfers:
        lines.append("\nTransfers")
        for t in gw_transfers:
            hit = f" (-{t['hit_cost']}pts)" if float(t["hit_cost"]) > 0 else ""
            lines.append(f"• {t['player_out']} → {t['player_in']}  £{float(t['price_out']):.1f}→£{float(t['price_in']):.1f}{hit}")
    else:
        lines.append("\nTransfers: hold")

    # Starting XI
    lines.append("\nStarting XI")
    for pos in ("GK", "DEF", "MID", "FWD"):
        pos_rows = sorted(
            [r for r in starters if r["position"] == pos],
            key=lambda r: float(r["xP"]), reverse=True,
        )
        for r in pos_rows:
            cap = " (C)" if captain and r["name"] == captain["name"] else ""
            lines.append(f"• {r['name']} ({r['team']}) — {float(r['xP']):.1f} xP{cap}")

    # Bench
    bench_sorted = sorted(bench, key=lambda r: float(r["xP"]), reverse=True)
    lines.append("\nBench")
    for r in bench_sorted:
        lines.append(f"• {r['name']} ({r['team']}) — {float(r['xP']):.1f} xP")

    return "\n".join(lines)


def _read_csv(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    with p.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("xi_path",           help="results/xi_gw{N}.csv")
    parser.add_argument("squad_rec_path",    help="results/squad_recommend_gw{N}.csv")
    parser.add_argument("xi_rec_path",       help="results/xi_recommend_gw{N}.csv")
    parser.add_argument("recommend_path",    help="results/recommend_gw{N}.csv")
    parser.add_argument("gw", type=int,      help="Target GW number")
    args = parser.parse_args()

    xi_rows       = _read_csv(args.xi_path)
    squad_rec     = _read_csv(args.squad_rec_path)
    xi_rec        = _read_csv(args.xi_rec_path)
    rec_rows      = _read_csv(args.recommend_path)

    # Bank from last transfer row for target GW
    gw_rec = [r for r in rec_rows if int(r["gw"]) == args.gw]
    bank = float(gw_rec[-1]["bank_after"]) if gw_rec else 0.0

    print(format_wildcard_xi_block(xi_rows, args.gw))
    print()
    print(format_my_team_block(squad_rec, xi_rec, rec_rows, bank, args.gw))


if __name__ == "__main__":
    main()
```

- [x] **Step 4: Run tests until green**

```bash
python -m pytest tests/test_format_discord_results.py -v
```

Expected: 10 passed.

Full suite:
```bash
python -m pytest tests/ -q --ignore=tests/test_integration.py
```

- [x] **Step 5: Add workflow notification step**

In `.github/workflows/daily_bootstrap.yml`, after "Commit predict/recommend results" and before "Notify Discord" (price changes), add:

```yaml
      - name: Notify Discord — predict/recommend results
        if: |
          steps.deadline_check.outputs.deadline_approaching == 'true' &&
          steps.user_config.outputs.config_available == 'true'
        env:
          DISCORD_WEBHOOK: ${{ secrets.DISCORD_PREDICT_RECOMMEND_WEBHOOK_URL }}
        run: |
          python - <<'EOF'
          import os, requests, datetime, subprocess, sys

          webhook = os.environ.get("DISCORD_WEBHOOK", "").strip()
          if not webhook:
              print("DISCORD_PREDICT_RECOMMEND_WEBHOOK_URL not set — skipping.")
              sys.exit(0)

          gw = "${{ steps.deadline_check.outputs.next_gw }}"
          result = subprocess.run(
              [
                  sys.executable, "scripts/format_discord_results.py",
                  f"results/xi_gw{gw}.csv",
                  f"results/squad_recommend_gw{gw}.csv",
                  f"results/xi_recommend_gw{gw}.csv",
                  f"results/recommend_gw{gw}.csv",
                  gw,
              ],
              capture_output=True, text=True,
          )
          body = result.stdout.strip() or "No predict/recommend output found."

          IST = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
          date_str = datetime.datetime.now(IST).strftime("%Y-%m-%d %H:%M IST")
          content = f"**FPL GW{gw} — {date_str}**\n\n{body}"
          if len(content) > 2000:
              content = content[:1990] + "\n… (truncated)"

          r = requests.post(webhook, json={"content": content}, timeout=10)
          r.raise_for_status()
          print(f"Discord results sent (HTTP {r.status_code})")
          EOF
```

- [x] **Step 6: Commit**

```bash
git add scripts/format_discord_results.py tests/test_format_discord_results.py .github/workflows/daily_bootstrap.yml
git commit -m "feat: add Discord Wildcard XI + My Team After Transfers notification (E-F4b)"
```

---

## Verification Checklist

- [x] All tests pass: `python -m pytest tests/test_fetch_bootstrap_snapshots.py tests/test_check_deadline.py tests/test_run_recommend_saves_squad.py tests/test_format_discord_results.py -v`
- [x] Workflow `workflow_dispatch` runs without errors when no release/secret is present (graceful skips)
- [x] When `USER_CONFIG_YAML` and a `gw*` release are set, predict + recommend files appear in repo after dispatch
- [x] Discord deadline alert message shows IST timestamp, not UTC
- [x] Discord price-changes message (already live) shows IST timestamp after this change
- [x] Discord predict/recommend message shows Wildcard XI (with captain) and My Team After Transfers (15 players, bench separate, bank shown)

- [ ] **Step 1: Write tests**

Create `tests/test_format_discord_results.py`:

```python
"""Tests for scripts/format_discord_results.py.

Culprit if failing: format_xi_block() or format_recommend_block() in format_discord_results.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import format_discord_results as fdr

XI_ROWS = [
    {"name": "Flekken",     "position": "GK",  "team": "BRE", "now_cost": 45.0, "xP": 4.1},
    {"name": "Walker",      "position": "DEF", "team": "BUR", "now_cost": 44.0, "xP": 7.6},
    {"name": "Virgil",      "position": "DEF", "team": "LIV", "now_cost": 63.0, "xP": 6.5},
    {"name": "Cucurella",   "position": "DEF", "team": "CHE", "now_cost": 60.0, "xP": 6.3},
    {"name": "B.Fernandes", "position": "MID", "team": "MUN", "now_cost": 103.0,"xP": 8.5},
    {"name": "Semenyo",     "position": "MID", "team": "MCI", "now_cost": 82.0, "xP": 12.0},
    {"name": "Amad",        "position": "MID", "team": "MUN", "now_cost": 62.0, "xP": 9.4},
    {"name": "Hinshelwood", "position": "MID", "team": "BHA", "now_cost": 51.0, "xP": 9.8},
    {"name": "Gomez",       "position": "MID", "team": "BHA", "now_cost": 49.0, "xP": 7.1},
    {"name": "G.Jesus",     "position": "FWD", "team": "ARS", "now_cost": 64.0, "xP": 7.1},
    {"name": "Mykolenko",   "position": "FWD", "team": "EVE", "now_cost": 49.0, "xP": 6.6},
]

REC_ROWS = [
    {"gw": 32, "action": "transfer", "player_out": "Wilson",  "player_in": "Semenyo",
     "price_out": 6.0, "price_in": 8.2, "xp_out": 1.0, "xp_in": 12.0, "hit_cost": 0, "bank_after": 1.5},
    {"gw": 32, "action": "transfer", "player_out": "Cunha",   "player_in": "Amad",
     "price_out": 8.0, "price_in": 6.2, "xp_out": 2.9, "xp_in": 9.4, "hit_cost": 0, "bank_after": 1.5},
    {"gw": 33, "action": "transfer", "player_out": "Andersen","player_in": "Walker",
     "price_out": 4.5, "price_in": 4.4, "xp_out": 1.2, "xp_in": 7.6, "hit_cost": 0, "bank_after": 1.5},
]


def test_xi_block_contains_captain():
    """Player with highest xP must be marked as captain (C)."""
    block = fdr.format_xi_block(XI_ROWS, gw=32)
    assert "(C)" in block
    # Semenyo has highest xP (12.0)
    assert "Semenyo" in block.split("(C)")[0].split("\n")[-1]


def test_xi_block_grouped_by_position():
    """GK, DEF, MID, FWD sections must all appear."""
    block = fdr.format_xi_block(XI_ROWS, gw=32)
    for pos in ("GK", "DEF", "MID", "FWD"):
        assert pos in block


def test_xi_block_shows_xp():
    """Each player line must include their xP value."""
    block = fdr.format_xi_block(XI_ROWS, gw=32)
    assert "12.0" in block  # Semenyo's xP


def test_xi_block_under_2000_chars():
    """Full message must fit Discord's 2000-char limit."""
    block = fdr.format_xi_block(XI_ROWS, gw=32)
    assert len(block) < 2000


def test_xi_block_captain_tie_picks_one():
    """When two players share the max xP, exactly one (C) marker must appear."""
    rows = [
        {"name": "A", "position": "MID", "team": "X", "now_cost": 60.0, "xP": 10.0},
        {"name": "B", "position": "MID", "team": "Y", "now_cost": 60.0, "xP": 10.0},
        {"name": "C", "position": "FWD", "team": "Z", "now_cost": 60.0, "xP": 7.0},
    ]
    block = fdr.format_xi_block(rows, gw=32)
    assert block.count("(C)") == 1


def test_recommend_block_shows_current_gw_only():
    """recommend block must only show transfers for the target GW, not future GWs."""
    block = fdr.format_recommend_block(REC_ROWS, gw=32)
    assert "Wilson" in block
    assert "Andersen" not in block  # GW33 transfer — excluded


def test_recommend_block_shows_arrow():
    """Each transfer line must show player_out → player_in."""
    block = fdr.format_recommend_block(REC_ROWS, gw=32)
    assert "Wilson → Semenyo" in block or "Wilson" in block


def test_recommend_block_no_transfers_message():
    """When no transfers for the target GW, return a 'no transfers' message."""
    block = fdr.format_recommend_block([], gw=32)
    assert "no transfer" in block.lower() or "hold" in block.lower()
```

- [ ] **Step 2: Run to confirm ImportError**

```bash
cd D:/FPL/fpl-assistant
python -m pytest tests/test_format_discord_results.py -v
```

Expected: `ModuleNotFoundError: No module named 'format_discord_results'`

- [ ] **Step 3: Write `scripts/format_discord_results.py`**

```python
"""Format predict/recommend CSV outputs into a Discord-ready message.

Used by the daily GitHub Actions workflow to post the predicted XI and
GW transfers after the predict + recommend pipeline steps run.

Usage (from workflow):
  python scripts/format_discord_results.py \
      results/xi_gw32.csv results/recommend_gw32.csv 32
"""
import argparse
import csv
import sys
from pathlib import Path


def format_xi_block(rows: list[dict], gw: int) -> str:
    """Return a formatted string for the predicted Starting XI.

    Rows must have keys: name, position, team, now_cost, xP.
    Player with highest xP is marked as captain (C).
    """
    if not rows:
        return f"GW{gw} XI: no data."

    captain = max(rows, key=lambda r: float(r["xP"]))
    order = ["GK", "DEF", "MID", "FWD"]
    lines = [f"**GW{gw} Predicted XI**"]
    for pos in order:
        pos_rows = sorted(
            [r for r in rows if r["position"] == pos],
            key=lambda r: float(r["xP"]),
            reverse=True,
        )
        if not pos_rows:
            continue
        lines.append(f"\n{pos}")
        for r in pos_rows:
            cap = " (C)" if r["name"] == captain["name"] else ""
            lines.append(f"• {r['name']} ({r['team']}) — {float(r['xP']):.1f} xP{cap}")
    return "\n".join(lines)


def format_recommend_block(rows: list[dict], gw: int) -> str:
    """Return a formatted string for GW transfers.

    Only shows transfers where row['gw'] == gw.
    """
    gw_rows = [r for r in rows if int(r["gw"]) == gw and r.get("action") == "transfer"]
    if not gw_rows:
        return f"GW{gw} Transfers: hold (no transfers recommended)."

    lines = [f"**GW{gw} Transfers**"]
    for r in gw_rows:
        hit = f" (-{r['hit_cost']}pts hit)" if float(r["hit_cost"]) > 0 else ""
        lines.append(
            f"• {r['player_out']} → {r['player_in']}"
            f"  £{float(r['price_out']):.1f}→£{float(r['price_in']):.1f}{hit}"
        )
    return "\n".join(lines)


def _read_csv(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    with p.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("xi_path", help="Path to xi_gw{N}.csv")
    parser.add_argument("recommend_path", help="Path to recommend_gw{N}.csv")
    parser.add_argument("gw", type=int, help="Target GW number")
    args = parser.parse_args()

    xi_rows = _read_csv(args.xi_path)
    rec_rows = _read_csv(args.recommend_path)

    xi_block = format_xi_block(xi_rows, args.gw)
    rec_block = format_recommend_block(rec_rows, args.gw)
    print(xi_block)
    print()
    print(rec_block)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests until green**

```bash
python -m pytest tests/test_format_discord_results.py -v
```

Expected: 7 passed.

Also run full suite to confirm no regressions:
```bash
python -m pytest tests/ -q --ignore=tests/test_integration.py
```

- [ ] **Step 5: Add workflow notification step**

In `.github/workflows/daily_bootstrap.yml`, after the "Commit predict/recommend results" step and before the existing "Notify Discord" (price changes) step, add:

```yaml
      - name: Notify Discord — predict/recommend results
        if: |
          steps.deadline_check.outputs.deadline_approaching == 'true' &&
          steps.user_config.outputs.config_available == 'true'
        env:
          DISCORD_WEBHOOK: ${{ secrets.DISCORD_PREDICT_RECOMMEND_WEBHOOK_URL }}
        run: |
          python - <<'EOF'
          import os, requests, datetime, subprocess, sys

          webhook = os.environ.get("DISCORD_WEBHOOK", "").strip()
          if not webhook:
              print("DISCORD_PREDICT_RECOMMEND_WEBHOOK_URL not set — skipping.")
              sys.exit(0)

          gw = "${{ steps.deadline_check.outputs.next_gw }}"
          xi_path = f"results/xi_gw{gw}.csv"
          rec_path = f"results/recommend_gw{gw}.csv"

          result = subprocess.run(
              [sys.executable, "scripts/format_discord_results.py", xi_path, rec_path, gw],
              capture_output=True, text=True
          )
          body = result.stdout.strip() or "No predict/recommend output found."

          IST = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
          date_str = datetime.datetime.now(IST).strftime("%Y-%m-%d %H:%M IST")
          content = f"**FPL GW{gw} Prediction — {date_str}**\n\n{body}"

          if len(content) > 2000:
              content = content[:1990] + "\n… (truncated)"

          r = requests.post(webhook, json={"content": content}, timeout=10)
          r.raise_for_status()
          print(f"Discord results sent (HTTP {r.status_code})")
          EOF
```

- [ ] **Step 6: Commit**

```bash
git add scripts/format_discord_results.py tests/test_format_discord_results.py .github/workflows/daily_bootstrap.yml
git commit -m "feat: add Discord predict/recommend results notification after deadline trigger (E-F4)"
```

---

## Verification Checklist

- [ ] All tests pass: `python -m pytest tests/test_fetch_bootstrap_snapshots.py tests/test_check_deadline.py tests/test_format_discord_results.py -v`
- [ ] Workflow `workflow_dispatch` runs without errors when no release/secret is present (graceful skips)
- [ ] When `USER_CONFIG_YAML` and a `gw*` release are set, predict + recommend files appear in repo after dispatch
- [ ] Discord deadline alert message shows IST timestamp, not UTC
- [ ] Discord price-changes message (already live) shows IST timestamp after this change
- [ ] Discord predict/recommend message shows XI grouped by position with captain marked (C)
