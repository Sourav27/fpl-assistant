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

- [ ] **Step 1: Write failing tests for `_price_change_summary`**

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

- [ ] **Step 2: Run to confirm failures**

```bash
cd D:/FPL/fpl-assistant
python -m pytest tests/test_fetch_bootstrap_snapshots.py::TestPriceChangeSummary tests/test_fetch_bootstrap_snapshots.py::TestLiveModeWritesPriceChangesFile -v
```

Expected: `FAILED` — `_price_change_summary` isn't imported or the `PRICE_CHANGES_FILE` patch target differs from the function being called.

Note: If `PRICE_CHANGES_FILE` is a module-level constant in `fetch_bootstrap_snapshots.py`, patch it as `fetch_bootstrap_snapshots.PRICE_CHANGES_FILE`. If it's computed inline (e.g., `SNAPSHOTS_DIR / "price_changes_latest.txt"`), patch `SNAPSHOTS_DIR` instead — already done above. Confirm by reading `live_mode()` source.

- [ ] **Step 3: Fix any import/patch issues and re-run until green**

Run: `python -m pytest tests/test_fetch_bootstrap_snapshots.py -v`
Expected: All existing tests still pass, new tests pass.

- [ ] **Step 4: Commit**

```bash
git add tests/test_fetch_bootstrap_snapshots.py
git commit -m "test: add E-F1 coverage for _price_change_summary and price_changes_latest.txt"
```

---

### Task 2: Deadline check helper (E-F2a)

**Files:**
- Create: `scripts/check_deadline.py`
- Create: `tests/test_check_deadline.py`

- [ ] **Step 1: Write the tests**

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

- [ ] **Step 2: Run to confirm ImportError**

```bash
python -m pytest tests/test_check_deadline.py -v
```

Expected: `ModuleNotFoundError: No module named 'check_deadline'`

- [ ] **Step 3: Write `scripts/check_deadline.py`**

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

- [ ] **Step 4: Run tests until green**

```bash
python -m pytest tests/test_check_deadline.py -v
```

Expected: `5 passed`

- [ ] **Step 5: Commit**

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

- [ ] **Step 1: Add the three new steps**

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

- [ ] **Step 2: Manual verification**

Trigger `workflow_dispatch` from the GitHub Actions tab. Confirm:
- "Check deadline proximity" step runs and prints hours correctly
- If no GitHub Release exists, "Download model" prints the "No GitHub Release found" message and does NOT fail the workflow
- If `USER_CONFIG_YAML` secret is not set, the "Write user_config" step prints the skip message

There is no unit test for the workflow YAML itself. The `check_deadline.py` helper is already unit-tested in Task 2.

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/daily_bootstrap.yml
git commit -m "feat: add deadline detection and predict/recommend trigger to daily workflow (E-F2b)"
```

---

### Task 4: Document model promotion (E-F3)

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Add model promotion section to CLAUDE.md**

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

- [ ] **Step 2: Commit**

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

## Verification Checklist

- [ ] All tests pass: `python -m pytest tests/test_fetch_bootstrap_snapshots.py tests/test_check_deadline.py -v`
- [ ] Workflow `workflow_dispatch` runs without errors when no release/secret is present (graceful skips)
- [ ] When `USER_CONFIG_YAML` and a `gw*` release are set, predict + recommend files appear in repo after dispatch
- [ ] Discord deadline alert message shows IST timestamp, not UTC
- [ ] Discord price-changes message (already live) shows IST timestamp after this change
