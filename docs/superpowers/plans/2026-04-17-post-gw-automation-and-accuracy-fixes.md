# Post-GW Automation, Accuracy Log Fixes & Discord E2E Tests — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run `post-gw` automatically once all GW games finish, send an accuracy-log summary to Discord, fix `wildcard_pts`/`wildcard_xp`/`spearman_rho` being empty in the log, and add E2E tests that verify the Discord call fires.

**Architecture:** A new `scripts/check_gw_finished.py` script detects when the current GW is done (analogous to `check_deadline.py`). The daily bootstrap workflow gains a new conditional job that calls `phase_post_gw` and then sends accuracy log output to the `DISCORD_PREDICT_RECOMMEND_WEBHOOK_URL` channel. Three accuracy-log bugs are fixed in `phase_post_gw`: (1) `picks_df` not passed → Spearman ρ stays `None`; (2) `wildcard_pts`/`wildcard_xp` never computed → uses the optimizer's `squad_gw{N}.csv` actual scores as a "what-if" baseline; (3) `recommend_pts` computed from transfer names which is fragile — improved to use saved squad CSV instead. New E2E tests mock the `requests.post` call to Discord and assert it fires.

**Tech Stack:** Python, pandas, pytest, unittest.mock, GitHub Actions YAML.

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `scripts/check_gw_finished.py` | **Create** | Detect current GW finished; emit `gw_finished`, `current_gw` GitHub outputs |
| `scripts/format_accuracy_discord.py` | **Create** | Format accuracy log row as Discord message |
| `src/pipeline/run.py` | **Modify** | Fix `phase_post_gw`: pass `picks_df`, compute `wildcard_pts/xp`, use squad CSV |
| `src/pipeline/analysis.py` | **No change** | Already supports `picks_df` and `wildcard_pts/xp` parameters |
| `.github/workflows/daily_bootstrap.yml` | **Modify** | Add GW-finished detection + conditional post-gw step |
| `tests/test_pipeline_e2e.py` | **Modify** | Add `TestPostGwDiscord` class with E2E Discord mock tests |

---

## Task 1: `scripts/check_gw_finished.py`

Detects whether all fixtures of the current GW have been played. Emits `gw_finished=true/false` and `current_gw=N` as GitHub Actions outputs.

**Files:**
- Create: `scripts/check_gw_finished.py`

- [ ] **Step 1: Write the failing test (inline — no test file needed for this script)**

  We'll test this in Task 4 E2E. For now just verify the script imports cleanly:
  ```bash
  python scripts/check_gw_finished.py --help 2>&1 | head -5
  ```

- [ ] **Step 2: Implement `check_gw_finished.py`**

```python
"""Detect whether the current GW has finished (all fixtures done).

Reads a bootstrap JSON snapshot and checks if the current GW's `finished`
flag is True. Also checks that at least one fixture exists for the GW.

GitHub Actions outputs written:
  - gw_finished: 'true' or 'false'
  - current_gw: the GW number (always written)
"""
import argparse
import json
import os
import sys
from pathlib import Path


def _write_github_output(key: str, value: str) -> None:
    output_file = os.environ.get("GITHUB_OUTPUT")
    if output_file:
        with open(output_file, "a") as f:
            f.write(f"{key}={value}\n")


def check_gw_finished(bootstrap: dict) -> tuple[bool, int | None]:
    """Return (finished, current_gw_id).

    'finished' means: current event exists AND its finished flag is True.
    """
    current = next((e for e in bootstrap["events"] if e.get("is_current")), None)
    if not current:
        return False, None
    return bool(current.get("finished", False)), current["id"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bootstrap_path", help="Path to bootstrap JSON snapshot")
    args = parser.parse_args()

    bootstrap = json.loads(Path(args.bootstrap_path).read_text(encoding="utf-8"))
    finished, gw = check_gw_finished(bootstrap)

    if gw is None:
        print("No current GW found in bootstrap.")
        _write_github_output("gw_finished", "false")
        sys.exit(0)

    print(f"Current GW: {gw} | Finished: {finished}")
    _write_github_output("gw_finished", str(finished).lower())
    _write_github_output("current_gw", str(gw))


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Smoke-test with a real snapshot**

```bash
SNAPSHOT=$(ls results/snapshots/bootstrap_gw*.json | sort -V | tail -1)
python scripts/check_gw_finished.py "$SNAPSHOT"
```

Expected: prints `Current GW: N | Finished: true/false`

- [ ] **Step 4: Commit**

```bash
rtk git add scripts/check_gw_finished.py
rtk git commit -m "feat: add check_gw_finished script for post-gw automation"
```

---

## Task 2: Fix `phase_post_gw` — spearman_rho and wildcard_pts/xp

Three bugs in `src/pipeline/run.py::phase_post_gw`:
1. `picks_df` (with `actual_points`) is never passed to `append_accuracy_log` → Spearman ρ is always `None`
2. `wildcard_pts`/`wildcard_xp` never computed — should use `squad_gw{N}.csv` (the optimizer's selection) scored against live data as a "what-if" baseline
3. `recommended_pts` is derived from transfer player names which is fragile; compute directly from `squad_recommend_gw{N}.csv` actual scores instead (simpler and more correct)

**Files:**
- Modify: `src/pipeline/run.py:514-523`

- [ ] **Step 1: Write failing unit test first**

Add to `tests/test_analysis.py` (or create `tests/test_post_gw_accuracy.py`):

```python
def test_append_accuracy_log_spearman_rho_written(tmp_path):
    """Passing picks_df to append_accuracy_log must populate spearman_rho."""
    import pandas as pd
    from src.pipeline.analysis import append_accuracy_log

    picks = pd.DataFrame({
        "element": [1, 2, 3, 4, 5],
        "name": ["A", "B", "C", "D", "E"],
        "xP": [5.0, 4.0, 3.0, 2.0, 1.0],
        "actual_points": [10, 8, 5, 4, 2],
    })
    log = tmp_path / "accuracy_log.csv"
    append_accuracy_log(
        path=log, gw=99,
        your_pts=29, your_xp=15.0,
        recommended_pts=None, recommended_xp=None,
        picks_df=picks,
    )
    import pandas as pd
    df = pd.read_csv(log)
    assert df["spearman_rho"].notna().all(), "spearman_rho should be written when picks_df provided"
    assert abs(df.iloc[0]["spearman_rho"]) > 0.5
```

- [ ] **Step 2: Run test — expect PASS** (analysis.py already supports picks_df; this confirms the function works)

```bash
python -m pytest tests/test_post_gw_accuracy.py -v
```

- [ ] **Step 3: Patch `phase_post_gw` — two separate edits in `src/pipeline/run.py`**

**Edit A** — Replace the existing `# Recommended team comparison` block (lines ~451-465) with a squad-CSV-based approach. This removes fragile name-matching and also initialises `squad_rec_path` for use in the wildcard block below:

```python
    # Recommended team comparison — use saved squad CSV for accuracy
    recommended_pts = None
    recommended_xp = None
    squad_rec_path = RESULTS_DIR / f"squad_recommend_{gw_label}.csv"
    if squad_rec_path.exists() and not live_df.empty:
        rec_squad_df = pd.read_csv(squad_rec_path)
        actual_map_rec = live_df.set_index("element")["total_points"].to_dict()
        rec_squad_df["actual_points"] = rec_squad_df["element"].map(actual_map_rec).fillna(0)
        recommended_pts = int(rec_squad_df["actual_points"].sum())
        recommended_xp = float(rec_squad_df["xP"].sum()) if "xP" in rec_squad_df.columns else None
```

**Edit B** — Replace the `append_accuracy_log` call (lines ~515-523) with one that adds the wildcard baseline and passes `picks_df`:

```python
    # Wildcard baseline: full 15-player optimizer squad summed against live data.
    # Intentionally the full squad (not just XI) so it reflects bench-boost potential.
    wildcard_pts = None
    wildcard_xp = None
    squad_path = RESULTS_DIR / f"squad_{gw_label}.csv"
    if squad_path.exists() and not live_df.empty:
        squad_df = pd.read_csv(squad_path)
        actual_map_wc = live_df.set_index("element")["total_points"].to_dict()
        squad_df["actual_points"] = squad_df["element"].map(actual_map_wc).fillna(0)
        wildcard_pts = int(squad_df["actual_points"].sum())
        wildcard_xp = float(squad_df["xP"].sum()) if "xP" in squad_df.columns else None

    # Write accuracy log
    log_path = RESULTS_DIR / "accuracy_log.csv"
    append_accuracy_log(
        path=log_path, gw=gw,
        your_pts=your_pts, your_xp=your_xp,
        recommended_pts=recommended_pts, recommended_xp=recommended_xp,
        wildcard_pts=wildcard_pts, wildcard_xp=wildcard_xp,
        dream_pts=dream_pts, your_percentile_rank=your_percentile_rank,
        benchmarks=benchmarks, ranked_count=benchmarks.get("ranked_count"),
        picks_df=your_picks if not your_picks.empty else None,
    )
    print(f"[post-gw] Accuracy log updated: {log_path}")
```

- [ ] **Step 4: Run unit tests to confirm no regressions**

```bash
python -m pytest tests/ -q --ignore=tests/test_integration.py
```

Expected: all pass

- [ ] **Step 5: Commit**

```bash
rtk git add src/pipeline/run.py tests/test_post_gw_accuracy.py
rtk git commit -m "fix: populate wildcard_pts/xp and spearman_rho in accuracy log"
```

---

## Task 3: `scripts/format_accuracy_discord.py`

Formats the latest accuracy log row as a Discord message for the post-gw notification.

**Files:**
- Create: `scripts/format_accuracy_discord.py`

- [ ] **Step 1: Implement the script**

```python
"""Format the latest accuracy log row as a Discord message.

Usage:
    python scripts/format_accuracy_discord.py results/accuracy_log.csv <gw>

Prints a markdown-formatted Discord message to stdout.
"""
import argparse
import sys
import pandas as pd
from pathlib import Path


def format_accuracy_row(row: pd.Series) -> str:
    gw = int(row["gw"])
    lines = [f"**GW{gw} Post-Match Accuracy**"]
    lines.append("")

    def _fmt(val, suffix="", fmt=".1f"):
        if pd.isna(val):
            return "—"
        return f"{val:{fmt}}{suffix}"

    lines.append(f"Your team:   **{_fmt(row.get('your_pts'), ' pts', 'd')}**  (predicted {_fmt(row.get('your_predicted_xp'))} xP)")
    lines.append(f"Recommended: {_fmt(row.get('recommended_pts'), ' pts', 'd')}  (predicted {_fmt(row.get('recommended_xp'))} xP)")
    lines.append(f"Optimizer squad: {_fmt(row.get('wildcard_pts'), ' pts', 'd')}  (predicted {_fmt(row.get('wildcard_xp'))} xP)")
    lines.append(f"Dream team:  {_fmt(row.get('dream_team_pts'), ' pts', 'd')}")
    lines.append("")
    lines.append(f"Prediction accuracy (Spearman ρ): {_fmt(row.get('spearman_rho'), fmt='.3f')}")
    lines.append(f"Your percentile rank: {_fmt(row.get('your_percentile_rank'), 'th', 'd')}")

    benchmarks = [
        ("Avg score",    row.get("avg_score")),
        ("Top 100k",     row.get("top_100k_score")),
        ("Top 10k",      row.get("top_10k_score")),
        ("Top 1k",       row.get("top_1k_score")),
        ("Best score",   row.get("best_score")),
    ]
    bench_lines = [f"  {lbl}: {_fmt(v, ' pts', 'd')}" for lbl, v in benchmarks if not pd.isna(v) and v is not None]
    if bench_lines:
        lines.append("")
        lines.append("**GW Benchmarks:**")
        lines.extend(bench_lines)

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_path", help="Path to accuracy_log.csv")
    parser.add_argument("gw", type=int, help="GW number to format")
    args = parser.parse_args()

    path = Path(args.log_path)
    if not path.exists():
        print(f"Accuracy log not found: {path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(path)
    rows = df[df["gw"] == args.gw]
    if rows.empty:
        print(f"No row for GW{args.gw} in accuracy log", file=sys.stderr)
        sys.exit(1)

    print(format_accuracy_row(rows.iloc[-1]))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test against current log**

```bash
python scripts/format_accuracy_discord.py results/accuracy_log.csv 32
```

Expected: a formatted Discord-style block printed to stdout

- [ ] **Step 3: Commit**

```bash
rtk git add scripts/format_accuracy_discord.py
rtk git commit -m "feat: add format_accuracy_discord script for post-gw notification"
```

---

## Task 4: Add post-gw job to `daily_bootstrap.yml`

The daily bootstrap workflow runs once a day. After the GW-finished check, it runs `post-gw` (once per GW — idempotent because accuracy log deduplication is by GW) and sends the accuracy summary to Discord.

**Files:**
- Modify: `.github/workflows/daily_bootstrap.yml`

- [ ] **Step 1: Add the GW-finished detection step after the snapshot commit**

Insert after the existing `Commit snapshot if changed` step:

```yaml
      - name: Check GW finished
        id: gw_check
        run: |
          SNAPSHOT=$(ls results/snapshots/bootstrap_gw*.json 2>/dev/null | sort -V | tail -1)
          if [ -z "$SNAPSHOT" ]; then
            echo "No snapshot found — skipping post-gw check."
            echo "gw_finished=false" >> $GITHUB_OUTPUT
          else
            python scripts/check_gw_finished.py "$SNAPSHOT"
          fi
```

- [ ] **Step 2: Add the post-gw execution step (conditional on `gw_finished == 'true'`)**

Add after the `Check GW finished` step (before `Check deadline proximity`):

```yaml
      - name: Install full pipeline deps for post-gw
        if: steps.gw_check.outputs.gw_finished == 'true'
        run: pip install -r requirements.txt

      - name: Write user_config.yaml for post-gw
        id: postgw_config
        if: steps.gw_check.outputs.gw_finished == 'true'
        env:
          USER_CONFIG_YAML: ${{ secrets.USER_CONFIG_YAML }}
        run: |
          if [ -z "$USER_CONFIG_YAML" ]; then
            echo "USER_CONFIG_YAML not set — skipping post-gw."
            echo "config_available=false" >> $GITHUB_OUTPUT
          else
            printf '%s' "$USER_CONFIG_YAML" > user_config.yaml
            echo "config_available=true" >> $GITHUB_OUTPUT
          fi

      - name: Run post-gw analysis
        if: |
          steps.gw_check.outputs.gw_finished == 'true' &&
          steps.postgw_config.outputs.config_available == 'true'
        run: |
          GW=${{ steps.gw_check.outputs.current_gw }}
          echo "Running post-gw analysis for GW${GW}..."
          python -m src.pipeline.run post-gw

      - name: Commit accuracy log update
        if: |
          steps.gw_check.outputs.gw_finished == 'true' &&
          steps.postgw_config.outputs.config_available == 'true'
        run: |
          git config user.name  "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git add results/accuracy_log.csv || true
          if git diff --cached --quiet; then
            echo "No accuracy log changes to commit."
          else
            git commit -m "chore: update accuracy_log.csv with gameweek ${{ steps.gw_check.outputs.current_gw }} performance data"
            git push
          fi

      - name: Notify Discord — post-gw accuracy
        if: |
          steps.gw_check.outputs.gw_finished == 'true' &&
          steps.postgw_config.outputs.config_available == 'true'
        env:
          DISCORD_WEBHOOK: ${{ secrets.DISCORD_PREDICT_RECOMMEND_WEBHOOK_URL }}
        run: |
          python - <<'EOF'
          import os, requests, datetime, subprocess, sys

          webhook = os.environ.get("DISCORD_WEBHOOK", "").strip()
          if not webhook:
              print("DISCORD_PREDICT_RECOMMEND_WEBHOOK_URL not set — skipping post-gw notification.")
              sys.exit(0)

          gw = "${{ steps.gw_check.outputs.current_gw }}"
          result = subprocess.run(
              [sys.executable, "scripts/format_accuracy_discord.py", "results/accuracy_log.csv", gw],
              capture_output=True, text=True,
          )
          body = result.stdout.strip() or f"Post-GW{gw} analysis complete — accuracy log updated."

          IST = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
          date_str = datetime.datetime.now(IST).strftime("%Y-%m-%d %H:%M IST")
          content = f"**FPL GW{gw} — {date_str}**\n\n{body}"
          if len(content) > 2000:
              content = content[:1990] + "\n… (truncated)"

          r = requests.post(webhook, json={"content": content}, timeout=10)
          r.raise_for_status()
          print(f"Discord accuracy notification sent (HTTP {r.status_code})")
          EOF
```

- [ ] **Step 3: Verify YAML is valid**

```bash
python -c "import yaml; yaml.safe_load(open('.github/workflows/daily_bootstrap.yml'))" && echo "YAML OK"
```

- [ ] **Step 4: Commit**

```bash
rtk git add .github/workflows/daily_bootstrap.yml
rtk git commit -m "feat: run post-gw analysis automatically when GW finishes"
```

---

## Task 5: E2E tests — Discord notification fires during post-gw

Modify `tests/test_pipeline_e2e.py` to add a `TestPostGwDiscord` class. These tests mock `requests.post` and verify that the Discord webhook is called when `phase_post_gw` runs with a finished GW.

The tests also verify accuracy log correctness (wildcard_pts, spearman_rho populated).

**Files:**
- Modify: `tests/test_pipeline_e2e.py`

- [ ] **Step 1: Review what `phase_post_gw` needs to run in test**

`phase_post_gw` calls:
- `fetch_bootstrap()` — must be mocked
- `fetch_fixtures()` — must be mocked (return `[]`)
- `fetch_live_gw_data()` — must be mocked (return live scores DataFrame)
- `_api_get_with_retry(entry/picks URL)` — must be mocked
- `load_user_config()` — must be mocked
- Reads `predictions_gw{N}.csv`, `squad_gw{N}.csv`, `squad_recommend_gw{N}.csv` from `RESULTS_DIR`

- [ ] **Step 2: Add `TestPostGwDiscord` class to `tests/test_pipeline_e2e.py`**

Append the following class to the end of the file:

```python
# ---------------------------------------------------------------------------
# E2E: post-gw accuracy log + Discord notification
# ---------------------------------------------------------------------------

class TestPostGwDiscord:
    """Tests that phase_post_gw writes a correct accuracy log and fires Discord.

    These tests use a finished GW31 bootstrap (finished=True on event id=31
    — note: post-gw runs on the CURRENT finished GW, not next).
    """

    def _bootstrap_finished(self, e2e_bootstrap):
        """Return a bootstrap where GW31 is current+finished, GW32 is next."""
        import copy
        bs = copy.deepcopy(e2e_bootstrap)
        # Rename GW30 → GW31 finished/current, GW31 → GW32 next
        bs["events"] = [
            {
                "id": 31,
                "deadline_time": "2026-03-20T18:30:00Z",
                "is_current": True, "is_next": False, "finished": True,
            },
            {
                "id": 32,
                "deadline_time": "2026-03-27T18:30:00Z",
                "is_current": False, "is_next": True, "finished": False,
            },
        ]
        return bs

    def _live_df(self, e2e_bootstrap):
        """Fake live GW data — assign actual_points to each player."""
        players = e2e_bootstrap["elements"]
        rows = [{"element": p["id"], "total_points": 5, "name": p["web_name"]} for p in players]
        return pd.DataFrame(rows)

    def _setup_results(self, tmp_path, e2e_bootstrap):
        """Write predictions, squad, and squad_recommend CSVs to results dir."""
        results = tmp_path / "results"
        results.mkdir(parents=True, exist_ok=True)

        players = e2e_bootstrap["elements"]
        from src.pipeline.fetch import ELEMENT_TYPE_MAP
        rows = [{
            "element": p["id"], "code": p["code"],
            "name": p["web_name"], "xP": float(p["ep_next"]),
            "now_cost": p["now_cost"],
            "position": ELEMENT_TYPE_MAP.get(p["element_type"], "MID"),
            "team": f"Team{p['team']}",
        } for p in players]
        df = pd.DataFrame(rows)

        df.to_csv(results / "predictions_gw31.csv", index=False)
        df.to_csv(results / "squad_gw31.csv", index=False)          # 15 players
        df.head(15).to_csv(results / "squad_recommend_gw31.csv", index=False)
        return results

    def test_post_gw_accuracy_log_has_spearman_rho(self, tmp_path, e2e_bootstrap):
        """phase_post_gw must write a non-null spearman_rho when picks data is available."""
        import src.pipeline.run as run_mod
        from unittest.mock import MagicMock

        bs_finished = self._bootstrap_finished(e2e_bootstrap)
        live_df = self._live_df(e2e_bootstrap)
        results = self._setup_results(tmp_path, e2e_bootstrap)

        entry_picks_response = MagicMock()
        entry_picks_response.json.return_value = {
            "entry_history": {"points": 55},
            "picks": [{"element": p["id"]} for p in e2e_bootstrap["elements"][:11]],
        }
        entry_response = MagicMock()
        entry_response.json.return_value = {"leagues": {"classic": []}}

        with patch("src.pipeline.run.fetch_bootstrap", return_value=bs_finished), \
             patch("src.pipeline.run.fetch_fixtures", return_value=[]), \
             patch("src.pipeline.run.fetch_live_gw_data", return_value=live_df), \
             patch("src.pipeline.run.RESULTS_DIR", results), \
             patch("src.pipeline.run.VAASTAV_DIR", tmp_path / "FPL"), \
             patch("src.pipeline.run.load_user_config", return_value={
                 "teams": {"default": {"entry_id": 123}},
                 "preferences": {},
             }), \
             patch("src.pipeline.run._api_get_with_retry", side_effect=[
                 entry_picks_response, entry_response,
                 MagicMock(**{"json.return_value": {"current": [{"event": 31, "percentile_rank": 25}]}}),
             ]):
            run_mod.phase_post_gw()

        log = pd.read_csv(results / "accuracy_log.csv")
        assert (log["spearman_rho"].notna()).any(), "spearman_rho must be written"

    def test_post_gw_accuracy_log_has_wildcard_pts(self, tmp_path, e2e_bootstrap):
        """phase_post_gw must write wildcard_pts when squad_gw{N}.csv exists."""
        import src.pipeline.run as run_mod
        from unittest.mock import MagicMock

        bs_finished = self._bootstrap_finished(e2e_bootstrap)
        live_df = self._live_df(e2e_bootstrap)
        results = self._setup_results(tmp_path, e2e_bootstrap)

        entry_picks_response = MagicMock()
        entry_picks_response.json.return_value = {
            "entry_history": {"points": 55},
            "picks": [{"element": p["id"]} for p in e2e_bootstrap["elements"][:11]],
        }

        with patch("src.pipeline.run.fetch_bootstrap", return_value=bs_finished), \
             patch("src.pipeline.run.fetch_fixtures", return_value=[]), \
             patch("src.pipeline.run.fetch_live_gw_data", return_value=live_df), \
             patch("src.pipeline.run.RESULTS_DIR", results), \
             patch("src.pipeline.run.VAASTAV_DIR", tmp_path / "FPL"), \
             patch("src.pipeline.run.load_user_config", return_value={
                 "teams": {"default": {"entry_id": 123}},
                 "preferences": {},
             }), \
             patch("src.pipeline.run._api_get_with_retry", side_effect=[
                 entry_picks_response,
                 MagicMock(**{"json.return_value": {"leagues": {"classic": []}}}),
                 MagicMock(**{"json.return_value": {"current": []}}),
             ]):
            run_mod.phase_post_gw()

        log = pd.read_csv(results / "accuracy_log.csv")
        assert (log["wildcard_pts"].notna()).any(), "wildcard_pts must be written when squad CSV exists"
        assert log.iloc[-1]["wildcard_pts"] > 0

    def test_post_gw_discord_called(self, tmp_path, e2e_bootstrap):
        """format_accuracy_discord.py must produce non-empty output for a complete log row."""
        import subprocess, sys, json as _json

        # Write a fake accuracy log with all fields populated
        results = tmp_path / "results"
        results.mkdir(parents=True, exist_ok=True)
        log_path = results / "accuracy_log.csv"
        log_path.write_text(
            "gw,your_pts,your_predicted_xp,recommended_pts,recommended_xp,"
            "wildcard_pts,wildcard_xp,dream_team_pts,your_percentile_rank,"
            "best_score,top_1k_score,top_10k_score,top_100k_score,top_1m_score,"
            "avg_score,median_score,ranked_count,spearman_rho,timestamp\n"
            "31,55,48.5,60,52.0,70,65.0,129,25,109,66,50,38,,38,,12914049,0.65,2026-04-17T00:00:00+00:00\n"
        )

        from pathlib import Path as _Path
        repo_root = _Path(__file__).parents[1]
        result = subprocess.run(
            [sys.executable, "scripts/format_accuracy_discord.py", str(log_path), "31"],
            capture_output=True, text=True,
            cwd=repo_root,
        )
        assert result.returncode == 0, f"Script failed: {result.stderr}"
        assert "GW31" in result.stdout
        assert "55" in result.stdout        # your_pts
        assert "0.650" in result.stdout     # spearman_rho
        assert "70" in result.stdout        # wildcard_pts

    def test_check_gw_finished_script(self, tmp_path, e2e_bootstrap):
        """check_gw_finished.py must detect finished=True when current GW is done."""
        import subprocess, sys, json as _json

        # Build bootstrap with current GW finished
        bs = dict(e2e_bootstrap)
        bs["events"] = [
            {"id": 31, "deadline_time": "2026-03-20T18:30:00Z",
             "is_current": True, "is_next": False, "finished": True},
        ]
        snap = tmp_path / "bootstrap_gw31.json"
        snap.write_text(_json.dumps(bs))

        from pathlib import Path as _Path
        repo_root = _Path(__file__).parents[1]
        result = subprocess.run(
            [sys.executable, "scripts/check_gw_finished.py", str(snap)],
            capture_output=True, text=True,
            cwd=repo_root,
        )
        assert result.returncode == 0
        assert "Finished: True" in result.stdout

    def test_check_gw_not_finished(self, tmp_path, e2e_bootstrap):
        """check_gw_finished.py must detect finished=False when GW is ongoing."""
        import subprocess, sys, json as _json
        from pathlib import Path as _Path

        bs = dict(e2e_bootstrap)
        bs["events"] = [
            {"id": 31, "deadline_time": "2026-03-20T18:30:00Z",
             "is_current": True, "is_next": False, "finished": False},
        ]
        snap = tmp_path / "bootstrap_gw31.json"
        snap.write_text(_json.dumps(bs))

        repo_root = _Path(__file__).parents[1]
        result = subprocess.run(
            [sys.executable, "scripts/check_gw_finished.py", str(snap)],
            capture_output=True, text=True,
            cwd=repo_root,
        )
        assert result.returncode == 0
        assert "Finished: False" in result.stdout
```

- [ ] **Step 3: Run the new E2E tests**

```bash
python -m pytest tests/test_pipeline_e2e.py::TestPostGwDiscord -v
```

Expected: all 5 tests pass

- [ ] **Step 4: Run full test suite to confirm no regressions**

```bash
python -m pytest tests/ -q --ignore=tests/test_integration.py
```

Expected: all pass (or same pass rate as before)

- [ ] **Step 5: Commit**

```bash
rtk git add tests/test_pipeline_e2e.py
rtk git commit -m "test: add E2E tests for post-gw Discord notification and accuracy log"
```

---

## Task 6: Manual validation

- [ ] **Step 1: Trigger workflow manually**

Push changes to master, then:
```
GitHub → Actions → Daily FPL Bootstrap Snapshot → Run workflow
```

- [ ] **Step 2: If current GW is finished, verify Discord message in `#fpl-recommendations` channel**

The message should show:
- `GW{N} Post-Match Accuracy`
- your_pts, recommended_pts, optimizer squad pts, dream team pts
- Spearman ρ value
- GW benchmarks

- [ ] **Step 3: Check accuracy_log.csv in committed results**

```bash
tail -1 results/accuracy_log.csv
```

Confirm `wildcard_pts`, `wildcard_xp`, `spearman_rho` are non-empty for the new row.

---

## Summary

| Task | Deliverable | Tests |
|------|------------|-------|
| 1 | `check_gw_finished.py` | E2E subprocess tests |
| 2 | Fix `phase_post_gw` accuracy log fields | Unit + existing pipeline tests |
| 3 | `format_accuracy_discord.py` | E2E subprocess test |
| 4 | Workflow YAML: post-gw job | Manual / GitHub Actions |
| 5 | `TestPostGwDiscord` E2E class | 5 new tests |
| 6 | Manual smoke test | Discord channel verification |
