"""Real official-game, accounting and CLI replay acceptance for the small demo."""

from copy import deepcopy
import json
from pathlib import Path
import random
import sqlite3

import pytest

from magical_athlete import demo
from magical_athlete.game import GameDiagnosticError


@pytest.fixture(scope="module")
def completed_demo():
    before = random.getstate()
    old_connect = sqlite3.connect

    def forbidden(*args, **kwargs):
        raise AssertionError("the demo must not open an experiment database")

    sqlite3.connect = forbidden
    try:
        value = demo.run_demo(1)
    finally:
        sqlite3.connect = old_connect
    assert random.getstate() == before
    return value


def test_official_demo_matches_small_checked_score_reference(completed_demo):
    summary, trace = completed_demo
    expected = json.loads((Path(__file__).resolve().parents[1] / "examples/demo-seed-1.expected.json").read_text(encoding="utf-8"))
    for key, value in expected.items():
        if key != "score_entries":
            assert summary[key] == value
    entries = [{key: entry[key] for key in ("sequence", "player_id", "race_index", "delta", "reason", "phase")}
               for entry in summary["score_ledger"]["entries"]]
    assert entries == expected["score_entries"]
    assert len(trace["draft_history"]) == 16
    assert len({card["card_id"] for race in trace["races"] for card in race["entries"]}) == 16
    assert [race["track"] for race in summary["races"]] == ["Mild", "Wild", "Mild", "Wild"]
    # This seed exercises genuine non-placement scoring, including Sisyphus
    # and Loveable Loser, rather than treating the chips as the whole reward.
    assert summary["races"][1]["score_breakdown"][3]["skills"] == 2
    assert summary["races"][3]["score_breakdown"][0]["skills"] == 2
    assert sum(summary["total_scores"]) == sum(entry["delta"] for entry in entries)


def test_same_seed_repeats_full_trace_and_strict_replay(completed_demo):
    first_summary, first_trace = completed_demo
    second_summary, second_trace = demo.run_demo(1)
    assert demo.canonical_json(first_trace) == demo.canonical_json(second_trace)
    assert first_summary == second_summary
    assert first_summary["verification"] == {
        "score_ledger_verified": True, "individual_races_verified": 4, "whole_game_verified": True,
    }


@pytest.mark.parametrize("mutation", ["score", "bool_delta", "completed_integer", "illegal_action"])
def test_strict_replay_rejects_real_trace_tampering(completed_demo, mutation):
    _, original = completed_demo
    trace = deepcopy(original)
    if mutation == "score":
        trace["scores"]["totals"]["0"] += 1
    elif mutation == "bool_delta":
        next(entry for entry in trace["scores"]["entries"] if entry["delta"] == 1)["delta"] = True
    elif mutation == "completed_integer":
        trace["completed"] = 1
    else:
        next(race for race in trace["race_traces"] if race["decisions"])["decisions"][0]["action"]["target_id"] = 999
    with pytest.raises((ValueError, TypeError)):
        demo.verify_trace(trace)


def test_cli_saves_full_evidence_and_replays_without_database(tmp_path, capsys, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("the demo must not open a database")

    monkeypatch.setattr(sqlite3, "connect", forbidden)
    output = tmp_path / "seed1"
    assert demo.main(["--seed", "1", "--output", str(output)]) == 0
    display = json.loads(capsys.readouterr().out)
    assert display["total_scores"] == [12, 10, 10, 3]
    assert display["strict_replay"]["whole_game_verified"] is True
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    trace = json.loads((output / "trace.json").read_text(encoding="utf-8"))
    assert summary["score_ledger"] == trace["scores"]
    assert demo.main(["--replay", str(output / "trace.json")]) == 0
    assert json.loads(capsys.readouterr().out)["total_scores"] == [12, 10, 10, 3]

    def no_new_game(*args, **kwargs):
        raise AssertionError("refuse an occupied directory before running a Game")

    monkeypatch.setattr(demo, "run_demo", no_new_game)
    with pytest.raises(SystemExit) as error:
        demo.main(["--output", str(output)])
    assert error.value.code == 2
    assert json.loads((output / "trace.json").read_text(encoding="utf-8")) == trace


def test_default_paths_are_relative_and_existing_evidence_is_preserved(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert demo.main([]) == 0
    display = json.loads(capsys.readouterr().out)
    assert display["files"] == {"summary": "demo-output/summary.json", "trace": "demo-output/trace.json"}
    assert (tmp_path / "demo-output/trace.json").is_file()


def test_diagnostic_is_saved_without_becoming_a_completed_game(tmp_path, capsys, monkeypatch):
    failure_trace = {"completed": False, "race_diagnostic_reason": "event_budget_exceeded"}

    def incomplete(seed):
        raise GameDiagnosticError("race_event_budget_exceeded", failure_trace)

    monkeypatch.setattr(demo, "run_demo", incomplete)
    output = tmp_path / "diagnostic"
    assert demo.main(["--output", str(output)]) == 2
    assert json.loads(capsys.readouterr().out)["completed"] is False
    assert json.loads((output / "trace.json").read_text(encoding="utf-8")) == failure_trace
    assert json.loads((output / "summary.json").read_text(encoding="utf-8"))["completed"] is False


@pytest.mark.parametrize("seed", [-1, True, 1.5])
def test_invalid_seed_is_rejected(seed):
    with pytest.raises(ValueError, match="nonnegative integer"):
        demo.seed_streams(seed)
