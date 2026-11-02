"""Report CSV scenario coverage separately from ordinary pytest success."""

import csv
import json
from pathlib import Path

import pytest

from magical_athlete.engine import RULE_VERSION

ROOT = Path(__file__).resolve().parents[1]


def pytest_configure(config):
    manifest = json.loads((ROOT / "docs/rule_manifest.json").read_text(encoding="utf-8-sig"))
    if manifest["canonical_version"] != RULE_VERSION:
        raise pytest.UsageError("Engine rule version differs from docs/rule_manifest.json")
    with (ROOT / "docs/test_vectors_curated.csv").open(encoding="utf-8-sig", newline="") as file:
        config.curated_ids = [row["test_id"] for row in csv.DictReader(file)]
    if sorted(config.curated_ids) != [f"T-{i:03d}" for i in range(1, 34)]:
        raise pytest.UsageError("Expected exactly T-001 through T-033 in the curated CSV")
    config.curated_nodes = {}
    config.curated_outcomes = {}


def pytest_collection_modifyitems(config, items):
    for item in items:
        for marker in item.iter_markers("rule_vector"):
            vector_id = marker.args[0]
            if vector_id not in config.curated_ids:
                raise pytest.UsageError(f"Unknown curated vector: {vector_id}")
            config.curated_nodes.setdefault(vector_id, []).append(item.nodeid)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    report = (yield).get_result()
    outcomes = item.config.curated_outcomes
    if report.failed or report.skipped:
        outcomes[item.nodeid] = report.outcome
    elif report.when == "call" and outcomes.get(item.nodeid) not in ("failed", "skipped"):
        outcomes[item.nodeid] = "xpassed" if hasattr(report, "wasxfail") else "passed"


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    coverage = {}
    for vector_id in config.curated_ids:
        nodes = config.curated_nodes.get(vector_id, [])
        outcomes = [config.curated_outcomes.get(node, "not_run") for node in nodes]
        if not nodes:
            status = "not_implemented"
        elif all(outcome == "passed" for outcome in outcomes):
            status = "passed"
        else:
            status = next(outcome for outcome in ("failed", "skipped", "xpassed", "not_run") if outcome in outcomes)
        coverage[vector_id] = {"status": status, "tests": nodes}
    passed = sum(v["status"] == "passed" for v in coverage.values())
    terminalreporter.write_sep("-", f"Curated real-engine scenarios: {passed}/{len(coverage)} passed")
    pending = [key for key, value in coverage.items() if value["status"] != "passed"]
    if pending:
        terminalreporter.write_line("Incomplete: " + ", ".join(pending))
    output = Path(config.cache.mkdir("rule_coverage")) / "coverage.json"
    output.write_text(json.dumps({
        "rule_version": RULE_VERSION,
        "pytest_exitstatus": int(exitstatus),
        "vectors": coverage,
    }, indent=2), encoding="utf-8")
