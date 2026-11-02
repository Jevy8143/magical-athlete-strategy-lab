# Reproducible local demo

From the repository root, with Python 3.12 or newer:

```console
python -m pip install -e .
python -m magical_athlete.demo --seed 1 --output demo-output
python -m magical_athlete.demo --replay demo-output/trace.json
```

The installed `magical-athlete-demo` command provides the same interface.
Use a new or empty output directory; existing evidence is never overwritten.
No experiment database, model checkpoint, network connection, or third-party
runtime library is needed.

This runs the standard four-player original-box game from the two-round draft
through Mild / Wild / Mild / Wild. Four independently seeded random legal
agents choose their cards and skill options from the engine's public menus.
The engine applies the project's documented rules and HOUSE rulings.

For seed `1`, player IDs `0, 1, 2, 3` score **12, 10, 10, 3**; player `0` wins.
The machine-readable reference is [demo-seed-1.expected.json](demo-seed-1.expected.json).
This is a reproducibility example, not a strategy-strength result.

`summary.json` contains every point change and separates placement chips,
Star points, and skill gains or penalties. `trace.json` preserves all four
Races, dice, public decision requests, choices, the draft, and the final
ledger. The command strictly replays each Race and the full Game before
reporting success. Replaying requires matching engine code; the trace stores
the code fingerprint. An event-budget failure is retained as an incomplete
diagnostic and exits nonzero, with no invented result.

The expected file omits the code fingerprint and full trace hash so that
formatting-only source changes do not change the documented score reference.
Generated output records both hashes. The local research checkpoints
is not bundled with this example; running the demo performs no training.

## Optional Model A pipeline smoke

```console
python examples/model_a_smoke.py --output smoke-output
```

This separate command starts from an empty prior and plays **4 TRAIN Games
and 8 frozen EVAL Games**, with a fixed master seed of `20261102`. It uses
the existing first-visit Monte Carlo / Thompson model, the V3 training guard,
and the explicitly versioned V4 mean-evaluation guard. It makes one real
update from the four complete training records. Evaluation records are not
used for that update. All twelve traces, model artifacts, source evidence,
and the detailed report are retained under the new output directory.
Existing output directories are refused.

One actual successful run is summarized in
[model-a-smoke.summary.json](model-a-smoke.summary.json). The generated report
path contains that run's source-dependent plan hash; another source revision
can produce a different hash. This small smoke checks the pipeline, not score
improvement or general playing strength. It does not load, inherit, reset,
or modify the separate local research checkpoints. Generated
experiment databases, logs and weights are not part of the example files.
