# Release validation

This record separates directly executed demonstrations from historical research results. The local release has independently verified installation, portable tests, replay, and failure handling. It is not a claim that the entire research program is finished.

## Executed in the development workspace

- Environment: Python 3.12.15.
- `python -m magical_athlete.demo --seed 1 --output <new directory>` completed the draft and all four official races. Final scores were **12, 10, 10, 3**, with player 0 winning.
- The generated ledger contains 20 real point entries. Every race and the complete game passed strict replay.
- The 12 new demo tests passed, including repeatability, tamper rejection, output preservation, and diagnostic retention.
- `python examples/model_a_smoke.py --output <new directory>` completed **4 TRAIN + 8 EVAL** games, retained all twelve, and committed one real update. The model started from zero historical training games.
- The smoke produced 193 state-action rows and 202 first-visit observations. Inspected estimates and action preferences changed; these are in-sample learning checks, not evidence of stronger play. Evaluation and failed games used for updates: **0**.

The source handoff incorporates focused fixes for long skill chains and a separately versioned mean-evaluation safeguard. Before this handoff, their combined 260-case suite and three specific complete-game regressions passed. These targeted results do not substitute for an exhaustive interaction test or a new full-suite run.

## Historical local research

A frozen checkpoint learned from **896 complete training games across seven updates**. The checkpoint and original database, model versions, replay traces, and failed attempts remain local; they are not distributed in this small source package.

That historical audit establishes real, traceable updates. It does not establish a reliable performance gain. Two small development comparisons each used 64 paired games and their score-difference intervals included zero. They used the earlier evaluation policy and must not be pooled with the new evaluation variant.

The previous broad 17,506-test run applied to an earlier engine snapshot. It is historical evidence, not the test count for this release. The earlier 512-game diagnostic pool retained 510 completed games and two failures. Both failures and the original records remain preserved.

## Completed fixed-coordinate engineering regression

The repaired engine and explicitly versioned mean-evaluation guard completed **512/512** retained diagnostic Games with zero simulation or validation failures in about 337 seconds. The study used 256 mean-policy Games and 256 Thompson-policy Games, balanced across four focal seats. All Games underwent policy-audit checks, individual-Race replay, and complete-Game replay. Both previously failing coordinates passed.

This regression reused the exact earlier 512 coordinates and all 5,120 random streams. It contains **zero new independent performance samples** and **zero model updates**. The frozen 896-game weights and three fixed prior opponents were retained; only the mean focal policy used the explicit V4 safeguard. At the time of that regression, the 54 shared production modules matched the then-working source; its extra demo and packaging metadata had a distinct complete-source identity. The later one-module repair and its current-source checks are recorded below. No extra full-suite rerun is implied by this regression.

A separate current-source certificate was issued for all **896 original TRAIN Games**. Each Game was actually rerun with its original frozen policies and seeds, passed strict replay, and matched its original trace except for permitted code-fingerprint stamps. The historical batches reconstructed the exact saved checkpoint bytes. The certificate and all unique Game identities were independently checked against the parent model and retained receipts. These diagnostic revalidations create **zero new learning samples**.

The bounded continuation completed **128 new TRAIN + 128 developer-EVAL Games**, all retained and verified. It committed one update to the selected **1,024-Game, eight-update checkpoint**, preserving the original 896-Game prefix. An independent arithmetic review reconstructed every new official remaining-score return, first-visit count, and the exact child checkpoint; EVAL, diagnostics, and failed attempts contributed zero update samples. The exposed 64-pair developer comparison changed mean score by **+0.125**, with pooled paired-bootstrap interval **[−0.6875, 1.0]**. This interval includes zero and is not independent confirmation.

The initial three-panel evaluation retained 639 completed Games, one native event-budget diagnostic, and 128 native Games left unstarted by the strict panel gate. Both exploratory panels completed 128 pairs and their intervals included zero. A narrowly scoped repair then passed 1,027 affected-family tests and the original diagnostic Game, followed by independent component review.

Under the repaired source, all **1,024 original training Games** were actually executed and replayed, the eight batches reconstructed the exact checkpoint, and original inputs were freshly checked before issuing a new certificate. All **512 completed exploratory Games** also passed 2,048 individual-Race and 512 full-Game compatibility replays with only code-fingerprint stamps projected in memory. Those are reused histories, not additional independent samples.

A separately predeclared native-only study then completed **128 new pairs / 256 Games**, with zero diagnostics or quarantines. Its mean difference was **−0.0078125**, with a seat-stratified pointwise 95% interval **[−0.546875, 0.5]**. Actual ledgers, random streams, model/context identities, and the independent numerical recomputation matched. See the [comparison report](CHECKPOINT_COMPARISON.md) for both studies and their separate public score tables; no initial failure was replaced or pooled away.

The final fixed 512-Game continuation did **not** enter training. Its initial developer evaluation retained 127 complete Games and one `race_event_budget_exceeded` diagnostic; all 640 later planned Games were unstarted. The actual process exited 1, with zero new TRAIN Games, consumed training samples, or updates. A bounded read-only audit checked all 128 physical trace hashes, planned streams, recorded integrity flags, the unstarted status coverage, and the unchanged 1,024-Game parent and source files. The diagnostic has no complete-Game replay verification or final score. This audit verifies evidence retention, **not** successful completion of the continuation, termination of all skill chains, or a performance improvement.

## Independent packaged verification

The first source archive was unpacked into a separate directory and installed in a new Python 3.12.15 virtual environment without system site packages. Installation using the README's `python -m pip install -e ".[test]"` succeeded. Imports and command execution were also checked from outside the extracted source directory.

The portable checks passed in two explicit partitions: **232 original release tests** plus **77 model and feature tests**, totaling **309 passed, 0 failed, 0 errors, 0 skipped**. The first partition executed all **33/33** curated rule scenarios against the real engine. The two added files are `test_model_a.py` and `test_model_a_features.py`; these are additional checks, not a second count of the original tests. The full packaged command is:

```console
python -m pytest -q
```

The independently installed module demo and installed `magical-athlete-demo` console replay both exited 0, reproducing **12, 10, 10, 3** and a complete four-race replay.

A separate real-engine check set the event budget to 5. It exited **2**, saved five events and an incomplete diagnostic, and awarded no invented places. Its partial Race trace could be replayed exactly using the Race API. The complete-Game replay command rejects an incomplete diagnostic; incomplete traces are retained for debugging, not presented as finished games.

An isolated temporary Git index with `core.autocrlf=true` preserved the original bytes of all four versioned guard modules. The packaged `.gitattributes` prevents automatic newline conversion of those fingerprinted files.

The final archive is checked against this verified runtime/test payload; documentation and package metadata changes are checked separately, with a final installation/import check. No unavailable historical test or excluded archive-dependent test is counted as a release pass.

## Repaired-source portable payload

After integrating the one-module native-cycle repair, a separate 99-file source snapshot was installed into a new virtual environment without system site packages. The README's editable-install command completed offline using seven locally cached dependency wheels. Imports from outside the source directory, distribution metadata, and the console entry point were checked.

The full portable command executed **1,336 tests: 1,336 passed, 0 failed, 0 skipped**, in 76.87 seconds, including all **33/33** curated scenarios. This is the original 309-check set plus the 31 new tests and their 996-case affected-family module. The original module is included because the new tests share its deterministic real-engine fixtures; no assertion was removed to make the tests portable.

The seed-1 demo and console full-Game replay exited 0 with scores **12, 10, 10, 3** and 20 ledger entries. All fields explicitly given by the compact expected file matched; the complete real ledger, including extra event/source metadata, remains retained. The validation wrapper initially compared that compact reference to a full dictionary and rejected the extra fields; its failed wrapper record was preserved, the projection check was corrected, and only the unfinished post-checks were run. Neither the fixture nor the engine changed, and the 1,336 tests and demo were not rerun.

The copied score-recomputation CLI also exited 0 and reproduced every initial-study status and statistic. A 78-member hash lock covers all packaged Python source, selected tests, `pyproject.toml`, and the generated minimal test metadata. The final source handoff will be checked against that tested payload; later prose or score-table additions do not imply another full test execution. The separate 12-Game learning smoke remains historical evidence and was not unnecessarily repeated.

## Interpretation

This release demonstrates deterministic simulation, explicit score accounting, replay, and an auditable Monte Carlo learning pipeline. It does not demonstrate exhaustive rules correctness, statistically significant strategy improvement, a completed Model B, or general optimal play.
