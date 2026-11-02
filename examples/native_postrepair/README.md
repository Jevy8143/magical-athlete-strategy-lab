# Separate post-repair native comparison

Run with Python 3.12 and its standard library:

```sh
python native_postrepair_scores.py --input native-scores-with-coverage.json
```

The output must equal `statistics.json`. Keep `paired_scores.py` next to the
reader: it provides the same frozen arithmetic used for the initial study.
No game database, model weights, simulation, or external packages are needed.

This is a new study declared after observing an engine diagnostic in the
initial native comparison. It uses 128 new paired seeds, 32 per focal seat,
and 256 completed four-Race Games. Its seeds are disjoint from the initial
384 pair coordinates, including their unstarted arms, and from checked
historical and reserved experiment streams. It does not complete, replace,
pool with, or erase the initial failed study.

The models remain frozen at 896 and 1024 verified training Games. Both arms
use the same mean V4 policy configuration, including `exploration_floor=0.05`
and public history/guard controls, against three identical frozen A0 V3
opponents. Actual and fixed lookup contexts are native C98. Only the model
weights differ between the arms. Mean evaluation is not pure greedy play.

| Quantity | Result |
| --- | ---: |
| Paired Games | 128 |
| Completed arm Games | 256 / 256 |
| 896-model mean official four-Race total | 9.9140625 |
| 1024-model mean official four-Race total | 9.90625 |
| Difference, 1024 minus 896 | -0.0078125 |
| 95% pointwise paired bootstrap interval | [-0.546875, 0.5] |

The interval uses 2,000 bootstrap replicates, resampled within focal seats.
There is no multiplicity correction. The unit is a paired Game seed and
focal seat; the three opponents are not three extra independent samples.
This result does not establish an improvement in expected official score.

The initial study remains separate and unchanged: its native parent arm had
127 completed Games and one diagnostic, and its 128 child-arm Games were
unstarted. Its complete random and public-memory exploratory panels showed
intervals containing zero. All 512 completed exploratory Games subsequently
passed strict compatibility replay under the repaired engine, retaining the
original actions, observations, and signed official score ledgers. Those
replays are not new performance samples.

The JSON contains exactly this study's 128 pair coordinates and 256 actual
statuses, with official four-by-four signed Race-score tables. Incomplete
arms would have `null` final scores and would prevent a study confidence
interval. No extra exploratory records are invented to fit the old schema.
The initial dataset is identified separately by its unchanged SHA256:
`d609931b608b5d9c75128d022687ace4710228f1af9b45a4204c06b75e277194`.

The standalone reader verifies the compact table and recomputes statistics.
A separate private review authenticated actual streams, frozen policies,
contexts, and official ledgers. This sample contains no weights, database,
personal names, private paths, or long traces. Distribute this README,
`native_postrepair_scores.py`, `paired_scores.py`,
`native-scores-with-coverage.json`, and `statistics.json`.
Exclude the local `private-export-verification.json`.
