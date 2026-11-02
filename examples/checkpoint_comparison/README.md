This directory preserves the **initial frozen study**, including its native diagnostic and unstarted arm. See the [checkpoint comparison report](../../docs/CHECKPOINT_COMPARISON.md) for the subsequent isolated repair, compatibility checks, and fresh native follow-up. These studies must not be pooled or counted twice.

# Frozen 896-to-1024 score comparison

This small dataset reproduces score-table statistics without a game database,
model weights, external dependencies, or simulation. Run with Python 3.12:

```sh
python paired_scores_with_coverage.py --input paired-scores-with-coverage.json
```

The printed JSON must equal `statistics-with-coverage.json`. The original
`paired_scores.py` supplies the frozen arithmetic; its complete-cohort entry
point is intentionally not used for this incomplete experiment.

Both models use Monte Carlo Control with the same frozen configuration,
fixed training-context lookup, and mean-evaluation V4 guard. Mean evaluation
retains `exploration_floor=0.05` and public history controls; it is not pure
greedy play. Only the frozen model weights differ between the two arms. The
parent has 896 verified training Games; the child adds 128, for 1024 total.
Training returns are official scores, without clipping or a stall penalty.

Three panels were declared before looking at their outcomes, each with
128 paired Game seeds and 32 pairs per focal seat. A Game has four Races in
Mild/Wild/Mild/Wild order. Each complete arm stores the four-by-four official
signed Race-score table; the focal player's four scores are summed. The
three opponents in one Game are not three independent observations.

| Panel | 896 mean | 1024 mean | Difference (1024 minus 896) | 95% pointwise bootstrap CI |
| --- | ---: | ---: | ---: | --- |
| Native, primary | unavailable | unavailable | unavailable | unavailable |
| Random, exploratory | 9.3984375 | 9.1484375 | -0.25 | [-0.875, 0.359375] |
| Public memory, exploratory | 10.4921875 | 10.6875 | 0.1953125 | [-0.2578125, 0.65625] |

The native panel stopped after the parent arm produced 127 complete Games
and one event-budget diagnostic. Its 128 child-arm Games were unstarted.
All 384 declared pair coordinates and 768 arm statuses remain in the data:
639 complete, one diagnostic, and 128 unstarted. Missing or diagnostic final
scores are `null`; partial scores are not treated as completed outcomes.
No seed was replaced, and no success-only native confidence interval is
reported. The primary incremental-learning comparison is therefore incomplete.

The two complete exploratory panels each retain all 128 pairs. Their
intervals use 2,000 paired bootstrap replicates, resampled separately within
each focal seat. They are pointwise intervals without multiplicity correction;
panels are not pooled. Both intervals include zero and do not establish a
score improvement. Successful model updates and compatible context lookup
are separate facts from performance improvement.

The dataset contains seeds, model and policy hashes, contexts, statuses, and
official score tables. It contains no weights, private filesystem paths,
player names, database, or long replay traces. A separate local exporter
authenticated actual arm seeds, frozen policies, and official score ledgers;
this standalone script verifies the compact table and recomputes its
statistics, not the underlying simulation or engine rules.

Files intended for distribution are this README, the two Python scripts,
`paired_scores_with_coverage.schema.json`, `paired-scores-with-coverage.json`,
and `statistics-with-coverage.json`. The local
`private-export-verification.json` is not part of this public sample.
