# Fixed512 extension: stopped before learning

This was the final fixed development learning-curve extension from the
already verified 1024-Game model. It planned 512 new TRAIN Games and initial
and final 128-Game DEV arms. It was exploratory, without adaptive additions,
and never replaced either earlier 896-to-1024 primary study.

Initial DEV produced 127 complete Games and one diagnostic. The strict pilot
then stopped before training. All 512 TRAIN Games and all 128 final DEV Games
were unstarted: 640 unstarted attempts remain explicitly recorded. There
were zero new training samples and zero updates. The verified 1024 model is
preserved. No new 1536-Game model or final-evaluation snapshot was produced.

No score gain or confidence interval is reported. The complete initial Games
are not selected into a success-only paired comparison. Every declared
attempt remains in `fixed512-development-scores.json`; incomplete or
unstarted arm scores are `null`. This failure does not replace or modify
the separately retained earlier research data.

`failure-summary.json` records the exact failed Game/run, actual seed streams,
trace SHA256, source fingerprints, and event-budget reason. It failed in
Race 2 on Wild Wilds, at the 10,000-event safety limit, with
Flip Flop, Scoocher and Suckerfish at space 5, and Huge Baby at space 6.
The trace was retained as a
diagnostic with no completed final score and `replay_verified=false`.
No engine repair, counterfactual Game, seed replacement, increased budget,
restart, or extra training was performed in this extension.

Reproduce the status report with Python 3.12 and its standard library:

```sh
python fixed512_development_scores.py --input fixed512-development-scores.json
```

The printed JSON must equal `statistics.json`. Keep `paired_scores.py`
next to the reader. Both the four-seat equal-weight stratified bootstrap
and the production Pilot's sorted-pairing-key pooled bootstrap are `null`
because the paired DEV study is incomplete. The two methods are retained
as distinct reporting conventions; neither creates an interval from the
127 successful initial Games.

This sample contains status metadata, seeds, model/source hashes and integer
score tables for actual complete initial Games. It contains no model weights,
database, private paths, personal names, or long traces. The standalone
script checks the compact table; separate private audits authenticated the
actual records and official ledgers without replaying Games.

Distribute this README, `fixed512_development_scores.py`, `paired_scores.py`,
`fixed512-development-scores.json`, `statistics.json`, and
`failure-summary.json`. Exclude `private-export-verification.json`.
