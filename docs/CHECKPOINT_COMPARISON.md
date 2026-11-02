# Frozen Model A checkpoint comparison

Status: the repaired snapshot is validated. The child checkpoint has 1,024 complete TRAIN Games and eight updates. All 1,024 training sources passed new-source certification, all 512 completed exploratory Games passed compatibility replay, and a separate fresh native comparison completed all 128 pairs. A reliable score improvement has not been established.

## Research question

Does one predeclared 128-Game update change expected official four-Race total score, starting from the selected 896-Game checkpoint? The comparison changes the frozen checkpoint weights while keeping the inference configuration and evaluation safeguard fixed.

Model A uses first-visit Monte Carlo control and a Dirichlet–Categorical score posterior. Returns are the final official total minus the focal player's total at the decision. They can be negative. Training samples are complete, persisted, verified TRAIN Games; EVAL and failed Games do not enter updates.

## Predeclared design

| Opponents in the other three seats | Role | Pairs (two Games each) |
| --- | --- | ---: |
| Three frozen untrained Model A policies | Primary incremental-learning comparison | 128 |
| Three uniform legal-action policies | Exploratory transfer | 128 |
| Three public-history first-legal policies | Exploratory transfer | 128 |

Each panel has 32 independent pair coordinates per focal seat. The two arms share all ten recorded random streams within each pair; different coordinates have disjoint streams. All seeds were declared before observing the child checkpoint or evaluation scores. The unit of analysis is one paired Game and focal seat, not each of its four players.

The statistic is the equally weighted four-seat mean difference, **1,024 minus 896**, with a 2,000-replicate seat-stratified paired-Game bootstrap and a pointwise 95% percentile interval. Panels are reported separately. These are not simultaneous family-level 95% intervals. No failed pair is replaced, imputed, or silently omitted. An incomplete panel receives full attempt coverage and no unconditional score-improvement claim; complete predeclared panels can be reported alongside it.

Both arms use the same mean-inference configuration, including the **0.05 exploration floor**, public repetition history, and explicit V4 safeguard. Training and the frozen native opponents retain the earlier V3 safeguard. This is not a purely greedy policy. Fixed streams make an experiment reproducible; different choices can still cause the two arms' subsequent trajectories to diverge.

The primary panel queries the model's native opponent context. The transfer panels explicitly reuse those learned estimates under different actual opponents. Actual opponent context and model lookup context are recorded separately; transfer is not described as native training against the new opponents.

The original 64-pair before/after developer evaluation reuses previously exposed coordinates. Its means were **9.390625 → 9.515625**, a difference of **+0.125**, with a pooled paired-bootstrap interval of **[−0.6875, 1.0]**. That interval includes zero. This developer result is separate from the new seat-stratified panels and is not pooled with them or called independent confirmation.

## Current independent results

| Panel | Complete pairs / planned | 896 mean | 1,024 mean | Difference | Pointwise 95% interval |
| --- | ---: | ---: | ---: | ---: | --- |
| Native priors, primary | 0 / 128 | — | — | — | Incomplete; not estimated |
| Uniform random, exploratory | 128 / 128 | 9.3984375 | 9.1484375 | −0.25 | [−0.875, 0.359375] |
| Public history, exploratory | 128 / 128 | 10.4921875 | 10.6875 | +0.1953125 | [−0.2578125, 0.65625] |

The native left arm retained 127 complete Games and one event-budget diagnostic at index 124. The strict panel gate left all 128 right-arm Games unstarted; they are explicitly included as unstarted in the full planned-attempt coverage. There is no conditional success-only native comparison. The initial study retained **639 completed Games, one diagnostic, and 128 unstarted planned Games**. No seed was replaced and no partial final score was imputed.

Both exploratory intervals include zero: these data do not establish improvement or deterioration under those opponent distributions. All 512 exploratory Games, both arms' actual streams, official four-Race score ledgers, model identities, and contexts were checked. Opponent panels are not pooled.

## Repair and follow-up boundary

The initial study used source archive `bc238f5f…`. Its diagnostic exposed an exact Romantic / Alchemist / Third Wheel / Copy Cat forced-cycle domain. A one-module repair passed 31 new tests, 996 affected-family regressions, and the original full-Game diagnostic with its four snapshots and ten streams preserved. Independent component review found no blocker. The integrated source archive is `2ddf3ee8…`; rule version, reward, model weights, and safeguards are unchanged.

The original diagnostic and unstarted arm remain part of the initial coverage. A separately predeclared **128-pair native-only study** used new streams under the repaired source. It does not fill the original missing arm or pool with the failed primary panel. All 1,024 training sources passed actual new-source semantic execution, strict replay, exact Monte Carlo checkpoint reconstruction, and final original-object rechecking. All 512 completed exploratory Games also matched 2,048 individual-Race replays and 512 full-Game replays after replacing only the five in-memory code-fingerprint stamps. Neither revalidation adds independent learning or performance samples.

The fresh native study completed **256/256 arm Games**, with zero diagnostics or quarantines:

| Quantity | Result |
| --- | ---: |
| Independent paired Game seeds | 128, with 32 per focal seat |
| 896-Game model mean official total | 9.9140625 |
| 1,024-Game model mean official total | 9.90625 |
| Difference, 1,024 minus 896 | −0.0078125 |
| Pointwise 95% interval | [−0.546875, 0.500000] |

The interval includes zero. This does not demonstrate improvement, deterioration, or that learning is ineffective. It measures a small, fixed increment under one native opponent configuration; it is separate from the original incomplete study and the two exploratory panels. Training and safeguards were fixed; none of the evaluation Games updated the models.

The small [initial score/coverage export](../examples/checkpoint_comparison/README.md) and [separate native follow-up export](../examples/native_postrepair/README.md) include executable standard-library recomputation instructions. Their actual ledgers, streams, policies, and contexts were privately authenticated; their statistics were independently recomputed. The initial failed dataset remains unchanged alongside the follow-up.

## Final bounded continuation: no new update

After preserving the validated source handoff, a separate fixed learning-curve plan attempted to extend the 1,024-Game checkpoint by four 128-Game training batches. Initial and final developer evaluations each had 128 planned Games. The pilot identifier was `deadline-c05-1024-1536-20261102-v1`, with master seed `202611023512`; size, parameters, and the strict stop policy were fixed before the latest native score difference was disclosed. This was exploratory developer evaluation, not a replacement primary study.

The initial evaluation produced **127 complete Games and one diagnostic**. Game index 62 in focal seat 1 reached `race_event_budget_exceeded` during Race 2 on Wild, in an unresolved chain involving Suckerfish, Scoocher, Huge Baby, and Flip Flop. The diagnostic is incomplete, has no final score, and is not strict-replay-certified. Budget exhaustion alone is not a proof of an infinite loop.

The strict initial-evaluation gate rejected training. The process actually closed with exit code 1 at **2026-11-02 08:39:57 UTC**, preserving all 128 attempted traces and the **640 unstarted** coordinates (512 TRAIN and 128 final DEV). There were **zero new training Games, zero consumed samples, and zero updates**. The selected model remains 1,024 Games / eight updates; no 1,536-Game model exists. There is no after-model, complete paired learning curve, or score-difference interval to report. No seed replacement, budget increase, or further engine repair was performed for this attempt.

The separate [stopped-continuation export](../examples/fixed512_stopped/README.md) includes every planned status and the failed seed/trace hash. Its standard-library reader reproduces coverage counts with both statistical reports explicitly `null`; the 127 complete initial Games are not turned into a success-only improvement estimate.

## Existing learning coverage

The selected parent has **896 unique TRAIN sources and seven updates**, with 38,333 state-action rows:

| Grouping | State-action rows | Stored observation counts | Single-observation rows |
| --- | ---: | ---: | ---: |
| Fine | 17,588 | 17,588 | 17,588 |
| Middle | 16,252 | 17,580 | 15,769 |
| Coarse | 4,493 | 14,991 | 3,280 |

The 50,159 stored counts include repeated bookkeeping across grouping levels; they are **not 50,159 independent Games**. Observed remaining returns range from −4 to 24. Fine groups are sparse, so shared coarse groups matter. Row counts alone do not show how often a complete legal-action set uses learned estimates, and neither coverage nor changed estimates proves stronger play.

All 896 original TRAIN Games were actually rerun under the current source with their original frozen policies and seeds. They matched the original traces except for permitted code-fingerprint stamps, passed strict replay, and reconstructed the saved checkpoint bytes. That certification adds zero learning samples. Separately, all 512 coordinates in the repaired engine's earlier diagnostic pool passed; those reused seeds are engineering evidence, not independent performance samples.

The additional 128 Games all completed and passed batch checks, committing one update. Stored hierarchical observation counts increased from 50,159 to 56,600. Of 2,275 collected decision probes, 2,274 showed changed mean estimates; 492 of 2,076 multi-action probes changed their greedy preference. These are **in-sample update checks**, not independent decision coverage or score-improvement evidence. The original 896-Game prefix and all failed attempts remain retained.

## Reproducibility boundary

The small source handoff runs a complete-game demo and an independent four-Game learning example. Historical weights, original databases, and full research replays remain local. The public exports reproduce their supplied score statistics; they cannot by themselves recreate the original training archive or authenticate an unavailable trace. The source and test payload was separately installed and checked with 1,336 passing portable tests.
