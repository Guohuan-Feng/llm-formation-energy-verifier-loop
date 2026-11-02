# H4 stage4.1: O one-shot feature generation
Status: complete; waiting for continue; stage4.1 only; stages4.2/4.3/4.4 wait for explicit continue.
Seeds0/1/2 reuse H2/H3 frozen S0500 plus uniform-random1500 POOL labels,2000 total; full synthetic TEST1000.
Each seed has20 independent candidates, each with exactly300 sampled labeled examples plus TRAIN2000 summary, <=20 functions.
As requested, examples/summary may include internal validation labels; this validation is not an untouched-label holdout.
No TEST/HOLDOUT label enters prompts, features, queries or selection; original label-free H3 geometry/Magpie132 cache is reused.
Fixed template-group80/20 internal split per seed; candidate selection uses row MAE, ties earliest candidate.
HGB uses original H3 unweighted fit and frozen parameters; TRAIN-only median imputation.
Magpie132 plus candidate functions; failure>20% on inner-fit TRAIN drops a function for validation.
After selection, recompute the failure mask on fullTRAIN2000; reuse cached TRAIN feature values, refit, predict TEST once.
H4-private H3b batch sandbox copy only;2 workers,50ms GO-to-result per function/structure; timeout/error→NaN.
Pinned CLI gpt-6.1-sol/high, no tools/network, at most4 calls; no claim of reproducible LLM output seed.
All60 prompts,300-example memberships, schemas, data/split/source hashes frozen before the first call; <=20000 input bound.
Global H4 call cap300 including repairs/retries;429 retries after60/120/240s; finite manual recovery only.
Terminal scoring: one TEST1000 read for all3 seeds; row/fullTEST-template group weights,2000 shared group bootstraps.
Per-seed metrics and arithmetic means of metrics, not an ensemble; private paired draws retained for later authorized comparisons.
Pilot8 requests plus their candidate validations estimate the full stage; stop if estimate exceeds10 hours.
H4A and all previous experiments remain read-only. Stage4.1 makes no query or real-data transfer test.
Pilot stage forecast: 1.373 h; measured pilot, not measured whole-stage runtime.
