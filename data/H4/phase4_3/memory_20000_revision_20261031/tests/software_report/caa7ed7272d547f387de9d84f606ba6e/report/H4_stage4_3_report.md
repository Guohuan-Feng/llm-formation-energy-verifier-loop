# H4 stage4.3 L1R2 revised results
Status: complete; stage4.4 waits for explicit continue.
All values below come from saved terminal metrics, paired draws, round metadata and request ledgers. No report-time fit, feature execution, TEST access or resampling occurs.
Fixed TEST1000; 333 original template groups; 2000 shared group bootstrap draws.
MAE/RMSE units: eV/atom. R2 is dimensionless. Seed means are arithmetic means of metrics, never prediction ensembles.

## Original freeze and later authorized revision

Original L1 freeze UTC: 2026-10-31T19:13:44+00:00; original config SHA256: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.
Authorized revision UTC: 2026-10-31T22:00:00+00:00. Reason: SYNTHETIC explicit memory revision
The original first three L1 responses are preserved but voided for this revised experiment. All60 formal L1 requests restart from S0 in the new L1R2 namespace; the original successful responses are not reused as revised rounds.
Only metadata memory content, its20000-token cap and the necessary memory-description text changed. The whole-input189000-byte guard, tokenizer/encoding, schema, query fields, HGB, seeds and template split rules remain unchanged.
Archive manifest: C:\Users\28908\Codex\fung-research\v2-diagnosis\work\h-data\H4\phase4_3\memory_20000_revision_20261031\tests\software_report\caa7ed7272d547f387de9d84f606ba6e\mock_archive_manifest.json

|Voided original request|Physical call index|Prompt SHA256|Canonical prompt SHA256|Validated response SHA256|
|---|---|---|---|---|
|L1_seed0_round01|121|e5f018aaa6795d52421a120d564534acdbf281b82130f9312966fcc2d379fb26|1a2ea3411c4976bfc09160d6bf0ead0a89f118105330dd57aa58eae6dd613437|1cf683ef507a9211ddbef9ab755ee36c5593ea256eb093ca06daa15e3b28ef7f|
|L1_seed1_round01|122|55ad39637240dea46f4c28ebcdabe4381bb553c6c0b7dae61a41a9745f0c7eba|962d5d7846b55bf7107a179f114c71d2fd539ad48da9becbf60804fb14f00c4d|a5227560d16ecd7f298d98d8077eda6bd9ca9761b4aecfc9e0e97c49fd57c4cf|
|L1_seed2_round01|123|e4fddde94f6fa5b738f914e8d5952fb942d17bd5946bb628d805349fb563a038|fc61786e53ceb3bf85f4c4e6455401225b359bad2a3fe71cefd4f59307be85b2|67b3eeb17e0f285947e9288a3622ba47c5ad01f8bc4216aaa96b3560a7ac1f1c|

## L1 terminal metrics

|Weight|Seed|MAE [95% CI]|RMSE [95% CI]|R2 [95% CI]|
|---|---|---|---|---|
|row_weighted|0|0.200000 [0.190000, 0.210000]|0.300000 [0.290000, 0.310000]|0.800000 [0.790000, 0.810000]|
|row_weighted|1|0.201000 [0.191000, 0.211000]|0.301000 [0.291000, 0.311000]|0.801000 [0.791000, 0.811000]|
|row_weighted|2|0.202000 [0.192000, 0.212000]|0.302000 [0.292000, 0.312000]|0.802000 [0.792000, 0.812000]|
|row_weighted|mean|0.201000 [0.191000, 0.211000]|0.301000 [0.291000, 0.311000]|0.801000 [0.791000, 0.811000]|
|group_weighted|0|0.200000 [0.190000, 0.210000]|0.300000 [0.290000, 0.310000]|0.800000 [0.790000, 0.810000]|
|group_weighted|1|0.201000 [0.191000, 0.211000]|0.301000 [0.291000, 0.311000]|0.801000 [0.791000, 0.811000]|
|group_weighted|2|0.202000 [0.192000, 0.212000]|0.302000 [0.292000, 0.312000]|0.802000 [0.792000, 0.812000]|
|group_weighted|mean|0.201000 [0.191000, 0.211000]|0.301000 [0.291000, 0.311000]|0.801000 [0.791000, 0.811000]|

## Prespecified paired differences

Definition: arm A minus arm B on the same TEST identities and shared stored group draws. Negative MAE/RMSE and positive R2 favor A. No new bootstrap draws or TEST reads were made.
L0-O six-decimal reference check: PASS.

### L1 minus L0

|Weight|Seed|Delta MAE [95% CI]|Delta RMSE [95% CI]|Delta R2 [95% CI]|
|---|---|---|---|---|
|row_weighted|0|+0.198000 [+0.188000, +0.208000]|+0.298000 [+0.288000, +0.308000]|+0.798000 [+0.788000, +0.808000]|
|row_weighted|1|+0.197900 [+0.187900, +0.207900]|+0.297900 [+0.287900, +0.307900]|+0.797900 [+0.787900, +0.807900]|
|row_weighted|2|+0.197800 [+0.187800, +0.207800]|+0.297800 [+0.287800, +0.307800]|+0.797800 [+0.787800, +0.807800]|
|row_weighted|mean|+0.197900 [+0.187900, +0.207900]|+0.297900 [+0.287900, +0.307900]|+0.797900 [+0.787900, +0.807900]|
|group_weighted|0|+0.198000 [+0.188000, +0.208000]|+0.298000 [+0.288000, +0.308000]|+0.798000 [+0.788000, +0.808000]|
|group_weighted|1|+0.197900 [+0.187900, +0.207900]|+0.297900 [+0.287900, +0.307900]|+0.797900 [+0.787900, +0.807900]|
|group_weighted|2|+0.197800 [+0.187800, +0.207800]|+0.297800 [+0.287800, +0.307800]|+0.797800 [+0.787800, +0.807800]|
|group_weighted|mean|+0.197900 [+0.187900, +0.207900]|+0.297900 [+0.287900, +0.307900]|+0.797900 [+0.787900, +0.807900]|

### L1 minus O

|Weight|Seed|Delta MAE [95% CI]|Delta RMSE [95% CI]|Delta R2 [95% CI]|
|---|---|---|---|---|
|row_weighted|0|+0.198000 [+0.188000, +0.208000]|+0.298000 [+0.288000, +0.308000]|+0.798000 [+0.788000, +0.808000]|
|row_weighted|1|+0.197900 [+0.187900, +0.207900]|+0.297900 [+0.287900, +0.307900]|+0.797900 [+0.787900, +0.807900]|
|row_weighted|2|+0.197800 [+0.187800, +0.207800]|+0.297800 [+0.287800, +0.307800]|+0.797800 [+0.787800, +0.807800]|
|row_weighted|mean|+0.197900 [+0.187900, +0.207900]|+0.297900 [+0.287900, +0.307900]|+0.797900 [+0.787900, +0.807900]|
|group_weighted|0|+0.198000 [+0.188000, +0.208000]|+0.298000 [+0.288000, +0.308000]|+0.798000 [+0.788000, +0.808000]|
|group_weighted|1|+0.197900 [+0.187900, +0.207900]|+0.297900 [+0.287900, +0.307900]|+0.797900 [+0.787900, +0.807900]|
|group_weighted|2|+0.197800 [+0.187800, +0.207800]|+0.297800 [+0.287800, +0.307800]|+0.797800 [+0.787800, +0.807800]|
|group_weighted|mean|+0.197900 [+0.187900, +0.207900]|+0.297900 [+0.287900, +0.307900]|+0.797900 [+0.787900, +0.807900]|

### L0 minus O

|Weight|Seed|Delta MAE [95% CI]|Delta RMSE [95% CI]|Delta R2 [95% CI]|
|---|---|---|---|---|
|row_weighted|0|+0.198000 [+0.188000, +0.208000]|+0.298000 [+0.288000, +0.308000]|+0.798000 [+0.788000, +0.808000]|
|row_weighted|1|+0.197900 [+0.187900, +0.207900]|+0.297900 [+0.287900, +0.307900]|+0.797900 [+0.787900, +0.807900]|
|row_weighted|2|+0.197800 [+0.187800, +0.207800]|+0.297800 [+0.287800, +0.307800]|+0.797800 [+0.787800, +0.807800]|
|row_weighted|mean|+0.197900 [+0.187900, +0.207900]|+0.297900 [+0.287900, +0.307900]|+0.797900 [+0.787900, +0.807900]|
|group_weighted|0|+0.198000 [+0.188000, +0.208000]|+0.298000 [+0.288000, +0.308000]|+0.798000 [+0.788000, +0.808000]|
|group_weighted|1|+0.197900 [+0.187900, +0.207900]|+0.297900 [+0.287900, +0.307900]|+0.797900 [+0.787900, +0.807900]|
|group_weighted|2|+0.197800 [+0.187800, +0.207800]|+0.297800 [+0.287800, +0.307800]|+0.797800 [+0.787800, +0.807800]|
|group_weighted|mean|+0.197900 [+0.187900, +0.207900]|+0.297900 [+0.287900, +0.307900]|+0.797900 [+0.787900, +0.807900]|

## Labels and final features

|Seed|Query labels|Uniform fill|Final labels|Winner round|Raw features|Kept features|
|---|---|---|---|---|---|---|
|0|1500|0|2000|20|3|2|
|1|1450|50|2000|20|3|2|
|2|1500|0|2000|17|3|2|

Failure rates below include all raw generated functions, including functions rejected by the AST policy. Kept rates are the mean of the saved per-function rates for final retained names; TEST failures are diagnostic only.

|Seed|Domain|Raw failed/total cells|Raw failure rate|Kept failure rate|Status counts|
|---|---|---|---|---|---|
|0|TRAIN|2020/6000|0.336667|0.005000|ast_rejected=2000, exception=20, ok=3980|
|0|TEST|1025/3000|0.341667|0.012500|ast_rejected=1000, exception=25, ok=1975|
|1|TRAIN|2020/6000|0.336667|0.005000|ast_rejected=2000, exception=20, ok=3980|
|1|TEST|1025/3000|0.341667|0.012500|ast_rejected=1000, exception=25, ok=1975|
|2|TRAIN|2020/6000|0.336667|0.005000|ast_rejected=2000, exception=20, ok=3980|
|2|TEST|1025/3000|0.341667|0.012500|ast_rejected=1000, exception=25, ok=1975|

## Memory and complete request sizes

Memory tokens use the unchanged pinned local counter on the exact retained metadata-only canonical JSON. Historical source/name/description/source summaries are absent from memory; current active feature source remains in its separate unchanged prompt section. Recent3 query/statistics/note records are retained; top30 source-hash ranking and historical-instance membership preserve the prior algorithm. Exact canonical/wire prompt and memory UTF8 sizes are separate from the whole-input bound, which adds fixed context and a1500-byte framing allowance. Byte bounds are not measured model-token counts.
|Seed|Requests|Max memory tokens|Max memory bytes|Prescribed trims|Max ordinary bound bytes|Max repair bound bytes|
|---|---|---|---|---|---|---|
|0|20|15000|1600|2|20200|20355|
|1|20|15000|1600|2|20200|20355|
|2|20|15000|1600|2|20200|20355|

The full60-request size ledger is [L1_request_input_ledger.csv](L1_request_input_ledger.csv), including each exact canonical/wire payload byte count, memory token/byte count, trim flag, retained prior-note characters and current response-note Unicode character count. The current response note is not part of its own request memory.
Frozen L1 complete-input bound: 189000 bytes.
Frozen accounting method: SYNTHETIC immutable UTF8 accounting
Active revised memory cap:20000 tokens. The same pinned tiktoken0.12.0/o200k_base.encode_ordinary counter and complete canonical-JSON counting method are retained.
The unchanged counter verification record still contains its original6000-token limit field as legacy provenance; that historical field is not the active revised memory cap.
Unchanged token-counter verification record SHA256: ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff.
L0 raised its input limit during round7 manual recovery. Original L1 was frozen and made3 calls before this explicit revision. This later revision is frozen before the new L1R2 run; it was not part of the original freeze. O/L0 results and original L1 records remain preserved.

Recorded pre-revision-freeze measurement basis (displayed unchanged):
```json
{
  "measurement": "SYNTHETIC fixture only",
  "estimated_bytes": 63000,
  "formula": "recorded estimate plus margin, rounded before revision freeze"
}
```

## Calls, tokens and elapsed time

Elapsed revised run time: 1234.500 seconds. Prior O/L0 physical attempts:120; voided original L1 attempts:3; valid L1R2 logical requests:60; revised L1 physical attempts:60; all L1 physical attempts:63; global physical attempts:183/300.
Formal valid logical requests across O/L0/L1R2:180. Global physical attempts excluding the voided3:180; revised repair/retry extra attempts:0. With no extra attempts these counts are183 physical,180 formal valid and63 physical L1 versus60 valid revised L1.
Reported provider token sums include every completed physical attempt in each scope, including the voided3 in all-L1/global scopes. Unknown usage is separately counted and is not claimed to be zero.
|Scope|Input tokens, known sum|Output tokens, known sum|Cached input tokens, known sum|Results with unknown usage|
|---|---|---|---|---|
|Valid revised L1R2|60000|120000|6000|0|
|Voided original L1|3000|6000|300|0|
|All physical L1|63000|126000|6300|0|
|Global H4 physical ledger|183000|366000|18300|0|

## Interpretation and audit limits

- O, L0 and L1 each have one independent terminal TEST1000 pass; the total is three. This report reads no TEST labels and does not rescore O or L0.
- Validation labels enter feedback and memory by design; they are not an untouched-label holdout. Validation membership grows, so aligned round curves compare different revealed sets.
- L1-L0 reflects the complete memory, feedback and query trajectories; it does not isolate a causal effect of notes alone.
- LOFO contributions below are saved winner-round internal-validation deltas. They do not use TEST performance, and no new LOFO fits were done for this report.
- The winner and its saved validation score were fixed before uniform fill. TEST feature failure rates did not select or rank features.
- Sample membership is seeded; local CLI output reproducibility is not claimed.

## Winner-round validation contributions and full source

Each seed lists up to five final-retained functions with finite saved LOFO deltas, descending delta; ties use original feature index. Positive delta means omission increased validation MAE. Negative deltas are retained in the ranking rather than hidden.

### Seed 0

Winner round:20; saved validation row MAE:0.200000000.

#### 1. sites

MAE_without minus saved full-model MAE: +0.002000 eV/atom.
Source SHA256: c937c1c58424b409f3c549c0834d0710a25115ab7a03a7dfe25c76f76609a1d0
```python
def feature(s):
    return float(s['n_sites'])

```

#### 2. volume

MAE_without minus saved full-model MAE: -0.001000 eV/atom.
Source SHA256: 34b3808ebbdb78a239a3a11abafb561eca641c5c2425070601502e6d9582faed
```python
def feature(s):
    return float(s['volume_per_atom_A3'])

```

### Seed 1

Winner round:20; saved validation row MAE:0.203000000.

#### 1. sites

MAE_without minus saved full-model MAE: +0.002000 eV/atom.
Source SHA256: c937c1c58424b409f3c549c0834d0710a25115ab7a03a7dfe25c76f76609a1d0
```python
def feature(s):
    return float(s['n_sites'])

```

#### 2. volume

MAE_without minus saved full-model MAE: -0.001000 eV/atom.
Source SHA256: 34b3808ebbdb78a239a3a11abafb561eca641c5c2425070601502e6d9582faed
```python
def feature(s):
    return float(s['volume_per_atom_A3'])

```

### Seed 2

Winner round:17; saved validation row MAE:0.206000000.

#### 1. sites

MAE_without minus saved full-model MAE: +0.002000 eV/atom.
Source SHA256: c937c1c58424b409f3c549c0834d0710a25115ab7a03a7dfe25c76f76609a1d0
```python
def feature(s):
    return float(s['n_sites'])

```

#### 2. volume

MAE_without minus saved full-model MAE: -0.001000 eV/atom.
Source SHA256: 34b3808ebbdb78a239a3a11abafb561eca641c5c2425070601502e6d9582faed
```python
def feature(s):
    return float(s['volume_per_atom_A3'])

```
