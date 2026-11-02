# H4 stage4.3 L1 results
Status: complete; stage4.4 waits for explicit continue.
All values below come from saved terminal metrics, paired draws, round metadata and request ledgers. No report-time fit, feature execution, TEST access or resampling occurs.
Fixed TEST1000; 333 original template groups; 2000 shared group bootstrap draws.
MAE/RMSE units: eV/atom. R2 is dimensionless. Seed means are arithmetic means of metrics, never prediction ensembles.

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

Memory tokens use the pre-frozen local counter on the exact retained canonical JSON. Memory UTF8 bytes and complete-request conservative UTF8 accounting are separate quantities; byte bounds are not measured model-token counts.
|Seed|Requests|Max memory tokens|Max memory bytes|Prescribed trims|Max ordinary bound bytes|Max repair bound bytes|
|---|---|---|---|---|---|---|
|0|20|300|1600|2|20200|20355|
|1|20|300|1600|2|20200|20355|
|2|20|300|1600|2|20200|20355|

The full 60-request size ledger is [L1_request_input_ledger.csv](L1_request_input_ledger.csv).
Frozen L1 complete-input bound: 70000 bytes.
Frozen accounting method: SYNTHETIC immutable UTF8 accounting
L0 raised its limit during its round7 manual recovery. The L1 limit and memory counter were set before the first L1 call; old L0 prompts, results and rules are preserved.

Recorded pre-freeze measurement basis (displayed unchanged):
```json
{
  "measurement": "SYNTHETIC fixture only",
  "estimated_bytes": 63000,
  "formula": "recorded estimate plus margin, rounded before freeze"
}
```

## Calls, tokens and elapsed time

Elapsed run time: 1234.500 seconds. Prior O/L0 CLI attempts:120; L1 attempts:60; global attempts:180/300, including repairs and retries.
|Scope|Input tokens|Output tokens|Cached input tokens|
|---|---|---|---|
|L1 ledger|30000|40000|100|
|Global H4 ledger|31000|42000|100|

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
