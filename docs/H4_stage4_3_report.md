# H4 stage4.3 L1R2 revised results
Status: complete; stage4.4 waits for explicit continue.
All values below come from saved terminal metrics, paired draws, round metadata and request ledgers. No report-time fit, feature execution, TEST access or resampling occurs.
Fixed TEST1000; 333 original template groups; 2000 shared group bootstrap draws.
MAE/RMSE units: eV/atom. R2 is dimensionless. Seed means are arithmetic means of metrics, never prediction ensembles.

## Original freeze and later authorized revision

Original L1 freeze UTC: 2026-10-31T19:13:43.077582+00:00; original config SHA256: ef187d78cebac452ff7f6f2b71c2350c0519393dbf742ca77286e0edd74396b5.
Authorized revision UTC: 2026-10-31T22:12:19.251306+00:00. Reason: Original seed0 round2 memory11873>6000; metadata-only 57-prefix preflight still max18057>6000. User authorized metadata-only memory20000 and truthful memory-description words.
The original first three L1 responses are preserved but voided for this revised experiment. All60 formal L1 requests restart from S0 in the new L1R2 namespace; the original successful responses are not reused as revised rounds.
Only metadata memory content, its20000-token cap and the necessary memory-description text changed. The whole-input189000-byte guard, tokenizer/encoding, schema, query fields, HGB, seeds and template split rules remain unchanged.
Archive manifest: C:\Users\28908\Codex\fung-research\v2-diagnosis\work\h-data\H4\phase4_3\voided_original_round1_20261031\manifest.json

|Voided original request|Physical call index|Prompt SHA256|Canonical prompt SHA256|Validated response SHA256|
|---|---|---|---|---|
|L1_seed0_round01|121|4ee26b9b15fb2e84cf0f24382fa7cdaf90c2c1af023531a7b36a0eadefac3129|4ee26b9b15fb2e84cf0f24382fa7cdaf90c2c1af023531a7b36a0eadefac3129|209b3ba481f5dd7b5cad2d2565f6fe4e797204a35af5707133229461a9652289|
|L1_seed1_round01|122|10b4933b841ebd66e2f7e8b5f564b4f606be543f34e6ef24c0bf0f39df8e4f58|10b4933b841ebd66e2f7e8b5f564b4f606be543f34e6ef24c0bf0f39df8e4f58|56713805129208737886e51f8a371adccf958ae9adfdada940f907a2451fec8b|
|L1_seed2_round01|123|79636097df3853a55177b5c557f1974692d59536b4af64a327c0b9f06de70da8|79636097df3853a55177b5c557f1974692d59536b4af64a327c0b9f06de70da8|8890ccca79984037c2a1c6f4f6aea8eb625b7822da16cc022a8df579aceda80f|

## L1 terminal metrics

|Weight|Seed|MAE [95% CI]|RMSE [95% CI]|R2 [95% CI]|
|---|---|---|---|---|
|row_weighted|0|0.204726 [0.191468, 0.219048]|0.271859 [0.252960, 0.291714]|0.934445 [0.922861, 0.943461]|
|row_weighted|1|0.210383 [0.195510, 0.225957]|0.289199 [0.264222, 0.314684]|0.925816 [0.910563, 0.938395]|
|row_weighted|2|0.210755 [0.197052, 0.225565]|0.285484 [0.263461, 0.310067]|0.927710 [0.914554, 0.939563]|
|row_weighted|mean|0.208621 [0.196488, 0.222161]|0.282181 [0.262281, 0.303787]|0.929324 [0.917002, 0.939513]|
|group_weighted|0|0.215682 [0.200692, 0.232253]|0.286839 [0.264324, 0.310023]|0.930647 [0.916926, 0.941923]|
|group_weighted|1|0.223577 [0.205123, 0.242238]|0.308418 [0.278494, 0.337611]|0.919819 [0.901202, 0.934279]|
|group_weighted|2|0.224689 [0.208045, 0.242097]|0.304466 [0.278558, 0.329808]|0.921861 [0.906214, 0.934518]|
|group_weighted|mean|0.221316 [0.206153, 0.237231]|0.299908 [0.275860, 0.323745]|0.924109 [0.909117, 0.935496]|

## Prespecified paired differences

Definition: arm A minus arm B on the same TEST identities and shared stored group draws. Negative MAE/RMSE and positive R2 favor A. No new bootstrap draws or TEST reads were made.
L0-O six-decimal reference check: PASS.

### L1 minus L0

|Weight|Seed|Delta MAE [95% CI]|Delta RMSE [95% CI]|Delta R2 [95% CI]|
|---|---|---|---|---|
|row_weighted|0|-0.002442 [-0.012623, +0.007721]|-0.004445 [-0.018363, +0.009235]|+0.002161 [-0.004498, +0.008926]|
|row_weighted|1|+0.006797 [-0.004823, +0.018136]|+0.020641 [+0.003112, +0.037724]|-0.010211 [-0.019762, -0.001540]|
|row_weighted|2|-0.010801 [-0.022715, +0.001353]|-0.016434 [-0.035701, +0.002436]|+0.008562 [-0.001334, +0.019020]|
|row_weighted|mean|-0.002149 [-0.008738, +0.004175]|-0.000079 [-0.010720, +0.009349]|+0.000171 [-0.004795, +0.005413]|
|group_weighted|0|+0.002421 [-0.008613, +0.013491]|+0.002680 [-0.012410, +0.019046]|-0.001290 [-0.009130, +0.006107]|
|group_weighted|1|+0.010585 [-0.002934, +0.024006]|+0.023808 [+0.003841, +0.044139]|-0.011901 [-0.023831, -0.001825]|
|group_weighted|2|-0.008714 [-0.022961, +0.004436]|-0.015529 [-0.040742, +0.007215]|+0.008174 [-0.003796, +0.020901]|
|group_weighted|mean|+0.001431 [-0.006507, +0.009364]|+0.003653 [-0.009255, +0.016180]|-0.001672 [-0.008593, +0.004972]|

### L1 minus O

|Weight|Seed|Delta MAE [95% CI]|Delta RMSE [95% CI]|Delta R2 [95% CI]|
|---|---|---|---|---|
|row_weighted|0|-0.014254 [-0.026661, -0.002800]|-0.026539 [-0.046107, -0.007231]|+0.013424 [+0.003542, +0.024033]|
|row_weighted|1|-0.005189 [-0.015939, +0.006434]|-0.008093 [-0.025455, +0.009053]|+0.004210 [-0.004882, +0.013186]|
|row_weighted|2|+0.003618 [-0.008846, +0.015247]|+0.006918 [-0.008943, +0.022683]|-0.003461 [-0.011606, +0.004630]|
|row_weighted|mean|-0.005275 [-0.013613, +0.002602]|-0.009238 [-0.021751, +0.002899]|+0.004724 [-0.001495, +0.011249]|
|group_weighted|0|-0.010829 [-0.023522, +0.001997]|-0.022077 [-0.040191, -0.002314]|+0.011086 [+0.001257, +0.020695]|
|group_weighted|1|-0.001334 [-0.014209, +0.011641]|-0.004401 [-0.024053, +0.015346]|+0.002305 [-0.008622, +0.012547]|
|group_weighted|2|+0.008889 [-0.004500, +0.021393]|+0.012304 [-0.004537, +0.028570]|-0.006188 [-0.015029, +0.002240]|
|group_weighted|mean|-0.001091 [-0.009915, +0.007640]|-0.004725 [-0.017394, +0.008482]|+0.002401 [-0.004546, +0.008912]|

### L0 minus O

|Weight|Seed|Delta MAE [95% CI]|Delta RMSE [95% CI]|Delta R2 [95% CI]|
|---|---|---|---|---|
|row_weighted|0|-0.011812 [-0.023744, -0.000675]|-0.022093 [-0.039673, -0.004081]|+0.011262 [+0.002077, +0.020932]|
|row_weighted|1|-0.011986 [-0.024331, +0.000177]|-0.028733 [-0.048275, -0.011177]|+0.014421 [+0.005548, +0.024989]|
|row_weighted|2|+0.014419 [+0.003056, +0.025433]|+0.023352 [+0.006985, +0.039997]|-0.012023 [-0.021375, -0.003466]|
|row_weighted|mean|-0.003126 [-0.010755, +0.004133]|-0.009158 [-0.021304, +0.002295]|+0.004553 [-0.001274, +0.010892]|
|group_weighted|0|-0.013250 [-0.026561, -0.001114]|-0.024757 [-0.043774, -0.007293]|+0.012376 [+0.003525, +0.022417]|
|group_weighted|1|-0.011919 [-0.024792, +0.000861]|-0.028209 [-0.047355, -0.010003]|+0.014206 [+0.004996, +0.024708]|
|group_weighted|2|+0.017604 [+0.004459, +0.031016]|+0.027833 [+0.007150, +0.050165]|-0.014362 [-0.026361, -0.003758]|
|group_weighted|mean|-0.002522 [-0.010568, +0.005686]|-0.008378 [-0.021102, +0.004061]|+0.004073 [-0.002422, +0.010818]|

## Labels and final features

|Seed|Query labels|Uniform fill|Final labels|Winner round|Raw features|Kept features|
|---|---|---|---|---|---|---|
|0|1500|0|2000|19|20|16|
|1|1500|0|2000|20|20|19|
|2|1500|0|2000|20|14|11|

Failure rates below include all raw generated functions, including functions rejected by the AST policy. Kept rates are the mean of the saved per-function rates for final retained names; TEST failures are diagnostic only.

|Seed|Domain|Raw failed/total cells|Raw failure rate|Kept failure rate|Status counts|
|---|---|---|---|---|---|
|0|TRAIN|8000/40000|0.200000|0.000000|ast_rejected=8000, ok=32000|
|0|TEST|4000/20000|0.200000|0.000000|ast_rejected=4000, ok=16000|
|1|TRAIN|2000/40000|0.050000|0.000000|ast_rejected=2000, ok=38000|
|1|TEST|1000/20000|0.050000|0.000000|ast_rejected=1000, ok=19000|
|2|TRAIN|6000/28000|0.214286|0.000000|ast_rejected=6000, ok=22000|
|2|TEST|3000/14000|0.214286|0.000000|ast_rejected=3000, ok=11000|

## Memory and complete request sizes

Memory tokens use the unchanged pinned local counter on the exact retained metadata-only canonical JSON. Historical source/name/description/source summaries are absent from memory; current active feature source remains in its separate unchanged prompt section. Recent3 query/statistics/note records are retained; top30 source-hash ranking and historical-instance membership preserve the prior algorithm. Exact canonical/wire prompt and memory UTF8 sizes are separate from the whole-input bound, which adds fixed context and a1500-byte framing allowance. Byte bounds are not measured model-token counts.
|Seed|Requests|Max memory tokens|Max memory bytes|Prescribed trims|Max ordinary bound bytes|Max repair bound bytes|
|---|---|---|---|---|---|---|
|0|20|18809|42307|5|82566|82721|
|1|20|19636|44178|2|84440|84595|
|2|20|19887|44622|1|74969|75124|

The full60-request size ledger is [L1_request_input_ledger.csv](L1_request_input_ledger.csv), including each exact canonical/wire payload byte count, memory token/byte count, trim flag, retained prior-note characters and current response-note Unicode character count. The current response note is not part of its own request memory.
Frozen L1 complete-input bound: 189000 bytes.
Frozen accounting method: Actual UTF8 strict wire schema+wire note+global/model instructions+1500 framing; ordinary and repair checked
Active revised memory cap:20000 tokens. The same pinned tiktoken0.12.0/o200k_base.encode_ordinary counter and complete canonical-JSON counting method are retained.
The unchanged counter verification record still contains its original6000-token limit field as legacy provenance; that historical field is not the active revised memory cap.
Unchanged token-counter verification record SHA256: f374eab336ae08f9bcb8a58cf0f8346c596ff8eb99e1b37357840f7393e6e99f.
L0 raised its input limit during round7 manual recovery. Original L1 was frozen and made3 calls before this explicit revision. This later revision is frozen before the new L1R2 run; it was not part of the original freeze. O/L0 results and original L1 records remain preserved.

Recorded pre-revision-freeze measurement basis (displayed unchanged):
```json
{
  "L1_config_not_yet_frozen": true,
  "LOFO_projection_index": {
    "path": "C:\\Users\\28908\\Codex\\fung-research\\v2-diagnosis\\work\\h-data\\H4\\phase4_3\\projection_data\\projection_index.json",
    "sha256": "d663045b66a95df77ca9ba2dbd8d43bd1ed66ba2da606fcb9510f80ef90146bf"
  },
  "actual_L0_notes": "All observed notes are empty; no synthetic future L1 notes are presented as observed data. All notes will be retained at runtime.",
  "created_client_time": "2026-10-31T15:11:30.468347-04:00",
  "created_utc": "2026-10-31T19:11:30.440212+00:00",
  "fixed_L1_extra_UTF8_bytes": 813,
  "fixed_L1_extra_components": {
    "ending_delta_UTF8": 6,
    "grammar_delta_UTF8": 25,
    "memory_JSON_trailing_prompt_newline_UTF8": 1,
    "memory_explanation_UTF8": 542,
    "memory_label_UTF8": 92,
    "prefix_delta_UTF8": 97,
    "strict_wire_schema_delta_UTF8": 50
  },
  "future_L1_notes_size": "Unknown; future real memory and request guards remain mandatory.",
  "margin_factor": 1.1,
  "max_memory_JSON_view": {
    "after_final_round_not_a_request": false,
    "all_historical_notes_and_query_statistics_preserved": true,
    "archive_sha256": "1fef50e817da37f2a84e29661739e5e5ee8dc632c24301336a2cb0762cb0af0d",
    "complete_recent_rounds": [
      5,
      6,
      7
    ],
    "completed_history_rounds": 7,
    "corresponding_next_request_round": 8,
    "memory_JSON_UTF8_bytes": 139379,
    "memory_overflow_6000": true,
    "memory_payload_sha256": "18f7627ed2292f2cba388561d0a5cf9645721b0a5d09d9ca634aca5cfdd656d0",
    "memory_tokens": 49796,
    "memory_trimmed": true,
    "seed": 2,
    "source_definitions_after_trim": 90,
    "source_definitions_before_trim": 140,
    "source_or_note_truncation": false,
    "untrimmed_memory_JSON_UTF8_bytes": 209390,
    "untrimmed_memory_tokens": 74693
  },
  "measured_memory_JSON_UTF8_max": 139379,
  "measured_request_history_prefixes": 60,
  "measurement_audit": {
    "path": "C:\\Users\\28908\\Codex\\fung-research\\v2-diagnosis\\work\\h-data\\H4\\phase4_3\\software\\memory_measurement_audit.json",
    "sha256": "42efc43da5587ece2bccc518b005dd6b2f06eac9dbe346983134f81c0939317c"
  },
  "memory_counter": {
    "TEST_targets_read": 0,
    "encoding": "o200k_base",
    "encoding_cache_files": 1,
    "limit_tokens": 6000,
    "scientific_LLM_calls": 0,
    "status": "PASS",
    "token_counter_sha256": "f374eab336ae08f9bcb8a58cf0f8346c596ff8eb99e1b37357840f7393e6e99f",
    "vendor_files": 151,
    "version": "0.12.0"
  },
  "memory_limit_tokens": 6000,
  "new_LLM_calls": 0,
  "new_TEST_target_accessors": 0,
  "old_L0_base_exact_fraction": "3233362/105",
  "old_L0_base_round20_UTF8_bytes": 30793.92380952381,
  "old_L0_input_projection_sha256": "c470adeedbf223b6bc2145398a16ed70d98a3d38e8ede1d83bdb4164a5370c63",
  "projected_exact_fraction": "17953522/105",
  "projected_maximum_request_UTF8_bytes": 170985.9238095238,
  "round_up_bytes": 1000,
  "runtime_policy": "Only actual emitted L1 requests are guarded. Prescribed memory retention still above6000 or actual complete request above selected limit stops the whole stage before dispatch; no further source/note truncation.",
  "selected_input_limit_bytes": 189000,
  "selection_formula": "ceil1000((old_L0_round20_base + directly_measured_memory_JSON_UTF8 + exact_L1_memory_wrapper_and_schema_delta)*1.10)",
  "source_maxLength_characters_per_feature": 20000,
  "stage": "4.3",
  "stage4_4_started": false,
  "status": "complete_pre_freeze_measurement",
  "step": 2,
  "surrogate_memory_may_exceed6000": "All57 nonempty hypothetical next-request L0 views exceed6000; they are measured size references, not actual L1 requests. This does not relax the L1 memory cap.",
  "theoretical_bounds_preserved_but_not_stop_conditions": true
}
```

## Calls, tokens and elapsed time

Elapsed revised run time: 9597.687 seconds. Prior O/L0 physical attempts:120; voided original L1 attempts:3; valid L1R2 logical requests:60; revised L1 physical attempts:60; all L1 physical attempts:63; global physical attempts:183/300.
Formal valid logical requests across O/L0/L1R2:180. Global physical attempts excluding the voided3:180; revised repair/retry extra attempts:0. With no extra attempts these counts are183 physical,180 formal valid and63 physical L1 versus60 valid revised L1.
Validated revised response requests:60; completed rounds with the unchanged admissible query-only empty-query fallback:0. Accepted fallbacks preserve valid features/notes and are not relabelled as fully validated provider responses.
Reported provider token sums include every completed physical attempt in each scope, including the voided3 in all-L1/global scopes. Unknown usage is separately counted and is not claimed to be zero.
|Scope|Input tokens, known sum|Output tokens, known sum|Cached input tokens, known sum|Results with unknown usage|
|---|---|---|---|---|
|Valid revised L1R2|1160798|617168|0|0|
|Voided original L1|11673|38865|0|0|
|All physical L1|1172471|656033|0|0|
|Global H4 physical ledger|2082407|1884035|0|0|

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

Winner round:19; saved validation row MAE:0.207343540.

#### 1. covalent_corrected_electron_balance

MAE_without minus saved full-model MAE: +0.009886 eV/atom.
Source SHA256: bce5946389375814337ee8a881b262bc1f47290feac51ef7d2eeeef3e98fc2c2
```python
def feature(s):
    if not s:
        return 0.0
    sites = s.get('sites') or []
    bonds = s.get('neighbors') or []
    props = s.get('element_properties') or {}
    n = len(sites)
    radii = [max(0.1, (props.get(x.get('element')) or {}).get('covalent_radius_A') or 1.3) for x in sites]
    occ = [max(0.0, x.get('occupancy', 1.0) or 0.0) for x in sites]
    total = 0.0
    for i in range(n):
        if occ[i] <= 0.0:
            continue
        p = props.get(sites[i].get('element')) or {}
        group = p.get('group')
        en = p.get('eneg')
        if group is None or en is None:
            continue
        if en < 2.0:
            if group == 1 or group == 2:
                total += occ[i] * group
            elif group == 13 or group == 14:
                total += occ[i] * (group - 10)
            continue
        if not (14 <= group <= 17):
            continue
        bond_sum = 0.0
        for b in bonds:
            if b.get('i', -1) != i:
                continue
            j = b.get('j', -1)
            d = b.get('distance_A') or 0.0
            if not (0 <= j < n and 0.0 < d <= 6.0) or occ[j] <= 0.0:
                continue
            q = props.get(sites[j].get('element')) or {}
            other_group = q.get('group')
            other_en = q.get('eneg')
            if other_group is None or other_en is None or other_en < 2.0:
                continue
            main_group = 14 <= other_group <= 17
            hydrogen = other_group == 1 and q.get('row') == 1
            if not (main_group or hydrogen):
                continue
            radius_sum = radii[i] + radii[j]
            if d <= 1.3 * radius_sum:
                ratio = min(2.0, radius_sum / d)
                bond_sum += occ[j] * min(3.0, ratio ** 6)
        total += occ[i] * (bond_sum - (18.0 - group))
    return total / max(sum(occ), 1e-12)
```

#### 2. atomic_hardness_polarization

MAE_without minus saved full-model MAE: +0.008031 eV/atom.
Source SHA256: a24a5bff78daead3f2a446f8d15a88a61774234764f3099c579a128c5ae63255
```python
def feature(s):
    if not s:
        return 0.0
    sites = s.get('sites') or []
    props = s.get('element_properties') or {}
    n = len(sites)
    radii = [max(0.1, (props.get(x.get('element')) or {}).get('covalent_radius_A') or 1.3) for x in sites]
    occ = [max(0.0, x.get('occupancy', 1.0) or 0.0) for x in sites]
    ion = [(props.get(x.get('element')) or {}).get('first_ionization_energy_eV') for x in sites]
    affinity = [(props.get(x.get('element')) or {}).get('electron_affinity_eV') for x in sites]
    total = 0.0
    for bond in s.get('neighbors') or []:
        i = bond.get('i', -1)
        j = bond.get('j', -1)
        d = bond.get('distance_A') or 0.0
        if not (0 <= i < n and 0 <= j < n and 0.0 < d < 6.0):
            continue
        if ion[i] is None or ion[j] is None or affinity[i] is None or affinity[j] is None:
            continue
        u = d / (1.5 * (radii[i] + radii[j]))
        w = occ[i] * occ[j] * max(0.0, 1.0 - u * u) ** 2
        contrast = 0.5 * (ion[i] + affinity[i] - ion[j] - affinity[j])
        hardness = max(0.5, ion[i] - affinity[i]) + max(0.5, ion[j] - affinity[j])
        total += w * contrast * contrast / hardness
    return -0.25 * total / max(sum(occ), 1e-12)
```

#### 3. occupied_covalent_packing_fraction

MAE_without minus saved full-model MAE: +0.005889 eV/atom.
Source SHA256: 20f16e86a128f59739168d3797f373487f69d7b8adbaa7e6ae3cc3ab641bed3d
```python
def feature(s):
    if not s:
        return 0.0
    sites = s.get('sites') or []
    if not sites:
        return 0.0
    props = s.get('element_properties') or {}
    volume = (s.get('lattice') or {}).get('volume_A3') or 0.0
    if volume <= 0.0:
        vpa = s.get('volume_per_atom_A3') or (s.get('basic_structure_statistics') or {}).get('volume_per_atom_A3') or 0.0
        volume = vpa * len(sites)
    if volume <= 0.0:
        return 0.0
    total = 0.0
    for site in sites:
        occ = max(0.0, site.get('occupancy', 1.0) or 0.0)
        radius = max(0.1, (props.get(site.get('element')) or {}).get('covalent_radius_A') or 1.3)
        total += occ * radius ** 3
    return 4.1887902047863905 * total / volume
```

#### 4. nearest_reduced_contact_heterogeneity

MAE_without minus saved full-model MAE: +0.005216 eV/atom.
Source SHA256: 12621277572610b3bdf5e0858202d44991314615d77d02fd3d081820c56cf782
```python
def feature(s):
    if not s:
        return 0.0
    sites = s.get('sites') or []
    bonds = s.get('neighbors') or []
    n = len(sites)
    props = s.get('element_properties') or {}
    radii = [max(0.1, (props.get(x.get('element')) or {}).get('covalent_radius_A') or 1.3) for x in sites]
    occ = [max(0.0, x.get('occupancy', 1.0) or 0.0) for x in sites]
    first = 0.0
    second = 0.0
    weight = 0.0
    for i in range(n):
        if occ[i] <= 0.0:
            continue
        best = 0.0
        for b in bonds:
            if b.get('i', -1) != i:
                continue
            j = b.get('j', -1)
            d = b.get('distance_A') or 0.0
            if 0 <= j < n and d > 0.0 and occ[j] > 0.0:
                reduced = d / (radii[i] + radii[j])
                if best == 0.0 or reduced < best:
                    best = reduced
        if best > 0.0:
            weight += occ[i]
            first += occ[i] * best
            second += occ[i] * best * best
    if weight <= 0.0 or first <= 0.0:
        return 0.0
    mean = first / weight
    return max(0.0, second / weight - mean * mean) ** 0.5 / mean
```

#### 5. soft_contact_ionicity

MAE_without minus saved full-model MAE: +0.004463 eV/atom.
Source SHA256: 13affba209a0647760afe89088f58a0d0e2f057169e1293a9a4c1ff19c76c9de
```python
def feature(s):
    if not s:
        return 0.0
    sites = s.get('sites') or []
    props = s.get('element_properties') or {}
    n = len(sites)
    radii = [max(0.1, (props.get(x.get('element')) or {}).get('covalent_radius_A') or 1.3) for x in sites]
    en = [(props.get(x.get('element')) or {}).get('eneg') for x in sites]
    occ = [max(0.0, x.get('occupancy', 1.0) or 0.0) for x in sites]
    total = 0.0
    weight = 0.0
    for b in s.get('neighbors') or []:
        i = b.get('i', -1)
        j = b.get('j', -1)
        d = b.get('distance_A') or 0.0
        if 0 <= i < n and 0 <= j < n and 0.0 < d < 6.0:
            if en[i] is not None and en[j] is not None:
                u = d / (1.5 * (radii[i] + radii[j]))
                w = occ[i] * occ[j] * max(0.0, 1.0 - u * u) ** 2
                contrast = (en[i] - en[j]) ** 2
                total += w * contrast / (1.0 + contrast)
                weight += w
    return total / max(weight, 1e-12)
```

### Seed 1

Winner round:20; saved validation row MAE:0.225256022.

#### 1. minimum_normalized_contact_separation

MAE_without minus saved full-model MAE: +0.007091 eV/atom.
Source SHA256: 7335ca689a253218559ab59751678393bd2ed179b0184d43ce5028a27f829d8a
```python
def feature(s):
    if not s:
        return 0.0
    sites = s.get('sites') or []
    edges = s.get('neighbors') or []
    props = s.get('element_properties') or {}
    n = len(sites)
    occ = [max(0.0, t.get('occupancy', 1.0) or 0.0) for t in sites]
    radii = [(props.get(t.get('element')) or {}).get('covalent_radius_A') for t in sites]
    total = 0.0
    result = 0.0
    for i in range(n):
        a = radii[i]
        if occ[i] > 0 and a is not None and a > 0:
            ratios = [edge.get('distance_A') / (a + radii[edge.get('j', -1)]) for edge in edges if edge.get('i', -1) == i and 0 <= edge.get('j', -1) < n and occ[edge.get('j', -1)] > 0 and radii[edge.get('j', -1)] is not None and radii[edge.get('j', -1)] > 0 and edge.get('distance_A') is not None and edge.get('distance_A') > 0]
            if ratios:
                result += occ[i] * min(ratios)
                total += occ[i]
    if total <= 0:
        return 0.0
    return result / total
```

#### 2. smooth_contact_coordination

MAE_without minus saved full-model MAE: +0.005480 eV/atom.
Source SHA256: 888b636875db2d4e729183882d25bc11e11e4881db8f40c70aab9ea433319172
```python
def feature(s):
    if not s:
        return 0.0
    sites = s.get('sites') or []
    props = s.get('element_properties') or {}
    n = len(sites)
    occ = [max(0.0, t.get('occupancy', 1.0) or 0.0) for t in sites]
    radii = [(props.get(t.get('element')) or {}).get('covalent_radius_A') for t in sites]
    atoms = sum(occ)
    if atoms <= 0:
        return 0.0
    total = 0.0
    for edge in s.get('neighbors') or []:
        i = edge.get('i', -1)
        j = edge.get('j', -1)
        d = edge.get('distance_A')
        if 0 <= i < n and 0 <= j < n and d is not None and d > 0:
            a = radii[i]
            b = radii[j]
            if a is not None and b is not None and a > 0 and b > 0:
                x = d / (a + b)
                x2 = x * x
                total += occ[i] * occ[j] / (1.0 + x2 * x2 * x2)
    return total / atoms
```

#### 3. coherent_local_electronegativity_contrast

MAE_without minus saved full-model MAE: +0.004970 eV/atom.
Source SHA256: a4a5a1e68768e38a68c1d3ea206c45165aa240a6dc4a6d07a4331deffc44559b
```python
def feature(s):
    if not s:
        return 0.0
    sites = s.get('sites') or []
    props = s.get('element_properties') or {}
    n = len(sites)
    occ = [max(0.0, t.get('occupancy', 1.0) or 0.0) for t in sites]
    radii = [(props.get(t.get('element')) or {}).get('covalent_radius_A') for t in sites]
    en = [(props.get(t.get('element')) or {}).get('eneg') for t in sites]
    pairs = [(edge.get('i', -1), edge.get('j', -1), edge.get('distance_A')) for edge in s.get('neighbors') or [] if 0 <= edge.get('i', -1) < n and 0 <= edge.get('j', -1) < n and edge.get('distance_A') is not None and edge.get('distance_A') > 0]
    valid = [(i, j, d) for i, j, d in pairs if occ[i] > 0 and occ[j] > 0 and radii[i] is not None and radii[j] is not None and radii[i] > 0 and radii[j] > 0 and en[i] is not None and en[j] is not None]
    weighted = [(i, j, occ[j] / (1.0 + (d / (radii[i] + radii[j])) ** 6)) for i, j, d in valid]
    result = 0.0
    total = 0.0
    for i in range(n):
        if occ[i] > 0 and en[i] is not None:
            weight = sum(w for center, j, w in weighted if center == i)
            if weight > 0:
                field = sum(w * (en[j] - en[i]) for center, j, w in weighted if center == i) / weight
                result += occ[i] * field * field
                total += occ[i]
    if total <= 0:
        return 0.0
    return result / total
```

#### 4. contact_hardness_scaled_electronic_contrast

MAE_without minus saved full-model MAE: +0.004697 eV/atom.
Source SHA256: 49108e28d9bbd582d2780ac75f8c9004c9b0f9ae40a1c2a003fd98b794900a8e
```python
def feature(s):
    if not s:
        return 0.0
    sites = s.get('sites') or []
    props = s.get('element_properties') or {}
    n = len(sites)
    occ = [max(0.0, t.get('occupancy', 1.0) or 0.0) for t in sites]
    atoms = sum(occ)
    if atoms <= 0:
        return 0.0
    radii = [(props.get(t.get('element')) or {}).get('covalent_radius_A') for t in sites]
    ion = [(props.get(t.get('element')) or {}).get('first_ionization_energy_eV') for t in sites]
    affinity = [(props.get(t.get('element')) or {}).get('electron_affinity_eV') for t in sites]
    result = 0.0
    for edge in s.get('neighbors') or []:
        i = edge.get('i', -1)
        j = edge.get('j', -1)
        d = edge.get('distance_A')
        if 0 <= i < n and 0 <= j < n and d is not None and d > 0 and occ[i] > 0 and occ[j] > 0:
            a = radii[i]
            b = radii[j]
            if a is not None and b is not None and a > 0 and b > 0:
                if ion[i] is not None and affinity[i] is not None and ion[j] is not None and affinity[j] is not None:
                    hi = ion[i] - affinity[i]
                    hj = ion[j] - affinity[j]
                    if hi > 1e-8 and hj > 1e-8:
                        delta = 0.5 * (ion[i] + affinity[i] - ion[j] - affinity[j])
                        x = d / (a + b)
                        x2 = x * x
                        w = occ[i] * occ[j] / (1.0 + x2 * x2 * x2)
                        result += w * delta * delta / (hi + hj)
    return 0.5 * result / atoms
```

#### 5. local_contact_direction_imbalance

MAE_without minus saved full-model MAE: +0.004221 eV/atom.
Source SHA256: 1a9335115190aa2a35aa8248f07c694dbe5282891957f12015c6b34d61f1962d
```python
def feature(s):
    if not s:
        return 0.0
    matrix = (s.get('lattice') or {}).get('matrix')
    if matrix is None or len(matrix) != 3:
        return 0.0
    sites = s.get('sites') or []
    props = s.get('element_properties') or {}
    edges = s.get('neighbors') or []
    n = len(sites)
    occ = [max(0.0, t.get('occupancy', 1.0) or 0.0) for t in sites]
    radii = [(props.get(t.get('element')) or {}).get('covalent_radius_A') for t in sites]
    lattice = np.asarray(matrix, dtype=float)
    result = 0.0
    total = 0.0
    for i in range(n):
        fi = sites[i].get('frac_coords')
        a = radii[i]
        if occ[i] > 0 and fi is not None and a is not None and a > 0:
            nearby = [edge for edge in edges if edge.get('i', -1) == i and 0 <= edge.get('j', -1) < n and occ[edge.get('j', -1)] > 0 and radii[edge.get('j', -1)] is not None and radii[edge.get('j', -1)] > 0 and edge.get('distance_A') is not None and edge.get('distance_A') > 0 and edge.get('image') is not None and sites[edge.get('j', -1)].get('frac_coords') is not None]
            if nearby:
                delta = np.asarray([[sites[edge.get('j')].get('frac_coords')[k] + edge.get('image')[k] - fi[k] for k in range(3)] for edge in nearby], dtype=float)
                vectors = np.matmul(delta, lattice)
                norms = np.sqrt(np.sum(vectors * vectors, axis=1))
                weights = np.asarray([occ[edge.get('j')] / (1.0 + (edge.get('distance_A') / (a + radii[edge.get('j')])) ** 6) for edge in nearby], dtype=float)
                weights = weights * np.where(norms > 1e-12, 1.0, 0.0)
                weight = float(np.sum(weights))
                if weight > 0:
                    unit = vectors / np.maximum(norms, 1e-12).reshape(-1, 1)
                    direction = np.sum(unit * weights.reshape(-1, 1), axis=0) / weight
                    value = float(np.sum(direction * direction))
                    result += occ[i] * max(0.0, min(1.0, value))
                    total += occ[i]
    if total <= 0:
        return 0.0
    return result / total
```

### Seed 2

Winner round:20; saved validation row MAE:0.233750687.

#### 1. neutral_main_group_contact_deficit

MAE_without minus saved full-model MAE: +0.002230 eV/atom.
Source SHA256: e95d8a62ef94e8395ae7f636e96d5b91483bc271dccecb95e212913dfd57544c
```python
def feature(s):
    if not s:
        return 0.0
    sites = s.get('sites') or []
    props = s.get('element_properties') or {}
    n = len(sites)
    radii = [(props.get(a.get('element')) or {}).get('covalent_radius_A') or 0.0 for a in sites]
    occ = [1.0 if a.get('occupancy') is None else max(0.0, a.get('occupancy')) for a in sites]
    edges = [(e.get('i'), e.get('j'), e.get('distance_A') or 0.0) for e in s.get('neighbors') or []]
    bonds = [(i, j, d) for i, j, d in edges if i is not None and j is not None and 0 <= i < n and 0 <= j < n and d > 0 and radii[i] > 0 and radii[j] > 0 and d < 1.35 * (radii[i] + radii[j])]
    total = 0.0
    amount = 0.0
    for i, a in enumerate(sites):
        if radii[i] <= 0 or occ[i] <= 0:
            continue
        p = props.get(a.get('element')) or {}
        group = p.get('group')
        if p.get('Z') == 1:
            reference = 1.0
        elif group is not None and 13 <= group <= 17:
            reference = float(min(group - 10, 18 - group))
        else:
            continue
        coordination = sum(occ[j] * ((1.35 - d / (radii[i] + radii[j])) / 0.35) ** 2 for ii, j, d in bonds if ii == i)
        deficit = max(0.0, 1.0 - coordination / reference)
        total += occ[i] * deficit * deficit
        amount += occ[i]
    return float(total / amount) if amount > 0 else 0.0
```

#### 2. soft_coordination_heterogeneity

MAE_without minus saved full-model MAE: +0.000818 eV/atom.
Source SHA256: 99bca00c41a8e14c4d8a530b1f609cb2c88a7274c6f0d640ff70622496387317
```python
def feature(s):
    if not s:
        return 0.0
    sites = s.get('sites') or []
    props = s.get('element_properties') or {}
    n = len(sites)
    radii = [(props.get(a.get('element')) or {}).get('covalent_radius_A') or 0.0 for a in sites]
    occ = [1.0 if a.get('occupancy') is None else max(0.0, a.get('occupancy')) for a in sites]
    edges = [(e.get('i'), e.get('j'), e.get('distance_A') or 0.0) for e in s.get('neighbors') or []]
    bonds = [(i, j, d) for i, j, d in edges if i is not None and j is not None and 0 <= i < n and 0 <= j < n and d > 0 and radii[i] > 0 and radii[j] > 0 and d < 1.35 * (radii[i] + radii[j])]
    first = 0.0
    second = 0.0
    amount = 0.0
    for i in range(n):
        if radii[i] <= 0 or occ[i] <= 0:
            continue
        coordination = sum(occ[j] * ((1.35 - d / (radii[i] + radii[j])) / 0.35) ** 2 for ii, j, d in bonds if ii == i)
        first += occ[i] * coordination
        second += occ[i] * coordination * coordination
        amount += occ[i]
    if amount <= 0 or first <= 0:
        return 0.0
    mean = first / amount
    return float(max(0.0, second / amount - mean * mean) / (mean * mean))
```

#### 3. local_chemical_field_squared

MAE_without minus saved full-model MAE: +0.000584 eV/atom.
Source SHA256: a018aa665a35fd418febd404a5ebcdc7835b359bdaca020ec1f73d9fe1bad682
```python
def feature(s):
    if not s:
        return 0.0
    sites = s.get('sites') or []
    props = s.get('element_properties') or {}
    n = len(sites)
    radii = [(props.get(a.get('element')) or {}).get('covalent_radius_A') or 0.0 for a in sites]
    eneg = [(props.get(a.get('element')) or {}).get('eneg') for a in sites]
    occ = [1.0 if a.get('occupancy') is None else max(0.0, a.get('occupancy')) for a in sites]
    edges = [(a.get('i'), a.get('j'), a.get('distance_A') or 0.0) for a in s.get('neighbors') or []]
    bonds = [(i, j, (1.35 - d / (radii[i] + radii[j])) ** 2) for i, j, d in edges if i is not None and j is not None and 0 <= i < n and 0 <= j < n and d > 0 and radii[i] > 0 and radii[j] > 0 and eneg[i] is not None and eneg[j] is not None and d < 1.35 * (radii[i] + radii[j])]
    total = 0.0
    amount = 0.0
    for i in range(n):
        if occ[i] <= 0 or eneg[i] is None or radii[i] <= 0:
            continue
        weight = sum(occ[j] * w for ii, j, w in bonds if ii == i)
        if weight <= 0:
            continue
        field = sum(occ[j] * w * (eneg[j] - eneg[i]) for ii, j, w in bonds if ii == i) / weight
        total += occ[i] * field * field
        amount += occ[i]
    return float(total / amount) if amount > 0 else 0.0
```

#### 4. all_contact_compression

MAE_without minus saved full-model MAE: +0.000258 eV/atom.
Source SHA256: ffc3b5ad82e7690f3b441c57eba7bf02bc65414c713896ed1520b6261b58ddf1
```python
def feature(s):
    if not s:
        return 0.0
    sites = s.get('sites') or []
    props = s.get('element_properties') or {}
    n = len(sites)
    radii = [(props.get(a.get('element')) or {}).get('covalent_radius_A') or 0.0 for a in sites]
    occ = [1.0 if a.get('occupancy') is None else max(0.0, a.get('occupancy')) for a in sites]
    amount = sum(occ[i] for i in range(n) if radii[i] > 0)
    if amount <= 0:
        return 0.0
    total = 0.0
    for edge in s.get('neighbors') or []:
        i = edge.get('i')
        j = edge.get('j')
        d = edge.get('distance_A') or 0.0
        if i is None or j is None or not 0 <= i < n or not 0 <= j < n or d <= 0:
            continue
        if radii[i] <= 0 or radii[j] <= 0:
            continue
        overlap = max(0.0, 1.0 - d / (radii[i] + radii[j]))
        total += occ[i] * occ[j] * overlap * overlap
    return float(total / amount)
```

#### 5. electronegative_framework_contact_fraction

MAE_without minus saved full-model MAE: -0.000026 eV/atom.
Source SHA256: 6ef44971628984007a49fbc9385cbac1740ae75f78baae1cf9c16674e400ce90
```python
def feature(s):
    if not s:
        return 0.0
    sites = s.get('sites') or []
    props = s.get('element_properties') or {}
    n = len(sites)
    p = [props.get(a.get('element')) or {} for a in sites]
    radii = [a.get('covalent_radius_A') or 0.0 for a in p]
    groups = [a.get('group') for a in p]
    eneg = [a.get('eneg') for a in p]
    selected = [groups[i] is not None and 14 <= groups[i] <= 17 and eneg[i] is not None and eneg[i] >= 2.5 for i in range(n)]
    occ = [1.0 if a.get('occupancy') is None else max(0.0, a.get('occupancy')) for a in sites]
    total = 0.0
    weight = 0.0
    for edge in s.get('neighbors') or []:
        i = edge.get('i')
        j = edge.get('j')
        d = edge.get('distance_A') or 0.0
        if i is None or j is None or not 0 <= i < n or not 0 <= j < n or d <= 0:
            continue
        if radii[i] <= 0 or radii[j] <= 0 or groups[i] is None or groups[j] is None or eneg[i] is None or eneg[j] is None:
            continue
        w = occ[i] * occ[j] * max(0.0, 1.35 - d / (radii[i] + radii[j])) ** 2
        weight += w
        if selected[i] and selected[j]:
            total += w
    return float(total / weight) if weight > 0 else 0.0
```

## 公开目录更新

用户明确批准方案A，保留并恢复O/L0配置与结果展示块；L1原状态和错误记录保留，终局评分尚待执行。
1300条清单路径只读检查全部成功、读取失败0；原PermissionError未记录失败文件名，确切原因未知，不能据此次重读成功推断为临时占用。
原主说明、私有配置、冻结记录、原保全manifest及更新前六公开文件已保留只读快照，见manual_terminal_resume_20261031/history_snapshot_manifest.json。原私有配置、frozen_config和原manifest不变。
predictions旧6000行/298912字节前缀逐字节相同；当前完整9000行含新增L1预测，所以完整文件hash变化。llm_calls及run.log同样保留原追加前缀。下表区分快照完整hash和更新后完整hash，不改历史hash。

### 第二步公开对象更新后哈希

|文件|旧阶段4.2快照 SHA256|当前完整文件 SHA256|
|---|---|---|
|config.json|bfff6e10ecf9b2d51c76fc20c59b7590e965a7c6988258c317ca5c0ef2fadb22|533308cc818653d3a1230d3d32c482fc1b64d131fdc0e8cf188a5e596ff34d60|
|predictions.csv|e11a27ecb3af3efe89a869b2db590a426583dde082fc4c353f0dd1f064546e20|6483c2088698a9cf7c9038d8e950b89f3a8a58d8690fd38ef15c5df0cd7936b0|
|llm_calls.jsonl|25d6bbf24ebc346f3bb574d4ebc22985eca6e6a3b9174d3995c837e713b98642|b4c7432837f4033d723bb1234030e990bc386ae266cec857d2c4bf5e5766b0ca|
|metrics.json|49dd1f0b17af1e7410bdf052b1b114d63d4bf122eacee9c443bbf652324e59df|07b0bbb0e56f2ea7b7f7063f26b4199e223bb7fbea6be60c9d177bdd5ec6b73f|
|run.log|4c7e92925c5cd0844f146ef121f52f2ba3cda95c6b0f7ca5aa41dff9326671a1|7fe820c178c8a16c0f28a6cb7b61d36497223d65e30892a9cd9003eaa6e067a8|
|README.md|58d02d6a6f065400a59ecf20df70b403298c8ab5c4d6895b109ff90c4b7c7152|0b8e02c722254fbcefe76f93803500f47285c4bbf12a99630427f14cb4141455|

### 终评分前公开文件哈希

|文件|旧阶段4.2快照 SHA256|当前完整文件 SHA256|
|---|---|---|
|config.json|bfff6e10ecf9b2d51c76fc20c59b7590e965a7c6988258c317ca5c0ef2fadb22|533308cc818653d3a1230d3d32c482fc1b64d131fdc0e8cf188a5e596ff34d60|
|predictions.csv|e11a27ecb3af3efe89a869b2db590a426583dde082fc4c353f0dd1f064546e20|6483c2088698a9cf7c9038d8e950b89f3a8a58d8690fd38ef15c5df0cd7936b0|
|llm_calls.jsonl|25d6bbf24ebc346f3bb574d4ebc22985eca6e6a3b9174d3995c837e713b98642|b4c7432837f4033d723bb1234030e990bc386ae266cec857d2c4bf5e5766b0ca|
|metrics.json|49dd1f0b17af1e7410bdf052b1b114d63d4bf122eacee9c443bbf652324e59df|07b0bbb0e56f2ea7b7f7063f26b4199e223bb7fbea6be60c9d177bdd5ec6b73f|
|run.log|4c7e92925c5cd0844f146ef121f52f2ba3cda95c6b0f7ca5aa41dff9326671a1|7fe820c178c8a16c0f28a6cb7b61d36497223d65e30892a9cd9003eaa6e067a8|
|README.md|58d02d6a6f065400a59ecf20df70b403298c8ab5c4d6895b109ff90c4b7c7152|93aea95aeed6ada9bb07a538662307c0be0eabafbdc500bbc291a60e8f4309d6|


### Legacy O reference handling correction before the TEST gate

The first terminal-only entry stopped during validation, before continuation_started.json or the canonical evaluation marker existed. Its new wrapper had treated the legacy prior_stage4_1_results path string as a result object. The failure log and old gate/source are retained; no TEST target was read. The reference and its original key remain unchanged. Its protected O snapshot results object exactly matches the immutable O terminal result; L0 also matches. Public preserved_condition_results now copies these original O/L0 objects without changing any numerical value. This corrects publication/reference handling only; frozen scientific sources, private config, prompts and all predictions remain unchanged.

#### Public hashes after reference-display correction (step2)

|文件|旧阶段4.2快照 SHA256|当前完整文件 SHA256|
|---|---|---|
|config.json|bfff6e10ecf9b2d51c76fc20c59b7590e965a7c6988258c317ca5c0ef2fadb22|533308cc818653d3a1230d3d32c482fc1b64d131fdc0e8cf188a5e596ff34d60|
|predictions.csv|e11a27ecb3af3efe89a869b2db590a426583dde082fc4c353f0dd1f064546e20|6483c2088698a9cf7c9038d8e950b89f3a8a58d8690fd38ef15c5df0cd7936b0|
|llm_calls.jsonl|25d6bbf24ebc346f3bb574d4ebc22985eca6e6a3b9174d3995c837e713b98642|b4c7432837f4033d723bb1234030e990bc386ae266cec857d2c4bf5e5766b0ca|
|metrics.json|49dd1f0b17af1e7410bdf052b1b114d63d4bf122eacee9c443bbf652324e59df|568ee65d5c8ccf150dd7dcd991dc7b44adc707e7e9d6326b092ff21fc86afee8|
|run.log|4c7e92925c5cd0844f146ef121f52f2ba3cda95c6b0f7ca5aa41dff9326671a1|7fe820c178c8a16c0f28a6cb7b61d36497223d65e30892a9cd9003eaa6e067a8|
|README.md|58d02d6a6f065400a59ecf20df70b403298c8ab5c4d6895b109ff90c4b7c7152|93aea95aeed6ada9bb07a538662307c0be0eabafbdc500bbc291a60e8f4309d6|

#### Sealed pre-score hashes after the reference correction

|文件|旧阶段4.2快照 SHA256|当前完整文件 SHA256|
|---|---|---|
|config.json|bfff6e10ecf9b2d51c76fc20c59b7590e965a7c6988258c317ca5c0ef2fadb22|533308cc818653d3a1230d3d32c482fc1b64d131fdc0e8cf188a5e596ff34d60|
|predictions.csv|e11a27ecb3af3efe89a869b2db590a426583dde082fc4c353f0dd1f064546e20|6483c2088698a9cf7c9038d8e950b89f3a8a58d8690fd38ef15c5df0cd7936b0|
|llm_calls.jsonl|25d6bbf24ebc346f3bb574d4ebc22985eca6e6a3b9174d3995c837e713b98642|b4c7432837f4033d723bb1234030e990bc386ae266cec857d2c4bf5e5766b0ca|
|metrics.json|49dd1f0b17af1e7410bdf052b1b114d63d4bf122eacee9c443bbf652324e59df|568ee65d5c8ccf150dd7dcd991dc7b44adc707e7e9d6326b092ff21fc86afee8|
|run.log|4c7e92925c5cd0844f146ef121f52f2ba3cda95c6b0f7ca5aa41dff9326671a1|7fe820c178c8a16c0f28a6cb7b61d36497223d65e30892a9cd9003eaa6e067a8|
|README.md|58d02d6a6f065400a59ecf20df70b403298c8ab5c4d6895b109ff90c4b7c7152|93aea95aeed6ada9bb07a538662307c0be0eabafbdc500bbc291a60e8f4309d6|


All ordered checks passed with the actual referenced O numeric result, L0 result, full prior configurations and exact6000-row byte prefix. The legacy O file-reference string is retained without alteration. The earlier entrypoint compatibility error occurred before both continuation_started.json and evaluation_started.json; it consumed no TEST read. The corrected entrypoint and its original readonly backup are both SHA-bound by this gate. Scientific frozen config/source and statistical rules remain unchanged.

## Public file hashes after terminal completion

The old files are the readonly completed-stage4.2 six-file snapshot. The new hashes below are taken after the final L1 metrics and README updates; this continuation makes no further public-file changes.
Prefix facts refer to literal old-file bytes and do not imply whole JSON/config equality. No LLM calls or prediction rows were added by the terminal-only continuation.

|File|Old SHA256|New SHA256|Old bytes|New bytes|Identical bytes|Old byte prefix preserved|
|---|---|---|---|---|---|---|
|config.json|bfff6e10ecf9b2d51c76fc20c59b7590e965a7c6988258c317ca5c0ef2fadb22|533308cc818653d3a1230d3d32c482fc1b64d131fdc0e8cf188a5e596ff34d60|751643|3140555|False|False|
|metrics.json|49dd1f0b17af1e7410bdf052b1b114d63d4bf122eacee9c443bbf652324e59df|569c0362ae76105a050ff1031da9c36fb70eff922ec3f4f9db1ae47b8a68ac8c|872868|371367|False|False|
|predictions.csv|e11a27ecb3af3efe89a869b2db590a426583dde082fc4c353f0dd1f064546e20|6483c2088698a9cf7c9038d8e950b89f3a8a58d8690fd38ef15c5df0cd7936b0|298912|451309|False|True|
|llm_calls.jsonl|25d6bbf24ebc346f3bb574d4ebc22985eca6e6a3b9174d3995c837e713b98642|b4c7432837f4033d723bb1234030e990bc386ae266cec857d2c4bf5e5766b0ca|12072609|21724439|False|True|
|run.log|4c7e92925c5cd0844f146ef121f52f2ba3cda95c6b0f7ca5aa41dff9326671a1|7fe820c178c8a16c0f28a6cb7b61d36497223d65e30892a9cd9003eaa6e067a8|20651|31433|False|True|
|README.md|58d02d6a6f065400a59ecf20df70b403298c8ab5c4d6895b109ff90c4b7c7152|6ff83360684cf68eddc9b10a9f419e29c814d1167c613ab658c97c76bd845688|2548|2444|False|False|
