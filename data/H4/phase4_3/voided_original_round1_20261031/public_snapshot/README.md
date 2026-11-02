# H4 stage4.3: L1 feature loop with memory
Status: stopped_requires_user before round2; required memory11873>6000tokens;3 of60 rounds saved; stage4.4 not started.
Original O and L0 results/markers/private trees and readonly six-file snapshots are preserved; old predictions and calls are not replayed.
Each seed starts at the same L0 S0500;20 rounds of75+carry, then uniform fill to2000, reported separately.
L0 query grammar/POOL-only space groups/template hash mapping/HGB parameters/unweighted fit/TRAIN median remain fixed.
Memory uses pinned tiktoken0.12.0/o200k_base encode_ordinary on exact compact sorted JSON;6000 tokens; not a model-internal-tokenizer claim.
Memory stores TRAIN/validation feature definitions, drop-one refit contributions, failure reasons, query statistics and <=300-character notes.
Over6000: deterministic top30 contributions plus complete latest3 memory records, with all query statistics and notes; deduplicate definitions, never truncate source.
If the required retained memory still exceeds6000 tokens, stop; never silently weaken the retention rule.
Real complete-request UTF8 guard including strict wire/schema/note/global/model instructions+1500 framing+repair is 189000 bytes.
The L1 limit is set before freeze from measured L0 surrogate memory+round20 base extrapolation with10% rounded-up margin; theoretical upper bounds are records only.
L0 changed its input limit during round7 recovery; L1 pre-freeze setting does not change any previous L0 request or result.
Internal validation labels enter feedback; its members grow with reveals. No TEST/HOLDOUT labels enter prompts, memory or selection.
Winner=min saved round validation rowMAE, earliest tie; persist before fill; original TRAIN cache reused before full2000 refit and TEST1000 prediction.
CLI max4 slots (3 seed dependencies); sandbox2workers/50ms; no CLI/sandbox overlap; global300 counts O60+L060+L1 and all repairs/retries.
Actual TEST scoring passes: O=1, L0=1, L1=0 (H4 total=2); L1 TEST gate was not created and targets were not read.
Row/fullTEST-template-group metrics;2000 shared group bootstraps; arithmetic seed means, not ensemble; paired comparisons use private cached draws only.
L1-L0 compares the complete memory/feedback/query trajectory; source, memory, request-byte ledgers and reports remain under phase4_3.
