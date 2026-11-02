# H4 stage4.3 pre-freeze input limit setting
Time: 2026-10-31T15:11:30.478237-04:00 (America/New_York).
User selected A: fixed local token counter; real emitted requests are guarded, theoretical maxima are records only.
Memory counter: tiktoken0.12.0/o200k_base encode_ordinary; compact sorted Unicode JSON;6000-token cap; counter SHA256 f374eab336ae08f9bcb8a58cf0f8346c596ff8eb99e1b37357840f7393e6e99f.
L1 whole-request limit is set to 189000 UTF8 budget bytes BEFORE freezing stage4.3 and BEFORE any L1 scientific call.
Measurement: all60 actual L0 histories with470 cached TRAIN/validation drop-one refits; max prescribed retained JSON 139379 UTF8bytes; source definitions are deduplicated verbatim.
Baseline extrapolation: 30793.923809523811 bytes (rounds1..6 OLS to20, including format repair); exact extra L1 wrapper/notes schema=813 bytes.
Formula: ceil1000((30793.923809523811+139379+813)*1.10)=189000.
Measurement projection SHA256: a357fa2eee5ac2363e8c9749aaf90a3874d0d9ab5de658dc23b447391bbc9100.
L0 had no notes, so measured notes are empty; future L1 notes and payload sizes are unknown and remain subject to actual guards. This is an empirical reference, not a future-size guarantee.
All nonempty L0 surrogate views exceed6000 tokens even after prescribed retention; the L1 memory cap is NOT increased. Actual L1 memory above6000 after prescribed trimming stops before the next request.
The old L0 limit changed from20000 to34000 during its round7 manual recovery. This L1 setting is pre-freeze, not an in-run amendment, and changes no prior L0 request/cache/result.
The original theoretical input_projection_L1.json and the prior L0 theoretical1720637-byte record are unchanged and cannot trigger a stop.
Whole-request guard includes actual strict wire schema+wire note+global/model instructions+1500 framing and the format-repair suffix; actual over-limit requests stop the whole stage, without truncating current valid source.
Prompt/query/HGB/seed/template split and per-source maxLength20000 are frozen only after this setting; all inherited scientific settings remain as specified. No TEST or HOLDOUT target was used in measurement.
O and L0 each retain their original one terminal scoring pass. L1 has not been scored. Stage4.4 is not started.
