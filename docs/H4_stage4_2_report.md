# H4 stage4.2 L0 results
Status: complete. Stage4.3 is not started.
TEST1000;333 template groups;2000 shared group bootstrap draws. Means are arithmetic means of seed metrics.
|Weight|Seed|MAE [95% CI]|RMSE [95% CI]|R2 [95% CI]|
|---|---|---|---|---|
|row_weighted|0|0.207168 [0.193436, 0.221132]|0.276304 [0.257817, 0.295173]|0.932284 [0.921139, 0.941389]|
|row_weighted|1|0.203586 [0.190959, 0.216650]|0.268558 [0.250991, 0.286395]|0.936028 [0.926130, 0.944309]|
|row_weighted|2|0.221556 [0.206492, 0.236991]|0.301918 [0.278346, 0.325686]|0.919147 [0.904369, 0.932208]|
|row_weighted|mean|0.210770 [0.198923, 0.223265]|0.282260 [0.264440, 0.300356]|0.929153 [0.918284, 0.938018]|
|group_weighted|0|0.213261 [0.198908, 0.227766]|0.284159 [0.264019, 0.304459]|0.931937 [0.920005, 0.941758]|
|group_weighted|1|0.212991 [0.197531, 0.228877]|0.284610 [0.260902, 0.307303]|0.931721 [0.920279, 0.941270]|
|group_weighted|2|0.233403 [0.215791, 0.252246]|0.319995 [0.290688, 0.349465]|0.913687 [0.898220, 0.926968]|
|group_weighted|mean|0.219885 [0.205919, 0.234362]|0.296255 [0.274493, 0.317438]|0.925781 [0.914250, 0.935191]|

|Seed|Query labels|Final uniform fill|Final labels|Winner round|Kept features|
|---|---|---|---|---|---|
|0|1500|0|2000|20|18|
|1|1450|50|2000|20|15|
|2|1500|0|2000|17|9|

Run elapsed seconds: 4190.578. Global CLI calls: 120; L0 CLI calls: 60.
All raw generated sets, query match/reveal/carry records, failure diagnostics, token ledgers and frozen hashes are retained privately.
Validation labels are allowed in feedback; validation memberships grow across rounds. Current code is the complete scientifically retained feature set, never budget truncation.
O was scored previously once; L0 is independently scored once. The original O marker, predictions and metrics are preserved; no O rescoring occurred.
This stage reports L0 only; paired arm and memory conclusions are reserved for later authorized stages.
