# H2 正式数据集（止于本阶段）

合成TEST1000/POOL5000，模板不重叠；各seed0/1/2以原S0 500+随机补1500训练，固定RF种子不变，各作一次终评。
真实固定2000→2000；严格可用CHGNet 1992→1998；accepted_physics仅诊断=1982/1985；缺失10个不补、不重算。

|固定 Magpie132 + RF300，min_leaf=2|MAE（不加权）|MAE（组加权）|RMSE|R²|符号 BA|
|---|---:|---:|---:|---:|---:|
|合成CHGNet seed0|0.2538|0.2686|0.3595|0.8854|0.8128|
|合成CHGNet seed1|0.2528|0.2701|0.3616|0.8840|0.8002|
|合成CHGNet seed2|0.2448|0.2590|0.3473|0.8930|0.8155|
|合成3种子指标均值|0.2505|0.2659|0.3561|0.8875|0.8095|
|真实CHGNet（1992→1998条件子集）|0.2305|0.2371|0.3785|0.8880|0.7778|
|真实DFT（完整2000→2000）|0.2377|0.2452|0.3906|0.8848|0.7790|
真实 HOLDOUT 的 CHGNet 标签对 DFT：MAE=0.0634 eV/atom；n=1998/2000；完整指标、95% CI 和同目标配对差值见 metrics.json。
CHGNet0.3.0真实标签：能量有限、有最终结构、收敛无error、原子/晶胞力≤0.1 eV/Å；不按合成窗口过滤，H3沿用。模板/真实成分分组，权重=1/当前组大小，配对组bootstrap2000次；种子均值CI逐draw平均指标，非预测集成。
POOL标签私存，公开清单仅ID/角色，predictions仅TEST/HOLDOUT；科学LLM调用0。Matbench未收录不证明LLM未见；负形成能不等于凸包稳定/可合成。本次授权hash核验复用旧结果；29旧GPU错新进程重试一次，新CUDA错仅回收对应worker并重试一次，两次失败跳过，累计重启>5降2并行；完整规则和恢复计数见metrics。
GPU基准（同40个候选，含启动；ΔE单位eV/atom）：1 workers=275.4个/h、1711.0MiB、最大ΔE=0.0000；2 workers=757.1个/h、2183.0MiB、最大ΔE=0.0014；4 workers=1190.9个/h、3627.0MiB、最大ΔE=0.0017。最大并行数4，大图准入降至2/1；总活跃耗时9.06/24 h。旧NHe3提交25.3GB显存的资源失败及整套基准重测记录已归档。
用户工程判据：并行ΔE门槛由1→5 meV/atom（≤），收敛及全部物理标志仍须一致；约为原P1 CHGNet-vs-DFT 58 meV/atom的1/10，仅作吞吐量取舍，非科学精度保证。
标签前筛（仅正式合成）：排He/Ne/Ar/Kr/Xe/Rn目标选项=11419；预缩放V/N<4 Å³=20提案，3 Å角对>20000=5提案（独立计数，交并集见metrics；等号保留）。 同40初态串行复跑（n=40，仅诊断运行间差异、不设门槛、不能确定唯一因果）：|ΔE|最大/均值=1.6848/0.0866 meV/atom，中位=0.0005，P95=0.2563，>1meV数=2，>5meV数=0；收敛/物理标志=一致/一致，错误=0。
状态：partial_real_label_coverage；H2 POOL5000 and all predeclared RF arms completed once; strict real CHGNet1992->1998 is conditional and full2000x2000 comparison unavailable; DFT2000->2000 complete. H3 must use strict_converged_real_v1. STOP after H2.
