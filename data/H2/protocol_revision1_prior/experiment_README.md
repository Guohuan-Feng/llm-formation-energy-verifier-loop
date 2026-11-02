# H2 已保存进度并停止
Synthetic-only point estimate 24.90h exceeds remaining total24h budget 23.52h; real4000 and RF cost not included. Saved before real calibration/formal labeling.
未启动 H3，未作尚未完成的 RF 外部评测。
正式保留TEST 0/1000、POOL 0/5000；真实标注 0/4000。
## GPU并行基准
|worker|个/小时（含启动）|峰值显存MiB|最大能量差eV/atom|
|---:|---:|---:|---:|
|1|275.37|1711.0|0.0|
|2|757.12|2183.0|0.0014131334092883918|
|4|1190.88|3627.0|0.001697540283203125|
仅合成部分点估计 24.90 h，已超过剩余预算 23.52 h；未计真实4000与RF。
输入、结构、标签及部分结果保存在 work/h-data/H2，POOL标签默认隐藏。
收敛要求原始原子力和原始晶胞过滤器力均≤0.1 eV/Å；最多300步。
首轮硬件基准遇显存溢出；同模板同替换路线更换1个基准候选后完整重跑40×3，失败耗时已计入24h，正式候选与真实4000不变；审计见work/h-data/H2/resource_revision.json。
