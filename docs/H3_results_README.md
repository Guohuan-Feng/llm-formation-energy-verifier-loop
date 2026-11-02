# H3：记忆效应与零样本特征
状态：partial_coverage；真实TRAIN1992CHGNet/2000DFT，合成主2000/匹配1992；TEST1000/HOLDOUT1998CHGNet或2000DFT。
真实标签沿H2严格收敛规则；accepted_physics仅诊断，10缺失不补、不重算；H3结束，未启动H4。
记忆探针各300：组权重LLM/RF：真实=0.6270，合成=0.7324；真实−合成=-0.1054，95%CI=[-0.2435170823192139, 0.034785913064398835]。
自报认识yes/partial/no分层及逐样本口径、DFT参考/匹配1992敏感性见metrics；自报与域差异不等于已验证记忆因果。
|固定HGB特征集（三seed指标均值）|合成CHGNet MAE行/组|真实CHGNet MAE行/组|真实DFT MAE行/组|
|---|---:|---:|---:|
|magpie|0.2335/0.2509|0.2136/0.2199|0.2221/0.2288|
|magpie_plus_structure9|0.2144/0.2215|0.1919/0.1964|0.2038/0.2091|
|magpie_plus_llm|0.2130/0.2231|0.1872/0.1920|0.1964/0.2021|
|llm_only|0.3086/0.3260|0.2660/0.2717|0.2736/0.2806|
CGCNN 2000各seed状态/超时与完整TRAIN checkpoint覆盖见metrics。
CGCNN 5000各seed状态/超时与完整TRAIN checkpoint覆盖见metrics。
三独立零样本请求各20函数，仅给schema/API；保留规则只看对应TRAIN失败率>20%剔除，TEST只报告。
沙箱14项CPU测试PASS：AST白名单、只读数据、math/np限制、子进程每函数每结构50ms（启动/准备分列），超时/NaN/异常计失败。
逐特征失败率与单结构耗时、每seed结果/均值/配对CI见metrics；仅LLM特征为空时该臂明确不可用，不伪造替代。
CGCNN各臂TRAIN内模板分组选择后fresh全TRAIN refit，两阶段合计≤2h；超时内选模型不能充当全量训练臂。
本机gpt-6.1-sol/high，4并发，禁工具/联网，无教授API；CLI版本/调用源码/输入≤20000token均冻结。
MAE/RMSE/R²/形成能<0的BA，行与逆组大小两口径；模板/成分组bootstrap2000次，域内配对、跨域独立。
总科学LLM调用/含探针总调用=609/610；耗时6.52h；完整provenance与限制见metrics。；26日主机重启、27日系统BlueScreen证据；本次复用3特征/57探针，固定CLI0.165人工续跑。
