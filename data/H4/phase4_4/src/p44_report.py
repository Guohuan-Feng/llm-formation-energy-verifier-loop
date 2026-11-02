"""Render only committed stage4.4 metrics and provenance; no science calls."""
from pathlib import Path
from collections import defaultdict
import json
import p44_common as c


def metric(value):
    def number(x):
        return 'undefined' if x is None else f'{x:.6f}'
    return number(value['point']) + ' [' + ', '.join(number(x) for x in value['ci95']) + ']'


def scores_table(lines, groups, *, paired=False):
    lines += ['|Group/comparison|Seed|Weighting|MAE [95% CI]|RMSE [95% CI]|R2 [95% CI]|',
              '|---|---|---|---|---|---|']
    for name, group in groups.items():
        items = [(seed, group['per_seed'][seed]) for seed in ('0', '1', '2')]
        items.append(('arithmetic mean', group['arithmetic_seed_mean']))
        for seed, row in items:
            row = row if paired or seed == 'arithmetic mean' else row['scores']
            for mode in ('row_weighted', 'group_weighted'):
                lines.append('|' + '|'.join([name, seed, mode] + [metric(row[mode][key]) for key in ('mae', 'rmse', 'r2')]) + '|')
    lines.append('')


def resources():
    totals = defaultdict(lambda: {'physical_calls': 0, 'input_tokens': 0, 'output_tokens': 0, 'cached_input_tokens': 0, 'unknown_fields': 0})
    with (c.PUBLIC / 'llm_calls.jsonl').open('rb') as stream:
        for line in stream:
            if not line.endswith(b'\n'):
                continue
            row = json.loads(line)
            if row.get('record_type') != 'result':
                continue
            rid = row['request_id']
            stage = '4.1 O' if rid.startswith('O_') else '4.2 L0' if rid.startswith('L0_') else '4.3 L1 (including void3)'
            record = totals[stage]
            record['physical_calls'] += 1
            for key in ('input_tokens', 'output_tokens', 'cached_input_tokens'):
                if type(row.get(key)) is int:
                    record[key] += row[key]
                else:
                    record['unknown_fields'] += 1
    totals['4.4 controls/transfer']
    return dict(totals)


def make_report(cfg, synthetic, real, controls, transfer, run_metadata):
    directory = c.safe(c.PHASE / 'report', write=True)
    directory.mkdir(exist_ok=True)
    previous = c.PHASE.parent / 'phase4_3/report'
    references = {}
    for name in ('L1_L0_validation_MAE.png', 'L1_winner_top5_features.json'):
        source = previous / name
        target = directory / name
        c.write(target, source.read_bytes(), exclusive=True)
        references[name] = {'source': str(source), 'sha256': c.sha(source), 'copied_sha256': c.sha(target)}
    call_usage = resources()
    c.write(directory / 'resource_summary.json', {'calls_and_tokens': call_usage, 'run': run_metadata,
        'prior_elapsed': cfg['prior_elapsed'], 'historical_active_totals_known': False,
        'note': 'Known stage durations are not a claim to include every interrupted historical active interval.'}, exclusive=True)
    lines = ['# H4 stage4.4 对照与迁移', '',
        '状态：complete。阶段4.4完成，H4结束；不启动H5或其他阶段，等待用户决定下一步。', '',
        '全部结果取自已提交的预测、一次性评分与私有组bootstrap抽样缓存；报告不训练、不重新预测、不读取评测标签。',
        'MAE/RMSE单位eV/atom；R²无量纲。2000次组bootstrap；三个种子的指标及抽样值取算术平均，不是预测ensemble。',
        '合成按完整TEST1000模板分组；真实按完整HOLDOUT2000约化成分分组。', '',
        '## 评分历史与范围', '',
        '本阶段合成TEST一次读取覆盖12对照臂，H4累计从3增至4；原O/L0/L1不重新评分。',
        '本阶段真实DFT HOLDOUT一次读取覆盖全部12迁移臂，不用HOLDOUT修改模型、特征或失败率筛选。',
        '历史HOLDOUT实际标签字段累计读取次数无法确认；可识别的H2/H3/H3b历史终评记录与本次新增1次分开记录，不能把marker数当成全部字段读取次数。',
        '4.3 PermissionError原trace未记录具体文件名，无法确认；当前可读性核验不能证明原因为临时占用。',
        '迁移使用H2初始/original DFT几何及DFT标签；不将H3的CHGNet弛豫结构缓存称为原始DFT结构。Magpie仅复用已核对成分的缓存。', '',
        '## 合成对照绝对指标', '']
    scores_table(lines, synthetic['per_group'])
    lines += ['历史参考：原H4协议给定的H3零样本LLM特征在2000标签下合成MAE约0.212，保留其原评测样本/结构口径。该数仅引用，不当成本阶段TEST1000新统计，不为它新增评分或配对CI。', '']
    lines += ['## 真实DFT迁移绝对指标', '',
        '原协议的各臂最终特征集包含O/L0/L1，每臂三个种子；另有三个Magpie-only HGB基线。只在真实TRAIN重新应用>20%失败剔除及训练中位数填补，不重新选择合成轮次。',
        '固定TRAIN成员及固定HGB参数可能使三个Magpie基线预测完全相同；这些不是独立随机重复，共享组抽样不会凭重复副本缩窄CI。', '']
    scores_table(lines, real['per_group'])
    lines += ['## 预登记的既有特征比较', '',
        '只引用4.3已存私有抽样结果并核对三项六位参考值，未重读标签、未重新bootstrap或修改参考。',
        'L1−L0是整套记忆、反馈和查询轨迹差异，不是纯笔记因果；内部validation标签进入反馈，成员随揭示增长。', '']
    scores_table(lines, controls['old_feature_pairs']['comparisons'], paired=True)
    lines += ['## 描述性比较：L1减对照', '']
    scores_table(lines, controls['comparisons'], paired=True)
    lines += ['## 迁移配对比较：L1减Magpie', '']
    scores_table(lines, transfer['comparisons'], paired=True)
    lines += ['## CGCNN复用与重新训练审计', '',
        'n5000三个种子复用H3b已冻结训练与预测；n2000三个种子因H4A探针重复评分历史而重新训练，不复用旧checkpoint或预测。',
        'H3b一次终评可产生full和probe多个统计scope，scope数不等于标签accessor读取次数。历史评分暴露与4.4明确授权的新统一评分门分开记录。',
        '详见[复用审计](../audit/cgcnn_reuse_audit.json)及[冻结配置](../config.json)。', '',
        '## 特征、曲线、源码与失败率', '',
        '固定的各臂源码、原winner和kept集合见配置feature_sets；真实TRAIN过滤与HOLDOUT仅报告的失败率见各臂model metadata和[训练预测元数据](../training_prediction_metadata.json)。',
        'L1合成最终有效函数16/19/11，已有TRAIN/TEST保留函数失败率0；迁移失败率单独报告，不能沿用合成0失败作为保证。',
        '![L1与L0验证曲线](L1_L0_validation_MAE.png)',
        '[4.3原winner top5完整源码与LOFO贡献](L1_winner_top5_features.json)。贡献来自合成内部验证集，不是HOLDOUT贡献；不保证每项delta均为正。', '',
        '|迁移臂|原定函数|保留函数|TRAIN原定函数平均失败率|HOLDOUT原定函数平均失败率|', '|---|---|---|---|---|']
    for arm in c.read(c.PHASE / 'training_prediction_metadata.json')['real_arms']:
        feature = arm['feature_selection']
        train_rates = feature['TRAIN_failure_rates']
        prediction_rates = feature['prediction_failure_rates']
        mean_train = f'{sum(train_rates)/len(train_rates):.6f}' if train_rates else 'not applicable'
        mean_predict = f'{sum(prediction_rates)/len(prediction_rates):.6f}' if prediction_rates else 'not applicable'
        lines.append('|' + '|'.join(str(x) for x in [arm['arm_id'], len(feature['original_features']), len(feature['kept_names']), mean_train, mean_predict]) + '|')
    lines += ['',
        '## 调用、token与耗时', '',
        '|阶段|物理LLM调用|input tokens|output tokens|cached tokens|未知字段|', '|---|---|---|---|---|---|']
    for name, row in call_usage.items():
        lines.append('|' + '|'.join(str(x) for x in [name] + list(row.values())) + '|')
    lines += ['', '总调用保持183/300；4.4新增0。原4.3作废3次仍保留计费记录，修订后60次有效调用未重复运行。',
        '旧阶段时间只报告既存记录，不把中断等待或睡眠当active，也不声称缺失的历史active总时已知。', '```json',
        json.dumps({'prior_elapsed': cfg['prior_elapsed'], 'stage4_4': run_metadata}, ensure_ascii=False, indent=2), '```', '',
        '## 文件与单次门', '',
        'H4_feature_loop六个公开文件与已完成4.1/4.2/4.3目录保持只读；所有本阶段产物均在phase4_4。',
        'Synthetic marker: ../evaluation_started.json；HOLDOUT marker: ../holdout_evaluation_started.json。',
        '配对阶段只减已有共享组抽样，新增评测标签读取0、bootstrap重抽样0。',
        '阶段4.4完成，H4结束。不启动H5，等待用户决定下一步。']
    path = directory / 'H4_stage4_4_report.md'
    c.write(path, '\n'.join(lines) + '\n', exclusive=True)
    c.write(directory / 'reference_artifacts.json', references, exclusive=True)
    return {'report': str(path), 'validation_curve': str(directory / 'L1_L0_validation_MAE.png'),
            'top5_sources': str(directory / 'L1_winner_top5_features.json'),
            'resources': str(directory / 'resource_summary.json'), 'reference_artifacts': str(directory / 'reference_artifacts.json')}
