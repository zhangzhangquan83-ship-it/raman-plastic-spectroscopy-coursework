"""汇总已经完成的15类固定测试；不拟合、不加载模型、不计算新预测。

仅在全部7条补充测试结果及完成回执齐备后，生成13运行明细和6方法主表。
未完成时输出pending状态并退出2，不把未完成实验标作通过。
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

PROJECT = Path(__file__).resolve().parents[2]
METRICS = ('accuracy', 'macro_f1_present', 'map_mean_accuracy', 'map_equal_weight_macro_f1')
ORDER = ('cosine_centroid', 'pca64_linear_svm', 'random_forest', 'cnn', 'mlp', 'resnet')
DISPLAY = dict(cosine_centroid='余弦类中心', pca64_linear_svm='PCA64加线性SVM',
               random_forest='随机森林300树', cnn='增强1D-CNN', mlp='增强MLP', resnet='增强1D-ResNet')
HISTORICAL = ('cosine_centroid', 'pca64_linear_svm', 'cnn_aug_seed42', 'cnn_aug_seed43',
              'cnn_aug_seed44', 'cnn_noaug_seed42')
SUPPLEMENT = ('random_forest_300_seed101', 'mlp_aug_seed42', 'mlp_aug_seed43', 'mlp_aug_seed44',
              'resnet_aug_seed42', 'resnet_aug_seed43', 'resnet_aug_seed44')
RUN_FIELDS = ('model', 'family', 'seed', 'training_condition', 'run_origin', 'accuracy',
              'balanced_accuracy', 'macro_f1_present', 'map_mean_accuracy',
              'map_equal_weight_macro_f1', 'evaluated_classes', 'pixels', 'maps')
SUMMARY_FIELDS = ('method', 'display_name', 'n_runs', 'seeds',
                  *(name for metric in METRICS for name in (metric + '_mean', metric + '_seed_sd')),
                  'evaluated_classes', 'pixels', 'maps', 'run_origin', 'posthoc_supplement')


class Pending(RuntimeError):
    """结果尚未齐备，不能发布正式汇总。"""


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        while block := f.read(1024 * 1024):
            h.update(block)
    return h.hexdigest()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(text, encoding='utf-8')
    temporary.replace(path)


def write_json(path: Path, value: dict) -> None:
    write_text(path, json.dumps(value, ensure_ascii=False, indent=2))


def write_csv(path: Path, fields, rows) -> None:
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def normalize_run(row: dict, origin: str) -> dict:
    name = row['model']
    if name in ('cosine_centroid', 'pca64_linear_svm'):
        family, seed, condition = name, '', 'not_applicable'
    elif name == 'random_forest_300_seed101':
        family, seed, condition = 'random_forest', 101, 'not_applicable'
    else:
        family = name.split('_', 1)[0]
        if family not in ('cnn', 'mlp', 'resnet') or '_seed' not in name:
            raise ValueError(f'未知测试运行：{name}')
        seed = int(name.rsplit('_seed', 1)[1])
        condition = 'no_augmentation_ablation' if '_noaug_' in name else 'augmentation_weighted_CE'
    output = dict(model=name, family=family, seed=seed, training_condition=condition, run_origin=origin)
    for metric in (*METRICS, 'balanced_accuracy'):
        value = float(row[metric])
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f'{name}的{metric}不是0至1范围内的有限数值。')
        output[metric] = value
    for field, expected in (('evaluated_classes', 15), ('pixels', 6399), ('maps', 6)):
        output[field] = int(row[field])
        if output[field] != expected:
            raise ValueError(f'{name}的{field}不符合固定15类/6399谱/6图。')
    return output


def aggregate_runs(rows: list[dict]) -> list[dict]:
    """CNN仅使用三次增强运行；无增强运行留在明细，不进入主表均值。"""
    summary = []
    for method in ORDER:
        selected = [r for r in rows if r['family'] == method and
                    r['training_condition'] != 'no_augmentation_ablation']
        expected_count = 3 if method in ('cnn', 'mlp', 'resnet') else 1
        if len(selected) != expected_count:
            raise ValueError(f'{method}应有{expected_count}个主表运行，实际为{len(selected)}。')
        if expected_count == 3 and sorted(r['seed'] for r in selected) != [42, 43, 44]:
            raise ValueError(f'{method}种子应完整包含42、43、44。')
        value = dict(method=method, display_name=DISPLAY[method], n_runs=len(selected),
                     seeds=','.join(str(r['seed']) for r in selected if r['seed'] != ''),
                     evaluated_classes=15, pixels=6399, maps=6,
                     run_origin='historical_stage1' if method in ('cosine_centroid', 'pca64_linear_svm', 'cnn') else 'posthoc_supplement',
                     posthoc_supplement=method in ('random_forest', 'mlp', 'resnet'))
        for metric in METRICS:
            values = [r[metric] for r in selected]
            value[metric + '_mean'] = statistics.mean(values)
            value[metric + '_seed_sd'] = statistics.stdev(values) if len(values) > 1 else None
        summary.append(value)
    return summary


def formatted(mean: float, sd: float | None, percentage=False) -> str:
    factor, digits = (100, 2) if percentage else (1, 4)
    text = f'{mean * factor:.{digits}f}'
    if sd is not None:
        text += f' ± {sd * factor:.{digits}f}'
    return text + ('%' if percentage else '')


def make_report(rows, summary, identity) -> str:
    lines = [
        '# 第一阶段十五类六方法补充比较', '',
        '本次在原有22张训练图、6张验证图和6张固定测试图的划分下，补充随机森林、MLP和一维ResNet，使第一阶段覆盖六类方法。测试均为6,399条光谱、15类。历史结果保持原样，新增结果独立保存。', '',
        '这是已查看过第一阶段测试表现之后开展的补充比较。新增设置在此次训练前冻结，测试不用于选参数、种子或轮次；该设计仍属于同一固定测试集的后续使用，不能称为从未接触过的新盲测或独立外部验证。', '',
        '## 六类方法主表', '',
        '| 方法 | 运行数 | 像素准确率 | 15类宏F1 | 图平均准确率 | 图等权宏F1 |',
        '| --- | ---: | ---: | ---: | ---: | ---: |',
    ]
    for row in summary:
        cells = [row['display_name'], str(row['n_runs'])]
        cells.extend(formatted(row[metric + '_mean'], row[metric + '_seed_sd'], metric in ('accuracy', 'map_mean_accuracy'))
                     for metric in METRICS)
        lines.append('| ' + ' | '.join(cells) + ' |')
    lines += [
        '',
        'CNN、MLP和ResNet各汇总预定种子42、43、44三次增强训练；±为同一测试集上训练种子间的样本标准差（ddof=1），不是独立物理样本的置信区间。准确率列的±采用百分点。三种传统方法各只有一个运行，因此不报告种子标准差，CSV对应单元格留空。', '',
        '无增强CNN seed42属于消融对照，保留在下面的完整运行表，不混入主表CNN均值。', '',
        '## 全部十三条正式测试记录', '',
        '| 运行 | 来源 | 像素准确率 | 15类宏F1 | 图平均准确率 | 图等权宏F1 |',
        '| --- | --- | ---: | ---: | ---: | ---: |',
    ]
    for row in rows:
        origin = '历史首阶段' if row['run_origin'] == 'historical_stage1' else '此次补充'
        cells = [row['model'], origin]
        cells.extend(formatted(row[metric], None, metric in ('accuracy', 'map_mean_accuracy')) for metric in METRICS)
        lines.append('| ' + ' | '.join(cells) + ' |')
    lines += [
        '', '## 训练与解释范围', '',
        '- 新增MLP和ResNet使用15类输出头，从随机初始化训练，最多80轮、早停耐心12轮、批量256、初始学习率0.001、AdamW权重衰减0.0001、余弦退火和训练类计数加权交叉熵，启用与原CNN相同的增强定义。',
        '- 最佳轮次严格按原6张验证图中13个有支持类别的宏F1选择；验证集中PU和Sil无支持。选定权重后不合并训练和验证图重新拟合。',
        '- 随机森林固定300棵树、最小叶节点样本数2、balanced_subsample和random_state=101，不按测试分数搜索参数。',
        '- 历史神经网络训练使用RTX 3090，新增神经网络使用RTX 3050 Ti Laptop GPU。即使都使用PyTorch 2.8.0+cu128，GPU、运行环境和硬件负载不同，训练用时不可直接用于架构速度排名。相同种子也不意味着不同架构逐样本经历完全相同的随机扰动。',
        '- 这13条记录不是13种算法，也不是13个独立数据集。主表包含六类方法，所有成绩仍来自相同6张测试采集图。采集图互斥不保证物理标本独立。',
        '- 图平均准确率是各图准确率的算术平均；图等权宏F1按每条谱的权重1/所在图谱数构造等图权重的分类评价，不是各图F1的简单平均。',
        '- 第一阶段15类固定测试与第二阶段12类开发图交叉验证属于不同任务范围，分数不能直接相减来宣称模型提升。', '',
        '## 原始结果与核查', '',
        '完整运行表来自历史与补充正式测试CSV；主表仅对预定运行计算均值和样本标准差。汇总脚本不调用模型，不计算新预测，不重新训练。输入哈希记录在summary_status.json，原始CSV和回执文件在汇总前后保持一致。', '',
        f"历史测试表SHA256：{identity['historical_comparison_sha256']}", '',
        f"补充测试表SHA256：{identity['supplement_comparison_sha256']}", '',
        f"冻结计划SHA256：{identity['plan_sha256']}", '',
    ]
    return '\n'.join(lines)


def main(args) -> int:
    root = args.root.resolve()
    out = args.output_dir.resolve() if args.output_dir else root / 'supplemental_study/results/stage1/summary'
    out.mkdir(parents=True, exist_ok=True)
    status_path = out / 'summary_status.json'
    old = root / 'results/final_test/model_comparison.csv'
    new = root / 'supplemental_study/results/stage1/final_test/model_comparison.csv'
    old_record_path = root / 'results/final_test/evaluation_record.json'
    new_record_path = new.parent / 'evaluation_record.json'
    plan_path = root / 'supplemental_study/protocol/stage1_plan.json'
    status = dict(status='pending', created_at_utc=datetime.now(timezone.utc).isoformat(),
                  real_training_started=False, new_predictions_computed=False)
    try:
        missing = [str(p.relative_to(root)) for p in (old, new, old_record_path, new_record_path, plan_path) if not p.is_file()]
        if missing:
            raise Pending('正式输入尚未齐备：' + ', '.join(missing))
        protected = {str(p.relative_to(root)): sha(p) for p in (old, new, old_record_path, new_record_path, plan_path)}
        plan, historical_record, supplement_record = read_json(plan_path), read_json(old_record_path), read_json(new_record_path)
        if supplement_record.get('status') != 'finished' or supplement_record.get('test_evaluated') is not True:
            raise Pending('补充正式测试回执尚未标记finished且test_evaluated=true。')
        historical_rows, supplement_rows = read_csv(old), read_csv(new)
        if len(supplement_rows) < 7:
            raise Pending(f'补充正式测试目前只有{len(supplement_rows)}行，需要7行。')
        if len(historical_rows) != 6 or {r['model'] for r in historical_rows} != set(HISTORICAL):
            raise ValueError('历史测试表应有预定的6条唯一记录。')
        if len(supplement_rows) != 7 or {r['model'] for r in supplement_rows} != set(SUPPLEMENT):
            raise ValueError('补充测试表应有预定的7条唯一记录。')
        if plan.get('posthoc_after_stage1_test_seen') is not True or supplement_record.get('posthoc_after_stage1_test_seen') is not True:
            raise ValueError('补充测试必须明确披露其发生在历史测试成绩已被查看之后。')
        if plan.get('prior_results_sha256') != sha(old):
            raise ValueError('历史测试表与冻结补充计划中prior_results_sha256不符。')
        evaluation_identity = supplement_record['identity']
        if evaluation_identity['plan_sha256'] != sha(plan_path):
            raise ValueError('正式补充评价的计划哈希不符。')
        if supplement_record['artifact_sha256'].get('model_comparison.csv') != sha(new):
            raise ValueError('补充测试表与完成回执中记录的哈希不符。')
        for key in ('dataset_sha256', 'split_sha256'):
            if historical_record[key] != evaluation_identity[key] or plan['expected_' + key] != evaluation_identity[key]:
                raise ValueError(f'两个测试阶段的{key}不一致。')
        if sorted(historical_record['test_maps']) != sorted(supplement_record['test_maps']) or len(supplement_record['test_maps']) != 6:
            raise ValueError('两组结果不对应相同六张固定测试图。')
        if set(evaluation_identity['model_sha256']) != set(SUPPLEMENT):
            raise ValueError('补充正式评价没有锁定全部7个预定运行。')
        for run in SUPPLEMENT:
            run_dir = root / 'supplemental_study/results/stage1/runs' / run
            metadata = read_json(run_dir / 'metadata.json')
            if metadata.get('status') != 'finished' or metadata['identity']['plan_sha256'] != sha(plan_path):
                raise ValueError(f'训练完成回执或计划不符：{run}')
            checkpoint = run_dir / ('model.joblib' if run.startswith('random_forest') else 'best.pt')
            if sha(checkpoint) != evaluation_identity['model_sha256'][run]:
                raise ValueError(f'正式测试权重哈希不符：{run}')
        original_by_name = {row['model']: row for row in historical_rows}
        supplement_by_name = {row['model']: row for row in supplement_rows}
        rows = [normalize_run(original_by_name[name], 'historical_stage1') for name in HISTORICAL]
        rows += [normalize_run(supplement_by_name[name], 'posthoc_supplement') for name in SUPPLEMENT]
        summary = aggregate_runs(rows)
        if len(rows) != 13 or len(summary) != 6:
            raise AssertionError('完整运行与主表行数异常。')
        identity = dict(historical_comparison_sha256=sha(old), supplement_comparison_sha256=sha(new),
                        plan_sha256=sha(plan_path), protected_input_sha256=protected)
        report = make_report(rows, summary, identity)
        # 在任何最终输出之前再次确认所有只读输入没有发生变化。
        if any(sha(root / relative) != expected for relative, expected in protected.items()):
            raise ValueError('汇总期间原始输入发生变化。')
        write_csv(out / 'combined_test_runs.csv', RUN_FIELDS, rows)
        write_csv(out / 'six_method_summary.csv', SUMMARY_FIELDS, summary)
        write_text(out / 'stage1_supplement_report.md', report)
        if any(sha(root / relative) != expected for relative, expected in protected.items()):
            raise ValueError('输出后原始输入哈希校验不符。')
        status.update(status='finished', passed=True, identity=identity,
                      combined_run_count=13, main_method_count=6,
                      cnn_main_runs=['cnn_aug_seed42', 'cnn_aug_seed43', 'cnn_aug_seed44'],
                      noaug_cnn_in_main_mean=False, traditional_seed_sd=None,
                      outputs_sha256={name: sha(out / name) for name in
                                      ('combined_test_runs.csv', 'six_method_summary.csv', 'stage1_supplement_report.md')})
        write_json(status_path, status)
        print(json.dumps(status, ensure_ascii=False))
        return 0
    except Pending as error:
        status.update(status='pending', passed=False, reason=str(error),
                      stale_final_outputs_present=any((out / name).exists() for name in
                                                     ('combined_test_runs.csv', 'six_method_summary.csv', 'stage1_supplement_report.md')))
        write_json(status_path, status)
        print(json.dumps(status, ensure_ascii=False))
        return 2
    except Exception as error:
        status.update(status='failed', passed=False, error_type=type(error).__name__, reason=str(error))
        write_json(status_path, status)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=PROJECT)
    parser.add_argument('--output-dir', type=Path)
    sys.exit(main(parser.parse_args()))
