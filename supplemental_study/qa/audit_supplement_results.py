"""补充实验的只读独立复算：不导入训练器、不训练、不执行模型前向。

使用 NumPy 手工混淆矩阵核对保存的预测；只在本目录写审计 JSON。
结果不全返回 pending（退出码 2），不一致返回 failed（退出码 1）。
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


DATASET_SHA = 'd0a4382e015099558f5e392869eeda626e8f085dfe269db36141842f4a09dbaf'
SPLIT_SHA = 'fe88a96ca11a03c72bbc3d47bbc80ecacd335bac0f78018e31d1f4990ba28557'
CV_SPLIT_SHA = 'f5f818e287bc0033afcc47f2729c27ba8194474be2430d3b9612d94b420259a6'
CV_PLAN_SHA = '864cf5c8efe54d8ee885b9cc848f9f9ac7cd1eb2c30b9bdb27970e5bc1a18004'
METRICS = ('accuracy', 'balanced_accuracy', 'macro_f1_present', 'map_mean_accuracy',
           'map_equal_weight_macro_f1', 'evaluated_classes', 'pixels', 'maps')
TEST_MAPS = {'Map_Env_PP_PTFE', 'Map_Nylon_PEs_PS', 'Map_PC_2',
             'Map_PMMA_PET_PE_ABS', 'Map_PVC_PU_POM', 'Map_Silicone'}
BLANK_ONLY = {'Map_PC_2', 'Map_Silicone'}
ORIGINAL_MODELS = ('cosine_centroid', 'pca64_linear_svm', 'cnn_aug_seed42',
                   'cnn_aug_seed43', 'cnn_aug_seed44', 'cnn_noaug_seed42')


class AuditError(ValueError):
    pass


class Pending(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise AuditError(message)


def need(path):
    if not path.is_file():
        raise Pending(f'尚缺产物：{path}')
    return path


def read_json(path):
    return json.loads(need(path).read_text(encoding='utf-8'))


def read_csv(path, **kwargs):
    return pd.read_csv(need(path), **kwargs)


def finished(record, context):
    status = record.get('status')
    if status in ('running', 'pending', 'subset_completed', None):
        raise Pending(f'{context} 尚未完成，status={status}')
    require(status in ('finished', 'completed', 'passed'), f'{context}: 非成功状态 {status}')


def canonical_hash(value):
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
    return hashlib.sha256(raw.encode('utf-8')).hexdigest()


class Files:
    def __init__(self):
        self.cache = {}

    def sha(self, path):
        path = need(path).resolve()
        stat = path.stat()
        fingerprint = (stat.st_size, stat.st_mtime_ns)
        if path not in self.cache or self.cache[path][0] != fingerprint:
            value = hashlib.sha256()
            with path.open('rb') as stream:
                while block := stream.read(1024 * 1024):
                    value.update(block)
            after = path.stat()
            require((after.st_size, after.st_mtime_ns) == fingerprint, f'读取期间文件变化：{path}')
            self.cache[path] = (fingerprint, value.hexdigest())
        return self.cache[path][1]

    def verify(self, base, mapping, context):
        require(bool(mapping), f'{context}: 哈希清单为空')
        for relative, expected in mapping.items():
            path = (base / relative).resolve()
            require(path.is_relative_to(base.resolve()), f'{context}: 清单路径越界')
            require(self.sha(path) == expected, f'{context}: 哈希不符 {relative}')


def matrix_metrics(truth, prediction, groups, labels):
    """不调用项目 helper 或 sklearn.metrics；F1 平均有真值支持的类别。"""
    truth, prediction, groups = map(np.asarray, (truth, prediction, groups))
    require(len(truth) > 0 and len(truth) == len(prediction) == len(groups), '预测数组长度异常')
    encoding = {label: number for number, label in enumerate(labels)}
    require(set(truth).issubset(encoding) and set(prediction).issubset(encoding), '预测/真值超出类别字典')
    y = np.asarray([encoding[v] for v in truth], dtype=np.int64)
    p = np.asarray([encoding[v] for v in prediction], dtype=np.int64)
    cm = np.zeros((len(labels), len(labels)), dtype=np.float64)
    np.add.at(cm, (y, p), 1.)
    names, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    weighted = np.zeros_like(cm)
    np.add.at(weighted, (y, p), 1. / counts[inverse])
    support = cm.sum(1)
    present = support > 0

    def f1(matrix):
        denominator = matrix.sum(0) + matrix.sum(1)
        return np.divide(2 * np.diag(matrix), denominator,
                         out=np.zeros(len(labels)), where=denominator > 0)

    precision = np.divide(np.diag(cm), cm.sum(0), out=np.zeros(len(labels)), where=cm.sum(0) > 0)
    recall = np.divide(np.diag(cm), support, out=np.zeros(len(labels)), where=support > 0)
    scores = dict(accuracy=float(np.trace(cm) / len(y)), balanced_accuracy=float(recall[present].mean()),
                  macro_f1_present=float(f1(cm)[present].mean()),
                  map_mean_accuracy=float(np.mean([np.mean(y[groups == g] == p[groups == g]) for g in names])),
                  map_equal_weight_macro_f1=float(f1(weighted)[present].mean()),
                  evaluated_classes=int(present.sum()), pixels=len(y), maps=len(names))
    classes = {label: dict(precision=float(precision[i]), recall=float(recall[i]),
                           **{'f1-score': float(f1(cm)[i])}, support=int(support[i]))
               for i, label in enumerate(labels)}
    return scores, cm, classes


def compare(actual, expected, context, tolerance=1e-10):
    largest = 0.
    for key, value in expected.items():
        require(key in actual, f'{context}: 缺少指标 {key}')
        observed = float(actual[key])
        require(np.isfinite(observed) and np.isfinite(value), f'{context}: 非有限指标 {key}')
        error = abs(observed - float(value))
        require(error <= tolerance, f'{context}: {key}={observed}，复算={value}，误差={error}')
        largest = max(largest, error)
    return largest


def unique_rows(frame, columns, context):
    require(not frame.duplicated(columns).any(), f'{context}: {columns}重复')


def check_predictions(frame, indices, data, context, original_index=True):
    unique_rows(frame, ['map_id', 'pixel_index'], context)
    require(len(frame) == len(indices), f'{context}: 谱数不符')
    if original_index:
        require(np.array_equal(frame.original_index, indices), f'{context}: 原始索引/行序不符')
    for column, key in [('map_id', 'groups'), ('pixel_index', 'pixel_index'), ('true_label', 'y')]:
        require(np.array_equal(frame[column].to_numpy(), data[key][indices]), f'{context}: {column}不符')


def probability_check(path, frame, indices, labels):
    with np.load(need(path), allow_pickle=False) as archive:
        require(np.array_equal(archive['original_indices'], indices), f'{path}: 概率原始索引不符')
        if 'labels' in archive:
            require(archive['labels'].astype(str).tolist() == labels, f'{path}: 类别顺序不符')
        p = archive['probabilities']
    require(p.shape == (len(indices), len(labels)) and np.isfinite(p).all(), f'{path}: 概率形状或有限值异常')
    require(np.all((p >= 0) & (p <= 1)) and np.allclose(p.sum(1), 1, rtol=0, atol=2e-6), f'{path}: 非法概率')
    require(np.array_equal(np.asarray(labels)[p.argmax(1)], frame.predicted_label), f'{path}: argmax不符')
    if 'confidence' in frame:
        require(np.allclose(frame.confidence, p.max(1), rtol=0, atol=2e-6), f'{path}: 置信度不符')
    return p


def checkpoint(path, labels, hashes):
    # 仅反序列化自有 tensor/checkpoint，不构建模型、不执行 forward，也不读取 RF pickle。
    import torch
    state = torch.load(need(path), map_location='cpu', weights_only=True)
    require(state['labels'] == labels, f'{path}: checkpoint类别不符')
    for key, expected in hashes.items():
        require(state.get(key) == expected, f'{path}: checkpoint {key}不符')
    require(all(torch.isfinite(v).all().item() for v in state['state_dict'].values()
                if v.is_floating_point()), f'{path}: 权重含非有限值')
    return int(state['epoch'])


def check_history(path, metadata, score_column, max_epochs, patience, selected_epoch):
    history = read_csv(path)
    require(np.array_equal(history.epoch, np.arange(1, len(history) + 1)), f'{path}: epoch不连续')
    require(len(history) == metadata['epochs_completed'] and 1 <= len(history) <= max_epochs, f'{path}: 训练轮数不符')
    values = history[score_column].to_numpy(dtype=float)
    require(np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all(), f'{path}: 验证分数异常')
    best_epoch = int(np.argmax(values)) + 1  # 严格 > 更新，平分选最早epoch。
    require(selected_epoch == best_epoch, f'{path}: 所选epoch不是内验证首次最大值')
    require(len(history) == max_epochs or len(history) - best_epoch == patience, f'{path}: 早停耐心不符')
    require(np.isfinite(history.train_loss).all() and (history.train_loss >= 0).all(), f'{path}: 训练loss异常')
    if score_column == 'macro_f1_present':
        for key, expected in [('evaluated_classes', 13), ('pixels', 5752), ('maps', 6)]:
            require(key in history and (history[key] == expected).all(), f'{path}: 原验证支持{key}不符')
    return float(values.max())


def context(root, files):
    require(files.sha(root / 'data/processed/dataset.npz') == DATASET_SHA, '原数据SHA发生变化')
    require(files.sha(root / 'data/processed/map_split.csv') == SPLIT_SHA, '原划分SHA发生变化')
    with np.load(root / 'data/processed/dataset.npz', allow_pickle=False) as archive:
        data = {key: archive[key].astype(str) if key in ('y', 'groups') else archive[key]
                for key in ('y', 'groups', 'pixel_index')}
    require(len(data['y']) == 33119, '原数据谱数不符')
    index = read_csv(root / 'data/processed/spectrum_index.csv')
    unique_rows(index, ['map_id', 'pixel_index'], '原始谱索引')
    for column, key in [('map_id', 'groups'), ('pixel_index', 'pixel_index'), ('label', 'y')]:
        require(np.array_equal(index[column], data[key]), f'原始索引CSV与NPZ的{key}不符')
    split = read_csv(root / 'data/processed/map_split.csv')
    require(split.map_id.is_unique and set(split.map_id) == set(data['groups']), 'map划分不完备')
    roles = dict(zip(split.map_id, split.split))
    assignment = np.asarray([roles[g] for g in data['groups']])
    require(set(assignment) == {'train', 'validation', 'test'}, '原划分角色异常')
    data['idx'] = {role: np.flatnonzero(assignment == role) for role in ['train', 'validation', 'test']}
    data['labels'] = sorted(set(data['y']))
    for role, pixels, maps, classes in [('train', 20968, 22, 15), ('validation', 5752, 6, 13), ('test', 6399, 6, 15)]:
        indices = data['idx'][role]
        require((len(indices), len(set(data['groups'][indices])), len(set(data['y'][indices]))) == (pixels, maps, classes), f'{role}: 支持范围不符')
    role_groups = [set(data['groups'][data['idx'][role]]) for role in data['idx']]
    require(all(a.isdisjoint(b) for i, a in enumerate(role_groups) for b in role_groups[i + 1:]), '训练/验证/测试map混入')
    require(set(data['groups'][data['idx']['test']]) == TEST_MAPS, '原六张测试图改变')
    receipt = read_json(root / 'extension_study/results/cloud_extension_execution.json')
    require(receipt['frozen_before'] == receipt['frozen_after'] and receipt['original_frozen_inputs_unchanged'] is True, '历史冻结回执不符')
    files.verify(root, receipt['frozen_after'], '原始数据及第一阶段结果冻结')
    return data


def per_map_check(path, predictions, prediction_column, indices, data, labels, name):
    frame = read_csv(path)
    part = frame[frame.model == name] if 'model' in frame else frame
    require(part.map_id.is_unique and set(part.map_id) == set(data['groups'][indices]), f'{path}/{name}: 逐图记录不符')
    for group in sorted(set(data['groups'][indices])):
        mask = predictions.map_id.to_numpy() == group
        scores, _, _ = matrix_metrics(predictions.true_label[mask], predictions[prediction_column][mask], predictions.map_id[mask], labels)
        compare(part.set_index('map_id').loc[group].to_dict(), scores, f'{path}/{name}/{group}')


def original_stage1(root, data, files):
    base = root / 'results/final_test'
    record = read_json(base / 'evaluation_record.json')
    require(record['dataset_sha256'] == DATASET_SHA and record['split_sha256'] == SPLIT_SHA, '历史评价数据版本不符')
    table = read_csv(base / 'model_comparison.csv')
    require(table.model.is_unique and set(table.model) == set(ORIGINAL_MODELS), '历史模型集合不符')
    predictions = read_csv(base / 'pixel_predictions.csv')
    indices = data['idx']['test']
    check_predictions(predictions, indices, data, '历史六模型预测', original_index=False)
    errors = []
    for name in ORIGINAL_MODELS:
        path = root / ('results/baselines/' + name + '.joblib' if name in ORIGINAL_MODELS[:2] else 'results/' + name + '/best.pt')
        require(files.sha(path) == record['model_sha256'][name], f'{name}: 历史模型哈希变化')
        scores, _, _ = matrix_metrics(predictions.true_label, predictions[name], predictions.map_id, data['labels'])
        errors.append(compare(table.set_index('model').loc[name].to_dict(), scores, name))
        per_map_check(base / 'per_map_metrics.csv', predictions, name, indices, data, data['labels'], name)
        if name.startswith('cnn_'):
            metadata = read_json(root / 'results' / name / 'metadata.json')
            finished(metadata, name)
            require(metadata['dataset_sha256'] == DATASET_SHA and metadata['split_sha256'] == SPLIT_SHA, f'{name}: 元数据哈希不符')
            require(metadata['train_pixels'] == 20968 and metadata['validation_pixels'] == 5752 and metadata['smoke_only'] is False, f'{name}: 历史拟合范围不符')
            epoch = checkpoint(path, data['labels'], {'dataset_sha256': DATASET_SHA, 'split_sha256': SPLIT_SHA})
            best = check_history(root / 'results' / name / 'history.csv', metadata, 'macro_f1_present', 80, 12, epoch)
            compare(metadata, {'best_validation_macro_f1': best}, name)
            expected_seed = int(name.rsplit('seed', 1)[1])
            require(metadata['config']['seed'] == expected_seed and metadata['test_evaluated'] is False, f'{name}: 训练seed或测试声明不符')
            require(metadata['config']['augment'] is name.startswith('cnn_aug_'), f'{name}: 历史增强配置不符')
    return dict(status='passed', verified_models=6, maximum_metric_error=max(errors))


def stage1(root, data, files):
    original = original_stage1(root, data, files)
    plan_path = root / 'supplemental_study/protocol/stage1_plan.json'
    plan = read_json(plan_path)
    require(plan['expected_dataset_sha256'] == DATASET_SHA and plan['expected_split_sha256'] == SPLIT_SHA, '补充第一阶段计划版本不符')
    require(plan['posthoc_after_stage1_test_seen'] is True, '缺少后续补测声明')
    require(plan['neural_seeds'] == [42, 43, 44] and plan['random_forest_seed'] == 101, '补充seed改变')
    require(files.sha(root / 'results/final_test/model_comparison.csv') == plan['prior_results_sha256'], '首轮模型比较表改变')
    files.verify(root, plan['source_sha256'], '第一阶段补充冻结源码')
    plan_sha = files.sha(plan_path)
    configs = [dict(run_id='random_forest_300_seed101', model='random_forest', seed=101, parameters=plan['random_forest'])]
    configs += [dict(run_id=f'{model}_aug_seed{seed}', model=model, seed=seed, training=plan['training'])
                for model in ['mlp', 'resnet'] for seed in [42, 43, 44]]
    base = root / 'supplemental_study/results/stage1'
    metadata_records = {}
    model_hashes = {}
    for config in configs:
        rid = config['run_id']
        out = base / 'runs' / rid
        metadata = read_json(out / 'metadata.json')
        finished(metadata, rid)
        identity = dict(config=config, dataset_sha256=DATASET_SHA, split_sha256=SPLIT_SHA,
                        plan_sha256=plan_sha, source_sha256=plan['source_sha256'], labels=data['labels'])
        require(metadata['config'] == config and metadata['identity'] == identity and metadata['identity_sha256'] == canonical_hash(identity), f'{rid}: 训练身份不符')
        require(metadata['test_evaluated'] is False and metadata['posthoc_after_stage1_test_seen'] is True, f'{rid}: 训练测试政策声明不符')
        compare(metadata, dict(train_pixels=20968, validation_pixels=5752, train_maps=22,
                               validation_maps=6, validation_supported_classes=13, output_classes=15), rid)
        required = {'validation_predictions.csv', 'validation_metrics.json', 'validation_probabilities.npz'}
        model_file = 'model.joblib' if config['model'] == 'random_forest' else 'best.pt'
        required.add(model_file)
        if config['model'] != 'random_forest':
            required.add('history.csv')
        require(required.issubset(metadata['artifact_sha256']), f'{rid}: 必需产物未锁哈希')
        files.verify(out, metadata['artifact_sha256'], rid)
        validation = read_csv(out / 'validation_predictions.csv')
        check_predictions(validation, data['idx']['validation'], data, rid + '/validation')
        scores, _, _ = matrix_metrics(validation.true_label, validation.predicted_label, validation.map_id, data['labels'])
        compare(read_json(out / 'validation_metrics.json'), scores, rid + '/validation')
        compare(metadata, {'best_validation_macro_f1': scores['macro_f1_present']}, rid)
        probability_check(out / 'validation_probabilities.npz', validation, data['idx']['validation'], data['labels'])
        if config['model'] != 'random_forest':
            epoch = checkpoint(out / model_file, data['labels'], dict(identity_sha256=metadata['identity_sha256'],
                               dataset_sha256=DATASET_SHA, split_sha256=SPLIT_SHA, plan_sha256=plan_sha))
            best = check_history(out / 'history.csv', metadata, 'macro_f1_present', 80, 12, epoch)
            require(epoch == metadata['best_epoch'], f'{rid}: best_epoch与权重不符')
            compare(metadata, {'best_validation_macro_f1': best}, rid)
            counts = np.asarray([(data['y'][data['idx']['train']] == label).sum() for label in data['labels']])
            require(np.array_equal(metadata['training_class_counts'], counts), f'{rid}: 类别权重计数非训练部分')
            require(np.allclose(metadata['class_weights'], len(data['idx']['train']) / (15 * counts), rtol=0, atol=1e-12), f'{rid}: 类别权重不符')
        else:
            require(metadata['hyperparameter_search'] is False, f'{rid}: RF不应搜索测试参数')
            require(all(metadata['sklearn_get_params'][k] == v for k, v in plan['random_forest'].items()), f'{rid}: RF参数不符')
        model_hashes[rid] = files.sha(out / model_file)
        metadata_records[rid] = metadata
    final = base / 'final_test'
    receipt = read_json(final / 'evaluation_record.json')
    finished(receipt, 'stage1 final_test')
    expected_identity = dict(plan_sha256=plan_sha, source_sha256=plan['source_sha256'], dataset_sha256=DATASET_SHA,
                             split_sha256=SPLIT_SHA, model_sha256=model_hashes, labels=data['labels'])
    require(receipt['identity'] == expected_identity and receipt['identity_sha256'] == canonical_hash(expected_identity), '补充正式评价身份不符')
    require(receipt['test_evaluated'] is True and receipt['posthoc_after_stage1_test_seen'] is True, '缺少补充评价声明')
    evaluation_time = datetime.fromisoformat(receipt['started_at'])
    require(all(datetime.fromisoformat(m['finished_at']) <= evaluation_time for m in metadata_records.values()), '未全部训练完成就启动正式评价')
    files.verify(final, receipt['artifact_sha256'], '补充第一阶段正式预测')
    table = read_csv(final / 'model_comparison.csv')
    require(table.model.is_unique and set(table.model) == set(model_hashes), '补充正式模型集合不符')
    combined = read_csv(final / 'pixel_predictions.csv')
    indices = data['idx']['test']
    check_predictions(combined, indices, data, '补充正式联合预测')
    errors, recalculated = [], {}
    for config in configs:
        rid = config['run_id']
        predictions = read_csv(final / f'{rid}_predictions.csv')
        check_predictions(predictions, indices, data, rid + '/test')
        require(np.array_equal(combined[rid], predictions.predicted_label), f'{rid}: 联合预测列不一致')
        probability_check(final / f'{rid}_probabilities.npz', predictions, indices, data['labels'])
        scores, cm, classes = matrix_metrics(predictions.true_label, predictions.predicted_label, predictions.map_id, data['labels'])
        errors.append(compare(read_json(final / f'{rid}_metrics.json'), scores, rid))
        errors.append(compare(table.set_index('model').loc[rid].to_dict(), scores, rid + '/CSV'))
        row = table.set_index('model').loc[rid]
        require(row.family == config['model'] and int(row.seed) == config['seed'], f'{rid}: 表中模型/seed不符')
        stored_cm = read_csv(final / f'{rid}_confusion.csv', index_col=0)
        require(stored_cm.index.tolist() == data['labels'] and stored_cm.columns.tolist() == data['labels'] and np.array_equal(stored_cm, cm), f'{rid}: 混淆矩阵不符')
        stored_classes = read_csv(final / f'{rid}_classes.csv', index_col=0)
        for label, values in classes.items():
            compare(stored_classes.loc[label].to_dict(), values, rid + '/' + label)
        per_map_check(final / 'per_map_metrics.csv', predictions, 'predicted_label', indices, data, data['labels'], rid)
        recalculated[rid] = scores
    summaries = read_json(final / 'neural_seed_summary.json')
    require({r['model'] for r in summaries} == {'mlp', 'resnet'} and len(summaries) == 2, '神经seed汇总集合不符')
    for row in summaries:
        require(row['seeds'] == [42, 43, 44], '汇总seed顺序不符')
        values = [recalculated[f"{row['model']}_aug_seed{s}"] for s in [42, 43, 44]]
        compare(row, {target: float(np.mean([v[key] for v in values])) for target, key in [('accuracy_mean', 'accuracy'), ('macro_f1_mean', 'macro_f1_present')]}, row['model'])
        compare(row, {target: float(np.std([v[key] for v in values], ddof=1)) for target, key in [('accuracy_seed_sd', 'accuracy'), ('macro_f1_seed_sd', 'macro_f1_present')]}, row['model'])
    files.verify(root, plan['source_sha256'], '第一阶段审计结束时源码仍冻结')
    return dict(status='passed', original=original, verified_additional_models=7, maximum_metric_error=max(errors),
                statistical_unit='同一已查看测试集；三seed的SD仅为训练随机性，不是三个独立测试集',
                source_plan_sha256=plan_sha, recalculated_metrics=recalculated)


def independent_calibration(truth, probabilities, labels):
    """补测器明确用float64十等宽箱；不沿用旧训练器的float32边界。"""
    p = probabilities.astype(np.float64)
    y = np.asarray([labels.index(v) for v in truth])
    confidence = p.max(1)
    correct = p.argmax(1) == y
    bins = np.minimum(np.floor(confidence * 10).astype(int), 9)
    ece = sum(float(mask.mean() * abs(correct[mask].mean() - confidence[mask].mean()))
              for i in range(10) if (mask := bins == i).any())
    target = np.eye(len(labels))[y]
    return dict(ece_10_bins=ece,
                negative_log_likelihood=float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-8, 1)).mean()),
                multiclass_brier=float(np.square(p - target).sum(1).mean()))


def check_hierarchical(rows, fold_path, summary_path, by_scope=False):
    """独立按记录索引取数，不调用evaluator的aggregate。"""
    keys = (['scope'] if by_scope else []) + ['family', 'model']
    folds, summary = read_csv(fold_path), read_csv(summary_path)
    unique_rows(folds, keys + ['fold_id'], str(fold_path))
    unique_rows(summary, keys, str(summary_path))
    conditions = sorted({tuple(r[k] for k in keys) for r in rows})
    require(len(folds) == len(conditions) * 3 and len(summary) == len(conditions), '层次汇总行数不符')
    metric_names = list(METRICS[:5]) + ['ece_10_bins', 'negative_log_likelihood', 'multiclass_brier']
    for identity in conditions:
        subset = [r for r in rows if tuple(r[k] for k in keys) == identity]
        condition = dict(zip(keys, identity))
        selector = np.ones(len(summary), dtype=bool)
        for key, value in condition.items():
            selector &= summary[key].to_numpy() == value
        require(selector.sum() == 1, f'缺少条件汇总：{identity}')
        summary_row = summary[selector].iloc[0].to_dict()
        expected_seeds = {101, 202, 303} if condition['family'] == 'architecture' else {101}
        fold_values = []
        for fold in [1, 2, 3]:
            selected = [r for r in subset if r['fold_id'] == fold]
            require(len(selected) == len(expected_seeds) and {r['seed'] for r in selected} == expected_seeds, f'{identity}/fold{fold}: seed不齐或重复')
            expected = {key: float(np.mean([r[key] for r in selected])) for key in metric_names if all(key in r for r in selected)}
            selector = folds.fold_id.to_numpy() == fold
            for key, value in condition.items():
                selector &= folds[key].to_numpy() == value
            require(selector.sum() == 1, f'缺少折汇总：{identity}/fold{fold}')
            compare(folds[selector].iloc[0].to_dict(), dict(n_seeds=len(expected_seeds), **expected), str(fold_path))
            fold_values.append(expected)
        expected_summary = dict(n_folds=3, n_model_fits=3 * len(expected_seeds))
        for key in fold_values[0]:
            values = [r[key] for r in fold_values]
            expected_summary[key + '_mean'] = float(np.mean(values))
            expected_summary[key + '_fold_sd'] = float(np.std(values, ddof=1))
        compare(summary_row, expected_summary, str(summary_path))
        require('非CI' in summary_row['dispersion_basis'] and '非OOF' in summary_row['dispersion_basis'], '层次离散度声明不符')


def stage2(root, data, files):
    plan_path = root / 'supplemental_study/protocol/stage2_holdout_plan.json'
    plan = read_json(plan_path)
    require(plan['already_viewed_test'] is True and plan['new_external_test'] is False and plan['training_performed'] is False, '第二阶段必须明确旧holdout补测且不训练')
    files.verify(root, plan['frozen_source_files'], '第二阶段原冻结文件')
    require(files.sha(root / 'supplemental_study/src/evaluate_stage2_frozen_holdout.py') == plan['evaluator_sha256'], '补测器与计划冻结SHA不符')
    require(files.sha(root / 'extension_study/protocol/splits.json') == CV_SPLIT_SHA and files.sha(root / 'extension_study/protocol/experiment_plan.json') == CV_PLAN_SHA, '原开发CV协议/计划发生变化')
    splits = read_json(root / 'extension_study/protocol/splits.json')
    cv_plan = read_json(root / 'extension_study/protocol/experiment_plan.json')
    labels = splits['labels']
    require(len(labels) == 12 and set(labels) == set(data['labels']) - {'polycarbonate', 'polyurethane', 'silicone'}, '12类资格发生变化')
    all_test = data['idx']['test']
    indices = all_test[np.isin(data['y'][all_test], labels)]
    excluded = all_test[~np.isin(data['y'][all_test], labels)]
    removed = {label: int(np.sum(data['y'][excluded] == label)) for label in sorted(set(data['y'][excluded]))}
    require((len(indices), len(excluded)) == (5425, 974) and removed == {'polycarbonate': 578, 'polyurethane': 78, 'silicone': 318}, '第二阶段过滤支持不符')
    require(set(data['y'][indices]) == set(labels), '主评价缺少12类支持')
    population = plan['population']
    compare(population, dict(original_test_pixels=6399, eligible_pixels=5425, excluded_pixels=974,
                            input_length=1024, plastic_supported_maps=4, blank_only_maps=2), '冻结评价总体')
    require(population['labels'] == labels and population['excluded_label_counts'] == removed and set(population['test_maps']) == TEST_MAPS, '总体类别/图列表不符')
    require(population['development_default_exact_equal'] is True and population['development_mapping_checked_pixels'] == 25541, '未声明既有表示映射核对')
    study_path = root / 'extension_study/data/study_data.npz'
    study_sha = files.sha(study_path)
    with np.load(study_path, allow_pickle=False) as archive:
        original = archive['original_indices']
        require(np.array_equal(original, splits['eligible_population_indices']) and len(original) == 25541, '开发原行号不符')
        for key in ['y', 'groups', 'pixel_index']:
            require(np.array_equal(archive[key], data[key][original]), f'开发{key}映射不符')
    folds = {f['fold_id']: f for f in splits['folds']}
    for fold in folds.values():
        roles = [np.asarray(fold[role + '_indices']) for role in ['inner_train', 'inner_val', 'outer_eval']]
        require(all(len(np.unique(a)) == len(a) and set(a).issubset(set(original)) for a in roles), 'CV角色原行号重复/越界')
        require(all(set(a).isdisjoint(b) for i, a in enumerate(roles) for b in roles[i + 1:]), 'CV角色像素泄漏')
        maps = [set(data['groups'][a]) for a in roles]
        require(all(g.isdisjoint(TEST_MAPS) for g in maps) and all(a.isdisjoint(b) for i, a in enumerate(maps) for b in maps[i + 1:]), 'CV训练/验证/评价map泄漏')
        require(set(np.concatenate(roles)) == set(original), '单CV折角色没有覆盖开发总体')
    expected_configs = {c['run_id']: c for c in cv_plan['configs'] if c['family'] == 'architecture'}
    for fold in [1, 2, 3]:
        for model in ['cosine_centroid', 'pca64_linear_svm', 'random_forest_300']:
            rid = f'baseline_{model}_f{fold}'
            expected_configs[rid] = dict(run_id=rid, family='baseline', model=model, fold_id=fold, seed=101)
    require(len(plan['runs']) == len(expected_configs) == 36 and {r['config']['run_id'] for r in plan['runs']} == set(expected_configs), '36模型冻结清单不符')
    require(len({r['config']['run_id'] for r in plan['runs']}) == 36, '第二阶段重复run')
    support = {}
    for group in sorted(TEST_MAPS):
        full = all_test[data['groups'][all_test] == group]
        kept = indices[data['groups'][indices] == group]
        truth = data['y'][kept]
        counts = {label: int(np.sum(truth == label)) for label in labels}
        support[group] = dict(map_id=group, original_pixels=len(full), eligible_pixels=len(kept),
                              excluded_pixels=len(full) - len(kept), blank_pixels=counts['blank'],
                              plastic_pixels=int(np.sum(truth != 'blank')), has_plastic_support=bool(np.any(truth != 'blank')),
                              classes_present=len(set(truth)), labels_present=sorted(set(truth)), label_counts=counts)
    require(len(population['map_support']) == 6 and {r['map_id']: r for r in population['map_support']} == support, '冻结逐图支持不符')
    require({g for g, r in support.items() if not r['has_plastic_support']} == BLANK_ONLY, '背景专用图不符')
    scope_masks = dict(all_eligible_pixels_6_maps=np.ones(len(indices), dtype=bool),
                       plastic_supported_maps_4_maps=~np.isin(data['groups'][indices], sorted(BLANK_ONLY)),
                       plastic_pixels_only_4_maps=data['y'][indices] != 'blank',
                       blank_only_maps_2_maps=np.isin(data['groups'][indices], sorted(BLANK_ONLY)))
    require(set(plan['scopes']) == set(scope_masks) and [int(m.sum()) for m in scope_masks.values()] == [5425, 4348, 2470, 1077], '四个预设评价scope不符')
    base = root / 'supplemental_study/results/stage2_frozen_holdout'
    receipt = read_json(base / 'execution.json')
    finished(receipt, '第二阶段补测执行')
    plan_sha = files.sha(plan_path)
    require(receipt['supplemental_plan_sha256'] == plan_sha and receipt['population'] == population, '第二阶段执行身份不符')
    require(receipt['training_performed'] is False and receipt['already_viewed_test'] is True and receipt['new_external_test'] is False, '第二阶段执行声明不符')
    require(receipt['original_frozen_sources_unchanged'] is True and receipt['original_cv_kept_separate'] is True and receipt['oof_predictions_created'] is False and receipt['ensembles_created'] is False, '不变性/非OOF/不集成声明不符')
    compare(receipt, dict(evaluated_model_fits=36, evaluated_neural_models=27, evaluated_traditional_models=9,
                          probabilities_saved_for=30, nonprobabilistic_model_fits=6), '第二阶段执行数量')
    files.verify(base, receipt['output_files'], '第二阶段所有输出')
    require(read_json(base / 'population.json') == population, '结果总体JSON与计划不符')
    check_predictions(read_csv(base / 'eligible_holdout_index.csv'), indices, data, '5425条公共holdout索引')
    support_csv = read_csv(base / 'map_support.csv')
    require(support_csv.map_id.is_unique and set(support_csv.map_id) == TEST_MAPS, '结果逐图支持集合不符')
    for row in support_csv.to_dict('records'):
        expected = support[row['map_id']]
        compare(row, {k: v for k, v in expected.items() if isinstance(v, (int, bool))}, '逐图支持')
        require(row['labels_present'] == '|'.join(expected['labels_present']) and json.loads(row['label_counts']) == expected['label_counts'], '逐图类别支持不符')
    table = read_csv(base / 'run_metrics.csv')
    scoped_table = read_csv(base / 'scope_run_metrics.csv')
    references = read_csv(base / 'original_cv_reference.csv')
    require(table.run_id.is_unique and set(table.run_id) == set(expected_configs) and len(table) == 36, '补测run表不齐/重复')
    require(references.run_id.is_unique and set(references.run_id) == set(expected_configs), '原CV参考表不齐/重复')
    unique_rows(scoped_table, ['run_id', 'scope'], 'scope逐run表')
    require(len(scoped_table) == 144, '四scope应各有36条记录')
    run_values, scoped_values, errors = [], [], []
    source_hashes = dict(data_sha256=study_sha, protocol_sha256=CV_SPLIT_SHA, plan_sha256=CV_PLAN_SHA)
    for item in plan['runs']:
        config = item['config']
        rid = config['run_id']
        require(config == expected_configs[rid], f'{rid}: 与原CV核心配置不符')
        for key in ['weight', 'metadata', 'original_cv_metrics']:
            record = item[key]
            path = (root / record['path']).resolve()
            require(path.is_relative_to(root.resolve()) and files.sha(path) == record['sha256'] and path.stat().st_size == record['bytes'], f'{rid}: 来源{key}版本不符')
        source = read_json(root / item['metadata']['path'])
        finished(source, rid + '/CV来源')
        require(source['config'] == config and source['labels'] == labels and source['original_test_used'] is False, f'{rid}: 来源配置/测试政策不符')
        require(all(source[k] == v for k, v in source_hashes.items()), f'{rid}: 来源三hash不符')
        fold = folds[config['fold_id']]
        compare(source, dict(train_maps=len(fold['inner_train_maps']), train_pixels=len(fold['inner_train_indices'])), rid + '/来源训练范围')
        if config['family'] == 'architecture':
            require(config['seed'] in [101, 202, 303] and config['variant'] == 'default' and config['map_level'] == 'maps100' and config['augment'] is True and config['weighted_loss'] is True, f'{rid}: 非预设核心神经配置')
            epoch = checkpoint(root / item['weight']['path'], labels, source_hashes)
            best = check_history((root / item['metadata']['path']).parent / 'history.csv', source, 'inner_val_macro_f1', 120, 18, epoch)
            require(epoch == source['best_epoch'], f'{rid}: 原CV最佳epoch不符')
            compare(source, {'best_inner_val_macro_f1': best}, rid + '/原CV选epoch')
        elif config['model'] == 'pca64_linear_svm':
            scores = source['parameter_search']
            require({r['C'] for r in scores} == {.1, 1., 10.}, f'{rid}: SVM候选C改变')
            compare(source['inner_val_metrics'], {'macro_f1_present': max(r['inner_val_macro_f1'] for r in scores)}, rid + '/仅内验证选C')
        original_cv = read_json(root / item['original_cv_metrics']['path'])
        cv_row = references.set_index('run_id').loc[rid].to_dict()
        compare(cv_row, {k: v for k, v in original_cv.items() if isinstance(v, (int, float))}, rid + '/原CV参考')
        require(cv_row['source_path'] == item['original_cv_metrics']['path'] and cv_row['source_sha256'] == item['original_cv_metrics']['sha256'], f'{rid}: 原CV参考来源不符')
        out = base / 'runs' / rid
        metadata = read_json(out / 'metadata.json')
        finished(metadata, rid + '/holdout')
        require(metadata['config'] == config and metadata['source'] == item and metadata['supplemental_plan_sha256'] == plan_sha, f'{rid}: 新评价身份不符')
        require(metadata['labels'] == labels and metadata['input_shape'] == [5425, 1, 1024], f'{rid}: 新评价输入类别/形状不符')
        require(metadata['training_performed'] is False and metadata['original_cv_results_modified'] is False and metadata['already_viewed_test'] is True, f'{rid}: 新评价声明不符')
        files.verify(out, metadata['output_files'], rid + '/新预测产物')
        predictions = read_csv(out / 'predictions.csv')
        check_predictions(predictions, indices, data, rid + '/5425谱')
        scores, cm, classes = matrix_metrics(predictions.true_label, predictions.predicted_label, predictions.map_id, labels)
        has_probabilities = config['family'] == 'architecture' or config['model'] == 'random_forest_300'
        require(metadata['probabilities_available'] is has_probabilities, f'{rid}: 概率可用性错误')
        probabilities = None
        if has_probabilities:
            probabilities = probability_check(out / 'probabilities.npz', predictions, indices, labels)
            scores.update(independent_calibration(predictions.true_label.to_numpy(), probabilities, labels))
        else:
            require(not (out / 'probabilities.npz').exists() and 'confidence' not in predictions, f'{rid}: 不应伪造概率')
        saved_metrics = read_json(out / 'metrics.json')
        if not has_probabilities:
            require(not set(saved_metrics).intersection({'ece_10_bins', 'negative_log_likelihood', 'multiclass_brier'}), f'{rid}: 非概率模型不应有校准指标')
        errors.append(compare(saved_metrics, scores, rid + '/JSON'))
        errors.append(compare(metadata['metrics'], scores, rid + '/metadata'))
        run_row = dict(run_id=rid, **{k: config[k] for k in ['family', 'model', 'fold_id', 'seed']}, **scores)
        for key in ['family', 'model', 'fold_id', 'seed']:
            require(table.set_index('run_id').loc[rid, key] == config[key], f'{rid}: run表身份不符')
        errors.append(compare(table.set_index('run_id').loc[rid].to_dict(), scores, rid + '/总表'))
        run_values.append(run_row)
        stored_cm = read_csv(out / 'confusion.csv', index_col=0)
        require(stored_cm.index.tolist() == labels and stored_cm.columns.tolist() == labels and np.array_equal(stored_cm, cm), f'{rid}: 补测混淆不符')
        per_class = read_csv(out / 'class_metrics.csv')
        require(per_class.label.is_unique and set(per_class.label) == set(labels), f'{rid}: 类别指标不齐')
        for number, label in enumerate(labels):
            values = classes[label]
            compare(per_class.set_index('label').loc[label].to_dict(), dict(precision=values['precision'], recall=values['recall'], f1=values['f1-score'], support=values['support'], predicted_pixels=int(cm[:, number].sum())), rid + '/' + label)
        per_map_check(out / 'per_map_metrics.csv', predictions, 'predicted_label', indices, data, labels, rid)
        map_table = read_csv(out / 'per_map_metrics.csv').set_index('map_id')
        for group, values in support.items():
            compare(map_table.loc[group].to_dict(), {k: values[k] for k in ['has_plastic_support', 'plastic_pixels', 'blank_pixels']}, rid + '/' + group)
            require(map_table.loc[group, 'labels_present'] == '|'.join(values['labels_present']), f'{rid}/{group}: 逐图标签不符')
        for scope, mask in scope_masks.items():
            part = predictions[mask]
            values, _, _ = matrix_metrics(part.true_label, part.predicted_label, part.map_id, labels)
            if probabilities is not None:
                values.update(independent_calibration(part.true_label.to_numpy(), probabilities[mask], labels))
            matches = scoped_table[(scoped_table.run_id == rid) & (scoped_table.scope == scope)]
            require(len(matches) == 1, f'{rid}/{scope}: scope记录不齐')
            compare(matches.iloc[0].to_dict(), values, rid + '/' + scope)
            scoped_values.append(dict(run_id=rid, scope=scope, **{k: config[k] for k in ['family', 'model', 'fold_id', 'seed']}, **values))
    check_hierarchical(run_values, base / 'condition_fold_means.csv', base / 'condition_summary.csv')
    check_hierarchical(scoped_values, base / 'scope_fold_means.csv', base / 'scope_summary.csv', by_scope=True)
    files.verify(root, plan['frozen_source_files'], '第二阶段审计结束时来源仍冻结')
    return dict(status='passed', verified_models=36, neural_models=27, baseline_models=9, probabilities_verified=30,
                maximum_metric_error=max(errors), eligible_pixels=5425, excluded_pixels=974, excluded_label_counts=removed,
                classes=12, blank_only_maps=sorted(BLANK_ONLY), map_support=list(support.values()),
                scopes={key: int(value.sum()) for key, value in scope_masks.items()},
                aggregation='每模型每原外折先平均3个神经seed（基线每折1个），再对3组拟合求均值和ddof=1 SD。',
                interpretation='36个已拟合模型反复评价相同holdout；不是36个独立测试样本，不是OOF或新的外部盲测。',
                source_plan_sha256=plan_sha)


def main(args):
    root = args.root.resolve()
    qa = root / 'supplemental_study/qa'
    output = (args.output or qa / 'supplement_results_audit.json').resolve()
    require(output.is_relative_to(qa.resolve()) and output.suffix == '.json', '审计输出必须位于supplemental_study/qa目录且为JSON')
    report = dict(status='pending', created_at=datetime.now(timezone.utc).isoformat(), stages={},
                  method='NumPy手工构造普通及图等权混淆矩阵；不调用项目metrics或sklearn.metrics。',
                  restrictions=['只读源数据、权重、配置、日志与既有预测；不训练、不执行模型前向、不覆盖实验结果。',
                                '不能证明未记录的人工行为；检查源代码冻结身份、训练日志及所有可用记录的一致性。'])
    files = Files()
    try:
        data = context(root, files)
        report['inputs'] = dict(dataset_sha256=DATASET_SHA, split_sha256=SPLIT_SHA, dataset_rows=len(data['y']))
        names = ['stage1', 'stage2'] if args.stage == 'all' else [args.stage]
        for name in names:
            try:
                report['stages'][name] = (stage1 if name == 'stage1' else stage2)(root, data, files)
            except Pending as error:
                report['stages'][name] = dict(status='pending', reason=str(error))
            except Exception as error:
                report['stages'][name] = dict(status='failed', error=f'{type(error).__name__}: {error}')
        statuses = {v['status'] for v in report['stages'].values()}
        report['status'] = 'failed' if 'failed' in statuses else 'pending' if 'pending' in statuses else 'passed'
    except Pending as error:
        report.update(status='pending', reason=str(error))
    except Exception as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
    report['files_hashed'] = len(files.cache)
    qa.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(dict(status=report['status'], output=str(output), stages=report['stages']), ensure_ascii=False))
    return {'passed': 0, 'failed': 1, 'pending': 2}[report['status']]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--stage', choices=['all', 'stage1', 'stage2'], default='all')
    parser.add_argument('--output', type=Path, help='仅允许写本项目supplemental_study/qa内的JSON')
    sys.exit(main(parser.parse_args()))
