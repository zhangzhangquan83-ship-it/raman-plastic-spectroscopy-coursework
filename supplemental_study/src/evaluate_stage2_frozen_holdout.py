"""对第二阶段已冻结模型评价此前已查看的六张测试图，不训练或挑选模型。

先以 --freeze-plan 核验输入并冻结方案；随后普通调用才进行36个已有模型的推理。
--verify-only 只核验冻结输入与已有输出。此评价不是新的外部测试，也不是外折OOF。
所有新增文件仅位于 supplemental_study，原首轮及 extension_study 产物保持不变。
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch


PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'src'))
sys.path.insert(0, str(PROJECT / 'extension_study/src'))
from study_models import build_model
# joblib中自主训练的类使用原始稳定模块名，不加载其他来源的模型。
from common import CosineCentroid  # noqa: F401

SCHEMA_VERSION = 1
CLASSIFICATION_METRICS = ['accuracy', 'balanced_accuracy', 'macro_f1_present',
                          'map_mean_accuracy', 'map_equal_weight_macro_f1']
CALIBRATION_METRICS = ['ece_10_bins', 'negative_log_likelihood', 'multiclass_brier']
CORE_MODELS = ['cnn', 'mlp', 'resnet']
BASELINES = ['cosine_centroid', 'pca64_linear_svm', 'random_forest_300']
SEEDS = [101, 202, 303]
WARNING = ('原六张测试图的首轮结果已被查看。本次只对冻结的第二阶段模型补充评价，'
           '不是新的外部测试、不是外折OOF，不使用本次分数选择模型/外折/种子/预处理或集成。'
           '每折先平均训练种子，再报告三种外折训练所得模型的均值与样本标准差；'
           '此离散度来自同一holdout上的模型拟合差异，不是独立测试样本的置信区间。')
SCOPE_NOTES = {
    'all_eligible_pixels_6_maps': '六图中属于冻结12类的全部5425条像素，包含背景。',
    'plastic_supported_maps_4_maps': '只保留仍有塑料真值的四图，但这些图内背景像素仍计入。',
    'plastic_pixels_only_4_maps': '只保留真值为非blank的像素；预测为blank仍计为错误，不能与含背景12类宏F1直接比较。',
    'blank_only_maps_2_maps': '类别过滤后只剩blank真值的两图；只能说明背景识别，不能支持塑料材料识别。',
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def short_file_record(root: Path, path: Path) -> dict[str, Any]:
    require(path.is_file(), f'缺少冻结源文件：{path}')
    return dict(path=path.relative_to(root).as_posix(), bytes=path.stat().st_size, sha256=sha256(path))


def frozen_source_snapshot(root: Path) -> dict[str, str]:
    """只读记录原始源文件；新增supplemental目录不会进入快照。"""
    files = {root/'data/processed/dataset.npz', root/'data/processed/map_split.csv', root/'src/common.py',
             root/'extension_study/data/study_data.npz', root/'extension_study/data/study_data_audit.json'}
    for directory in [root/'results/final_test', root/'extension_study/protocol',
                      root/'extension_study/results', root/'extension_study/src']:
        require(directory.is_dir(), f'缺少原始目录：{directory}')
        files.update(path for path in directory.rglob('*') if path.is_file()
                     and '__pycache__' not in path.parts and path.suffix not in {'.pyc', '.tmp'})
    return {path.relative_to(root).as_posix(): sha256(path) for path in sorted(files)}


def confusion_metrics(y: np.ndarray, pred: np.ndarray, maps: np.ndarray,
                      n_classes: int) -> dict[str, float | int]:
    """从numpy混淆矩阵独立计算；图等权F1不是各图F1的算术均值。"""
    y = np.asarray(y, dtype=np.int64)
    pred = np.asarray(pred, dtype=np.int64)
    maps = np.asarray(maps).astype(str)
    require(y.ndim == pred.ndim == maps.ndim == 1 and len(y) == len(pred) == len(maps) > 0,
            '指标输入必须为长度相同的非空一维数组。')
    require(np.all((y >= 0) & (y < n_classes)) and np.all((pred >= 0) & (pred < n_classes)),
            '类别编码超出冻结范围。')
    cm = np.bincount(y*n_classes+pred, minlength=n_classes*n_classes).reshape(n_classes, n_classes)
    present = cm.sum(axis=1) > 0
    tp = np.diag(cm).astype(np.float64)
    recall = np.divide(tp, cm.sum(axis=1), out=np.zeros(n_classes), where=cm.sum(axis=1) != 0)
    denom = cm.sum(axis=0) + cm.sum(axis=1)
    f1 = np.divide(2*tp, denom, out=np.zeros(n_classes), where=denom != 0)
    unique, inverse, counts = np.unique(maps, return_inverse=True, return_counts=True)
    weighted = np.bincount(y*n_classes+pred, weights=1.0/counts[inverse],
                           minlength=n_classes*n_classes).reshape(n_classes, n_classes)
    w_denom = weighted.sum(axis=0)+weighted.sum(axis=1)
    w_f1 = np.divide(2*np.diag(weighted), w_denom, out=np.zeros(n_classes), where=w_denom != 0)
    return dict(accuracy=float(np.mean(y == pred)), balanced_accuracy=float(recall[present].mean()),
                macro_f1_present=float(f1[present].mean()),
                map_mean_accuracy=float(np.mean([np.mean(pred[maps == g] == y[maps == g]) for g in unique])),
                map_equal_weight_macro_f1=float(w_f1[present].mean()), evaluated_classes=int(present.sum()),
                pixels=int(len(y)), maps=int(len(unique)))


def class_table(y: np.ndarray, pred: np.ndarray, labels: list[str]) -> pd.DataFrame:
    k = len(labels)
    cm = np.bincount(y*k+pred, minlength=k*k).reshape(k, k)
    tp = np.diag(cm).astype(np.float64)
    true_n, predicted_n = cm.sum(axis=1), cm.sum(axis=0)
    precision = np.divide(tp, predicted_n, out=np.zeros(k), where=predicted_n != 0)
    recall = np.divide(tp, true_n, out=np.zeros(k), where=true_n != 0)
    denom = true_n+predicted_n
    f1 = np.divide(2*tp, denom, out=np.zeros(k), where=denom != 0)
    return pd.DataFrame(dict(label=labels, precision=precision, recall=recall, f1=f1,
                             support=true_n, predicted_pixels=predicted_n))


def calibration(y: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    require(probabilities.ndim == 2 and len(probabilities) == len(y), '概率矩阵行数不符。')
    require(np.isfinite(probabilities).all() and np.all(probabilities >= 0)
            and np.all(probabilities <= 1) and np.allclose(probabilities.sum(axis=1), 1, atol=5e-6),
            '概率矩阵存在非有限值、负值或行和不为1。')
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == y
    ece = 0.0
    for i in range(10):
        mask = (confidence >= i/10) & ((confidence < (i+1)/10) if i < 9 else (confidence <= 1))
        if mask.any():
            ece += float(mask.mean()*abs(correct[mask].mean()-confidence[mask].mean()))
    target = np.eye(probabilities.shape[1], dtype=np.float64)[y]
    return dict(ece_10_bins=ece,
                negative_log_likelihood=float(-np.log(np.clip(probabilities[np.arange(len(y)), y], 1e-8, 1)).mean()),
                multiclass_brier=float(np.sum((probabilities-target)**2, axis=1).mean()))


def load_population(root: Path, splits: dict[str, Any]) -> dict[str, Any]:
    """从主NPZ映射开发输入并逐值核对，然后只选既有六图中的12类像素。"""
    labels = [str(s) for s in splits['labels']]
    require(len(labels) == 12 and len(set(labels)) == 12 and 'blank' in labels, '冻结标签应为含blank的12类。')
    with np.load(root/'data/processed/dataset.npz', allow_pickle=False) as source:
        X = source['X']
        text_y = source['y'].astype(str)
        maps = source['groups'].astype(str)
        pixels = source['pixel_index'].astype(np.int64)
        axis = source['wavenumbers']
    require(X.dtype == np.float32 and X.shape == (33119, 1024) and np.isfinite(X).all(), '主数据特征不符。')
    require(axis.shape == (1024,) and np.isfinite(axis).all() and np.all(np.diff(axis) > 0), '频轴必须升序且为1024点。')
    with np.load(root/'extension_study/data/study_data.npz', allow_pickle=False) as study:
        original = study['original_indices'].astype(np.int64)
        default = study['X_default']
        require(original.shape == (25541,) and len(np.unique(original)) == len(original)
                and np.all(np.diff(original) > 0) and np.all((original >= 0) & (original < len(X))),
                '开发original_indices必须为25541条唯一升序全局行号。')
        require(default.dtype == np.float32 and default.shape == (25541, 1024), '开发默认特征形状不符。')
        require(np.array_equal(X[original], default), '主NPZ开发映射与X_default不逐值相等，禁止混用表示。')
        require(np.array_equal(text_y[original], study['y'].astype(str))
                and np.array_equal(maps[original], study['groups'].astype(str))
                and np.array_equal(pixels[original], study['pixel_index']), '开发标签/map/原像素行号映射不一致。')
    require(np.array_equal(original, np.asarray(splits['eligible_population_indices'], dtype=np.int64)),
            '开发原行号与冻结划分不同。')
    table = pd.read_csv(root/'data/processed/map_split.csv')
    require(table.map_id.is_unique and set(table.map_id) == set(maps), '首阶段map划分与主NPZ不同。')
    test_maps = sorted(table.loc[table.split == 'test', 'map_id'].astype(str))
    require(len(test_maps) == 6 and test_maps == sorted(splits['original_test_maps_excluded']), '原六测试图不符。')
    require(set(maps[original]).isdisjoint(test_maps), '开发数据含原测试图。')
    original_test = np.flatnonzero(np.isin(maps, test_maps))
    eligible = original_test[np.isin(text_y[original_test], labels)]
    excluded = original_test[~np.isin(text_y[original_test], labels)]
    excluded_counts = {label: int(np.sum(text_y[excluded] == label)) for label in sorted(np.unique(text_y[excluded]))}
    require(len(original_test) == 6399 and len(eligible) == 5425 and len(excluded) == 974,
            '6399→5425的冻结过滤数量不符。')
    require(excluded_counts == {'polycarbonate': 578, 'polyurethane': 78, 'silicone': 318},
            '排除类别/数量不符，不可依据评价结果改变过滤规则。')
    require(set(text_y[eligible]) == set(labels), '过滤后评价必须覆盖全部12类。')
    support = []
    for group in test_maps:
        full = original_test[maps[original_test] == group]
        kept = eligible[maps[eligible] == group]
        truth = text_y[kept]
        plastic = int(np.sum(truth != 'blank'))
        support.append(dict(map_id=group, original_pixels=len(full), eligible_pixels=len(kept),
                            excluded_pixels=len(full)-len(kept), blank_pixels=int(np.sum(truth == 'blank')),
                            plastic_pixels=plastic, has_plastic_support=plastic > 0,
                            classes_present=len(np.unique(truth)), labels_present=sorted(np.unique(truth).tolist()),
                            label_counts={label:int(np.sum(truth == label)) for label in labels}))
    plastic_maps = [s['map_id'] for s in support if s['has_plastic_support']]
    blank_only_maps = [s['map_id'] for s in support if not s['has_plastic_support']]
    require(len(plastic_maps) == 4 and len(blank_only_maps) == 2, '过滤后应为4张塑料支持图和2张blank-only图。')
    frame = pd.DataFrame(dict(original_index=eligible, map_id=maps[eligible], pixel_index=pixels[eligible],
                              true_label=text_y[eligible]))
    y = np.array([labels.index(s) for s in text_y[eligible]], dtype=np.int64)
    scopes = {'all_eligible_pixels_6_maps':np.ones(len(y), dtype=bool),
              'plastic_supported_maps_4_maps':np.isin(maps[eligible], plastic_maps),
              'plastic_pixels_only_4_maps':text_y[eligible] != 'blank',
              'blank_only_maps_2_maps':np.isin(maps[eligible], blank_only_maps)}
    return dict(X=X[eligible].copy(), y=y, frame=frame, labels=labels, test_maps=test_maps,
                excluded_counts=excluded_counts, map_support=support, scopes=scopes,
                development_mapping_checked_pixels=len(original), development_default_exact_equal=True)


def source_inventory(root: Path, plan: dict[str, Any], splits: dict[str, Any],
                     snapshot: dict[str, str]) -> list[dict[str, Any]]:
    configs = [dict(c) for c in plan['configs'] if c['family'] == 'architecture']
    require(len(configs) == 27, '只能选择冻结architecture核心27次神经拟合。')
    identities = {(c['model'], int(c['fold_id']), int(c['seed'])) for c in configs}
    expected = {(m, f, s) for m in CORE_MODELS for f in [1, 2, 3] for s in SEEDS}
    require(identities == expected, 'architecture核心模型/外折/种子集合不符。')
    require(set(plan['baselines']) == set(BASELINES), '传统基线集合不符。')
    for c in configs:
        require(c['variant'] == 'default' and c['map_level'] == 'maps100'
                and c['augment'] is True and c['weighted_loss'] is True, '核心模型必须是默认全图增强加权设置。')
        require(c['run_id'] == f"core_{c['model']}_f{c['fold_id']}_s{c['seed']}", '核心run_id不符。')
    configs += [dict(run_id=f'baseline_{m}_f{f}', family='baseline', model=m, fold_id=f, seed=101)
                for f in [1, 2, 3] for m in BASELINES]
    result = []
    study_hashes = {'plan_sha256':snapshot['extension_study/protocol/experiment_plan.json'],
                    'protocol_sha256':snapshot['extension_study/protocol/splits.json'],
                    'data_sha256':snapshot['extension_study/data/study_data.npz']}
    for config in sorted(configs, key=lambda c:c['run_id']):
        directory = root/'extension_study/results/runs'/config['run_id']
        metadata_path = directory/'metadata.json'
        metadata = read_json(metadata_path)
        require(metadata['status'] == 'finished' and metadata['labels'] == splits['labels']
                and metadata.get('original_test_used') is False, f"{config['run_id']}来源状态/标签不符。")
        require(all(metadata.get(k) == v for k, v in study_hashes.items()), f"{config['run_id']}来源哈希不符。")
        require(all(metadata['config'].get(k) == v for k, v in config.items()), f"{config['run_id']}配置不符。")
        weight = directory/('model.joblib' if config['family'] == 'baseline' else 'best.pt')
        cv = directory/'outer_metrics.json'
        result.append(dict(config=config, weight=short_file_record(root, weight),
                           metadata=short_file_record(root, metadata_path),
                           original_cv_metrics=short_file_record(root, cv),
                           probability_method=('softmax' if config['family'] == 'architecture' else
                                               'predict_proba' if config['model'] == 'random_forest_300' else None)))
    require(len(result) == 36 and len({r['config']['run_id'] for r in result}) == 36, '应为36个唯一已有模型。')
    return result


def build_plan(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    splits = read_json(root/'extension_study/protocol/splits.json')
    original_plan = read_json(root/'extension_study/protocol/experiment_plan.json')
    snapshot = frozen_source_snapshot(root)
    require(original_plan['split_sha256'] == snapshot['extension_study/protocol/splits.json']
            and original_plan['task_classes'] == splits['labels'], '原扩展计划与划分不符。')
    audit = read_json(root/'extension_study/data/study_data_audit.json')
    require(audit['source_dataset_sha256'] == snapshot['data/processed/dataset.npz']
            and audit['study_dataset_sha256'] == snapshot['extension_study/data/study_data.npz'], '主/开发NPZ与审计不同。')
    first_evaluation = read_json(root/'results/final_test/evaluation_record.json')
    require(first_evaluation['dataset_sha256'] == snapshot['data/processed/dataset.npz']
            and first_evaluation['split_sha256'] == snapshot['data/processed/map_split.csv'],
            '主NPZ或首阶段划分与原正式评价版本不符。')
    require(sha256(Path(sys.modules['study_models'].__file__)) == snapshot['extension_study/src/study_models.py']
            and sha256(Path(sys.modules['common'].__file__)) == snapshot['src/common.py'],
            '当前导入的模型/自定义类代码与--root冻结源码不同。')
    population = load_population(root, splits)
    runs = source_inventory(root, original_plan, splits, snapshot)
    value = dict(schema_version=SCHEMA_VERSION, study='第二阶段冻结模型在已查看六测试图上的补充评价',
                 warning=WARNING, already_viewed_test=True, new_external_test=False, training_performed=False,
                 checkpoint_selection='固定全部27个architecture核心权重与9个传统模型，不挑最佳外折/种子，不集成。',
                 population=dict(original_test_pixels=6399, eligible_pixels=5425, excluded_pixels=974,
                                 excluded_label_counts=population['excluded_counts'], labels=population['labels'],
                                 input_length=1024, test_maps=population['test_maps'],
                                 plastic_supported_maps=4, blank_only_maps=2,
                                 map_support=population['map_support'],
                                 original_index_order='主NPZ全局行号升序，所有36个模型使用相同5425行。',
                                 development_default_exact_equal=True, development_mapping_checked_pixels=25541),
                 aggregation=dict(neural='每model/原外折先平均种子101/202/303，再平均3组拟合并计算ddof=1样本SD。',
                                  baseline='每model/原外折1次拟合，再平均3组拟合并计算ddof=1样本SD。',
                                  dispersion='固定同一5425样本上的拟合差异；不增加独立测试数据，不是CI或OOF。'),
                 scopes=SCOPE_NOTES, runs=runs, frozen_source_files=snapshot,
                 evaluator_sha256=sha256(Path(__file__)))
    return value, population


def safe_output_root(root: Path) -> tuple[Path, Path]:
    supplement = (root/'supplemental_study').resolve()
    output = (supplement/'results/stage2_frozen_holdout').resolve()
    plan = (supplement/'protocol/stage2_holdout_plan.json').resolve()
    require(output.is_relative_to(supplement) and plan.is_relative_to(supplement), '补充输出路径越界。')
    require(not output.is_relative_to((root/'extension_study').resolve())
            and not output.is_relative_to((root/'results').resolve()), '不得写入原首轮/扩展结果目录。')
    return output, plan


def validate_checkpoint(checkpoint: dict[str, Any], metadata: dict[str, Any], labels: list[str]) -> None:
    require(checkpoint.get('labels') == labels and len(labels) == 12, 'checkpoint类别及顺序必须精确对应冻结12类。')
    for key in ['data_sha256', 'protocol_sha256', 'plan_sha256']:
        require(checkpoint.get(key) == metadata.get(key), f'checkpoint内部{key}不符。')
    require(checkpoint.get('epoch') == metadata['best_epoch'], '必须使用内验证已锁定最佳epoch。')
    require(all(torch.isfinite(v).all().item() for v in checkpoint['state_dict'].values() if v.is_floating_point()),
            'checkpoint含非有限参数。')


def predict_existing(root: Path, item: dict[str, Any], X: np.ndarray, labels: list[str],
                     device: torch.device) -> tuple[np.ndarray, np.ndarray | None]:
    config = item['config']
    metadata = read_json(root/item['metadata']['path'])
    path = root/item['weight']['path']
    require(sha256(path) == item['weight']['sha256'], '推理前权重SHA变化。')
    if config['family'] == 'architecture':
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        validate_checkpoint(checkpoint, metadata, labels)
        model = build_model(config['model'], len(labels), input_length=1024)
        require(sum(p.numel() for p in model.parameters()) == metadata['parameters'], '网络参数数量不符。')
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        model.to(device).eval()
        batches = []
        with torch.inference_mode():
            for start in range(0, len(X), 512):
                logits = model(torch.from_numpy(X[start:start+512, None, :]).to(device))
                require(logits.ndim == 2 and logits.shape[1] == 12 and torch.isfinite(logits).all().item(),
                        '网络必须输出有限的12类logits。')
                batches.append(logits.softmax(dim=1).cpu().numpy())
        probabilities = np.concatenate(batches).astype(np.float32)
        pred = probabilities.argmax(axis=1).astype(np.int64)
        del model, checkpoint, batches
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    else:
        model = joblib.load(path)
        require(np.array_equal(np.asarray(model.classes_), np.arange(12)), '传统模型classes_应为冻结12类编码。')
        pred = np.asarray(model.predict(X))
        require(np.all(np.isfinite(pred)) and np.all(pred == pred.astype(np.int64)), '传统模型预测应为整数编码。')
        pred = pred.astype(np.int64)
        probabilities = None
        if config['model'] == 'random_forest_300':
            # 保留RF原始float64概率，避免float32舍入在极近的并列类别间改变argmax。
            probabilities = np.asarray(model.predict_proba(X), dtype=np.float64)
            require(np.array_equal(probabilities.argmax(axis=1), pred), 'RF预测与其概率argmax不符。')
        del model
    require(pred.shape == (len(X),) and np.all((pred >= 0) & (pred < 12)), '预测形状或类别超界。')
    if probabilities is not None:
        require(probabilities.shape == (len(X), 12), '概率矩阵必须为N×12。')
        calibration(np.zeros(len(X), dtype=np.int64), probabilities)
    return pred, probabilities


def evaluate_run(root: Path, output: Path, item: dict[str, Any], population: dict[str, Any],
                 plan_hash: str, device: torch.device) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config = item['config']
    directory = output/'runs'/config['run_id']
    directory.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    pred, probabilities = predict_existing(root, item, population['X'], population['labels'], device)
    y = population['y']
    truth = population['frame']
    groups = truth.map_id.to_numpy()
    scores = confusion_metrics(y, pred, groups, len(population['labels']))
    if probabilities is not None:
        scores.update(calibration(y, probabilities))
    frame = truth.copy()
    frame['predicted_label'] = np.array(population['labels'])[pred]
    if probabilities is not None:
        frame['confidence'] = probabilities.max(axis=1)
        np.savez_compressed(directory/'probabilities.npz', original_indices=frame.original_index.to_numpy(),
                            probabilities=probabilities, labels=np.array(population['labels']))
    frame.to_csv(directory/'predictions.csv', index=False, encoding='utf-8-sig')
    class_table(y, pred, population['labels']).to_csv(directory/'class_metrics.csv', index=False, encoding='utf-8-sig')
    cm = np.bincount(y*12+pred, minlength=144).reshape(12, 12)
    pd.DataFrame(cm, index=population['labels'], columns=population['labels']).to_csv(directory/'confusion.csv', encoding='utf-8-sig')
    per_map = []
    for support in population['map_support']:
        mask = groups == support['map_id']
        per_map.append(dict(map_id=support['map_id'], **confusion_metrics(y[mask], pred[mask], groups[mask], 12),
                            has_plastic_support=support['has_plastic_support'], plastic_pixels=support['plastic_pixels'],
                            blank_pixels=support['blank_pixels'], labels_present='|'.join(support['labels_present'])))
    pd.DataFrame(per_map).to_csv(directory/'per_map_metrics.csv', index=False, encoding='utf-8-sig')
    scoped = []
    for name, mask in population['scopes'].items():
        values = confusion_metrics(y[mask], pred[mask], groups[mask], 12)
        if probabilities is not None:
            values.update(calibration(y[mask], probabilities[mask]))
        scoped.append(dict(run_id=config['run_id'], model=config['model'], family=config['family'],
                           fold_id=config['fold_id'], seed=config['seed'], scope=name, **values))
    write_json(directory/'metrics.json', scores)
    output_files = {p.name:sha256(p) for p in sorted(directory.iterdir()) if p.is_file()}
    metadata = dict(status='completed', config=config, supplemental_plan_sha256=plan_hash,
                    evaluated_at=now(), labels=population['labels'], input_shape=[5425, 1, 1024],
                    training_performed=False, original_cv_results_modified=False, already_viewed_test=True,
                    source=item, device=str(device), torch_version=str(torch.__version__), sklearn_version=sklearn.__version__,
                    probabilities_available=probabilities is not None,
                    probability_note=('模型原始softmax/predict_proba，未经本holdout后验校准。' if probabilities is not None else
                                      '原余弦/LinearSVC没有校准概率；未把decision score伪造为概率。'),
                    warning=WARNING, metrics=scores, output_files=output_files,
                    runtime_seconds=time.perf_counter()-started)
    write_json(directory/'metadata.json', metadata)
    record = dict(run_id=config['run_id'], family=config['family'], model=config['model'],
                  fold_id=config['fold_id'], seed=config['seed'], probabilities_available=probabilities is not None,
                  runtime_seconds=metadata['runtime_seconds'], **scores)
    print(json.dumps(dict(completed=config['run_id'], macro_f1_present=scores['macro_f1_present']), ensure_ascii=False), flush=True)
    return record, scoped


def aggregate(frame: pd.DataFrame, by_scope: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """同一holdout上的拟合模型层次汇总，绝不拼接为OOF或概率集成。"""
    keys = (['scope'] if by_scope else []) + ['family', 'model']
    fold_rows = []
    for identity, part in frame.groupby(keys+['fold_id'], sort=True):
        identity = dict(zip(keys+['fold_id'], identity))
        expected = set(SEEDS) if identity['family'] == 'architecture' else {101}
        require(len(part) == len(expected) and set(part.seed.astype(int)) == expected, '折内种子缺失/重复。')
        values = {key:float(part[key].mean()) for key in CLASSIFICATION_METRICS}
        for key in CALIBRATION_METRICS:
            if key in part and part[key].notna().all():
                values[key] = float(part[key].mean())
        fold_rows.append(dict(identity, n_seeds=len(part), **values))
    folds = pd.DataFrame(fold_rows)
    rows = []
    for identity, part in folds.groupby(keys, sort=True):
        if not isinstance(identity, tuple):
            identity = (identity,)
        values = dict(zip(keys, identity))
        require(len(part) == 3 and set(part.fold_id.astype(int)) == {1, 2, 3}, '只能汇总完整三个拟合外折。')
        for key in CLASSIFICATION_METRICS+CALIBRATION_METRICS:
            if key in part and part[key].notna().all():
                values[key+'_mean'] = float(part[key].mean())
                values[key+'_fold_sd'] = float(part[key].std(ddof=1))
        rows.append(dict(values, n_folds=3, n_model_fits=int(part.n_seeds.sum()),
                         dispersion_basis='固定同一holdout上的模型拟合差异；非CI、非OOF'))
    return folds, pd.DataFrame(rows)


def self_check() -> dict[str, Any]:
    """仅用小数组核验非线性图等权F1与层次聚合，不加载任何真实模型。"""
    y = np.array([0, 0, 0, 1])
    pred = np.array([0, 1, 0, 1])
    maps = np.array(['A', 'A', 'A', 'B'])
    score = confusion_metrics(y, pred, maps, 3)
    expected = dict(accuracy=.75, balanced_accuracy=5/6, macro_f1_present=11/15,
                    map_mean_accuracy=5/6, map_equal_weight_macro_f1=29/35)
    require(all(np.isclose(score[k], v, atol=1e-14) for k, v in expected.items()), '小数组指标核查失败。')
    require(not np.isclose(score['map_equal_weight_macro_f1'], .9), '不能误用各图F1简单平均。')
    table = class_table(y, pred, ['a', 'b', 'absent'])
    require(int(table.loc[2, 'support']) == 0 and np.isfinite(table[['precision', 'recall', 'f1']]).all().all(),
            '无支持类别处理失败。')
    probabilities = np.array([[.9, .1], [.4, .6], [.8, .2], [.1, .9]])
    require(all(np.isfinite(v) for v in calibration(y, probabilities).values()), '小数组校准指标失败。')
    records = []
    for f in [1, 2, 3]:
        for s in SEEDS:
            records.append(dict(family='architecture', model='cnn', fold_id=f, seed=s,
                                **{key:f/10+(s == 202)*.01 for key in CLASSIFICATION_METRICS}))
    folds, summary = aggregate(pd.DataFrame(records))
    require(len(folds) == 3 and np.isclose(summary.iloc[0].accuracy_mean, .2+.01/3)
            and np.isclose(summary.iloc[0].accuracy_fold_sd, .1), '层次汇总核查失败。')
    return dict(status='passed', real_inference_performed=False, small_array_metrics=True,
                absent_class_handling=True, probability_checks=True, hierarchical_aggregation=True)


def verify_existing_output(output: Path, plan_hash: str, source_snapshot: dict[str, str], root: Path) -> dict[str, Any]:
    receipt = read_json(output/'execution.json')
    require(receipt['status'] == 'completed' and receipt['supplemental_plan_sha256'] == plan_hash,
            '已有评价不是当前冻结计划下的完成结果。')
    require(receipt['original_frozen_sources_unchanged'] is True, '已有回执未通过原源文件不变核查。')
    for relative, expected in receipt['output_files'].items():
        path = output/relative
        require(path.is_file() and sha256(path) == expected, f'已有补充结果变化：{relative}')
    require(frozen_source_snapshot(root) == source_snapshot, '核对期间原冻结源文件发生变化。')
    return dict(status='verified', real_inference_performed=False, model_fits=36,
                existing_output=str(output), warning=WARNING)


def main(args: argparse.Namespace) -> None:
    root = args.root.resolve()
    torch.set_num_threads(args.cpu_threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    output, plan_path = safe_output_root(root)
    self_check()
    proposed, population = build_plan(root)
    if plan_path.exists():
        require(read_json(plan_path) == proposed, '现有stage2_holdout_plan.json与输入/源码/清单不符，禁止静默修改冻结方案。')
    elif args.freeze_plan:
        write_json(plan_path, proposed)
    else:
        raise FileNotFoundError('先以--freeze-plan只核验并冻结方案，再运行推理；不会自动边选方案边评价。')
    plan_hash = sha256(plan_path)
    if args.freeze_plan:
        print(json.dumps(dict(status='plan_frozen', plan=str(plan_path), sha256=plan_hash,
                              neural_models=27, traditional_models=9, eligible_pixels=5425,
                              source_mapping_pixels_verified=25541, real_inference_performed=False), ensure_ascii=False, indent=2))
        return
    if args.verify_only:
        if (output/'execution.json').exists():
            result = verify_existing_output(output, plan_hash, proposed['frozen_source_files'], root)
        else:
            result = dict(status='inputs_verified', real_inference_performed=False, plan_sha256=plan_hash,
                          original_test_pixels=6399, eligible_pixels=5425, existing_output=False)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if output.exists():
        if (output/'execution.json').exists() and read_json(output/'execution.json').get('status') == 'completed':
            print(json.dumps(verify_existing_output(output, plan_hash, proposed['frozen_source_files'], root), ensure_ascii=False, indent=2))
            return
        require(not any(output.iterdir()), '存在未完成补充输出；请先保留该失败目录，勿覆盖原始结果，再明确重新评价。')
    require(args.device != 'cuda' or torch.cuda.is_available(), '请求CUDA但当前不可用。')
    device = torch.device(args.device)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    receipt = dict(status='running', started_at=now(), supplemental_plan_sha256=plan_hash,
                   training_performed=False, already_viewed_test=True, new_external_test=False,
                   device=str(device), gpu=torch.cuda.get_device_name(0) if device.type == 'cuda' else None,
                   torch_version=str(torch.__version__), sklearn_version=sklearn.__version__,
                   population=proposed['population'], warning=WARNING)
    write_json(output/'execution.json', receipt)
    try:
        support = pd.DataFrame(population['map_support'])
        support['labels_present'] = support.labels_present.map(lambda x:'|'.join(x))
        support['label_counts'] = support.label_counts.map(lambda x:json.dumps(x, ensure_ascii=False, sort_keys=True))
        support.to_csv(output/'map_support.csv', index=False, encoding='utf-8-sig')
        population['frame'].to_csv(output/'eligible_holdout_index.csv', index=False, encoding='utf-8-sig')
        write_json(output/'population.json', proposed['population'])
        records, scopes, cv_rows = [], [], []
        for item in proposed['runs']:
            record, scoped = evaluate_run(root, output, item, population, plan_hash, device)
            records.append(record)
            scopes.extend(scoped)
            c = item['config']
            cv_rows.append(dict(run_id=c['run_id'], family=c['family'], model=c['model'], fold_id=c['fold_id'], seed=c['seed'],
                                evaluation_population='原28开发图的该外评价折；不是六图holdout',
                                source_path=item['original_cv_metrics']['path'],
                                source_sha256=item['original_cv_metrics']['sha256'],
                                **read_json(root/item['original_cv_metrics']['path'])))
        frame = pd.DataFrame(records)
        frame.to_csv(output/'run_metrics.csv', index=False, encoding='utf-8-sig')
        scoped_frame = pd.DataFrame(scopes)
        scoped_frame.to_csv(output/'scope_run_metrics.csv', index=False, encoding='utf-8-sig')
        pd.DataFrame(cv_rows).to_csv(output/'original_cv_reference.csv', index=False, encoding='utf-8-sig')
        folds, summary = aggregate(frame)
        folds.to_csv(output/'condition_fold_means.csv', index=False, encoding='utf-8-sig')
        summary.to_csv(output/'condition_summary.csv', index=False, encoding='utf-8-sig')
        scope_folds, scope_summary = aggregate(scoped_frame, by_scope=True)
        scope_folds.to_csv(output/'scope_fold_means.csv', index=False, encoding='utf-8-sig')
        scope_summary.to_csv(output/'scope_summary.csv', index=False, encoding='utf-8-sig')
        require(frozen_source_snapshot(root) == proposed['frozen_source_files'], '评价后原冻结源文件变化。')
        outputs = {path.relative_to(output).as_posix():sha256(path) for path in sorted(output.rglob('*'))
                   if path.is_file() and path.name != 'execution.json' and path.suffix != '.tmp'}
        receipt.update(status='completed', finished_at=now(), elapsed_seconds=time.perf_counter()-started,
                       evaluated_model_fits=36, evaluated_neural_models=27, evaluated_traditional_models=9,
                       original_frozen_sources_unchanged=True, output_files=outputs,
                       probabilities_saved_for=30, nonprobabilistic_model_fits=6,
                       original_cv_kept_separate=True, oof_predictions_created=False, ensembles_created=False)
        write_json(output/'execution.json', receipt)
        print(json.dumps(dict(status='completed', output=str(output), model_fits=36, pixels=5425,
                              warning=WARNING), ensure_ascii=False, indent=2))
    except Exception as exc:
        receipt.update(status='failed', finished_at=now(), elapsed_seconds=time.perf_counter()-started,
                       error=f'{type(exc).__name__}: {exc}',
                       original_frozen_sources_unchanged=frozen_source_snapshot(root) == proposed['frozen_source_files'])
        write_json(output/'execution.json', receipt)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=PROJECT)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--cpu-threads', type=int, default=8)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--freeze-plan', action='store_true', help='只核验源数据/权重清单并冻结方案，不加载模型推理。')
    mode.add_argument('--verify-only', action='store_true', help='只复核冻结输入及已有输出，不推理。')
    args = parser.parse_args()
    require(args.cpu_threads > 0, 'cpu-threads必须为正整数。')
    main(args)
