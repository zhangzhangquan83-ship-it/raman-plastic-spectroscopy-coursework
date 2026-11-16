"""首阶段15类补充比较；训练与固定测试评价分离，不覆盖历史实验。

正式运行前须保存 ``supplemental_study/protocol/stage1_plan.json``。
该补充实验发生在历史测试结果已被查看之后，应报告为后续补充比较。
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import random
import sys
import time
from typing import Any
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'src'))
sys.path.insert(0, str(PROJECT / 'extension_study/src'))
from common import file_hash, load_data, metrics
from models import augment_spectra
from study_models import build_model

TRAINING = dict(epochs=80, patience=12, batch_size=256, learning_rate=0.001,
                weight_decay=0.0001, gradient_clip=5.0, augment=True)
MODELS = ('random_forest', 'mlp', 'resnet')
SEEDS = (42, 43, 44)
SOURCE_PATHS = ('src/common.py', 'src/models.py', 'src/train_cnn.py',
                'extension_study/src/study_models.py',
                'supplemental_study/src/run_stage1_supplement.py')


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
    return hashlib.sha256(raw.encode('utf-8')).hexdigest()


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def save_torch(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + '.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def load_plan(root: Path, plan_path: Path | None):
    path = plan_path or root / 'supplemental_study/protocol/stage1_plan.json'
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f'正式运行需要预先冻结的计划：{path}')
    plan = json.loads(path.read_text(encoding='utf-8'))
    if plan.get('stage') != 'stage1_supplement':
        raise ValueError('计划stage必须为stage1_supplement。')
    if plan.get('posthoc_after_stage1_test_seen') is not True:
        raise ValueError('须明确设置posthoc_after_stage1_test_seen=true。')
    if plan.get('random_forest_seed') != 101:
        raise ValueError('已约定随机森林沿用扩展方案random_state=101。')
    if tuple(plan.get('models', MODELS)) != MODELS:
        raise ValueError(f'已约定补充方法为{MODELS}，顺序也须固定。')
    if tuple(plan.get('neural_seeds', SEEDS)) != SEEDS:
        raise ValueError('已约定神经网络训练种子为42、43、44。')
    supplied = plan.get('training', {})
    if set(supplied) - set(TRAINING):
        raise ValueError('training含未知设置，请显式修改并审查runner。')
    training = {**TRAINING, **supplied}
    if training != TRAINING:
        raise ValueError('正式参数须匹配首阶段80/12/batch256等已约定设置；变更需另立协议。')
    return path, plan, file_hash(path), training


def checked_data(root: Path, plan: dict):
    d = load_data(root)
    if d['X'].shape != (33119, 1024) or len(d['labels']) != 15:
        raise ValueError('数据范围应为33119条、1024点、15类。')
    for role, expected in [('train', (20968, 22)), ('validation', (5752, 6)), ('test', (6399, 6))]:
        idx = d['idx'][role]
        if (len(idx), len(np.unique(d['groups'][idx]))) != expected:
            raise ValueError(f'{role}谱数或图数不匹配固定首阶段划分。')
    if len(np.unique(d['y'][d['idx']['validation']])) != 13:
        raise ValueError('原验证集应仅有13个有支持类别。')
    if len(np.unique(d['y'][d['idx']['test']])) != 15:
        raise ValueError('原测试集应覆盖15类。')
    for key in ('dataset_sha256', 'split_sha256'):
        expected = plan.get('expected_' + key)
        if expected is not None and expected != d[key]:
            raise ValueError(f'{key}与冻结计划不符。')
    return d


def source_hashes(root: Path) -> dict:
    return {relative: file_hash(root / relative) for relative in SOURCE_PATHS}


def make_configs(training: dict) -> list[dict]:
    result = [dict(run_id='random_forest_300_seed101', model='random_forest', seed=101,
                   parameters=dict(n_estimators=300, min_samples_leaf=2,
                                   class_weight='balanced_subsample', random_state=101,
                                   criterion='gini', max_features='sqrt', bootstrap=True))]
    result.extend(dict(run_id=f'{model}_aug_seed{seed}', model=model, seed=seed,
                       training=deepcopy(training)) for model in ('mlp', 'resnet') for seed in SEEDS)
    return result


def identity(config: dict, d: dict, plan_sha: str, sources: dict) -> dict:
    return dict(config=config, dataset_sha256=d['dataset_sha256'],
                split_sha256=d['split_sha256'], plan_sha256=plan_sha, source_sha256=sources,
                labels=d['labels'])


def verify_artifacts(out: Path, record: dict) -> None:
    for name, expected in record.get('artifact_sha256', {}).items():
        path = out / name
        if not path.is_file() or file_hash(path) != expected:
            raise ValueError(f'已保存产物缺失或哈希变化：{path}')
    required = {'metadata.json', 'validation_metrics.json', 'validation_predictions.csv',
                'validation_probabilities.npz'}
    required.add('model.joblib' if record['config']['model'] == 'random_forest' else 'best.pt')
    if record['config']['model'] != 'random_forest':
        required.add('history.csv')
    if not all((out / name).is_file() for name in required):
        raise ValueError(f'完整产物缺失：{out}')


def finished_record(out: Path, expected: dict) -> dict:
    record = json.loads((out / 'metadata.json').read_text(encoding='utf-8'))
    if record.get('status') != 'finished' or record.get('test_evaluated') is not False:
        raise ValueError(f'不是已完成且尚未正式测试的训练：{out}')
    if record.get('identity_sha256') != canonical_hash(expected) or record.get('identity') != expected:
        raise ValueError(f'计划、源代码、配置或数据哈希与已保存运行不符：{out}')
    verify_artifacts(out, record)
    return record


def predict_neural(model, X: np.ndarray, device, batch: int) -> np.ndarray:
    model.eval()
    result = []
    with torch.inference_mode():
        for start in range(0, len(X), batch):
            inputs = torch.from_numpy(X[start:start + batch, None, :]).to(device)
            result.append(model(inputs).softmax(1).cpu().numpy())
    return np.concatenate(result).astype(np.float32)


def prediction_frame(d: dict, indices: np.ndarray, predictions: np.ndarray, probabilities: np.ndarray):
    labels = np.asarray(d['labels'])
    return pd.DataFrame(dict(original_index=indices, map_id=d['groups'][indices],
                             pixel_index=d['pixel_index'][indices], true_label=labels[d['y'][indices]],
                             predicted_label=labels[predictions], confidence=probabilities.max(axis=1)))


def save_validation(d: dict, out: Path, probs: np.ndarray) -> dict:
    idx = d['idx']['validation']
    pred = probs.argmax(axis=1)
    value = metrics(d['y'][idx], pred, d['groups'][idx])
    save_json(out / 'validation_metrics.json', value)
    prediction_frame(d, idx, pred, probs).to_csv(out / 'validation_predictions.csv', index=False, encoding='utf-8-sig')
    np.savez_compressed(out / 'validation_probabilities.npz', original_indices=idx,
                        probabilities=probs, labels=np.asarray(d['labels']))
    return value


def train_run(root, d, config, plan_path, plan_sha, sources, args, device):
    out = root / 'supplemental_study/results/stage1/runs' / config['run_id']
    expected = identity(config, d, plan_sha, sources)
    if out.exists() and any(out.iterdir()):
        if args.skip_completed:
            record = finished_record(out, expected)
            print(json.dumps(dict(skipped=config['run_id'], identity_sha256=record['identity_sha256'])), flush=True)
            return record
        raise FileExistsError(f'运行目录已存在；拒绝覆盖。完成产物可显式--skip-completed：{out}')
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    record = dict(status='running', config=config, identity=expected,
                  identity_sha256=canonical_hash(expected), test_evaluated=False,
                  posthoc_after_stage1_test_seen=True, started_at=now(),
                  device=str(device) if config['model'] != 'random_forest' else 'cpu',
                  gpu=torch.cuda.get_device_name(0) if device.type == 'cuda' and config['model'] != 'random_forest' else None,
                  torch_version=str(torch.__version__), sklearn_version=sklearn.__version__,
                  cpu_threads=args.cpu_threads, prediction_batch_size=args.prediction_batch_size,
                  train_pixels=20968, validation_pixels=5752, train_maps=22, validation_maps=6,
                  validation_supported_classes=13, output_classes=15)
    save_json(out / 'metadata.json', record)
    try:
        a, b = d['idx']['train'], d['idx']['validation']
        if config['model'] == 'random_forest':
            model = RandomForestClassifier(**config['parameters'], n_jobs=args.cpu_threads)
            record['sklearn_get_params'] = model.get_params(deep=True)
            model.fit(d['X'][a], d['y'][a])
            if not np.array_equal(model.classes_, np.arange(15)):
                raise ValueError('随机森林训练类别顺序错误。')
            temporary = out / 'model.joblib.tmp'
            joblib.dump(model, temporary, compress=3)
            temporary.replace(out / 'model.joblib')
            probs = model.predict_proba(d['X'][b])
            val = save_validation(d, out, probs)
            record.update(best_validation_macro_f1=val['macro_f1_present'], hyperparameter_search=False)
        else:
            seed = config['seed']
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if device.type == 'cuda':
                torch.cuda.manual_seed_all(seed)
                torch.cuda.reset_peak_memory_stats(device)
            training = config['training']
            model = build_model(config['model'], 15).to(device)
            record['parameters'] = sum(p.numel() for p in model.parameters())
            counts = np.bincount(d['y'][a], minlength=15)
            if np.any(counts == 0):
                raise ValueError('训练集必须覆盖全部15类。')
            weights = len(a) / (15 * counts)
            record['training_class_counts'] = counts.tolist()
            record['class_weights'] = weights.tolist()
            loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
            optimizer = torch.optim.AdamW(model.parameters(), lr=training['learning_rate'],
                                         weight_decay=training['weight_decay'])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=training['epochs'])
            loader = DataLoader(TensorDataset(torch.from_numpy(d['X'][a, None, :]),
                                             torch.from_numpy(d['y'][a])),
                                batch_size=training['batch_size'], shuffle=True, num_workers=0,
                                pin_memory=device.type == 'cuda', drop_last=False,
                                generator=torch.Generator().manual_seed(seed))
            if len(a) % training['batch_size'] == 1:
                raise ValueError('最后一批仅1条，BatchNorm不可用；不得静默丢弃样本或改变协议。')
            best, stale, history = -np.inf, 0, []
            for epoch in range(1, training['epochs'] + 1):
                tick = time.perf_counter()
                model.train()
                total_loss, seen = 0.0, 0
                for x, y in loader:
                    x, y = x.to(device), y.to(device)
                    x = augment_spectra(x)
                    optimizer.zero_grad(set_to_none=True)
                    loss = loss_fn(model(x), y)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), training['gradient_clip'])
                    optimizer.step()
                    total_loss += float(loss.item()) * len(y)
                    seen += len(y)
                probs = predict_neural(model, d['X'][b], device, args.prediction_batch_size)
                val = metrics(d['y'][b], probs.argmax(axis=1), d['groups'][b])
                history.append(dict(epoch=epoch, train_loss=total_loss / seen,
                                    seconds=time.perf_counter() - tick, **val))
                pd.DataFrame(history).to_csv(out / 'history.csv', index=False, encoding='utf-8-sig')
                score = val['macro_f1_present']
                if score > best:
                    best, stale = score, 0
                    save_torch(out / 'best.pt', dict(state_dict=model.state_dict(), labels=d['labels'],
                               epoch=epoch, identity_sha256=record['identity_sha256'],
                               dataset_sha256=d['dataset_sha256'], split_sha256=d['split_sha256'],
                               plan_sha256=plan_sha))
                else:
                    stale += 1
                scheduler.step()
                print(json.dumps(dict(run=config['run_id'], epoch=epoch, epochs_max=training['epochs'],
                                      validation_macro_f1=score, best_validation_macro_f1=best,
                                      seconds=history[-1]['seconds'])), flush=True)
                if stale >= training['patience']:
                    break
            state = torch.load(out / 'best.pt', map_location=device, weights_only=True)
            if state['identity_sha256'] != record['identity_sha256'] or state['labels'] != d['labels']:
                raise ValueError('保存的最佳权重身份校验不符。')
            model.load_state_dict(state['state_dict'])
            val = save_validation(d, out, predict_neural(model, d['X'][b], device, args.prediction_batch_size))
            if val['macro_f1_present'] != best:
                raise ValueError('重新加载最佳权重后验证分数与所选分数不符。')
            record.update(best_epoch=int(state['epoch']), epochs_completed=len(history),
                          best_validation_macro_f1=best)
            if device.type == 'cuda':
                record['peak_cuda_memory_allocated_bytes'] = torch.cuda.max_memory_allocated(device)
        if file_hash(plan_path) != plan_sha or source_hashes(root) != sources:
            raise ValueError('训练期间冻结计划或源代码发生变化，拒绝将运行标为完成。')
        record['artifact_sha256'] = {p.name: file_hash(p) for p in out.iterdir()
                                     if p.is_file() and p.name != 'metadata.json'}
        record.update(status='finished', finished_at=now(), runtime_seconds=time.perf_counter() - started)
        save_json(out / 'metadata.json', record)
        print(json.dumps(dict(finished=config['run_id'], test_evaluated=False,
                              runtime_seconds=record['runtime_seconds'])), flush=True)
        return record
    except Exception as error:
        record.update(status='failed', failed_at=now(), error_type=type(error).__name__, error=str(error))
        save_json(out / 'metadata.json', record)
        raise
    finally:
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()


def evaluate(root, d, configs, plan_path, plan_sha, sources, args, device):
    base = root / 'supplemental_study/results/stage1'
    records = {}
    checkpoints = {}
    # 所有预定方法及种子先完成并锁定，再开始正式测试推理。
    for config in configs:
        run = base / 'runs' / config['run_id']
        record = finished_record(run, identity(config, d, plan_sha, sources))
        filename = 'model.joblib' if config['model'] == 'random_forest' else 'best.pt'
        records[config['run_id']] = record
        checkpoints[config['run_id']] = file_hash(run / filename)
    evaluation_identity = dict(plan_sha256=plan_sha, source_sha256=sources,
                               dataset_sha256=d['dataset_sha256'], split_sha256=d['split_sha256'],
                               model_sha256=checkpoints, labels=d['labels'])
    out = base / 'final_test'
    if out.exists() and any(out.iterdir()):
        path = out / 'evaluation_record.json'
        if args.skip_completed and path.is_file():
            old = json.loads(path.read_text(encoding='utf-8'))
            if old.get('status') == 'finished' and old.get('identity') == evaluation_identity:
                for name, sha in old['artifact_sha256'].items():
                    if file_hash(out / name) != sha:
                        raise ValueError('已保存正式评价产物发生变化。')
                print(json.dumps(dict(skipped='final_test', identity_sha256=old['identity_sha256'])), flush=True)
                return old
        raise FileExistsError('正式测试结果已存在或不完整，拒绝覆盖或重复评价。')
    out.mkdir(parents=True, exist_ok=True)
    record = dict(status='running', started_at=now(), identity=evaluation_identity,
                  identity_sha256=canonical_hash(evaluation_identity),
                  posthoc_after_stage1_test_seen=True, test_reuse_disclosure='原6张固定测试图的结果在首阶段已被查看；本次为后续补充比较，并非新的盲测或外部数据集',
                  test_maps=sorted(np.unique(d['groups'][d['idx']['test']]).tolist()),
                  physical_specimen_independence='未知；采集图互斥不保证物理标本互斥',
                  evaluation_device=str(device), prediction_batch_size=args.prediction_batch_size)
    save_json(out / 'evaluation_record.json', record)
    rows, per_map = [], []
    indices = d['idx']['test']
    labels = np.asarray(d['labels'])
    detail = pd.DataFrame(dict(original_index=indices, map_id=d['groups'][indices],
                               pixel_index=d['pixel_index'][indices], true_label=labels[d['y'][indices]]))
    try:
        for config in configs:
            name = config['run_id']
            run = base / 'runs' / name
            if config['model'] == 'random_forest':
                model = joblib.load(run / 'model.joblib')
                if not np.array_equal(model.classes_, np.arange(15)):
                    raise ValueError('随机森林类别顺序不符。')
                probs = model.predict_proba(d['X'][indices])
            else:
                state = torch.load(run / 'best.pt', map_location=device, weights_only=True)
                if state['identity_sha256'] != records[name]['identity_sha256'] or state['labels'] != d['labels']:
                    raise ValueError('测试权重身份不符。')
                model = build_model(config['model'], 15).to(device)
                model.load_state_dict(state['state_dict'])
                probs = predict_neural(model, d['X'][indices], device, args.prediction_batch_size)
            if probs.shape != (6399, 15) or not np.isfinite(probs).all() or not np.allclose(probs.sum(1), 1, atol=1e-5):
                raise ValueError('预测概率的形状、有限性或归一性错误。')
            pred = probs.argmax(1)
            score = metrics(d['y'][indices], pred, d['groups'][indices])
            rows.append(dict(model=name, family=config['model'], seed=config['seed'], **score))
            detail[name] = labels[pred]
            prediction_frame(d, indices, pred, probs).to_csv(out / f'{name}_predictions.csv', index=False, encoding='utf-8-sig')
            np.savez_compressed(out / f'{name}_probabilities.npz', original_indices=indices, probabilities=probs, labels=labels)
            report = classification_report(d['y'][indices], pred, labels=np.arange(15), target_names=d['labels'], output_dict=True, zero_division=0)
            pd.DataFrame(report).T.to_csv(out / f'{name}_classes.csv', encoding='utf-8-sig')
            cm = confusion_matrix(d['y'][indices], pred, labels=np.arange(15))
            pd.DataFrame(cm, index=d['labels'], columns=d['labels']).to_csv(out / f'{name}_confusion.csv', encoding='utf-8-sig')
            save_json(out / f'{name}_metrics.json', score)
            for group in sorted(np.unique(d['groups'][indices])):
                mask = d['groups'][indices] == group
                per_map.append(dict(model=name, map_id=group, **metrics(d['y'][indices][mask], pred[mask], d['groups'][indices][mask])))
            print(json.dumps(dict(evaluated=name, **score)), flush=True)
            del model
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        pd.DataFrame(rows).to_csv(out / 'model_comparison.csv', index=False, encoding='utf-8-sig')
        pd.DataFrame(per_map).to_csv(out / 'per_map_metrics.csv', index=False, encoding='utf-8-sig')
        detail.to_csv(out / 'pixel_predictions.csv', index=False, encoding='utf-8-sig')
        summary = []
        for family in ('mlp', 'resnet'):
            selected = [row for row in rows if row['family'] == family]
            summary.append(dict(model=family, seeds=[row['seed'] for row in selected],
                                accuracy_mean=float(np.mean([row['accuracy'] for row in selected])),
                                accuracy_seed_sd=float(np.std([row['accuracy'] for row in selected], ddof=1)),
                                macro_f1_mean=float(np.mean([row['macro_f1_present'] for row in selected])),
                                macro_f1_seed_sd=float(np.std([row['macro_f1_present'] for row in selected], ddof=1))))
        save_json(out / 'neural_seed_summary.json', summary)
        if file_hash(plan_path) != plan_sha or source_hashes(root) != sources:
            raise ValueError('评价期间计划或源代码变化。')
        record['artifact_sha256'] = {p.name: file_hash(p) for p in out.iterdir() if p.name != 'evaluation_record.json' and p.is_file()}
        record.update(status='finished', finished_at=now(), test_evaluated=True,
                      neural_seed_summary_definition='同一固定测试集上3个训练种子的均值与样本标准差；不是独立测试样本的置信区间')
        save_json(out / 'evaluation_record.json', record)
        return record
    except Exception as error:
        record.update(status='failed', failed_at=now(), error_type=type(error).__name__, error=str(error))
        save_json(out / 'evaluation_record.json', record)
        raise


def main(args):
    if args.cpu_threads < 1 or args.prediction_batch_size < 1:
        raise ValueError('线程数和预测批量必须为正整数。')
    root = args.root.resolve()
    plan_path, plan, plan_sha, training = load_plan(root, args.plan)
    d = checked_data(root, plan)
    sources = source_hashes(root)
    expected_sources = plan.get('source_sha256')
    if expected_sources is not None and expected_sources != sources:
        raise ValueError('代码哈希与冻结计划中source_sha256不符。')
    configs = make_configs(training)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('请求CUDA但不可用；不自动回落CPU或改变批量。')
    device = torch.device(args.device)
    torch.set_num_threads(args.cpu_threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    models = args.models or list(MODELS)
    seeds = args.seeds or list(SEEDS)
    if len(set(models)) != len(models) or len(set(seeds)) != len(seeds) or not set(seeds).issubset(SEEDS):
        raise ValueError('模型或种子重复，或种子不在冻结计划中。')
    if args.phase == 'evaluate' and (set(models) != set(MODELS) or set(seeds) != set(SEEDS)):
        raise ValueError('正式评价必须在全部7个预定运行完成后统一执行，不允许挑选模型或种子。')
    selected = [config for config in configs if config['model'] in models and
                (config['model'] == 'random_forest' or config['seed'] in seeds)]
    invocation = root / 'supplemental_study/results/stage1/invocations' / f'{args.phase}_{uuid4().hex}.json'
    execution = dict(status='running', phase=args.phase, started_at=now(), arguments={**vars(args), 'root': str(root), 'plan': str(plan_path)},
                     plan_sha256=plan_sha, dataset_sha256=d['dataset_sha256'], split_sha256=d['split_sha256'],
                     source_sha256=sources, selected_runs=[config['run_id'] for config in selected])
    save_json(invocation, execution)
    try:
        if args.phase == 'train':
            for config in selected:
                train_run(root, d, config, plan_path, plan_sha, sources, args, device)
        else:
            evaluate(root, d, configs, plan_path, plan_sha, sources, args, device)
        execution.update(status='finished', finished_at=now())
    except Exception as error:
        execution.update(status='failed', failed_at=now(), error_type=type(error).__name__, error=str(error))
        raise
    finally:
        save_json(invocation, execution)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=PROJECT)
    p.add_argument('--plan', type=Path)
    p.add_argument('--phase', choices=['train', 'evaluate'], required=True)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--models', choices=MODELS, nargs='+')
    p.add_argument('--seeds', type=int, nargs='+')
    p.add_argument('--cpu-threads', type=int, default=8)
    p.add_argument('--prediction-batch-size', type=int, default=128)
    p.add_argument('--skip-completed', action='store_true')
    return p


if __name__ == '__main__':
    main(parser().parse_args())
