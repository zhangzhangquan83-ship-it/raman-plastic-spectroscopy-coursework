"""只用原28开发map的标签计数，冻结共同类任务的嵌套按图验证协议。

本脚本不读取模型预测、不读取任何训练结果，也不使用原六个测试map
决定类别、划分、学习曲线或参数。indices始终指原始dataset.npz的全局行号。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import itertools
import json
import math
from pathlib import Path

import numpy as np

DEFAULT_SEED = 20261113
EXPECTED_DATASET_SHA256 = 'd0a4382e015099558f5e392869eeda626e8f085dfe269db36141842f4a09dbaf'
ROLES = ('inner_train', 'inner_val', 'outer_eval')


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')


def read_source(root):
    dataset_path = root / 'data/processed/dataset.npz'
    dataset_sha = file_sha256(dataset_path)
    if dataset_sha != EXPECTED_DATASET_SHA256:
        raise ValueError('主数据快照与既定SHA256不符，不能静默改变研究总体。')
    with np.load(dataset_path, allow_pickle=False) as dataset:
        labels = dataset['y'].astype(str)
        groups = dataset['groups'].astype(str)
        pixel_index = dataset['pixel_index'].astype(np.int64)
    with (root / 'data/processed/map_split.csv').open(encoding='utf-8-sig', newline='') as stream:
        original = list(csv.DictReader(stream))
    roles = {row['map_id']: row['split'] for row in original}
    if len(roles) != 34 or set(roles) != set(groups.tolist()):
        raise ValueError('原协议的map集合与主数据不符。')
    if set(roles.values()) != {'train', 'validation', 'test'}:
        raise ValueError('原协议包含未知角色。')
    test_maps = sorted(group for group, role in roles.items() if role == 'test')
    dev_maps = sorted(group for group, role in roles.items() if role in {'train', 'validation'})
    if len(test_maps) != 6 or len(dev_maps) != 28:
        raise ValueError('必须严格保留原六图测试集，仅对其余28图做开发研究。')
    dev_mask = np.isin(groups, dev_maps)
    # 类别资格完全由开发总体决定；不统计原测试图的标签表现或覆盖。
    dev_labels = sorted(np.unique(labels[dev_mask]).tolist())
    all_counts = np.array([[np.sum((groups == group) & (labels == label))
                            for label in dev_labels] for group in dev_maps], dtype=np.int64)
    coverage = np.sum(all_counts > 0, axis=0)
    eligible = [label for label, count in zip(dev_labels, coverage) if count >= 3]
    excluded = [dict(label=label, development_map_coverage=int(coverage[i]),
                     development_pixels=int(all_counts[:, i].sum()),
                     reason='开发map覆盖少于3，无法支持每个外评价折都有真值的三折按图研究。')
                for i, label in enumerate(dev_labels) if label not in eligible]
    if len(eligible) != 12:
        raise ValueError(f'既定共同类规则应得到12类，目前得到{len(eligible)}类。')
    counts = all_counts[:, [dev_labels.index(label) for label in eligible]]
    eligible_mask = dev_mask & np.isin(labels, eligible)
    population = np.flatnonzero(eligible_mask)
    semantic_split = [{'map_id': group, 'split': roles[group]} for group in sorted(roles)]
    return dict(dataset_sha256=dataset_sha, original_split_semantic_sha256=hashlib.sha256(canonical_json(semantic_split)).hexdigest(),
                labels=labels, groups=groups, pixel_index=pixel_index, dev_maps=dev_maps,
                original_test_maps=test_maps, dev_labels=dev_labels, all_counts=all_counts,
                counts=counts, eligible_classes=eligible, excluded_classes=excluded,
                eligible_mask=eligible_mask, population_indices=population,
                development_pixels=int(dev_mask.sum()))


def balanced_outer_candidate(counts, rng, target_sizes):
    """先满足稀有类别的逐折出现，再随机补齐map数量；不使用任何模型分数。"""
    n_maps, n_classes = counts.shape
    present = counts > 0
    assigned = np.full(n_maps, -1, dtype=np.int64)
    sizes = np.zeros(3, dtype=np.int64)
    coverage_order = np.argsort(present.sum(axis=0), kind='stable')
    for label in coverage_order:
        covered = {int(assigned[group]) for group in np.flatnonzero(present[:, label]) if assigned[group] >= 0}
        missing = [fold for fold in range(3) if fold not in covered]
        rng.shuffle(missing)
        available = np.flatnonzero(present[:, label] & (assigned < 0)).tolist()
        rng.shuffle(available)
        for fold in missing:
            if sizes[fold] >= target_sizes[fold] or not available:
                return None
            group = available.pop()
            assigned[group] = fold
            sizes[fold] += 1
    remaining = np.flatnonzero(assigned < 0)
    slots = np.repeat(np.arange(3), np.asarray(target_sizes) - sizes)
    if len(remaining) != len(slots):
        return None
    rng.shuffle(slots)
    assigned[remaining] = slots
    fold_counts = np.stack([counts[assigned == fold].sum(axis=0) for fold in range(3)])
    if np.any(fold_counts <= 0):
        return None
    return assigned, fold_counts


def outer_balance_score(fold_counts):
    fractions = fold_counts / fold_counts.sum(axis=0, keepdims=True)
    class_imbalance = float(np.mean((fractions - 1 / 3) ** 2))
    pixel_fractions = fold_counts.sum(axis=1) / fold_counts.sum()
    return class_imbalance + .2 * float(np.mean((pixel_fractions - 1 / 3) ** 2))


def inner_split(outer_train, counts):
    """穷举5/6图内验证集；先保证内训练覆盖，再最大化验证覆盖。"""
    total = counts[outer_train].sum(axis=0)
    candidates = []
    for validation_size in (5, 6):
        choices = np.asarray(list(itertools.combinations(outer_train.tolist(), validation_size)), dtype=np.int64)
        if not len(choices):
            continue
        validation_counts = counts[choices].sum(axis=1)
        train_complete = np.all(total[None, :] - validation_counts > 0, axis=1)
        choices, validation_counts = choices[train_complete], validation_counts[train_complete]
        if not len(choices):
            continue
        n_present = np.sum(validation_counts > 0, axis=1)
        max_present = int(n_present.max())
        eligible = np.flatnonzero(n_present == max_present)
        target_fraction = validation_size / len(outer_train)
        fractions = validation_counts[eligible] / total[None, :]
        class_score = np.mean((fractions - target_fraction) ** 2, axis=1)
        pixel_fraction = validation_counts[eligible].sum(axis=1) / total.sum()
        scores = class_score + .2 * (pixel_fraction - target_fraction) ** 2
        selected = int(eligible[int(np.argmin(scores))])
        result = dict(validation=choices[selected], supported_classes=max_present,
                      balance_score=float(scores.min()), validation_maps=validation_size)
        candidates.append(result)
        # 5图已经覆盖全部共同类时，保留更多内训练map，不再扩大验证集。
        if max_present == counts.shape[1]:
            return result
    if not candidates:
        raise ValueError('没有保持内训练全类覆盖的5图或6图内验证集。')
    return min(candidates, key=lambda item: (-item['supported_classes'], item['validation_maps'], item['balance_score']))


def select_nested_partition(counts, seed, candidate_budget=12000, retained_candidates=64):
    rng = np.random.default_rng(seed)
    target_sizes = [10, 9, 9]
    pool, seen = [], set()
    feasible_count = 0
    for candidate in range(candidate_budget):
        result = balanced_outer_candidate(counts, rng, target_sizes)
        if result is None:
            continue
        assigned, fold_counts = result
        key = tuple(assigned.tolist())
        if key in seen:
            continue
        seen.add(key)
        feasible_count += 1
        score = outer_balance_score(fold_counts)
        item = (-score, candidate, assigned.copy())
        if len(pool) < retained_candidates:
            heapq.heappush(pool, item)
        elif score < -pool[0][0]:
            heapq.heapreplace(pool, item)
    if not pool:
        raise ValueError('固定标签搜索预算内没有找到外三折全类覆盖划分。')
    examined = 0
    best = None
    for negative_score, candidate, assigned in sorted(pool, key=lambda item: (-item[0], item[1])):
        inners = []
        for fold in range(3):
            outer_train = np.flatnonzero(assigned != fold)
            inners.append(inner_split(outer_train, counts))
        examined += 1
        supported = sum(item['supported_classes'] for item in inners)
        rank = (-supported, -negative_score, candidate)
        if best is None or rank < best[0]:
            best = (rank, assigned, inners, candidate)
        if supported == 3 * counts.shape[1]:
            break
    _, assigned, inners, selected_candidate = best
    search = dict(seed=seed, candidate_budget=candidate_budget, feasible_unique_outer_candidates=feasible_count,
                  retained_outer_candidates=len(pool), nested_candidates_examined=examined,
                  selected_candidate=selected_candidate, outer_eval_map_sizes=target_sizes,
                  selection_uses='仅开发map标签覆盖、标签像素计数及map/pixel规模平衡，无模型预测或分数。',
                  outer_balance_score=outer_balance_score(np.stack([counts[assigned == fold].sum(axis=0) for fold in range(3)])),
                  inner_policy='穷举5图，保持内训练全类且内验证全类时优先5图；否则穷举6图；若仍不全则报告最多覆盖及缺失。')
    return assigned, inners, search


def role_description(map_indices, source):
    maps = [source['dev_maps'][i] for i in sorted(map_indices)]
    indices = np.flatnonzero(source['eligible_mask'] & np.isin(source['groups'], maps))
    coverage = {}
    for label in source['eligible_classes']:
        mask = source['labels'][indices] == label
        coverage[label] = dict(pixels=int(mask.sum()), maps=int(len(np.unique(source['groups'][indices][mask]))))
    missing = [label for label, values in coverage.items() if values['pixels'] == 0]
    return dict(maps=maps, indices=indices.tolist(), pixels=len(indices), map_count=len(maps),
                classes_present=len(coverage) - len(missing), missing_classes=missing, class_counts=coverage)


def learning_curve(inner_train, source, seed):
    """精确最小覆盖集作为共同起点，其余map按同一随机顺序嵌套补充。"""
    rng = np.random.default_rng(seed)
    train_order = np.asarray(sorted(inner_train), dtype=np.int64)
    present = source['counts'] > 0
    minimal_choices = None
    for size in range(1, len(train_order) + 1):
        choices = np.asarray(list(itertools.combinations(train_order.tolist(), size)), dtype=np.int64)
        complete = np.all(np.any(present[choices], axis=1), axis=1)
        if np.any(complete):
            minimal_choices = choices[complete]
            break
    if minimal_choices is None:
        raise ValueError('内训练集不能覆盖共同类别。')
    minimum = minimal_choices[int(rng.integers(len(minimal_choices)))].tolist()
    remaining = sorted(set(train_order.tolist()) - set(minimum))
    rng.shuffle(remaining)
    nested_order = minimum + remaining
    n_train = len(train_order)
    requests = [('maps60', .6), ('maps80', .8), ('maps100', 1.)]
    provisional = []
    for level, fraction in requests:
        size = max(len(minimum), math.ceil(fraction * n_train))
        selected = sorted(nested_order[:size])
        description = role_description(selected, source)
        assert not description['missing_classes']
        provisional.append(dict(level=level, target_fraction=fraction, actual_map_fraction=size / n_train,
                                maps=description['maps'], indices=description['indices'], pixels=description['pixels'],
                                map_count=size, missing_classes=[], minimum_cover_map_count=len(minimum),
                                raised_to_minimum_cover=math.ceil(fraction * n_train) < len(minimum)))
    # 若罕见覆盖使两个目标变成相同map集合，只保留最高目标水平并显式记录复用。
    retained, omitted = [], []
    for item in reversed(provisional):
        same = next((other for other in retained if other['maps'] == item['maps']), None)
        if same is not None:
            omitted.append(dict(level=item['level'], target_fraction=item['target_fraction'],
                                reason='最小类覆盖约束使map集合与另一水平重复，不重复训练。',
                                reuse_of=same['level'], actual_map_fraction=item['actual_map_fraction']))
        else:
            retained.append(item)
    retained.reverse()
    for first, second in zip(retained, retained[1:]):
        assert set(first['maps']) < set(second['maps'])
    return retained, dict(seed=seed, minimum_cover_map_count=len(minimum),
                          minimum_cover_maps=[source['dev_maps'][i] for i in sorted(minimum)],
                          nested_addition_order=[source['dev_maps'][i] for i in nested_order],
                          omitted_duplicate_levels=omitted)


def build_protocol(source, seed, candidate_budget):
    assigned, inner_choices, search = select_nested_partition(source['counts'], seed, candidate_budget)
    folds = []
    role_count_rows, curve_count_rows = [], []
    for fold in range(3):
        outer_eval = np.flatnonzero(assigned == fold)
        outer_train = np.flatnonzero(assigned != fold)
        inner_val = inner_choices[fold]['validation']
        inner_train = np.asarray(sorted(set(outer_train) - set(inner_val)), dtype=np.int64)
        descriptions = {role: role_description(maps, source) for role, maps in
                        [('inner_train', inner_train), ('inner_val', inner_val), ('outer_eval', outer_eval)]}
        if descriptions['inner_train']['missing_classes'] or descriptions['outer_eval']['missing_classes']:
            raise ValueError('强制类覆盖约束未满足。')
        curve, curve_policy = learning_curve(inner_train, source, seed + 1000 + fold + 1)
        fold_record = dict(fold_id=fold + 1,
                           outer_train_maps=[source['dev_maps'][i] for i in outer_train],
                           outer_train_indices=sorted(descriptions['inner_train']['indices'] + descriptions['inner_val']['indices']),
                           coverage={}, learning_curve=curve, learning_curve_policy=curve_policy)
        for role, description in descriptions.items():
            fold_record[role + '_maps'] = description['maps']
            fold_record[role + '_indices'] = description['indices']
            fold_record['coverage'][role] = {key: value for key, value in description.items() if key not in {'maps', 'indices'}}
            role_count_rows.append(dict(fold_id=fold + 1, role=role, maps=description['map_count'],
                                        pixels=description['pixels'], classes_present=description['classes_present'],
                                        missing_classes=';'.join(description['missing_classes'])))
        for item in curve:
            curve_count_rows.append(dict(fold_id=fold + 1, level=item['level'], maps=item['map_count'],
                                         pixels=item['pixels'], target_fraction=item['target_fraction'],
                                         actual_map_fraction=item['actual_map_fraction'], classes=12))
        folds.append(fold_record)
    population_keys = [dict(index=int(i), map_id=str(source['groups'][i]), pixel_index=int(source['pixel_index'][i]),
                            label=str(source['labels'][i])) for i in source['population_indices']]
    protocol = dict(schema_version=1, protocol_id='RaMPI_dev28_common12_nested3_seed20261113_v1', seed=seed,
                    dataset_sha256=source['dataset_sha256'], generator_sha256=file_sha256(__file__),
                    original_map_split_semantic_sha256=source['original_split_semantic_sha256'],
                    labels=source['eligible_classes'], eligible_classes=source['eligible_classes'],
                    class_eligibility_rule='仅原28开发map中出现于至少3个map的标签；类别资格不参考原测试图或任何模型表现。',
                    excluded_classes=source['excluded_classes'], development_maps=source['dev_maps'],
                    development_population_pixels=source['development_pixels'], eligible_population_pixels=len(population_keys),
                    eligible_population_indices=source['population_indices'].tolist(),
                    population_keys_sha256=hashlib.sha256(canonical_json(population_keys)).hexdigest(),
                    original_test_maps_excluded=source['original_test_maps'], original_test_reused=False,
                    indices_definition='原始data/processed/dataset.npz的全局零基行号，非子集重编号；已过滤不合共同类资格的像素。',
                    nested_fitting_rule='全部模型只在inner_train拟合；inner_val仅用于epoch或超参数选择；outer_eval仅作评价。',
                    refit_outer_train_after_selection=False, outer_eval_partition='三个外评价集合互斥，合计恰好覆盖全部合资格开发像素一次。',
                    independent_unit='采集map，未知物理标本是否跨map共享。',
                    not_comparable_to_original_15_class_test=True,
                    comparison_warning='这是原28开发map上的12类共同任务，不能与第一轮原6图15类测试F1直接比较；不改变第一轮结论。',
                    protocol_hash_scheme='SHA256(splits.json精确字节)，保存于外部splits.sha256，避免自引用。',
                    search=search, folds=folds)
    return protocol, role_count_rows, curve_count_rows, population_keys


def verify_protocol(protocol, source):
    if protocol['dataset_sha256'] != source['dataset_sha256'] or protocol['labels'] != source['eligible_classes']:
        raise ValueError('协议的数据版本或共同类别与当前定义不一致。')
    if protocol['development_maps'] != source['dev_maps'] or protocol['original_test_maps_excluded'] != source['original_test_maps']:
        raise ValueError('开发总体或受保护旧测试图发生变化。')
    all_outer = []
    forbidden = set(source['original_test_maps'])
    for fold in protocol['folds']:
        role_sets = {role: set(fold[role + '_maps']) for role in ROLES}
        if any(role_sets[a] & role_sets[b] for a, b in itertools.combinations(ROLES, 2)):
            raise ValueError('同一外折的map角色重叠。')
        if set.union(*role_sets.values()) != set(source['dev_maps']):
            raise ValueError('某外折没有恰好划分全部28开发map。')
        if any(maps & forbidden for maps in role_sets.values()):
            raise ValueError('原六图测试集被错误复用。')
        for role in ROLES:
            expected = np.flatnonzero(source['eligible_mask'] & np.isin(source['groups'], sorted(role_sets[role]))).tolist()
            if fold[role + '_indices'] != expected:
                raise ValueError('全局索引与map/共同类规则不一致。')
            present = set(source['labels'][expected].tolist())
            if role != 'inner_val' and present != set(source['eligible_classes']):
                raise ValueError('强制角色未覆盖共同12类。')
        all_outer.extend(fold['outer_eval_indices'])
        curves = fold['learning_curve']
        seen = set()
        for curve in curves:
            maps = set(curve['maps'])
            key = tuple(sorted(maps))
            if key in seen or not maps <= role_sets['inner_train']:
                raise ValueError('学习曲线重复或越出内训练集合。')
            seen.add(key)
            expected = np.flatnonzero(source['eligible_mask'] & np.isin(source['groups'], sorted(maps))).tolist()
            if curve['indices'] != expected or set(source['labels'][expected].tolist()) != set(source['eligible_classes']):
                raise ValueError('学习曲线索引或类别覆盖错误。')
        for first, second in zip(curves, curves[1:]):
            if not set(first['maps']) < set(second['maps']):
                raise ValueError('学习曲线map集合未严格嵌套。')
    if sorted(all_outer) != source['population_indices'].tolist() or len(set(all_outer)) != len(all_outer):
        raise ValueError('外评价未对共同类开发总体构成互斥完整划分。')


def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def main(args):
    root = Path(args.root).resolve()
    output = Path(args.output).resolve() if args.output else root / 'extension_study/protocol'
    source = read_source(root)
    path = output / 'splits.json'
    if path.exists() and not args.overwrite:
        protocol = json.loads(path.read_text(encoding='utf-8'))
        verify_protocol(protocol, source)
        expected_hash = (output / 'splits.sha256').read_text(encoding='ascii').split()[0]
        if file_sha256(path) != expected_hash:
            raise ValueError('已冻结协议文件的SHA256不匹配。')
        print('已校验并复用冻结协议，不重新搜索划分。')
    else:
        if args.verify_only:
            raise FileNotFoundError('没有待校验的冻结协议。')
        protocol, role_rows, curve_rows, keys = build_protocol(source, args.seed, args.candidate_budget)
        verify_protocol(protocol, source)
        output.mkdir(parents=True, exist_ok=True)
        encoded = (json.dumps(protocol, ensure_ascii=False, indent=2, sort_keys=True) + '\n').encode('utf-8')
        path.write_bytes(encoded)
        digest = hashlib.sha256(encoded).hexdigest()
        (output / 'splits.sha256').write_text(digest + '  splits.json\n', encoding='ascii')
        write_csv(output / 'role_counts.csv', role_rows)
        write_csv(output / 'learning_curve_counts.csv', curve_rows)
        write_csv(output / 'eligible_pixel_keys.csv', keys)
        class_rows = []
        coverage = (source['all_counts'] > 0).sum(axis=0)
        for i, label in enumerate(source['dev_labels']):
            class_rows.append(dict(label=label, development_maps=int(coverage[i]),
                                   development_pixels=int(source['all_counts'][:, i].sum()),
                                   eligible=label in source['eligible_classes'],
                                   exclusion_reason='' if label in source['eligible_classes'] else 'development_map_coverage_below_3'))
        write_csv(output / 'class_eligibility.csv', class_rows)
    print(json.dumps(dict(protocol_id=protocol['protocol_id'], sha256=file_sha256(path),
                         eligible_classes=len(protocol['labels']), eligible_pixels=protocol['eligible_population_pixels'],
                         folds=[dict(fold_id=fold['fold_id'],
                                     inner_train_maps=len(fold['inner_train_maps']), inner_val_maps=len(fold['inner_val_maps']),
                                     outer_eval_maps=len(fold['outer_eval_maps']),
                                     missing_inner_val=fold['coverage']['inner_val']['missing_classes'],
                                     learning_curve=[dict(level=item['level'], maps=item['map_count'], pixels=item['pixels'])
                                                     for item in fold['learning_curve']]) for fold in protocol['folds']]),
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--output', type=Path)
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument('--candidate-budget', type=int, default=12000)
    parser.add_argument('--overwrite', action='store_true', help='显式重建协议；正式实验启动后不要使用')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    if args.verify_only and args.overwrite:
        parser.error('verify-only不能与overwrite同时使用')
    if args.seed != DEFAULT_SEED:
        parser.error('本研究冻结seed为20261113，不应随意更改')
    main(args)
