"""只准备开发图共同类别的逐谱预处理变体，原独立测试图不进入本研究。"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import zipfile

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


VARIANTS = {
    'X_default': '精确复用冻结主NPZ开发行：SG(11,3)→端点线性基线→逐谱SNV',
    'X_no_sg': '插值→端点线性基线→逐谱SNV，关闭SG',
    'X_no_baseline': '插值→SG(11,3)→逐谱SNV，不扣除端点线性基线',
    'X_l2': '插值→SG(11,3)→端点线性基线→逐谱L2，不减均值',
    'X_quadratic': '插值→SG(11,3)→全频轴二阶最小二乘基线→逐谱SNV',
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def save_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def source_helpers(root: Path):
    """只导入原脚本的下载校验函数，不调用会重建主NPZ的prepare。"""
    path = root / 'src/prepare_data.py'
    spec = importlib.util.spec_from_file_location('rampi_frozen_source_helpers', path)
    if spec is None or spec.loader is None:
        raise ImportError(f'无法读取源数据工具：{path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_selected_rows(path: Path, key: str, indices: np.ndarray) -> np.ndarray:
    """从NPZ内的NPY读取选中开发行，不把旧测试行物化为数值数组。"""
    if indices.ndim != 1 or not len(indices) or np.any(np.diff(indices) <= 0):
        raise ValueError('开发行索引必须为非空、升序、无重复的一维数组')
    with zipfile.ZipFile(path) as archive, archive.open(key + '.npy') as stream:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
        elif version == (2, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
        else:
            raise ValueError(f'不支持的源NPY版本：{version}')
        if len(shape) != 2 or fortran or dtype != np.dtype('float32'):
            raise ValueError('冻结源X应为按行保存的二维float32数组')
        if indices[0] < 0 or indices[-1] >= shape[0]:
            raise IndexError('开发索引超出冻结源数组范围')
        data_start = stream.tell()
        row_bytes = shape[1] * dtype.itemsize
        result = np.empty((len(indices), shape[1]), dtype=np.float32)
        blocks = np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
        written = 0
        for block in blocks:
            # ZipExtFile跳过未选行字节；跳过的旧测试行不被解释为浮点特征。
            stream.seek(data_start + int(block[0]) * row_bytes)
            needed = len(block) * row_bytes
            payload = stream.read(needed)
            if len(payload) != needed:
                raise ValueError('冻结源NPZ内容长度不足')
            result[written:written + len(block)] = np.frombuffer(payload, dtype=dtype).reshape(len(block), shape[1])
            written += len(block)
    return result


def snv(values: np.ndarray, group: str, variant: str) -> np.ndarray:
    centered = values - values.mean(axis=1, keepdims=True)
    std = centered.std(axis=1, keepdims=True, ddof=0)
    if not np.isfinite(std).all() or np.any(std <= 1e-8):
        raise ValueError(f'{group}/{variant}发现近零或非有限标准差，不静默删除行')
    return (centered / std).astype(np.float32)


def linear_baseline(values: np.ndarray) -> np.ndarray:
    left = values[:, :5].mean(axis=1, keepdims=True)
    right = values[:, -5:].mean(axis=1, keepdims=True)
    return values - (left + (right - left) * np.linspace(0., 1., values.shape[1])[None, :])


def prepare(root: Path, workers: int = 4, skip_download: bool = False, reuse: bool = False) -> dict:
    started = time.perf_counter()
    root = root.resolve()
    study = root / 'extension_study'
    protocol_path = study / 'protocol/splits.json'
    protocol = json.loads(protocol_path.read_text(encoding='utf-8-sig'))
    protocol_sha = sha256_file(protocol_path)
    checksum_path = protocol_path.with_suffix('.sha256')
    if checksum_path.exists() and checksum_path.read_text(encoding='utf-8').strip().split()[0] != protocol_sha:
        raise ValueError('协议SHA256与旁存校验值不匹配')
    source_path = root / 'data/processed/dataset.npz'
    source_sha = sha256_file(source_path)
    if source_sha != protocol['dataset_sha256']:
        raise ValueError('冻结主NPZ校验失败；不能混用其他数据快照')
    dev_maps = sorted(str(s) for s in protocol['development_maps'])
    excluded_maps = sorted(str(s) for s in protocol['original_test_maps_excluded'])
    labels = sorted(str(s) for s in protocol['labels'])
    if len(dev_maps) != 28 or len(excluded_maps) != 6 or set(dev_maps) & set(excluded_maps) or len(labels) != 12:
        raise ValueError('预期28个开发图、6个隔离测试图和12个共同类别')
    with np.load(source_path, allow_pickle=False) as source:
        all_groups = source['groups'].astype(str)
        all_labels = source['y'].astype(str)
        all_pixels = source['pixel_index'].astype(np.int32)
        stored_grid = source['wavenumbers'].astype(np.float32)
    if set(all_groups) != set(dev_maps) | set(excluded_maps):
        raise ValueError('协议与冻结数据的图清单不一致')
    mask = np.isin(all_groups, dev_maps) & np.isin(all_labels, labels)
    original_indices = np.flatnonzero(mask).astype(np.int64)
    if original_indices.tolist() != protocol['eligible_population_indices']:
        raise ValueError('共同类开发行与冻结协议不完全一致')
    groups, y, pixels = all_groups[mask], all_labels[mask], all_pixels[mask]
    if set(groups) & set(excluded_maps):
        raise AssertionError('原测试图意外进入开发研究')
    out = study / 'data'
    out.mkdir(parents=True, exist_ok=True)
    audit_path = out / 'study_data_audit.json'
    dataset_path = out / 'study_data.npz'
    if reuse:
        audit = json.loads(audit_path.read_text(encoding='utf-8'))
        if audit['protocol_sha256'] != protocol_sha or audit['source_dataset_sha256'] != source_sha:
            raise ValueError('已有开发数据与当前源或协议不一致')
        if sha256_file(dataset_path) != audit['study_dataset_sha256']:
            raise ValueError('已有开发数据校验失败')
        with np.load(dataset_path, allow_pickle=False) as cache:
            for key in VARIANTS:
                value = cache[key]
                if value.shape != (len(original_indices), 1024) or value.dtype != np.float32 or not np.isfinite(value).all():
                    raise ValueError(f'已有{key}的形状、类型或有限性失败')
                if np.any(value.std(axis=1) <= 1e-8):
                    raise ValueError(f'已有{key}包含近零标准差行')
            if not np.array_equal(cache['original_indices'], original_indices):
                raise ValueError('已有开发数据索引不一致')
            for key, expected in [('y', y), ('groups', groups), ('pixel_index', pixels),
                                  ('wavenumbers', stored_grid)]:
                if not np.array_equal(cache[key], expected):
                    raise ValueError(f'已有开发数据{key}与冻结源不一致')
        print(f'复用开发研究数据：{len(original_indices)}条、28图、12类、5种表示', flush=True)
        return audit
    helpers = source_helpers(root)
    manifest_path = root / 'data/manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
    selected = [row for row in manifest if row['map_id'] in set(dev_maps)]
    if len(selected) != 28 or {row['map_id'] for row in selected} != set(dev_maps):
        raise ValueError('源清单不含全部28个开发图')
    if any(row['map_id'] in set(excluded_maps) for row in selected):
        raise AssertionError('禁止下载原独立测试图')
    raw = root / 'data/raw'
    raw.mkdir(parents=True, exist_ok=True)
    downloads = []
    if skip_download:
        for row in selected:
            path = raw / (row['map_id'] + '.csv')
            if helpers.md5_file(path) != row['md5']:
                raise ValueError(f'{path.name}的MD5不匹配')
            downloads.append({'map_id': row['map_id'], 'file_id': row['file_id'], 'status': 'verified', 'md5': row['md5']})
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            tasks = {pool.submit(helpers.download_one, row, raw / (row['map_id'] + '.csv')): row for row in selected}
            for count, future in enumerate(as_completed(tasks), 1):
                row = tasks[future]
                result = future.result()
                downloads.append({'map_id': row['map_id'], **result})
                print(f'开发图下载校验 {count}/28：{row["map_id"]}', flush=True)
    source_audit = json.loads((root / 'data/processed/data_audit.json').read_text(encoding='utf-8'))
    lo, hi = map(float, source_audit['common_axis_cm_inverse'])
    grid = np.linspace(lo, hi, 1024, dtype=np.float64)
    if not np.array_equal(grid.astype(np.float32), stored_grid):
        raise ValueError('冻结频轴与预处理定义不一致')
    arrays = {key: np.empty((len(original_indices), 1024), dtype=np.float32) for key in VARIANTS}
    arrays['X_default'] = read_selected_rows(source_path, 'X', original_indices)
    # 二阶基线的固定设计矩阵只由已固定的频轴构造，不拟合跨谱统计量。
    u = np.linspace(-1., 1., 1024, dtype=np.float64)
    polynomial = np.column_stack([np.ones_like(u), u, u ** 2])
    projection = np.linalg.pinv(polynomial)
    per_map = []
    for count, group in enumerate(dev_maps, 1):
        locations = np.flatnonzero(groups == group)
        selected_pixels = pixels[locations]
        path = raw / (group + '.csv')
        frame = pd.read_csv(path)
        raw_labels = frame.iloc[:, 0].map(helpers.normalize_text).to_numpy(dtype=str)
        if not np.array_equal(raw_labels[selected_pixels], y[locations]):
            raise ValueError(f'{group}源标签与冻结行索引不一致')
        axis, order = helpers.native_axis(path)
        if lo < axis[0] - 1e-8 or hi > axis[-1] + 1e-8:
            raise ValueError(f'{group}不覆盖既定频段，禁止外推')
        values = frame.iloc[selected_pixels, 1:].to_numpy(dtype=np.float64)[:, order]
        if not np.isfinite(values).all():
            raise ValueError(f'{group}原开发谱有非有限值，不静默删除行')
        interpolated = np.stack([np.interp(grid, axis, row) for row in values])
        smooth = savgol_filter(interpolated, window_length=11, polyorder=3, axis=1)
        corrected = linear_baseline(smooth)
        recomputed = snv(corrected, group, 'default_recalculation')
        maximum_difference = float(np.max(np.abs(recomputed - arrays['X_default'][locations])))
        if not np.allclose(recomputed, arrays['X_default'][locations], rtol=3e-6, atol=3e-6):
            raise ValueError(f'{group}默认处理复算不一致，最大差{maximum_difference}')
        arrays['X_no_sg'][locations] = snv(linear_baseline(interpolated), group, 'no_sg')
        arrays['X_no_baseline'][locations] = snv(smooth, group, 'no_baseline')
        norm = np.linalg.norm(corrected, axis=1, keepdims=True)
        if not np.isfinite(norm).all() or np.any(norm <= 1e-8):
            raise ValueError(f'{group}/l2发现近零或非有限范数')
        arrays['X_l2'][locations] = (corrected / norm).astype(np.float32)
        coefficients = projection @ smooth.T
        quadratic_corrected = smooth - (polynomial @ coefficients).T
        arrays['X_quadratic'][locations] = snv(quadratic_corrected, group, 'quadratic')
        for key in VARIANTS:
            if not np.isfinite(arrays[key][locations]).all():
                raise ValueError(f'{group}/{key}处理后出现非有限值')
        per_map.append({'map_id': group, 'pixels': len(locations), 'raw_file_md5': helpers.md5_file(path),
                        'default_recalculation_max_abs_difference': maximum_difference,
                        'labels': {str(k): int(v) for k, v in pd.Series(y[locations]).value_counts().items()}})
        print(f'开发图变体 {count}/28：{group} {len(locations)}条', flush=True)
    for key, value in arrays.items():
        if value.dtype != np.float32 or value.shape != (len(original_indices), 1024):
            raise AssertionError(f'{key}输出形状或类型不一致')
        if not np.isfinite(value).all() or np.any(value.std(axis=1) <= 1e-8):
            raise ValueError(f'{key}输出含非有限值或近零标准差行')
    execution_dir = root / 'work/execution'
    execution_dir.mkdir(parents=True, exist_ok=True)
    temporary = execution_dir / 'study_data.tmp.npz'
    np.savez_compressed(temporary, original_indices=original_indices, y=y, groups=groups,
                        pixel_index=pixels, wavenumbers=stored_grid, **arrays)
    temporary.replace(dataset_path)
    audit = {
        'schema_version': 1, 'source_dataset_sha256': source_sha, 'protocol_sha256': protocol_sha,
        'source_manifest_sha256': sha256_file(manifest_path), 'study_dataset_sha256': sha256_file(dataset_path),
        'map_count': len(dev_maps), 'pixels': len(original_indices), 'class_count': len(labels),
        'features': 1024, 'dtype': 'float32', 'labels': labels,
        'eligibility_rule': protocol.get('class_eligibility_rule'), 'excluded_classes': protocol.get('excluded_classes'),
        'development_maps': dev_maps, 'original_test_maps_excluded': excluded_maps,
        'original_test_raw_files_opened_or_downloaded': [], 'original_test_rows_in_output': 0,
        'source_feature_read_policy': '主NPZ仅对选择的开发行读取NPY数值；其余行字节跳过。源文件整体字节仅用于SHA256校验。',
        'normalization_fitted_across_spectra': False, 'variants': VARIANTS,
        'variant_ranges': {key: {'minimum': float(value.min()), 'maximum': float(value.max())} for key, value in arrays.items()},
        'common_axis_cm_inverse': [lo, hi], 'per_map': per_map, 'downloads': downloads,
        'original_indices_semantics': '冻结主data/processed/dataset.npz的全局行号，保持升序，不是本文件的局部行号',
        'rows_dropped_by_this_script': 0, 'runtime_seconds': time.perf_counter() - started,
        'limitations': ['本研究只有开发图，不重用原6幅测试图来挑选新增模型。',
                        '共同类别研究为12类含背景，不能与原15类独立测试分数直接比较。',
                        '图间物理标本是否独立未知；逐谱处理不等于新增独立样品。'],
    }
    save_json(audit_path, audit)
    print(json.dumps({key: audit[key] for key in ['pixels', 'map_count', 'class_count', 'features', 'study_dataset_sha256', 'runtime_seconds']}, ensure_ascii=False), flush=True)
    return audit


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--skip-download', action='store_true', help='只校验已存在的28个开发图原始CSV')
    parser.add_argument('--reuse', action='store_true', help='校验并复用已准备的开发研究NPZ')
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error('workers必须在1到8之间')
    prepare(args.root, workers=args.workers, skip_download=args.skip_download, reuse=args.reuse)
