"""下载并预处理RaMPI像素光谱；逐光谱处理，不拟合跨样品统计参数。"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time
import urllib.request

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

METADATA_FILE = {
    'file_id': 1049720,
    'md5': 'af0071eb9897cb40bc823840ad48ec08',
    'url': 'https://borealisdata.ca/api/access/datafile/1049720?format=original',
}


def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open('rb') as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def download_one(record: dict, path: Path, retries: int = 4) -> dict:
    """临时文件校验成功后再替换，失败的下载不会冒充完整CSV。"""
    if path.exists() and md5_file(path) == record['md5']:
        return {'file_id': record['file_id'], 'path': path.name, 'status': 'cached', 'md5': record['md5']}
    temporary = path.with_suffix(path.suffix + '.part')
    for attempt in range(retries):
        try:
            request = urllib.request.Request(record['url'], headers={'User-Agent': 'RaMPI-Course-Project/1.0'})
            with urllib.request.urlopen(request, timeout=90) as response, temporary.open('wb') as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
            actual = md5_file(temporary)
            if actual != record['md5']:
                raise ValueError(f"MD5不匹配 {path.name}: {actual}")
            temporary.replace(path)
            return {'file_id': record['file_id'], 'path': path.name, 'status': 'downloaded', 'md5': actual}
        except Exception as exc:
            if attempt + 1 == retries:
                raise RuntimeError(f'{path.name}下载或校验失败，已尝试{retries}次') from exc
            time.sleep(min(2 ** attempt, 8))
    raise AssertionError('不可达的下载分支')


def download_all(root: Path, manifest: list[dict], workers: int = 4) -> list[dict]:
    raw = root / 'data/raw'
    raw.mkdir(parents=True, exist_ok=True)
    records = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(download_one, row, raw / (row['map_id'] + '.csv')): row['map_id'] for row in manifest}
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            records.append(result)
            print(f"下载校验 {i}/{len(manifest)}: {futures[future]} {result['status']}", flush=True)
    records.append(download_one(METADATA_FILE, raw / 'RaMPI_metadata.csv'))
    return records


def normalize_text(value) -> str:
    return str(value).replace('\u00a0', ' ').strip().lower()


def native_axis(path: Path) -> tuple[np.ndarray, np.ndarray]:
    columns = pd.read_csv(path, nrows=0).columns.tolist()
    if normalize_text(columns[0]) != 'assignment':
        raise ValueError(f'{path.name}首列不是assignment')
    axis = np.array(columns[1:], dtype=np.float64)
    if not np.isfinite(axis).all() or len(np.unique(axis)) != len(axis):
        raise ValueError(f'{path.name}频轴有非有限值或重复坐标')
    order = np.argsort(axis)
    return axis[order], order


def duplicate_summary(hashes: np.ndarray, labels: np.ndarray, groups: np.ndarray,
                      pixel_indices: np.ndarray, kind: str) -> tuple[np.ndarray, list[dict], dict]:
    """隔离所有异标签完全重复谱；同标签跨图重复交由训练层合并隔离组。"""
    occurrences = defaultdict(list)
    for i, digest in enumerate(hashes):
        occurrences[digest].append(i)
    conflict = np.zeros(len(labels), dtype=bool)
    rows, edges = [], set()
    duplicate_sets = conflict_sets = cross_map_sets = 0
    for digest, indices in occurrences.items():
        if len(indices) < 2:
            continue
        duplicate_sets += 1
        unique_labels = sorted(set(labels[indices]))
        unique_groups = sorted(set(groups[indices]))
        bad = len(unique_labels) > 1
        conflict[indices] = bad
        conflict_sets += int(bad)
        cross_map_sets += int(len(unique_groups) > 1)
        if len(unique_groups) > 1 and not bad:
            for other in unique_groups[1:]:
                edges.add((unique_groups[0], other))
        for i in indices:
            rows.append({'kind': kind, 'sha256': digest, 'map_id': groups[i],
                         'pixel_index': int(pixel_indices[i]), 'label': labels[i],
                         'label_conflict': bad, 'cross_map': len(unique_groups) > 1})
    summary = {'duplicate_sets': duplicate_sets, 'duplicate_rows': len(rows),
               'conflicting_label_sets': conflict_sets, 'conflicting_label_rows': int(conflict.sum()),
               'cross_map_duplicate_sets': cross_map_sets,
               'same_label_cross_map_edges': [list(edge) for edge in sorted(edges)]}
    return conflict, rows, summary


def read_metadata(root: Path, manifest: list[dict]) -> pd.DataFrame:
    """名称显式对应；无颗粒ID、空间尺寸或物理样品ID时保持未知。"""
    frame = pd.read_csv(root / 'data/raw/RaMPI_metadata.csv')
    frame.columns = [str(c).replace('\u00a0', ' ').strip() for c in frame.columns]
    frame.iloc[:, 0] = frame.iloc[:, 0].astype(str).str.replace('\u00a0', ' ', regex=False).str.strip()
    names = ['map_id', 'laser_power_percent', 'scan_time_ms', 'scan_accumulations',
             'reported_spectra', 'reported_plastic_spectra', 'step_size_um']
    if len(frame.columns) != len(names):
        raise ValueError('元数据结构变化，请审查后再读取')
    frame.columns = names
    if frame.map_id.duplicated().any():
        raise ValueError('元数据包含重复map_id')
    result = pd.DataFrame({'map_id': [row['map_id'] for row in manifest]}).merge(frame, on='map_id', how='left', validate='one_to_one')
    result['physical_sample_id'] = ''
    result['particle_id_available'] = False
    result['spatial_shape_known'] = False
    result['is_environmental'] = result['map_id'].str.startswith('Map_Env_')
    return result


def prepare(root: Path, skip_download: bool = False, workers: int = 4, reuse_processed: bool = False) -> dict:
    started = time.perf_counter()
    manifest_path = root / 'data/manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if len(manifest) != 34 or len({row['map_id'] for row in manifest}) != 34:
        raise ValueError('清单应为34个唯一map')
    out = root / 'data/processed'
    out.mkdir(parents=True, exist_ok=True)
    if reuse_processed:
        audit = json.loads((out / 'data_audit.json').read_text(encoding='utf-8'))
        if audit.get('manifest_sha256') != manifest_sha256:
            raise ValueError('已处理数据的清单校验不匹配，请重新准备数据')
        with np.load(out / 'dataset.npz', allow_pickle=False) as cached:
            required = {'X', 'y', 'groups', 'wavenumbers', 'pixel_index'}
            if not required.issubset(cached.files):
                raise ValueError('NPZ缺少必要数组')
            X = cached['X']
            n = len(X)
            if X.shape != (audit['retained_spectra_count'], 1024) or X.dtype != np.float32 or not np.isfinite(X).all():
                raise ValueError('NPZ特征形状、数值或类型校验失败')
            if any(len(cached[key]) != n for key in ['y', 'groups', 'pixel_index']):
                raise ValueError('NPZ标签或索引长度不匹配')
            if cached['y'].dtype.kind != 'U' or cached['groups'].dtype.kind != 'U':
                raise ValueError('标签和组ID必须使用Unicode数组')
            if not np.isfinite(cached['wavenumbers']).all() or not np.all(np.diff(cached['wavenumbers']) > 0):
                raise ValueError('NPZ频轴校验失败')
        print(f'复用已处理数据，清单与数值校验通过：{n}条光谱×1024点', flush=True)
        return audit
    if skip_download:
        downloads = []
        for row in manifest:
            path = root / 'data/raw' / (row['map_id'] + '.csv')
            actual = md5_file(path)
            if actual != row['md5']:
                raise ValueError(f'{path.name}校验失败')
            downloads.append({'file_id': row['file_id'], 'path': path.name, 'status': 'verified', 'md5': actual})
        if md5_file(root / 'data/raw/RaMPI_metadata.csv') != METADATA_FILE['md5']:
            raise ValueError('元数据校验失败')
    else:
        downloads = download_all(root, manifest, workers)
    axes = {}
    for record in manifest:
        axis, order = native_axis(root / 'data/raw' / (record['map_id'] + '.csv'))
        axes[record['map_id']] = (axis, order)
    lower = max(axis.min() for axis, _ in axes.values())
    upper = min(axis.max() for axis, _ in axes.values())
    if lower >= upper:
        raise ValueError('各map不存在共同频轴覆盖范围')
    grid = np.linspace(lower, upper, 1024, dtype=np.float64)
    all_X, all_y, all_groups, all_pixel, all_native_hash = [], [], [], [], []
    excluded, map_audits = [], []
    for number, record in enumerate(manifest, 1):
        group = record['map_id']
        frame = pd.read_csv(root / 'data/raw' / (group + '.csv'))
        labels = frame.iloc[:, 0].map(normalize_text).to_numpy(dtype=str)
        values = frame.iloc[:, 1:].to_numpy(dtype=np.float64)
        axis, order = axes[group]
        values = values[:, order]
        finite = np.isfinite(values).all(axis=1)
        valid_label = ~np.isin(labels, ['nan', 'none', ''])
        valid = finite & valid_label
        for idx in np.flatnonzero(~valid):
            excluded.append({'map_id': group, 'pixel_index': int(idx), 'label': labels[idx],
                             'reason': 'nonfinite_raw' if not finite[idx] else 'missing_label'})
        pixels = np.flatnonzero(valid).astype(np.int32)
        native_values = values[valid]
        axis_bytes = axis.astype('<f8').tobytes()
        native_hashes = [hashlib.sha256(axis_bytes + row.astype('<f8').tobytes()).hexdigest() for row in native_values]
        # 插值、平滑、基线与SNV均独立作用于每一条谱，不使用图级或类级参数。
        interpolated = np.stack([np.interp(grid, axis, row) for row in native_values])
        smoothed = savgol_filter(interpolated, window_length=11, polyorder=3, axis=1)
        left = smoothed[:, :5].mean(axis=1, keepdims=True)
        right = smoothed[:, -5:].mean(axis=1, keepdims=True)
        baseline = left + (right - left) * np.linspace(0, 1, 1024)[None, :]
        corrected = smoothed - baseline
        centered = corrected - corrected.mean(axis=1, keepdims=True)
        std = centered.std(axis=1, keepdims=True, ddof=0)
        valid_std = (std[:, 0] > 1e-8) & np.isfinite(std[:, 0])
        for idx in np.flatnonzero(~valid_std):
            excluded.append({'map_id': group, 'pixel_index': int(pixels[idx]), 'label': labels[valid][idx],
                             'reason': 'near_zero_or_nonfinite_std'})
        processed = (centered[valid_std] / std[valid_std]).astype(np.float32)
        if not np.isfinite(processed).all():
            raise ValueError(f'{group}预处理后出现非有限值')
        all_X.append(processed)
        all_y.append(labels[valid][valid_std])
        all_groups.append(np.full(len(processed), group, dtype=f'<U{len(group)}'))
        all_pixel.append(pixels[valid_std])
        all_native_hash.extend(np.asarray(native_hashes)[valid_std].tolist())
        map_audits.append({'map_id': group, 'raw_rows': len(frame), 'retained_before_duplicate_audit': len(processed),
                           'raw_features': len(axis), 'native_axis_min': float(axis.min()), 'native_axis_max': float(axis.max()),
                           'raw_label_counts': dict(Counter(labels.tolist())), 'nonfinite_raw_rows': int((~finite).sum()),
                           'missing_label_rows': int((~valid_label).sum()), 'near_zero_std_rows': int((~valid_std).sum())})
        print(f'预处理 {number}/34: {group} {len(frame)}条', flush=True)
    X = np.concatenate(all_X)
    y = np.concatenate(all_y)
    groups = np.concatenate(all_groups)
    pixel = np.concatenate(all_pixel)
    native_hash = np.asarray(all_native_hash, dtype='<U64')
    # 保存float32输入的精确哈希，使训练层可在分组后复查泄漏。
    processed_hash = np.array([hashlib.sha256(row.astype('<f4').tobytes()).hexdigest() for row in X], dtype='<U64')
    native_conflict, native_duplicate_rows, native_duplicates = duplicate_summary(native_hash, y, groups, pixel, 'native')
    processed_conflict, processed_duplicate_rows, processed_duplicates = duplicate_summary(processed_hash, y, groups, pixel, 'processed_float32')
    conflict = native_conflict | processed_conflict
    for idx in np.flatnonzero(conflict):
        excluded.append({'map_id': groups[idx], 'pixel_index': int(pixel[idx]), 'label': y[idx], 'reason': 'identical_spectrum_label_conflict'})
    keep = ~conflict
    X, y, groups, pixel = X[keep], y[keep], groups[keep], pixel[keep]
    native_hash, processed_hash = native_hash[keep], processed_hash[keep]
    np.savez_compressed(out / 'dataset.npz', X=X, y=y, groups=groups, wavenumbers=grid.astype(np.float32), pixel_index=pixel)
    duplicate_columns = ['kind', 'sha256', 'map_id', 'pixel_index', 'label', 'label_conflict', 'cross_map']
    pd.DataFrame(native_duplicate_rows + processed_duplicate_rows, columns=duplicate_columns).to_csv(out / 'duplicate_spectra.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(excluded, columns=['map_id', 'pixel_index', 'label', 'reason']).to_csv(out / 'excluded_spectra.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame({'map_id': groups, 'pixel_index': pixel, 'label': y, 'native_sha256': native_hash, 'processed_sha256': processed_hash}).to_csv(out / 'spectrum_index.csv', index=False, encoding='utf-8-sig')
    metadata = read_metadata(root, manifest)
    actual = pd.DataFrame({'map_id': groups, 'label': y}).groupby('map_id').agg(kept_spectra=('label', 'size'), kept_plastic_spectra=('label', lambda values: int((values != 'blank').sum()))).reset_index()
    metadata = metadata.merge(actual, on='map_id', how='left', validate='one_to_one')
    metadata.to_csv(out / 'map_metadata.csv', index=False, encoding='utf-8-sig')
    mismatch = metadata[(metadata['reported_spectra'] != metadata['kept_spectra']) |
                        (metadata['reported_plastic_spectra'] != metadata['kept_plastic_spectra'])]
    reconciliation = {
        'metadata_total_spectra': int(metadata['reported_spectra'].sum()),
        'metadata_plastic_spectra': int(metadata['reported_plastic_spectra'].sum()),
        'actual_retained_plastic_spectra': int(np.sum(y != 'blank')),
        'actual_retained_blank_spectra': int(np.sum(y == 'blank')),
        'mismatch_maps': mismatch[['map_id', 'reported_spectra', 'kept_spectra',
                                   'reported_plastic_spectra', 'kept_plastic_spectra']].to_dict(orient='records'),
        'policy': '以下载CSV的逐行标签计数为本作业实际统计，不为匹配论文或元数据总数而改标签。',
    }
    final_labels = sorted(np.unique(y).tolist())
    per_map = pd.crosstab(pd.Series(groups, name='map_id'), pd.Series(y, name='label')).reindex(columns=final_labels, fill_value=0)
    per_map.to_csv(out / 'map_label_counts.csv', encoding='utf-8-sig')
    audit = {
        'source_doi': '10.5683/SP3/8UQQQN', 'paper_doi': '10.1038/s41597-026-07103-8', 'license': 'CC BY 4.0',
        'manifest_sha256': manifest_sha256,
        'map_count': len(manifest), 'raw_spectra_count': sum(item['raw_rows'] for item in map_audits),
        'retained_spectra_count': len(X), 'features': X.shape[1], 'dtype': str(X.dtype),
        'class_count': len(final_labels), 'class_labels': final_labels, 'label_counts': dict(Counter(y.tolist())),
        'common_axis_cm_inverse': [float(lower), float(upper)], 'axis_policy': '所有34幅图原始频轴交集，升序等间距1024点，不外推',
        'preprocessing': {'interpolation': 'linear', 'savgol_window': 11, 'savgol_polyorder': 3,
                          'baseline': '两端各5点均值之间的线性基线', 'normalization': '逐谱SNV ddof=0', 'zero_std_threshold': 1e-8,
                          'fitted_across_spectra_statistics': False},
        'excluded_rows': len(excluded), 'excluded_reason_counts': dict(Counter(row['reason'] for row in excluded)),
        'nonfinite_final_values': int((~np.isfinite(X)).sum()), 'native_exact_duplicates': native_duplicates,
        'processed_exact_duplicates': processed_duplicates, 'per_map': map_audits,
        'metadata_reconciliation': reconciliation,
        'per_map_label_counts': {str(index): {str(k): int(v) for k, v in row.items()} for index, row in per_map.to_dict(orient='index').items()},
        'downloads': downloads, 'runtime_seconds': time.perf_counter() - started,
        'limitations': ['33119条是像素光谱，不是33119个独立物理样品。', '图与图之间可能共享RaSPI中的物理样品，元数据没有物理样品ID。',
                        'CSV没有空间坐标或可靠二维图尺寸，不从像素数猜测二维形状或连通粒子ID。',
                        '同标签完全重复谱若跨图出现，训练层必须将有关图合并为隔离组或剔除重复，不能跨训练和测试。'],
        'metadata_file': METADATA_FILE,
    }
    (out / 'data_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({key: audit[key] for key in ['raw_spectra_count', 'retained_spectra_count', 'map_count', 'class_count', 'features', 'runtime_seconds']}, ensure_ascii=False), flush=True)
    return audit


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--skip-download', action='store_true', help='只校验已下载文件并重新预处理')
    parser.add_argument('--reuse-processed', action='store_true', help='校验并复用随包提供的处理后NPZ，不重新下载原始CSV')
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 8:
        parser.error('workers应在1到8之间')
    prepare(args.root, args.skip_download, args.workers, args.reuse_processed)
