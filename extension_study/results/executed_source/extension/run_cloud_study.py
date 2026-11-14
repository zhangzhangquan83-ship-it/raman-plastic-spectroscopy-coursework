"""云端扩展实验批处理：记录输入、源码和冻结测试哈希，不操作实例开关。"""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import hashlib
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / 'extension_study'
OUT = STUDY / 'results'
OUT.mkdir(parents=True, exist_ok=True)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(1024 * 1024):
            h.update(block)
    return h.hexdigest()


def frozen_inputs():
    paths = [ROOT / 'data/processed/dataset.npz', ROOT / 'data/processed/map_split.csv']
    paths += sorted(p for p in (ROOT / 'results/final_test').rglob('*') if p.is_file())
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def main():
    env = os.environ.copy()
    env.update(PYTHONUNBUFFERED='1', OMP_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8')
    record = dict(status='running', started_at=now(), original_test_evaluated=False,
                  frozen_before=frozen_inputs(), source_hashes={p.name: sha(p) for p in sorted((STUDY/'src').glob('*.py'))})
    started = time.perf_counter()
    receipt = OUT/'cloud_extension_execution.json'
    def save():
        temp = receipt.with_suffix('.tmp')
        temp.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
        temp.replace(receipt)
    save()
    prepare_command=[sys.executable, 'extension_study/src/prepare_study_data.py']
    if (STUDY/'data/study_data.npz').exists():
        prepare_command.append('--reuse')
        record['prepared_data_delivery']='复用已逐文件校验、从本机传入的开发特征；云端再核SHA与所有矩阵。'
    commands = [prepare_command,
                [sys.executable, 'extension_study/src/run_study.py']]
    try:
        for stage, command in zip(['prepare', 'train'], commands):
            record['stage'] = stage
            save()
            with (OUT/f'cloud_{stage}.log').open('w', encoding='utf-8') as log:
                result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            record[stage+'_exit_code'] = result.returncode
            save()
            if result.returncode:
                raise RuntimeError(f'{stage}失败：退出码{result.returncode}')
        record['status'] = 'completed'
    except Exception as exc:
        record.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        record.update(finished_at=now(), elapsed_seconds=time.perf_counter()-started,
                      frozen_after=frozen_inputs())
        record['original_frozen_inputs_unchanged'] = record['frozen_before']==record['frozen_after']
        save()
    if not record['original_frozen_inputs_unchanged']:
        raise RuntimeError('原始冻结输入或结果被改变。')


if __name__ == '__main__':
    main()
