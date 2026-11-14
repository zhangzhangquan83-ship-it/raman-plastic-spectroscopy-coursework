"""在已检查的云实例上执行冻结实验；本程序不操作云实例开关机。"""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import json
import os
import subprocess
import sys
import time
import hashlib

root=Path(__file__).resolve().parent
out=root/'results'
out.mkdir(exist_ok=True)
record_path=out/'cloud_execution.json'
if record_path.exists() or (out/'final_test/evaluation_record.json').exists():
    raise FileExistsError('已有首轮回执或正式结果，请在不含results的新目录从零复现，保留原结果。')
def now():return datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=8))).isoformat()
def save(obj):
    tmp=record_path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
    tmp.replace(record_path)

def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        while block:=stream.read(1024*1024):digest.update(block)
    return digest.hexdigest()

# 新目录也可从零执行；只准备数据和工程检查，不提前计算测试表现。
prepare=[sys.executable,'src/prepare_data.py']
if (root/'data/processed/dataset.npz').exists():prepare.append('--reuse-processed')
subprocess.run(prepare,cwd=root,check=True)
subprocess.run([sys.executable,'src/make_split.py'],cwd=root,check=True)
subprocess.run([sys.executable,'src/sanity_check.py','--device','cuda'],cwd=root,check=True)

record=dict(status='running',started_at=now(),command=['bash','run_cloud.sh'],
            historical_initial_archive_sha256='8c4ac64e9dd60b5df2ceb1fd17263409e1adb0c9f9e94425d59c42a47585fe39',
            source_dataset_sha256=sha(root/'data/processed/dataset.npz'),
            source_split_sha256=sha(root/'data/processed/map_split.csv'),
            actual_source_sha256={p.relative_to(root).as_posix():sha(p) for p in sorted((root/'src').glob('*.py'))},
            hardware_preflight=json.loads((out/'sanity_check/verification.json').read_text(encoding='utf-8')),
            gpu_training_executed=True,formal_test_evaluated=False)
save(record)
started=time.perf_counter()
env=os.environ.copy()
env['PATH']='/root/miniconda3/bin:'+env.get('PATH','')
env['PYTHONUNBUFFERED']='1'
with (out/'cloud_run.log').open('w',encoding='utf-8') as log:
    result=subprocess.run(['bash','run_cloud.sh'],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
record.update(finished_at=now(),elapsed_seconds=time.perf_counter()-started,
              exit_code=result.returncode,status='completed' if result.returncode==0 else 'failed',
              formal_test_evaluated=(out/'final_test/evaluation_record.json').exists())
save(record)
raise SystemExit(result.returncode)
