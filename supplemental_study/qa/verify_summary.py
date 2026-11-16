"""人工值验证汇总口径及pending退出；不会生成虚构的正式结果。"""
import csv
import importlib.util
import io
import json
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'supplemental_study/src/summarize_stage1_supplement.py'
spec = importlib.util.spec_from_file_location('stage1_summary', SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

old = module.read_csv(ROOT / 'results/final_test/model_comparison.csv')
rows = [module.normalize_run(row, 'historical_stage1') for row in old]
for i, name in enumerate(module.SUPPLEMENT):
    # 明确为人工构造的测试值，仅传入汇总函数，不写入任何正式CSV或报告。
    number = 0.4 if i == 0 else 0.5 + ((i - 1) % 3) * 0.1
    fake = dict(model=name, accuracy=number, balanced_accuracy=number,
                macro_f1_present=number, map_mean_accuracy=number,
                map_equal_weight_macro_f1=number, evaluated_classes=15, pixels=6399, maps=6)
    rows.append(module.normalize_run(fake, 'posthoc_supplement'))
summary = module.aggregate_runs(rows)
by_method = {row['method']: row for row in summary}
assert [r['method'] for r in summary] == list(module.ORDER)
assert len(rows) == 13 and len(summary) == 6
assert abs(by_method['mlp']['macro_f1_present_mean'] - 0.6) < 1e-15
assert abs(by_method['mlp']['macro_f1_present_seed_sd'] - 0.1) < 1e-15
expected_cnn = statistics.mean(float(row['macro_f1_present']) for row in old if row['model'].startswith('cnn_aug_'))
assert by_method['cnn']['macro_f1_present_mean'] == expected_cnn
assert by_method['cnn']['n_runs'] == 3
assert all(by_method[m]['accuracy_seed_sd'] is None for m in module.ORDER[:3])
buffer = io.StringIO(newline='')
writer = csv.DictWriter(buffer, fieldnames=module.SUMMARY_FIELDS)
writer.writeheader()
writer.writerows(summary)
buffer.seek(0)
serialized = list(csv.DictReader(buffer))
assert all(row['accuracy_seed_sd'] == '' for row in serialized[:3])
assert serialized[3]['n_runs'] == '3'

try:
    module.aggregate_runs([row for row in rows if row['model'] != 'mlp_aug_seed44'])
except ValueError:
    pass
else:
    raise AssertionError('缺少预定种子必须失败。')

with tempfile.TemporaryDirectory(prefix='pending_summary_', dir=ROOT / 'supplemental_study/qa') as directory:
    temporary_root = Path(directory).resolve()
    assert temporary_root.is_relative_to((ROOT / 'supplemental_study/qa').resolve())
    result = subprocess.run([sys.executable, str(SOURCE), '--root', str(temporary_root)],
                            capture_output=True, text=True, encoding='utf-8')
    assert result.returncode == 2, result.stderr
    status = json.loads(result.stdout)
    assert status['status'] == 'pending' and status['passed'] is False
    assert not list(temporary_root.rglob('combined_test_runs.csv'))
    assert not list(temporary_root.rglob('stage1_supplement_report.md'))

report = dict(status='passed', synthetic_aggregation_only=True, no_real_prediction_or_training=True,
              checks=['13运行及6方法顺序', '三种子样本标准差', '无增强CNN未混入主表',
                      '传统SD写空白', '缺种子阻断', '未齐结果pending且退出2，不生成最终报告'])
module.write_json(ROOT / 'supplemental_study/qa/summary_verification.json', report)
print(json.dumps(report, ensure_ascii=False))
