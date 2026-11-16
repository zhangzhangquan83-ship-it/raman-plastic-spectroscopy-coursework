"""补充runner的人工数组/权重往返检查；不拟合真实数据、不计算测试成绩。"""
import importlib.util
import json
from pathlib import Path
import py_compile
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / 'supplemental_study/src/run_stage1_supplement.py'
py_compile.compile(str(RUNNER), doraise=True)
spec = importlib.util.spec_from_file_location('stage1_supplement', RUNNER)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
torch.set_num_threads(2)

assert module.canonical_hash({'a': 1, 'b': 2}) == module.canonical_hash({'b': 2, 'a': 1})
configs = module.make_configs(module.TRAINING)
assert len(configs) == 7 and len({c['run_id'] for c in configs}) == 7
assert configs[0]['parameters']['random_state'] == 101
assert [c['seed'] for c in configs if c['model'] == 'mlp'] == [42, 43, 44]

shape_results = []
temporary = ROOT / 'supplemental_study/qa/synthetic_checkpoint.pt'
for name, parameters in [('mlp', 661519), ('resnet', 1227759)]:
    torch.manual_seed(913)
    model = module.build_model(name, 15)
    assert sum(p.numel() for p in model.parameters()) == parameters
    model.train()
    x = torch.randn(4, 1, 1024)
    logits = model(x)
    assert logits.shape == (4, 15) and torch.isfinite(logits).all()
    torch.nn.CrossEntropyLoss()(logits, torch.tensor([0, 1, 12, 14])).backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    test_x = np.zeros((3, 1024), dtype=np.float32)
    probabilities = module.predict_neural(model, test_x, torch.device('cpu'), batch=2)
    assert probabilities.shape == (3, 15) and np.allclose(probabilities.sum(1), 1, atol=1e-6)
    module.save_torch(temporary, {'state_dict': model.state_dict(), 'labels': [str(i) for i in range(15)]})
    restored = module.build_model(name, 15)
    restored.load_state_dict(torch.load(temporary, map_location='cpu', weights_only=True)['state_dict'])
    assert np.array_equal(module.predict_neural(restored, test_x, torch.device('cpu'), batch=2), probabilities)
    shape_results.append({'model': name, 'parameters': parameters, 'shape': [4, 15]})
temporary.unlink()

y = np.repeat(np.arange(13), 2)
groups = np.asarray(['mapA', 'mapB'] * 13)
value = module.metrics(y, y.copy(), groups)
assert value['macro_f1_present'] == 1 and value['evaluated_classes'] == 13
assert value['map_equal_weight_macro_f1'] == 1
args = module.parser().parse_args(['--phase', 'train', '--models', 'random_forest', '--device', 'cpu'])
assert args.models == ['random_forest'] and args.prediction_batch_size == 128
report = {'status': 'passed', 'synthetic_only': True, 'real_training_started': False,
          'real_test_evaluated': False, 'torch_version': str(torch.__version__),
          'checks': ['语法', '冻结配置7运行', '15类模型形状与参数量', '人工数组反向传播有限性',
                     'CPU推理与批量1尾批', '自有权重保存加载往返', '13类验证宏F1口径', 'CLI解析'],
          'models': shape_results}
module.save_json(ROOT / 'supplemental_study/qa/runner_verification.json', report)
print(json.dumps(report, ensure_ascii=False))
