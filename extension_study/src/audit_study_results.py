"""从逐谱预测独立构造混淆矩阵，核对84组结果及冻结数据边界。"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
import torch


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        while block:=stream.read(1024*1024):h.update(block)
    return h.hexdigest()


def f1_from_matrix(cm,present):
    diagonal=np.diag(cm)
    denominator=cm.sum(0)+cm.sum(1)
    f1=np.divide(2*diagonal,denominator,out=np.zeros(len(cm)),where=denominator>0)
    return float(f1[present].mean())


def independent_metrics(y,pred,groups,k):
    cm=np.zeros((k,k),dtype=np.float64)
    np.add.at(cm,(y,pred),1.)
    present=cm.sum(1)>0
    names,inverse,counts=np.unique(groups,return_inverse=True,return_counts=True)
    weighted=np.zeros_like(cm)
    np.add.at(weighted,(y,pred),1./counts[inverse])
    return dict(accuracy=float(np.mean(y==pred)),
                balanced_accuracy=float((np.diag(cm)[present]/cm.sum(1)[present]).mean()),
                macro_f1_present=f1_from_matrix(cm,present),
                map_mean_accuracy=float(np.mean([np.mean(y[groups==g]==pred[groups==g]) for g in names])),
                map_equal_weight_macro_f1=f1_from_matrix(weighted,present),
                evaluated_classes=int(present.sum()),pixels=len(y),maps=len(names))


def main(root):
    study=root/'extension_study';protocol=study/'protocol/splits.json';plan_path=study/'protocol/experiment_plan.json'
    splits=json.loads(protocol.read_text(encoding='utf-8'));plan=json.loads(plan_path.read_text(encoding='utf-8'))
    data_path=study/'data/study_data.npz'
    data_sha=sha(data_path);plan_sha=sha(plan_path);protocol_sha=sha(protocol)
    with np.load(data_path,allow_pickle=False) as archive:
        original=archive['original_indices'];labels_text=archive['y'].astype(str)
        groups=archive['groups'].astype(str);pixels=archive['pixel_index']
    labels=splits['labels'];encoding={name:i for i,name in enumerate(labels)}
    lookup={int(index):position for position,index in enumerate(original)}
    configs={c['run_id']:c for c in plan['configs']}
    for fold in [1,2,3]:
        for name in plan['baselines']:
            rid=f'baseline_{name}_f{fold}'
            configs[rid]=dict(run_id=rid,fold_id=fold,model=name,family='baseline')
    records=[];largest_error=0.;neural_count=0
    for rid,config in configs.items():
        out=study/'results/runs'/rid
        metadata=json.loads((out/'metadata.json').read_text(encoding='utf-8'))
        assert metadata['status']=='finished',rid
        assert metadata['data_sha256']==data_sha and metadata['plan_sha256']==plan_sha and metadata['protocol_sha256']==protocol_sha,rid
        assert metadata['labels']==labels and metadata['original_test_used'] is False,rid
        frame=pd.read_csv(out/'outer_predictions.csv')
        assert frame.original_index.is_unique and not frame[['map_id','pixel_index']].duplicated().any(),rid
        fold=next(f for f in splits['folds'] if int(f['fold_id'])==config['fold_id'])
        assert frame.original_index.tolist()==fold['outer_eval_indices'],rid
        locations=np.array([lookup[int(i)] for i in frame.original_index])
        assert np.array_equal(frame.true_label,labels_text[locations]),rid
        assert np.array_equal(frame.map_id,groups[locations]),rid
        assert np.array_equal(frame.pixel_index,pixels[locations]),rid
        assert set(frame.map_id).isdisjoint(splits['original_test_maps_excluded']),rid
        y=np.array([encoding[v] for v in frame.true_label]);pred=np.array([encoding[v] for v in frame.predicted_label])
        expected=independent_metrics(y,pred,frame.map_id.to_numpy(),len(labels))
        actual=json.loads((out/'outer_metrics.json').read_text(encoding='utf-8'))
        assert all(key in actual and np.isfinite(actual[key]) for key in expected),rid
        differences={key:abs(actual[key]-value) for key,value in expected.items()}
        error=max(differences.values());largest_error=max(largest_error,error)
        assert error<1e-10,(rid,differences)
        if config['family']!='baseline':
            neural_count+=1
            assert metadata['device']=='cuda' and metadata['config']==config,rid
            history=pd.read_csv(out/'history.csv');best_epoch=int(metadata['best_epoch'])
            assert np.isfinite(history[['train_loss','inner_val_macro_f1','seconds']].to_numpy()).all(),rid
            assert len(history)==metadata['epochs_completed']<=config['epochs'],rid
            best=float(history.inner_val_macro_f1.max())
            assert abs(history.iloc[best_epoch-1].inner_val_macro_f1-best)<1e-12,rid
            assert abs(metadata['best_inner_val_macro_f1']-best)<1e-12,rid
            assert len(history)==config['epochs'] or len(history)-best_epoch>=config['patience'],rid
            state=torch.load(out/'best.pt',map_location='cpu',weights_only=True)
            assert state['labels']==labels and state['epoch']==best_epoch,rid
            assert state['data_sha256']==data_sha and state['plan_sha256']==plan_sha and state['protocol_sha256']==protocol_sha,rid
            assert all(torch.isfinite(v).all() for v in state['state_dict'].values() if v.is_floating_point()),rid
            with np.load(out/'outer_probabilities.npz',allow_pickle=False) as archive:
                native_probability=archive['probabilities']
                probability=native_probability.astype(np.float64)
                assert np.array_equal(archive['original_indices'],frame.original_index),rid
            assert probability.shape==(len(y),len(labels)) and np.isfinite(probability).all(),rid
            assert np.allclose(probability.sum(1),1.,atol=3e-7) and np.all(probability>=0),rid
            assert np.array_equal(probability.argmax(1),pred),rid
            confidence=probability.max(1);native_confidence=native_probability.max(1);correct=(pred==y)
            assert np.allclose(frame.confidence,confidence,atol=1e-7,rtol=1e-7),rid
            ece=0.
            for bin_id in range(10):
                # 分箱遵循保存float32概率的边界定义，避免升精度改变0.7/0.9等边界归属。
                mask=(native_confidence>=bin_id/10)&((native_confidence<(bin_id+1)/10) if bin_id<9 else (native_confidence<=1.))
                if mask.any():ece+=mask.mean()*abs(correct[mask].mean()-confidence[mask].mean())
            nll=float(-np.log(np.clip(probability[np.arange(len(y)),y],1e-8,1)).mean())
            target=np.eye(len(labels))[y];brier=float(((probability-target)**2).sum(1).mean())
            assert all(np.isfinite(actual[key]) for key in ['ece_10_bins','negative_log_likelihood','multiclass_brier']),rid
            assert abs(ece-actual['ece_10_bins'])<3e-7 and abs(nll-actual['negative_log_likelihood'])<3e-6,rid
            assert abs(brier-actual['multiclass_brier'])<3e-6,rid
        records.append(dict(run_id=rid,maximum_metric_error=error,pixels=len(frame),maps=frame.map_id.nunique(),prediction_sha256=sha(out/'outer_predictions.csv')))
    receipt=json.loads((study/'results/cloud_extension_execution.json').read_text(encoding='utf-8'))
    assert receipt['status']=='completed' and receipt['original_frozen_inputs_unchanged'] is True
    assert receipt['frozen_before']==receipt['frozen_after']
    for relative,expected in receipt['frozen_after'].items():assert sha(root/relative)==expected,relative
    core_source=['build_plan.py','prepare_study_data.py','run_study.py','split_study.py','study_models.py']
    for name in core_source:assert sha(study/'src'/name)==receipt['source_hashes'][name],name
    audit=dict(status='passed',expected_neural_runs=75,verified_neural_runs=neural_count,
               verified_baseline_evaluations=len(records)-neural_count,verified_total=len(records),
               independent_calculation='numpy手工混淆矩阵与逐图权重，不调用训练或评价程序的metric helper。',
               maximum_metric_error=largest_error,source_dataset_sha256=sha(root/'data/processed/dataset.npz'),
               study_data_sha256=data_sha,protocol_sha256=protocol_sha,plan_sha256=plan_sha,
               audit_source_sha256=sha(Path(__file__).resolve()),
               original_test_used_in_extension=False,original_inputs_and_test_results_unchanged=True,
               training_source_hashes_match_execution=True,verified_training_source_files=core_source,
               caveats=['核对全部结果预测、元数据、概率、checkpoint和早停记录；未独立重复全部训练。',
                        '折间标准差是3折的描述性离散程度，种子不是独立采样单位。'],runs=records)
    path=study/'results/independent_audit.json';path.write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({key:audit[key] for key in ['status','verified_neural_runs','verified_baseline_evaluations','maximum_metric_error']},ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2])
    main(parser.parse_args().root)
