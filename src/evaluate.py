"""训练与选型完成后统一评价独立测试map。禁止用本输出反复调参。"""
import argparse
from pathlib import Path
import json
import time
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report,confusion_matrix
from common import ROOT,load_data,metrics,save_json,file_hash,CosineCentroid
from models import SpectralCNN
# joblib中自主训练的类需从其稳定模块路径导入；不得加载不可信外部模型。

DEFAULT_RUNS=['cnn_aug_seed42','cnn_aug_seed43','cnn_aug_seed44','cnn_noaug_seed42']


def predict_cnn(model,X,device,batch_size=512):
    predictions=[]
    model.eval()
    with torch.inference_mode():
        for start in range(0,len(X),batch_size):
            x=torch.from_numpy(X[start:start+batch_size,None,:]).to(device)
            predictions.append(model(x).argmax(1).cpu().numpy())
    return np.concatenate(predictions)


def main(args):
    root=Path(args.root)
    device=torch.device('cuda' if args.device=='auto' and torch.cuda.is_available() else 'cpu' if args.device=='auto' else args.device)
    torch.set_num_threads(args.cpu_threads)
    d=load_data(root);idx=d['idx']['test'];X=d['X'][idx];y=d['y'][idx];groups=d['groups'][idx]
    # 全部训练完成后再开始读取测试输入进行预测。
    for name in args.runs:
        metadata=json.loads((root/'results'/name/'metadata.json').read_text(encoding='utf-8'))
        if metadata.get('smoke_only') or metadata.get('status')!='finished':
            raise ValueError(f'{name}不是完成的正式训练，不能冒充最终模型。')
        if metadata['dataset_sha256']!=d['dataset_sha256'] or metadata['split_sha256']!=d['split_sha256']:
            raise ValueError('模型、数据与划分版本不一致。')
    baseline_meta=json.loads((root/'results/baselines/metadata.json').read_text(encoding='utf-8'))
    if baseline_meta['dataset_sha256']!=d['dataset_sha256'] or baseline_meta['split_sha256']!=d['split_sha256']:
        raise ValueError('基线数据版本不一致。')
    predictions={};checkpoints={};start=time.perf_counter()
    for name in ['cosine_centroid','pca64_linear_svm']:
        path=root/'results/baselines'/(name+'.joblib')
        model=joblib.load(path)
        predictions[name]=model.predict(X)
        checkpoints[name]=file_hash(path)
    for name in args.runs:
        path=root/'results'/name/'best.pt'
        checkpoint=torch.load(path,map_location=device,weights_only=True)
        if checkpoint['labels']!=d['labels']:raise ValueError('模型类别顺序不一致。')
        if checkpoint['dataset_sha256']!=d['dataset_sha256'] or checkpoint['split_sha256']!=d['split_sha256']:
            raise ValueError('checkpoint内部数据或划分校验值不一致。')
        model=SpectralCNN(len(d['labels'])).to(device)
        model.load_state_dict(checkpoint['state_dict'])
        predictions[name]=predict_cnn(model,X,device)
        checkpoints[name]=file_hash(path)
    out=root/'results/final_test';out.mkdir(parents=True,exist_ok=True)
    rows=[];map_rows=[]
    detail=pd.DataFrame(dict(map_id=groups,pixel_index=d['pixel_index'][idx],true_label=d['text_y'][idx]))
    for name,pred in predictions.items():
        rows.append(dict(model=name,**metrics(y,pred,groups)))
        detail[name]=np.array(d['labels'])[pred]
        report=classification_report(y,pred,labels=np.arange(len(d['labels'])),target_names=d['labels'],output_dict=True,zero_division=0)
        pd.DataFrame(report).T.to_csv(out/(name+'_classes.csv'),encoding='utf-8-sig')
        cm=confusion_matrix(y,pred,labels=np.arange(len(d['labels'])))
        pd.DataFrame(cm,index=d['labels'],columns=d['labels']).to_csv(out/(name+'_confusion.csv'),encoding='utf-8-sig')
        for g in np.unique(groups):
            m=groups==g
            map_rows.append(dict(model=name,map_id=g,**metrics(y[m],pred[m],groups[m])))
    comparison=pd.DataFrame(rows)
    comparison.to_csv(out/'model_comparison.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(map_rows).to_csv(out/'per_map_metrics.csv',index=False,encoding='utf-8-sig')
    detail.to_csv(out/'pixel_predictions.csv',index=False,encoding='utf-8-sig')
    seed_rows=comparison[comparison.model.str.match(r'^cnn_aug_seed\d+$')]
    if len(seed_rows)>1:
        seed_rows.drop(columns=['model']).agg(['mean','std']).to_csv(out/'cnn_seed_summary.csv',encoding='utf-8-sig')
    save_json(out/'evaluation_record.json',dict(dataset_sha256=d['dataset_sha256'],split_sha256=d['split_sha256'],
        model_sha256=checkpoints,test_maps=sorted(np.unique(groups)),runtime_seconds=time.perf_counter()-start,
        independent_test_units='6个采集map；物理样品是否独立未知',
        metric_definitions={'macro_f1_present':'测试所有15类有真值，故为15类macro-F1',
          'map_mean_accuracy':'6张图各自准确率的算术均值',
          'map_equal_weight_macro_f1':'像素权重=该图像素数的倒数，构造图等权混淆后计算macro-F1'},
        note='所有训练完成且参数锁定后统一评价。不能据此调参，再次结果应视为同一测试集的重复使用。'))
    print(comparison.round(4).to_string(index=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    p.add_argument('--runs',nargs='+',default=DEFAULT_RUNS)
    p.add_argument('--cpu-threads',type=int,default=4)
    main(p.parse_args())
