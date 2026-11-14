"""仅在28个开发map进行嵌套研究，原始6个测试map不进入本程序的数据集。"""
from pathlib import Path
import argparse,csv,gc,hashlib,json,os,random,sys,time
import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer,StandardScaler
from sklearn.svm import LinearSVC

PROJECT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT/'src'))
from common import metrics,save_json,file_hash,CosineCentroid
from models import augment_spectra
from study_models import build_model


def calibration(y,probs):
    pred=probs.argmax(1);confidence=probs.max(1);correct=(pred==y)
    ece=0.
    for k in range(10):
        m=(confidence>=k/10)&((confidence<(k+1)/10) if k<9 else (confidence<=1.))
        if m.any():ece+=m.mean()*abs(correct[m].mean()-confidence[m].mean())
    target=np.eye(probs.shape[1],dtype=np.float32)[y]
    return dict(ece_10_bins=float(ece),negative_log_likelihood=float(-np.log(np.clip(probs[np.arange(len(y)),y],1e-8,1)).mean()),
                multiclass_brier=float(np.sum((probs-target)**2,axis=1).mean()))


class StudyData:
    def __init__(self,root):
        self.root=Path(root);self.study=self.root/'extension_study'
        self.split_path=self.study/'protocol/splits.json'
        self.splits=json.loads(self.split_path.read_text(encoding='utf-8'))
        self.plan_path=self.study/'protocol/experiment_plan.json'
        self.plan=json.loads(self.plan_path.read_text(encoding='utf-8'))
        self.protocol_sha=file_hash(self.split_path);self.plan_sha=file_hash(self.plan_path)
        assert self.plan['split_sha256']==self.protocol_sha
        self.path=self.study/'data/study_data.npz'
        self.data_sha=file_hash(self.path)
        audit=json.loads((self.study/'data/study_data_audit.json').read_text(encoding='utf-8'))
        assert audit['study_dataset_sha256']==self.data_sha and audit['protocol_sha256']==self.protocol_sha
        self.archive=np.load(self.path,allow_pickle=False)
        self.original=self.archive['original_indices'].astype(np.int64)
        self.labels=[str(s) for s in self.splits['labels']]
        text=self.archive['y'].astype(str)
        self.y=np.array([self.labels.index(s) for s in text],dtype=np.int64)
        self.groups=self.archive['groups'].astype(str)
        self.pixels=self.archive['pixel_index'].astype(np.int32)
        assert set(self.groups).isdisjoint(self.splits['original_test_maps_excluded'])
        assert set(self.groups)==set(self.splits['development_maps'])
        assert len(self.original)==len(set(self.original))==25541
        self.reverse=np.full(int(self.original.max())+1,-1,dtype=np.int64)
        self.reverse[self.original]=np.arange(len(self.original))
        self.cache={}

    def X(self,variant):
        if variant not in self.cache:
            value=self.archive['X_'+variant]
            assert value.dtype==np.float32 and value.shape==(len(self.y),1024) and np.isfinite(value).all()
            self.cache[variant]=value
        return self.cache[variant]

    def local(self,indices):
        result=self.reverse[np.array(indices,dtype=np.int64)]
        assert np.all(result>=0)
        return result

    def role(self,fold_id,level='maps100'):
        fold=next(f for f in self.splits['folds'] if int(f['fold_id'])==fold_id)
        curve=next(item for item in fold['learning_curve'] if item['level']==level)
        a=self.local(curve['indices']);b=self.local(fold['inner_val_indices']);c=self.local(fold['outer_eval_indices'])
        assert set(self.groups[a]).isdisjoint(self.groups[b])
        assert set(self.groups[a]).isdisjoint(self.groups[c])
        assert set(self.groups[b]).isdisjoint(self.groups[c])
        assert set(self.y[a])==set(range(len(self.labels)))
        return a,b,c


def predict(model,X,device,batch=512):
    result=[];model.eval()
    with torch.inference_mode():
        for start in range(0,len(X),batch):
            x=torch.from_numpy(X[start:start+batch,None,:]).to(device)
            result.append(model(x).softmax(1).cpu().numpy())
    return np.concatenate(result)


def outputs(data,out,c,pred,probs=None):
    value=metrics(data.y[c],pred,data.groups[c])
    if probs is not None:value.update(calibration(data.y[c],probs))
    frame=pd.DataFrame(dict(original_index=data.original[c],map_id=data.groups[c],pixel_index=data.pixels[c],
                            true_label=np.array(data.labels)[data.y[c]],predicted_label=np.array(data.labels)[pred]))
    if probs is not None:
        frame['confidence']=probs.max(1)
        np.savez_compressed(out/'outer_probabilities.npz',original_indices=data.original[c],probabilities=probs.astype(np.float32))
    frame.to_csv(out/'outer_predictions.csv',index=False,encoding='utf-8-sig')
    per_map=[]
    for group in sorted(np.unique(data.groups[c])):
        m=data.groups[c]==group
        per_map.append(dict(map_id=group,**metrics(data.y[c][m],pred[m],data.groups[c][m])))
    pd.DataFrame(per_map).to_csv(out/'outer_map_metrics.csv',index=False,encoding='utf-8-sig')
    save_json(out/'outer_metrics.json',value)
    return value


def neural(data,config,device):
    out=data.study/'results/runs'/config['run_id'];out.mkdir(parents=True,exist_ok=True)
    metadata=out/'metadata.json'
    if metadata.exists():
        old=json.loads(metadata.read_text(encoding='utf-8'))
        if old.get('status')=='finished':
            assert old['plan_sha256']==data.plan_sha and old['data_sha256']==data.data_sha and old['protocol_sha256']==data.protocol_sha
            assert all((out/name).exists() for name in ['best.pt','outer_predictions.csv','outer_metrics.json','outer_probabilities.npz','history.csv'])
            return old
    seed=int(config['seed']);random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if device.type=='cuda':torch.cuda.manual_seed_all(seed)
    a,b,c=data.role(int(config['fold_id']),config['map_level'])
    X=data.X(config['variant'])
    record=dict(config=config,status='running',labels=data.labels,data_sha256=data.data_sha,
                protocol_sha256=data.protocol_sha,plan_sha256=data.plan_sha,device=str(device),
                train_maps=len(np.unique(data.groups[a])),train_pixels=len(a),inner_val_pixels=len(b),outer_eval_pixels=len(c),
                original_test_used=False,torch_version=str(torch.__version__))
    save_json(metadata,record);started=time.perf_counter()
    model=build_model(config['model'],len(data.labels)).to(device)
    record['parameters']=sum(p.numel() for p in model.parameters())
    weights=None
    if config['weighted_loss']:
        counts=np.bincount(data.y[a],minlength=len(data.labels))
        weights=torch.tensor(len(a)/(len(counts)*counts),dtype=torch.float32,device=device)
    loss_fn=nn.CrossEntropyLoss(weight=weights)
    opt=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=config['epochs'])
    batch=config['batch_size']
    loader=DataLoader(TensorDataset(torch.from_numpy(X[a,None,:]),torch.from_numpy(data.y[a])),
                     batch_size=batch,shuffle=True,num_workers=0,pin_memory=device.type=='cuda',
                     drop_last=(len(a)%batch==1),generator=torch.Generator().manual_seed(seed))
    best=-np.inf;stale=0;history=[]
    for epoch in range(1,config['epochs']+1):
        tick=time.perf_counter();model.train();total=0.;seen=0
        for x,y in loader:
            x,y=x.to(device),y.to(device)
            if config['augment']:x=augment_spectra(x)
            opt.zero_grad(set_to_none=True);loss=loss_fn(model(x),y)
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step()
            total+=loss.item()*len(y);seen+=len(y)
        probs=predict(model,X[b],device)
        val=metrics(data.y[b],probs.argmax(1),data.groups[b])
        score=val['macro_f1_present']
        history.append(dict(epoch=epoch,train_loss=total/seen,inner_val_macro_f1=score,seconds=time.perf_counter()-tick))
        pd.DataFrame(history).to_csv(out/'history.csv',index=False)
        if score>best:
            best=score;stale=0
            torch.save(dict(state_dict=model.state_dict(),labels=data.labels,epoch=epoch,
                            data_sha256=data.data_sha,protocol_sha256=data.protocol_sha,plan_sha256=data.plan_sha),out/'best.pt')
        else:stale+=1
        schedule.step()
        if epoch%10==0:print(json.dumps({'run':config['run_id'],'epoch':epoch,'best_inner_val':best}),flush=True)
        if stale>=config['patience']:break
    state=torch.load(out/'best.pt',map_location=device,weights_only=True)
    assert state['data_sha256']==data.data_sha and state['protocol_sha256']==data.protocol_sha
    assert state['plan_sha256']==data.plan_sha and state['labels']==data.labels
    model.load_state_dict(state['state_dict'])
    # 外折在epoch锁定后才计算，不用于early stopping。
    probs=predict(model,X[c],device)
    score=outputs(data,out,c,probs.argmax(1),probs)
    if config['family']=='architecture':
        robustness=[]
        for sigma in data.plan['noise_levels']:
            rng=np.random.default_rng(8800+int(config['fold_id'])*100+int(sigma*1000))
            noise=rng.normal(0,sigma,size=X[c].shape).astype(np.float32)
            value=predict(model,X[c]+noise,device)
            robustness.append(dict(noise_std=sigma,**metrics(data.y[c],value.argmax(1),data.groups[c]),**calibration(data.y[c],value)))
        pd.DataFrame(robustness).to_csv(out/'noise_robustness.csv',index=False)
    record.update(status='finished',best_epoch=int(state['epoch']),epochs_completed=len(history),
                  best_inner_val_macro_f1=best,runtime_seconds=time.perf_counter()-started,outer_metrics=score)
    save_json(metadata,record)
    print(json.dumps({'finished':config['run_id'],'epochs':len(history),'outer_macro_f1':score['macro_f1_present']},ensure_ascii=False),flush=True)
    del model,opt,loader,loss_fn
    gc.collect()
    if device.type=='cuda':torch.cuda.empty_cache()
    return record


def baselines(data):
    X=data.X('default')
    for fold in data.splits['folds']:
        f=int(fold['fold_id']);a,b,c=data.role(f)
        for name in data.plan['baselines']:
            out=data.study/'results/runs'/f'baseline_{name}_f{f}';out.mkdir(parents=True,exist_ok=True)
            if (out/'metadata.json').exists():
                old=json.loads((out/'metadata.json').read_text(encoding='utf-8'))
                if old.get('status')=='finished':
                    assert old['plan_sha256']==data.plan_sha and old['data_sha256']==data.data_sha and old['protocol_sha256']==data.protocol_sha
                    assert all((out/name).exists() for name in ['model.joblib','outer_predictions.csv','outer_metrics.json'])
                    continue
            tick=time.perf_counter();search=[]
            if name=='cosine_centroid':
                model=Pipeline([('norm',Normalizer()),('model',CosineCentroid())]).fit(X[a],data.y[a])
            elif name=='random_forest_300':
                model=RandomForestClassifier(n_estimators=300,min_samples_leaf=2,class_weight='balanced_subsample',n_jobs=8,random_state=101).fit(X[a],data.y[a])
            else:
                pca=PCA(n_components=64,svd_solver='randomized',random_state=101).fit(X[a])
                za,zb=pca.transform(X[a]),pca.transform(X[b]);scale=StandardScaler().fit(za)
                za,zb=scale.transform(za),scale.transform(zb);best=-np.inf
                for C in data.plan['baseline_C_candidates']:
                    clf=LinearSVC(C=C,class_weight='balanced',dual='auto',max_iter=10000,random_state=101).fit(za,data.y[a])
                    v=metrics(data.y[b],clf.predict(zb),data.groups[b])['macro_f1_present'];search.append(dict(C=C,inner_val_macro_f1=v))
                    if v>best:best=v;chosen=clf
                model=Pipeline([('pca',pca),('scale',scale),('model',chosen)])
            val=metrics(data.y[b],model.predict(X[b]),data.groups[b])
            value=outputs(data,out,c,model.predict(X[c]))
            joblib.dump(model,out/'model.joblib',compress=3)
            save_json(out/'metadata.json',dict(status='finished',config=dict(run_id=out.name,model=name,fold_id=f,family='baseline',seed=101),
                labels=data.labels,plan_sha256=data.plan_sha,protocol_sha256=data.protocol_sha,data_sha256=data.data_sha,
                inner_val_metrics=val,outer_metrics=value,parameter_search=search,original_test_used=False,
                runtime_seconds=time.perf_counter()-tick,train_maps=len(np.unique(data.groups[a])),train_pixels=len(a)))
            print(json.dumps({'finished':out.name,'outer_macro_f1':value['macro_f1_present']}),flush=True)


def main(args):
    if args.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('请求CUDA但不可用。')
    device=torch.device(args.device)
    torch.set_num_threads(8);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    data=StudyData(args.root);out=data.study/'results';out.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter()
    status=dict(status='running',started_at=pd.Timestamp.now(tz='Asia/Shanghai').isoformat(),
                original_test_used=False,plan_sha256=data.plan_sha,data_sha256=data.data_sha,
                gpu=torch.cuda.get_device_name(0) if device.type=='cuda' else None,total_neural_runs=len(data.plan['configs']))
    save_json(out/'study_execution.json',status)
    configs=data.plan['configs']
    if args.only_run:
        configs=[c for c in configs if c['run_id']==args.only_run]
        if len(configs)!=1:raise ValueError('only-run必须是计划中唯一有效的run_id。')
    if not args.neural_only:baselines(data)
    for number,config in enumerate(configs,1):
        if time.perf_counter()-started>data.plan['max_wall_hours']*3600:
            status.update(status='runtime_limit_reached',completed_in_this_process=number-1)
            save_json(out/'study_execution.json',status);raise RuntimeError('达到预设运行时间上限，请检查预算后再决定续跑。')
        neural(data,config,device)
        status.update(last_completed_run=config['run_id'],completed_in_this_process=number,elapsed_seconds=time.perf_counter()-started)
        save_json(out/'study_execution.json',status)
    expected=[c['run_id'] for c in data.plan['configs']]+[f'baseline_{name}_f{fold}' for fold in [1,2,3] for name in data.plan['baselines']]
    finished=[]
    for run in expected:
        path=out/'runs'/run/'metadata.json'
        if path.exists():
            record=json.loads(path.read_text(encoding='utf-8'))
            if record.get('status')=='finished' and record.get('plan_sha256')==data.plan_sha and record.get('data_sha256')==data.data_sha:
                finished.append(run)
    status.update(status='completed' if len(finished)==len(expected) else 'subset_completed',
                  completed_total=len(finished),expected_total=len(expected),
                  finished_at=pd.Timestamp.now(tz='Asia/Shanghai').isoformat(),elapsed_seconds=time.perf_counter()-started)
    save_json(out/'study_execution.json',status)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=PROJECT)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--only-run')
    p.add_argument('--neural-only',action='store_true')
    main(p.parse_args())
