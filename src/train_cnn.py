"""CNN训练与验证；本文件不计算测试集指标。"""
import argparse
import json
import random
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from common import ROOT,load_data,metrics,save_json
from models import SpectralCNN,augment_spectra


def main(args):
    root=Path(args.root)
    if args.device=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('请求CUDA但GPU不可用，请检查所选PyTorch镜像和驱动；本机验证可用--device cpu。')
    device=torch.device(args.device if args.device!='auto' else 'cuda' if torch.cuda.is_available() else 'cpu')
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(args.cpu_threads)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    d=load_data(root)
    a,b=d['idx']['train'],d['idx']['validation']
    if args.max_train_pixels:
        # 本地快速验证只抽取训练像素，测试集合不用于拟合或验证指标；分层保证每类有例子。
        chosen=[]
        rng=np.random.default_rng(args.seed)
        for c in range(len(d['labels'])):
            pool=a[d['y'][a]==c]
            chosen.extend(rng.choice(pool,size=min(len(pool),max(2,args.max_train_pixels//len(d['labels']))),replace=False).tolist())
        a=np.array(chosen)
    run_name=args.run_name or f"cnn_{'aug' if args.augment else 'noaug'}_seed{args.seed}"
    out=root/'results'/run_name
    out.mkdir(parents=True,exist_ok=True)
    if (out/'best.pt').exists() and not args.overwrite:
        raise FileExistsError(f'{out}/best.pt已存在，请使用新run-name或明确--overwrite。')
    train=DataLoader(TensorDataset(torch.from_numpy(d['X'][a,None,:]),torch.from_numpy(d['y'][a])),
                     batch_size=args.batch_size,shuffle=True,num_workers=0,pin_memory=device.type=='cuda',
                     generator=torch.Generator().manual_seed(args.seed))
    validation=DataLoader(TensorDataset(torch.from_numpy(d['X'][b,None,:]),torch.from_numpy(d['y'][b])),batch_size=args.batch_size*2)
    model=SpectralCNN(len(d['labels'])).to(device)
    counts=np.bincount(d['y'][a],minlength=len(d['labels']))
    weights=len(a)/(len(counts)*np.maximum(counts,1))
    loss_fn=nn.CrossEntropyLoss(weight=torch.tensor(weights,dtype=torch.float32,device=device))
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=max(args.epochs,1))
    best=-np.inf;stale=0;history=[];start=time.perf_counter()
    configuration=vars(args).copy();configuration['root']=str(root)
    metadata=dict(config=configuration,labels=d['labels'],dataset_sha256=d['dataset_sha256'],split_sha256=d['split_sha256'],
                  device=str(device),gpu=torch.cuda.get_device_name(0) if device.type=='cuda' else None,
                  parameters=sum(p.numel() for p in model.parameters()),torch_version=torch.__version__,
                  train_pixels=len(a),validation_pixels=len(b),test_evaluated=False,
                  smoke_only=bool(args.max_train_pixels or args.probe),status='running')
    save_json(out/'metadata.json',metadata)
    for epoch in range(1,args.epochs+1):
        tick=time.perf_counter();model.train();total=0.
        for x,y in train:
            x,y=x.to(device),y.to(device)
            if args.augment:x=augment_spectra(x)
            optimizer.zero_grad(set_to_none=True)
            loss=loss_fn(model(x),y)
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
            total+=loss.item()*len(y)
        model.eval();pred=[]
        with torch.inference_mode():
            for x,_ in validation:pred.append(model(x.to(device)).argmax(1).cpu().numpy())
        pred=np.concatenate(pred)
        scores=metrics(d['y'][b],pred,d['groups'][b])
        row=dict(epoch=epoch,train_loss=total/len(a),seconds=time.perf_counter()-tick,**scores)
        history.append(row)
        pd.DataFrame(history).to_csv(out/'history.csv',index=False)
        score=scores['macro_f1_present']
        print(f"epoch {epoch:03d}/{args.epochs} loss={row['train_loss']:.4f} val_macroF1={score:.4f} {row['seconds']:.1f}s",flush=True)
        if score>best:
            best=score;stale=0
            torch.save(dict(state_dict=model.state_dict(),labels=d['labels'],dataset_sha256=d['dataset_sha256'],
                            split_sha256=d['split_sha256'],epoch=epoch),out/'best.pt')
        else:stale+=1
        scheduler.step()
        if stale>=args.patience:break
    metadata.update(status='finished',best_validation_macro_f1=best,epochs_completed=len(history),runtime_seconds=time.perf_counter()-start)
    save_json(out/'metadata.json',metadata)
    print('仅训练/验证完成，测试尚未评价。',out,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--device',choices=['auto','cuda','cpu'],default='auto')
    p.add_argument('--epochs',type=int,default=80)
    p.add_argument('--batch-size',type=int,default=256)
    p.add_argument('--learning-rate',type=float,default=.001)
    p.add_argument('--patience',type=int,default=12)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--cpu-threads',type=int,default=4)
    p.add_argument('--augment',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--max-train-pixels',type=int,default=0)
    p.add_argument('--probe',action='store_true',help='标记为测时检查，不纳入最终测试模型')
    p.add_argument('--run-name')
    p.add_argument('--overwrite',action='store_true')
    main(p.parse_args())
