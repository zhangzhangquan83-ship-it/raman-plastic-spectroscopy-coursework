"""正式计费训练前验证数据、划分、前向/反向传播和安全权重读写。"""
import argparse
from pathlib import Path
import importlib.metadata
import time
import torch
import numpy as np
from common import ROOT,load_data,save_json
from models import SpectralCNN


def main(args):
    start=time.perf_counter()
    if args.device=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA不可用。请先更换为支持GPU的PyTorch镜像，避免在CPU上误跑正式实验。')
    device=torch.device('cuda' if args.device=='auto' and torch.cuda.is_available() else 'cpu' if args.device=='auto' else args.device)
    torch.set_num_threads(4)
    d=load_data(args.root)
    assert len(d['groups'])==33119 and len(np.unique(d['groups']))==34 and len(d['labels'])==15
    assert all(type(s) is str for s in d['labels'])
    a=d['idx']['train'][:32]
    x=torch.from_numpy(d['X'][a,None,:]).to(device)
    y=torch.from_numpy(d['y'][a]).to(device)
    model=SpectralCNN(len(d['labels'])).to(device)
    loss=torch.nn.CrossEntropyLoss()(model(x),y)
    loss.backward()
    assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    out=Path(args.root)/'results/sanity_check';out.mkdir(parents=True,exist_ok=True)
    model.eval()
    with torch.inference_mode():expected=model(x).detach().cpu()
    path=out/'roundtrip.pt'
    torch.save({'state_dict':model.state_dict(),'labels':d['labels']},path)
    checkpoint=torch.load(path,map_location=device,weights_only=True)
    restored=SpectralCNN(len(d['labels'])).to(device)
    restored.load_state_dict(checkpoint['state_dict']);restored.eval()
    with torch.inference_mode():actual=restored(x).detach().cpu()
    assert torch.allclose(expected,actual,rtol=1e-5,atol=1e-6)
    record=dict(status='passed',device=str(device),gpu=torch.cuda.get_device_name(0) if device.type=='cuda' else None,
                cuda_available=torch.cuda.is_available(),torch_cuda_build=torch.version.cuda,
                versions={p:importlib.metadata.version(p) for p in ['torch','numpy','pandas','scipy','scikit-learn']},
                samples=33119,maps=34,classes=15,parameters=sum(p.numel() for p in model.parameters()),
                forward_backward_finite=True,weights_only_roundtrip=True,test_evaluated=False,
                runtime_seconds=time.perf_counter()-start)
    save_json(out/'verification.json',record)
    print(record)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    main(p.parse_args())
