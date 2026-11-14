"""扩大版公共工具：以map划分，所有预处理参数逐谱计算。"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import Normalizer

ROOT = Path(__file__).resolve().parents[1]
SHORT={'blank':'Blank','acrylonitrile butadiene styrene':'ABS','nylon':'PA',
       'poly (methyl methacrylate)':'PMMA','poly(methyl methacrylate)':'PMMA',
       'polycarbonate':'PC','polyester':'PEs','polyethylene':'PE',
       'polyethylene terephthalate':'PET','polyoxymethylene':'POM',
       'polypropylene':'PP','polystyrene':'PS','polytetrafluoroethylene':'PTFE',
       'polyurethane':'PU','polyvinyl chloride':'PVC','silicone':'Sil'}


class CosineCentroid(ClassifierMixin,BaseEstimator):
    def fit(self,X,y):
        self.classes_=np.unique(y)
        self.centers_=Normalizer().fit_transform(np.stack([X[y==c].mean(axis=0) for c in self.classes_]))
        return self
    def predict(self,X):
        return self.classes_[(X@self.centers_.T).argmax(axis=1)]


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while block := f.read(1024 * 1024):
            h.update(block)
    return h.hexdigest()


def load_data(root=ROOT):
    root = Path(root)
    path = root/'data/processed/dataset.npz'
    with np.load(path, allow_pickle=False) as d:
        X = d['X'].astype(np.float32)
        text_y = d['y'].astype(str)
        groups = d['groups'].astype(str)
        wn = d['wavenumbers']
        pixel = d['pixel_index']
    labels = sorted(np.unique(text_y).tolist())
    y = np.array([labels.index(s) for s in text_y], dtype=np.int64)
    assert np.isfinite(X).all() and X.shape[1] == 1024
    split = pd.read_csv(root/'data/processed/map_split.csv')
    if not split.map_id.is_unique or set(split.map_id) != set(groups):
        raise ValueError('map划分与数据不一致。')
    assignment = dict(zip(split.map_id, split.split))
    membership = np.array([assignment[g] for g in groups])
    idx = {s: np.flatnonzero(membership == s) for s in ['train', 'validation', 'test']}
    for a,b in [('train','test'),('train','validation'),('validation','test')]:
        assert set(groups[idx[a]]).isdisjoint(groups[idx[b]])
    if set(y[idx['train']]) != set(range(len(labels))):
        raise ValueError('训练集未覆盖全部类别，请检查预先定义的划分。')
    return dict(X=X, y=y, text_y=text_y, groups=groups, wavenumbers=wn,
                pixel_index=pixel, labels=labels, idx=idx,
                dataset_sha256=file_hash(path), split_sha256=file_hash(root/'data/processed/map_split.csv'))


def metrics(y, pred, groups):
    """macro-F1只平均该评价集合实际有真值的类别；总是同时报告覆盖。"""
    present = np.unique(y)
    maps = np.unique(groups)
    map_acc = [float(np.mean(pred[groups == g] == y[groups == g])) for g in maps]
    _,inverse,counts=np.unique(groups,return_inverse=True,return_counts=True)
    weights=1.0/counts[inverse]
    return dict(accuracy=float(accuracy_score(y, pred)),
                balanced_accuracy=float(recall_score(y,pred,labels=present,average='macro',zero_division=0)),
                macro_f1_present=float(f1_score(y, pred, labels=present, average='macro', zero_division=0)),
                map_mean_accuracy=float(np.mean(map_acc)),
                map_equal_weight_macro_f1=float(f1_score(y,pred,labels=present,average='macro',sample_weight=weights,zero_division=0)),
                evaluated_classes=len(present), pixels=len(y), maps=len(maps))
