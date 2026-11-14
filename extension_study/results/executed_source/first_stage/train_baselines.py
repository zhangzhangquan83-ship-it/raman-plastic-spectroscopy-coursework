"""仅用训练及验证map开发基线，不使用测试特征拟合或报告测试分数。"""
import argparse
from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.base import BaseEstimator, ClassifierMixin
from common import ROOT, load_data, metrics, save_json, CosineCentroid


def main(root):
    root=Path(root)
    d=load_data(root)
    a,b=d['idx']['train'],d['idx']['validation']
    out=root/'results/baselines'
    out.mkdir(parents=True,exist_ok=True)
    models={'cosine_centroid':Pipeline([('norm',Normalizer()),('model',CosineCentroid())])}
    rows=[]
    start=time.perf_counter()
    models['cosine_centroid'].fit(d['X'][a],d['y'][a])
    rows.append(dict(model='cosine_centroid',**metrics(d['y'][b],models['cosine_centroid'].predict(d['X'][b]),d['groups'][b])))
    # PCA与标准化只在训练map拟合；维度预先设为64。
    pca=PCA(n_components=64,svd_solver='randomized',random_state=42)
    za=pca.fit_transform(d['X'][a])
    zb=pca.transform(d['X'][b])
    scale=StandardScaler().fit(za)
    za,zb=scale.transform(za),scale.transform(zb)
    best_score=-np.inf
    best=None
    for C in [0.1,1.,10.]:
        clf=LinearSVC(C=C,class_weight='balanced',dual='auto',max_iter=10000,random_state=42).fit(za,d['y'][a])
        scores=metrics(d['y'][b],clf.predict(zb),d['groups'][b])
        rows.append(dict(model=f'pca64_linear_svm_C{C}',C=C,**scores))
        if scores['macro_f1_present']>best_score:
            best_score=scores['macro_f1_present']
            best=clf
    models['pca64_linear_svm']=Pipeline([('pca',pca),('scale',scale),('model',best)])
    for name,model in models.items():
        joblib.dump(model,out/(name+'.joblib'))
    pd.DataFrame(rows).to_csv(out/'validation_comparison.csv',index=False,encoding='utf-8-sig')
    save_json(out/'metadata.json',dict(dataset_sha256=d['dataset_sha256'],split_sha256=d['split_sha256'],
             labels=d['labels'],test_evaluated=False,runtime_seconds=time.perf_counter()-start,
             selected_C=best.C,selection_metric='validation macro_f1_present（13个有支持类别）'))
    print(pd.DataFrame(rows).round(4).to_string(index=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT)
    main(p.parse_args().root)
