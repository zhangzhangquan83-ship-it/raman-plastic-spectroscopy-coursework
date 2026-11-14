"""预先指定的跨map划分，不查看模型结果挑选测试图。"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from common import ROOT, save_json

TEST_MAPS = {
    'Map_PC_2', 'Map_Nylon_PEs_PS', 'Map_PMMA_PET_PE_ABS',
    'Map_Env_PP_PTFE', 'Map_PVC_PU_POM', 'Map_Silicone',
}
VALIDATION_MAPS = {
    'Map_PC_1', 'Map_PMMA_PET', 'Map_Nylon_PTFE_POM',
    'Map_PS_PE_PEs_2', 'Map_PVC_ABS_PEs', 'Map_PP',
}


def main(root):
    root=Path(root)
    with np.load(root/'data/processed/dataset.npz',allow_pickle=False) as d:
        X,y,groups=d['X'],d['y'].astype(str),d['groups'].astype(str)
    all_maps=set(groups)
    if not (TEST_MAPS | VALIDATION_MAPS) <= all_maps:
        raise ValueError('当前快照缺少协议指定的map。')
    rows=[]
    for g in sorted(all_maps):
        s='test' if g in TEST_MAPS else 'validation' if g in VALIDATION_MAPS else 'train'
        rows.append(dict(map_id=g,split=s,pixels=int(np.sum(groups==g)),classes=int(len(np.unique(y[groups==g])))))
    table=pd.DataFrame(rows)
    table.to_csv(root/'data/processed/map_split.csv',index=False,encoding='utf-8-sig',lineterminator='\n')
    lookup=dict(zip(table.map_id,table.split))
    membership=np.array([lookup[g] for g in groups])
    coverage=pd.crosstab(pd.Series(y,name='label'),pd.Series(membership,name='split'))
    coverage.to_csv(root/'data/processed/class_split_coverage.csv',encoding='utf-8-sig',lineterminator='\n')
    missing={s:coverage.index[coverage[s]==0].tolist() for s in ['train','validation','test']}
    if missing['train'] or missing['test']:
        raise ValueError(f'预定训练/测试划分未覆盖全部类别，不能静默更改协议：{missing}')
    # 任何跨map完全相同的输入均会影响独立评价，检查而不是用模型结果修复。
    import hashlib
    seen={}
    for i,row in enumerate(X):
        key=hashlib.sha256(np.ascontiguousarray(row).tobytes()).digest()
        if key in seen and membership[i]!=membership[seen[key]]:
            raise ValueError('同一处理后光谱跨越数据集，请审查重复记录。')
        seen[key]=i
    summary=dict(strategy='预先固定22个训练map、6个验证map、6个测试map；不按模型分数选择划分。',
                 missing_classes=missing,map_counts=table.split.value_counts().to_dict(),
                 pixel_counts={s:int(np.sum(membership==s)) for s in ['train','validation','test']},
                 independent_unit='采集map；未知物理样品ID，不能声称不同map具有独立物理标本。',
                 metric='像素macro-F1（该集合有真值的类别）及map等权平均准确率；同时报告类别覆盖。')
    save_json(root/'data/processed/split_protocol.json',summary)
    print(table.groupby('split').agg(maps=('map_id','size'),pixels=('pixels','sum')).to_string())
    print('未覆盖类别：',missing)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT)
    main(p.parse_args().root)
