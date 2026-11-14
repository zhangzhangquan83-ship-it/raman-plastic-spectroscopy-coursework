"""预注册开发研究的75次神经网络训练及9组基线评价。"""
from pathlib import Path
import hashlib,json

ROOT=Path(__file__).resolve().parents[2]


def main():
    protocol=ROOT/'extension_study/protocol'
    splits=json.loads((protocol/'splits.json').read_text(encoding='utf-8'))
    configs=[]
    for fold in splits['folds']:
        f=int(fold['fold_id'])
        base=dict(fold_id=f,variant='default',map_level='maps100',augment=True,weighted_loss=True,
                  epochs=120,patience=18,batch_size=256,learning_rate=.001,weight_decay=.0001)
        for model in ['cnn','mlp','resnet']:
            for seed in [101,202,303]:
                configs.append(dict(base,run_id=f'core_{model}_f{f}_s{seed}',model=model,seed=seed,family='architecture'))
        for setting,changes in [('noaug',{'augment':False}),('unweighted',{'weighted_loss':False})]:
            for seed in [101,202,303]:
                configs.append(dict(base,**changes,run_id=f'{setting}_cnn_f{f}_s{seed}',model='cnn',seed=seed,family=setting))
        for variant in ['no_sg','no_baseline','l2','quadratic']:
            configs.append(dict(base,variant=variant,augment=False,run_id=f'prep_{variant}_f{f}_s101',model='cnn',seed=101,family='preprocessing'))
        for level in ['maps60','maps80']:
            for seed in [101,202,303]:
                configs.append(dict(base,map_level=level,run_id=f'curve_{level}_f{f}_s{seed}',model='cnn',seed=seed,family='learning_curve'))
    if len(configs)!=75:raise ValueError(f'预期75个神经训练配置，实际{len(configs)}')
    value=dict(study='开发图上的共同类别嵌套交叉验证，不重新评价原6张测试图。',
               task_classes=splits['labels'],outer_folds=3,training_seeds=[101,202,303],
               configs=configs,baselines=['cosine_centroid','pca64_linear_svm','random_forest_300'],
               baseline_C_candidates=[.1,1.,10.],noise_levels=[.02,.05,.1,.2],
               preprocessing_reference='每折noaug_cnn_f{fold}_s101；五种预处理均关闭增强，避免L2尺度与固定绝对噪声的混杂。',
               selection='只以inner_val macro-F1选择epoch或SVM C；outer_eval仅评价。',
               aggregation='每外折先平均3个训练种子，再报告3外折均值及折间标准差；不把9次当9个独立测试。',
               max_wall_hours=4,original_test_reuse=False,
               split_sha256=hashlib.sha256((protocol/'splits.json').read_bytes()).hexdigest())
    path=protocol/'experiment_plan.json'
    if path.exists() and json.loads(path.read_text(encoding='utf-8'))!=value:
        raise FileExistsError('计划已存在且不同，不能静默修改已冻结配置。')
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    (protocol/'experiment_plan.sha256').write_text(hashlib.sha256(path.read_bytes()).hexdigest()+'\n',encoding='utf-8')
    print('冻结神经训练配置：',len(configs),'；基线评价：',3*3)


if __name__=='__main__':main()
