"""读取已完成的实际实验，生成最终图表及Markdown结果说明。"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import ROOT,SHORT


def main(root):
    root=Path(root)
    out=root/'results/final_test'
    path=out/'model_comparison.csv'
    if not path.exists():raise FileNotFoundError('正式训练及独立测试尚未完成，请先执行训练和evaluate.py；不生成虚构结果。')
    comparison=pd.read_csv(path)
    figures=out/'figures';figures.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'figure.dpi':150,
                        'axes.spines.top':False,'axes.spines.right':False})
    display_names={'cosine_centroid':'Cosine centroid','pca64_linear_svm':'PCA64 + linear SVM',
                   'cnn_aug_seed42':'CNN aug seed42','cnn_aug_seed43':'CNN aug seed43',
                   'cnn_aug_seed44':'CNN aug seed44','cnn_noaug_seed42':'CNN no augmentation'}
    fig,ax=plt.subplots(figsize=(10,5),layout='constrained')
    x=np.arange(len(comparison));ax.barh(x,comparison.macro_f1_present,color='#446A91')
    ax.set(yticks=x,yticklabels=[display_names.get(s,s) for s in comparison.model],xlim=(0,1.06),
           xlabel='Pixel macro-F1 (15 classes)',title='Independent test on six held-out acquisition maps')
    for i,v in enumerate(comparison.macro_f1_present):ax.text(v+.01,i,f'{v:.3f}',va='center')
    fig.savefig(figures/'01_model_comparison.png',bbox_inches='tight');plt.close(fig)
    # 主展示预先指定seed42，不按测试最高者选择图。
    name='cnn_aug_seed42'
    cm=pd.read_csv(out/(name+'_confusion.csv'),index_col=0)
    fig,ax=plt.subplots(figsize=(9,8),layout='constrained')
    ax.imshow(cm,cmap='Blues')
    names=[SHORT.get(s,s) for s in cm.index]
    ax.set(xticks=np.arange(len(cm)),yticks=np.arange(len(cm)),xticklabels=names,yticklabels=names,
           xlabel='Predicted class',ylabel='True class',title='CNN seed42 confusion matrix (pixel counts)')
    plt.setp(ax.get_xticklabels(),rotation=45,ha='right')
    for i in range(len(cm)):
        for j in range(len(cm)):
            if cm.iloc[i,j]:ax.text(j,i,str(cm.iloc[i,j]),ha='center',va='center',fontsize=8,
                                   color='white' if cm.iloc[i,j]>.5*cm.to_numpy().max() else 'black')
    fig.savefig(figures/'02_confusion_matrix.png',bbox_inches='tight');plt.close(fig)
    history=pd.read_csv(root/'results/cnn_aug_seed42/history.csv')
    fig,ax=plt.subplots(figsize=(9,4.6),layout='constrained')
    ax.plot(history.epoch,history.macro_f1_present,color='#446A91',marker='o',markersize=3)
    ax.set(xlabel='Epoch',ylabel='Validation macro-F1 (13 supported classes)',ylim=(0,1),title='Validation-only early stopping')
    fig.savefig(figures/'03_training_curve.png',bbox_inches='tight');plt.close(fig)
    maps=pd.read_csv(out/'per_map_metrics.csv')
    focal=maps[maps.model=='cnn_aug_seed42']
    fig,ax=plt.subplots(figsize=(11,5),layout='constrained')
    ax.barh(focal.map_id,focal.accuracy,color='#446A91')
    ax.set(xlabel='Accuracy within map',xlim=(0,1.05),title='CNN seed42: variation across six held-out maps')
    fig.savefig(figures/'04_per_map_accuracy.png',bbox_inches='tight');plt.close(fig)
    lines=['# 大规模拉曼成像塑料识别实验结果','',
           '以下数字由已完成的训练和独立测试文件生成，没有预填性能。所有指标对应跨采集map协议，不代表独立物理标本或全部环境塑料。','',
           '| 模型 | 像素准确率 | 15类宏平均F1 | 图等权准确率 | 图等权宏平均F1 |',
           '| --- | ---: | ---: | ---: | ---: |']
    for row in comparison.itertuples():
        lines.append(f'| {row.model} | {row.accuracy:.4f} | {row.macro_f1_present:.4f} | {row.map_mean_accuracy:.4f} | {row.map_equal_weight_macro_f1:.4f} |')
    lines+=['','## 必须同时说明的适用范围','',
            '- 数据约3.3万像素谱，独立采集map只有34张，最终测试map只有6张。',
            '- 验证集中PU与Sil无支持，仅用其余13类选择训练轮次；两类各只有一个测试map，不能认为有充分验证。',
            '- 不同map可能使用相同物理材料；未提供逐粒子ID，不能保证物理样品层面的完全独立。',
            '- 参考库中的高质量光谱分类分数不能与本任务的噪声成像谱分数直接比较。',
            '- 最终图表固定展示seed42，不根据测试分数挑选最高的随机种子。三种子结果应报告均值和标准差。',
            '- CSV没有可靠二维形状，像素行号只能用于定位；本项目不把一维顺序伪装成二维粒子分割图。','']
    (root/'扩大版实验结果.md').write_text('\n'.join(lines),encoding='utf-8')
    print('实际结果与4张最终图表已生成。')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT)
    main(p.parse_args().root)
