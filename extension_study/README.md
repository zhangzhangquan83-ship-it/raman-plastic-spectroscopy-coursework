# 开发图上的共同类别扩展研究

这部分回答网络结构、增强、类别加权、预处理及训练图数量会怎样影响跨图识别。它使用原28幅开发图，原6幅最终测试图完全不进入这部分训练或评价。它不是对已经看过的原测试集继续调参。

## 数据与评价口径

- 根据开发图标签覆盖预先保留至少出现在3幅图的类别，共12类含blank、25,541条像素光谱；排除PC、PU和Sil。
- 三外折的内训练／内验证／外评价图数分别为13／5／10、14／5／9、14／5／9。每个角色都覆盖全部12类。
- 训练仅拟合内训练；内验证选择epoch或SVM的C；外评价仅在选择完成后计算。没有选择后重拟合外训练全集。
- 相邻像素不是独立样品。主表先在每折内平均三个训练种子，再报告三外折的均值和样本标准差。该标准差是描述性折间波动，不是置信区间。
- 外评价三折互斥，合计覆盖共同类别总体一次。不能把这部分12类分数与第一轮15类6图分数直接比较。

## 冻结实验

| 问题 | 配置 | 神经训练次数 |
| --- | --- | ---: |
| 网络结构 | CNN、MLP、1D-ResNet，各3折×3种子 | 27 |
| 光谱增强 | CNN关闭增强，各3折×3种子；参照复用核心CNN | 9 |
| 类别不平衡 | CNN不加权交叉熵，各3折×3种子 | 9 |
| 预处理 | 去SG、去基线、L2、二阶基线，各3折×seed101 | 12 |
| 训练图数量 | 约60%与80%内训练图，各3折×3种子；100%复用核心CNN | 18 |
| 合计 | 不重复计算已复用配置 | 75 |

另外每折比较余弦类中心、PCA64＋线性SVM、300棵随机森林，共9组基线评价。SVM的C候选为0.1、1、10，只根据内验证选择。学习曲线子图严格嵌套并保持全12类；实际图比例见protocol/learning_curve_counts.csv。

神经训练统一AdamW，学习率0.001、weight decay 0.0001、batch256、最多120轮、早停耐心18轮；验证指标为macro-F1。加权损失的类别权重仅从内训练计数计算。训练种子101、202、303。所有配置逐项保存在protocol/experiment_plan.json。

增强包含幅度0.9–1.1、标准差0.02的加性噪声和±2点非循环平移。预处理研究全部关闭增强，以同折无增强CNN seed101为参照，避免L2尺度改变造成相对噪声强度混杂。该修订发生在首次拟合前，见protocol/plan_revision.md。

核心27次神经训练另外在固定噪声标准差0.02、0.05、0.1、0.2下推理，各模型同折使用同一噪声实现；噪声加在已SNV的谱上。这是人工扰动压力测试，不代表仪器噪声模型。校准诊断报告10等宽分箱ECE、NLL和多类Brier分数，没有用外评价结果进行温度缩放。

## 预处理表示

默认矩阵精确复用主项目冻结X。四种变体从28幅开发图源CSV生成，均使用固定共同频轴与1024点插值。

| 键 | 表示 |
| --- | --- |
| X_default | SG11/3→端点线性基线→SNV |
| X_no_sg | 不做SG，端点线性基线→SNV |
| X_no_baseline | SG11/3→SNV |
| X_l2 | SG11/3→端点线性基线→L2，不另中心化 |
| X_quadratic | SG11/3→全点二阶最小二乘基线→SNV |

每条谱独立处理，没有拟合跨谱归一化。original_indices保留主NPZ全局行号，pixel_index保留源CSV零基行号。

## 复现

在项目根目录、已有兼容CUDA版PyTorch的环境中执行。requirements.txt不重新安装torch。
以下命令用于复核已有完整结果包，其中已经含云执行回执。若在不含results的新目录从零训练，先按主README运行run_frozen_experiment.py完成首轮，再运行extension_study/src/run_cloud_study.py；后者会生成独立审计所需的cloud_extension_execution.json。

```bash
python extension_study/src/split_study.py --verify-only
python extension_study/src/build_plan.py
# 包含study_data.npz时核验复用；缺失时去掉--reuse以获取28图并重建
python extension_study/src/prepare_study_data.py --reuse
python extension_study/src/run_study.py --device cuda
python extension_study/src/audit_study_results.py
python extension_study/src/summarize_study.py
```

run_study默认跳过同数据、同协议、同计划的已完成任务；中断任务从epoch1重新训练，不恢复优化器或随机数状态。--only-run接受计划中的单一run_id，子集完成不会标记整项研究完成。4小时软限制在任务间检查，不会自动关闭实例。

results/runs保存全部权重、逐轮日志、外折预测、概率、逐图指标与元数据；results/cloud_extension_execution.json记录云端执行与原始冻结文件的前后SHA；results/independent_audit.json独立核对逐谱预测、指标、概率、早停及checkpoint。

## 解释边界

这是一组预先固定对照的描述性研究，仅3个外折，图之间仍可能共享材料来源。预处理每折只用一个训练种子，结论须保留这个限制。不能将外折比较后较高的配置宣称为在原6图上已经取得改进；如果据此选择未来模型，仍需全新的独立测试数据。
学习曲线每折只使用一条预先固定的嵌套子集序列，三个训练种子不代表三种图子集；图数量与新增图的材料组合、采集难度仍有混杂，不能据此保证任意追加同样数量的图都会得到相同收益。
