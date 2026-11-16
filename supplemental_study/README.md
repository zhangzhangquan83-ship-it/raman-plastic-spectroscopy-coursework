# 塑料拉曼分类补充实验说明

本目录保存2026年11月16日按用户要求新增的两项工作：**首阶段补训7个模型，使原15类任务覆盖六类方法；以及对第二阶段36个已冻结CV拟合做旧六图补充评价，不新增第二阶段训练。** 原首轮与原扩展CV的协议、权重、预测和分数保持原样。

两项工作都发生在原六张测试图的表现已经被查看之后。新增配置在各自训练或推理之前冻结，仍属于同一固定测试集的后续使用，不能称为从未接触过的新盲测或新外部验证。这里的“六类方法”指算法家族，与“15类/12类材料标签”不是同一概念。

## 两份计划和实际完成内容

| 任务 | 冻结计划 | 本次执行 | 评价总体 |
| --- | --- | --- | --- |
| 首阶段补齐方法 | [stage1_plan.json](protocol/stage1_plan.json) | RF 1次、MLP 3种子、ResNet 3种子，共7次新拟合；全部训练完成后统一测试 | 原22/6/6图划分；测试6图、6,399条、15类 |
| 第二阶段冻结模型补测 | [stage2_holdout_plan.json](protocol/stage2_holdout_plan.json) | 27个既有神经模型＋9个既有传统模型的36次推理；不拟合、不挑最好项、不集成 | 同一旧6图中的5,425条、12类；排除974条 |

首阶段MLP/ResNet与历史CNN对齐80轮上限、patience12、batch256、lr0.001、AdamW权重衰减0.0001、梯度裁剪5、余弦退火、训练类别计数加权CE和增强定义，种子42/43/44。最佳epoch只按原6张验证图中13个有支持类别的宏F1选择；PU、Sil在验证中无支持，且选好权重后不合并训练和验证图重拟合。RF固定300树、最小叶节点2、balanced_subsample、random_state101，不按测试分数搜参数。

第二阶段复用原12类核心配置：CNN/MLP/ResNet各3原外折×3种子101/202/303，3种传统方法各3原外折。这里的原外折编号标识权重来自哪组开发训练/验证划分，**不是把旧6图重新分折**。这些神经模型只拟合了原内训练图，没有全28图重拟合；所有36个模型均独立预测相同5,425条谱。

## 首阶段六方法结果

| 方法 | 运行数 | 15类测试像素准确率 % | 15类宏F1 | 来源 |
| --- | ---: | ---: | ---: | --- |
| 余弦类中心 | 1 | 64.87 | 0.4471 | 历史首阶段 |
| PCA64加线性SVM | 1 | 72.53 | 0.5873 | 历史首阶段 |
| 随机森林300树 | 1 | 68.92 | 0.5632 | 此次追加 |
| 增强1D-CNN | 3 | 70.50 ± 0.58 | 0.5725 ± 0.0181 | 历史首阶段 |
| 增强MLP | 3 | 71.30 ± 1.22 | 0.5866 ± 0.0080 | 此次追加 |
| 增强1D-ResNet | 3 | 71.35 ± 0.87 | 0.5913 ± 0.0079 | 此次追加 |


数据：[six_method_summary.csv](results/stage1/summary/six_method_summary.csv)。神经方法的±为同一测试集上3训练种子的样本SD；准确率±的单位为百分点，单运行传统方法不编造SD。无增强CNN seed42保留在13条完整明细中，不混入CNN主表均值。13条记录由历史6条＋新增7条组成，并不表示13种算法。

ResNet的本表平均宏F1为0.5913，SVM为0.5873；这个差距很小，且新增架构属于后验追加，不能据此宣布显著或普遍优势。历史训练使用RTX3090，新增训练与补测使用RTX3050 Ti Laptop GPU，训练用时不应用来直接做跨硬件架构速度排名。完整解释见[首阶段补充报告](results/stage1/summary/stage1_supplement_report.md)。

## 第二阶段旧六图补测结果

| 方法 | 既有拟合数 | 像素准确率 % | 12类宏F1 | 图平均准确率 % | 图等权宏F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 余弦类中心 | 3 | 68.70 ± 1.67 | 0.5096 ± 0.0298 | 82.24 ± 0.83 | 0.5666 ± 0.0260 |
| PCA64＋线性SVM | 3 | 77.82 ± 0.66 | 0.6620 ± 0.0134 | 87.85 ± 0.51 | 0.7127 ± 0.0077 |
| 随机森林300树 | 3 | 74.87 ± 1.79 | 0.6369 ± 0.0144 | 85.83 ± 0.99 | 0.6738 ± 0.0039 |
| 增强1D-CNN | 9 | 77.93 ± 1.44 | 0.6691 ± 0.0102 | 87.63 ± 0.60 | 0.7070 ± 0.0071 |
| 增强MLP | 9 | 79.21 ± 1.62 | 0.6837 ± 0.0201 | 88.15 ± 0.93 | 0.7177 ± 0.0148 |
| 增强1D-ResNet | 9 | 77.00 ± 1.81 | 0.6481 ± 0.0190 | 87.12 ± 0.87 | 0.6841 ± 0.0150 |

数据：[condition_summary.csv](results/stage2_frozen_holdout/condition_summary.csv)。本表所有±来自同一holdout上的拟合差异：神经每原外折先平均3种子，传统每原外折1次，再对3组拟合计算均值和样本SD；不是独立测试样本的置信区间，也不是OOF。

原6,399条中按已冻结12类规则排除PC578、PU78、Sil318，共974条，保留5,425条。`Map_PC_2`与`Map_Silicone`只剩1,077条blank；4张图仍有塑料支持，且每一保留塑料类只在1张holdout图中有真值。MLP全体像素准确率79.21%，仅塑料像素准确率57.34%，需要同时报告这两个不同总体。四scope、逐图支持和完整解释见[第二阶段补测报告](results/stage2_frozen_holdout/stage2_supplement_report.md)。

原开发CV、第一阶段15类测试和本次12类补测的评价总体、训练模型与统计口径不同，分数不能混为一个排行榜或直接相减宣称性能提升。按采集图隔离也不等于按独立物理标本隔离。

## 先查看结果或仅复核 已完成任务无需重训

所有以下路径均从项目根目录执行。完整输入齐全时，以下命令不训练模型：

```bash
# 从保存CSV汇总首阶段六方法；未执行模型前向
python supplemental_study/src/summarize_stage1_supplement.py --root .
# 核对冻结计划、源输入及已有第二阶段补测输出；不重新推理
python supplemental_study/src/evaluate_stage2_frozen_holdout.py --root . --verify-only
# 独立核对两项补充结果；仅将审计JSON写到supplemental_study/qa
python supplemental_study/qa/audit_supplement_results.py --root . --stage all
```

独立核查[两阶段审计JSON](qa/supplement_results_audit.json)为passed：首阶段核对历史6条与新增7条；第二阶段核对36模型及30份概率，最大指标重算误差分别小于1×10⁻¹⁵。独立核对不是独立重训，误差接近浮点精度也不表示分类错误接近零。

## 从头重做补充实验时的执行顺序

使用**包含本次补充资料的新版完整复现包**或相同输入快照。新建独立工作副本，保留原主数据、原首轮结果、原扩展CV全部模型/结果/协议及本目录源码和两份冻结计划；不复制该副本中的`supplemental_study/results/stage1/`和`supplemental_study/results/stage2_frozen_holdout/`，让新增输出目录从空开始。不要删除或覆盖原交付结果、修改冻结计划来绕过哈希检查。

```bash
# 1. 仅训练首阶段7个预定新增模型，默认就是完整RF/MLP/ResNet及约定种子
python supplemental_study/src/run_stage1_supplement.py --root . --phase train --device cuda --cpu-threads 8
# 2. 七个训练全部完成后才统一评价；不得按结果只选择部分模型或种子
python supplemental_study/src/run_stage1_supplement.py --root . --phase evaluate --device cuda --cpu-threads 8 --prediction-batch-size 128
# 3. 将历史6条与新增7条整理为六方法主表和13运行明细；只读预测
python supplemental_study/src/summarize_stage1_supplement.py --root .
# 4. 核验已有第二阶段补测计划；若已存在，只验证完全一致，不覆盖计划
python supplemental_study/src/evaluate_stage2_frozen_holdout.py --root . --freeze-plan
# 5. 只对36个既有CV模型进行补测，没有第二阶段训练或全开发集重拟合
python supplemental_study/src/evaluate_stage2_frozen_holdout.py --root . --device cuda --cpu-threads 8
# 6. 核对全部结果，检查输出为passed后再引用数值
python supplemental_study/qa/audit_supplement_results.py --root . --stage all
```

已完成首阶段run默认拒绝覆盖；明确使用`--skip-completed`只会验证并复用合格完成项，未实现逐epoch优化器断点续训。第二阶段已有完成回执时只校验并返回，不会偷偷重算；存在未完成目录则拒绝覆盖。两份计划都必须与实际数据、源码和源权重一致，不能看了分数再替换某个fold、seed或参数。

第二阶段计划针对的是**这批精确的既有CV权重及原始产物哈希**。若从零重训原CV，浮点运算、环境或序列化可能改变文件字节，原补测计划不会继续匹配。那应作为另一个明确区分的重训复现实验另立记录，不能改动当前计划硬称是同一份冻结权重补测。

## 环境和缺失二进制的条件

实际补充执行使用Python3.12、PyTorch2.8.0+cu128、scikit-learn1.9.1以及NVIDIA GeForce RTX3050 Ti Laptop GPU；版本与设备以各运行metadata/回执为准。可用兼容GPU运行，CPU也可推理但耗时不同；同一随机种子不保证不同GPU、不同架构逐位相同或增强实现完全相同。优先沿用已经兼容的GPU版PyTorch，安装根目录`requirements.txt`时不要覆盖其CUDA构建。

```bash
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
```

轻量提交包或只包含可浏览文本的仓库通常省略NPZ、PT、joblib及概率矩阵。它们可以查看本报告和保存输出，**不能直接执行上述数据/权重复核命令**。执行需要两套处理NPZ、原首轮模型与结果、原扩展完整目录（75个神经拟合与9个传统基线及其结果、协议和源码），以及新增7个模型和对应预测。第二阶段实际只对其中36个模型推理，但冻结来源快照会核对原扩展全目录，不能仅拣出36个权重；请取得本次新版完整复现包。只下载原CSV并重建数据不会恢复模型权重。

27个神经模型保存softmax概率，3个RF保存predict_proba；余弦/LinearSVC共6个拟合没有校准概率，metadata明确标记不可用，不将decision score充作概率。原CV分数保留在`original_cv_reference.csv`，不与这批同一holdout预测混成OOF。

## 文件导航与交付覆盖

- `protocol/stage1_plan.json`、`protocol/stage2_holdout_plan.json`：两份在新训练/推理前冻结的计划与输入身份。
- `results/stage1/runs/`：新增7模型的配置、训练日志、权重/传统模型；`results/stage1/final_test/`保存统一评价预测与回执。
- `results/stage1/summary/`：六方法表、13运行明细及首阶段补充报告。
- `results/stage2_frozen_holdout/runs/`：36个来源run对应的本次预测、指标、逐类/逐图和可用概率。
- `results/stage2_frozen_holdout/map_support.csv`、`scope_summary.csv`：两张blank-only图、四张塑料支持图以及不同像素总体的分层诊断。
- `results/stage2_frozen_holdout/condition_fold_means.csv`、`condition_summary.csv`：先种子后拟合折的汇总，样本SD不冒充CI。
- `results/stage2_frozen_holdout/original_cv_reference.csv`：原开发CV结果的独立引用，含源路径/SHA；原CSV仍保留在`extension_study/results/`。
- `results/stage2_frozen_holdout/execution.json`：36模型推理回执与当时列出的产物哈希；原始协议/结果前后未变。
- `qa/supplement_results_audit.json`：独立NumPy混淆矩阵与概率核查，不执行训练或模型前向。
- `results/figures/`：首阶段和第二阶段补充比较PNG/PDF/SVG，保留相应图注和统计口径。

本README及`stage2_supplement_report.md`是在模型推理与独立审计完成后编写的解释材料。**没有修改任何冻结runner/evaluator、原阶段结果或回执；新文档不在原`execution.output_files`的历史清单中，不声称被该清单覆盖**。后续打包由交付清单单独记录其SHA256。原指标、原CV引用和冻结来源保持只读。

数据来源、署名、DOI、CC BY4.0及派生处理声明见[数据来源与研究边界](../数据来源与研究边界.md)。成员姓名、学号、实际操作和复核贡献仍需真实填写，代码与材料使用AI辅助，不代填虚构贡献。
