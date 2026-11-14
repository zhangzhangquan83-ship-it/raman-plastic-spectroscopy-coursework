# 拉曼塑料光谱识别与跨图评价

《机器学习在物质科学中的应用》课程作业完成版：RaMPI公开拉曼成像谱，**34图、33,119条像素光谱**，已完成RTX3090真实计算、独立指标核查及展示材料。

**所有课程提交材料均已提供。** 仓库中可浏览展示、代码、协议、图表、全部预测CSV、日志、指标、数据获取说明及分工引用。处理数据、全部权重与概率矩阵另外保存在[Release完整包](https://github.com/zhangzhangquan83-ship-it/raman-plastic-spectroscopy-coursework/releases/download/v1.0.0/raman-coursework-full.zip)中。

## 下载与阅读

| 用途 | 入口 | 包含内容 |
| --- | --- | --- |
| 课堂展示 | [已执行Notebook](%E6%89%A9%E5%A4%A7%E7%89%88_%E6%95%B0%E6%8D%AE%E4%B8%8E%E5%AE%9E%E9%AA%8C%E5%B1%95%E7%A4%BA.ipynb) | 25个已执行代码单元，15幅已检查的嵌入图 |
| 离线阅读 | [HTML文件](%E6%89%A9%E5%A4%A7%E7%89%88%E5%B1%95%E7%A4%BA%E9%A2%84%E8%A7%88.html)、[10分钟讲稿与5分钟答问](10%E5%88%86%E9%92%9F%E6%B1%87%E6%8A%A5%E8%AE%B2%E7%A8%BF%E4%B8%8E%E7%AD%94%E9%97%AE.md) | HTML请下载后用浏览器打开；整页浏览器预览未验证 |
| 课程提交 | [提交版ZIP，13.87 MiB](https://github.com/zhangzhangquan83-ship-it/raman-plastic-spectroscopy-coursework/releases/download/v1.0.0/raman-coursework-submission.zip) | 展示、代码、说明、图表、全部CSV预测/日志/指标，不含大二进制 |
| 完整复现 | [完整ZIP，775.22 MiB](https://github.com/zhangzhangquan83-ship-it/raman-plastic-spectroscopy-coursework/releases/download/v1.0.0/raman-coursework-full.zip) | 再含两套处理NPZ、全部模型权重及概率矩阵 |
| 校验 | [SHA256SUMS.txt](https://github.com/zhangzhangquan83-ship-it/raman-plastic-spectroscopy-coursework/releases/download/v1.0.0/SHA256SUMS.txt) | 两个ZIP的SHA256；包内另有逐文件清单 |

**仅git clone可直接查看已保存输出。运行Notebook全部单元、checkpoint推理、84组独立审计或重画扩展图，都应先下载并解压完整复现包。** 仅重新下载原CSV不能恢复已经训练的权重；也可在新目录按冻结计划重新训练。

GitHub普通Git文件有100 MiB上限，所以大数据和模型采用Release附件分发，未放入Git历史。[GitHub大文件说明](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)

## 主要结果与评价范围

- 第一阶段：固定22/6/6图、15类。PCA64＋线性SVM测试准确率**72.53%**、宏F1 **0.5873**；增强CNN三训练种子宏F1 **0.5725 ± 0.0181**，±表示训练种子样本标准差。
- 第二阶段：仅原28开发图的12共同类、25,541条谱，3外折与内验证，75次神经拟合＋9组基线，累计**3769轮**。每折先平均种子，再统计三折均值与折间样本标准差。
- 新12类CV宏F1：SVM **0.8742 ± 0.0721**，CNN **0.8687 ± 0.0629**，1D-ResNet **0.8560 ± 0.0644**，MLP **0.8490 ± 0.0813**。这里的±为折间标准差，未作显著性声明。

**两阶段类别、评价图和目标范围不同，不能把分数差直接称为性能提升。** 原6图测试未参与追加调参。像素谱数不等于独立物理样品数；公开数据没有可靠粒子ID或二维网格，项目不宣称独立标本/跨仪器验证、浓度定量或二维分割。

## 课程要求对应文件

详见[提交材料核对表](SUBMISSION_CHECKLIST.md)。

| 要求 | 文件 |
| --- | --- |
| 展示文件 | [Notebook](%E6%89%A9%E5%A4%A7%E7%89%88_%E6%95%B0%E6%8D%AE%E4%B8%8E%E5%AE%9E%E9%AA%8C%E5%B1%95%E7%A4%BA.ipynb)、[讲稿与答问](10%E5%88%86%E9%92%9F%E6%B1%87%E6%8A%A5%E8%AE%B2%E7%A8%BF%E4%B8%8E%E7%AD%94%E9%97%AE.md) |
| 项目代码 | [首轮src](src/)、[扩展src](extension_study/src/) |
| 数据说明/获取方式 | [数据来源与研究边界](%E6%95%B0%E6%8D%AE%E6%9D%A5%E6%BA%90%E4%B8%8E%E7%A0%94%E7%A9%B6%E8%BE%B9%E7%95%8C.md)、[源清单](data/manifest.json)、[完整包](https://github.com/zhangzhangquan83-ship-it/raman-plastic-spectroscopy-coursework/releases/download/v1.0.0/raman-coursework-full.zip) |
| 环境与运行 | [完整运行说明](docs/%E8%AF%BE%E7%A8%8B%E4%BD%9C%E4%B8%9A%E5%AE%8C%E6%95%B4%E8%BF%90%E8%A1%8C%E8%AF%B4%E6%98%8E.md)、[requirements](requirements.txt)、[代码导读](%E4%BB%A3%E7%A0%81%E5%AF%BC%E8%AF%BB%E4%B8%8E%E8%AF%84%E4%BB%B7%E9%80%BB%E8%BE%91.md) |
| 结果与核查 | [首轮结果](%E6%89%A9%E5%A4%A7%E7%89%88%E5%AE%9E%E9%AA%8C%E7%BB%93%E6%9E%9C.md)、[扩展报告](extension_study/extension_report.md)、[独立审计](extension_study/results/independent_audit.json) |
| 成员贡献与引用 | Notebook和讲稿中的真实填写表、参考文献、数据许可 |

成员姓名、学号与亲自完成的贡献须据实填写，仓库没有编造成员经历。代码与材料有AI辅助。

## 数据来源与许可

数据：Smith等，Borealis，[DOI 10.5683/SP3/8UQQQN](https://doi.org/10.5683/SP3/8UQQQN)，**CC BY 4.0**。
论文：Hogan等，*Scientific Data* 13, 765 (2026)，[DOI 10.1038/s41597-026-07103-8](https://doi.org/10.1038/s41597-026-07103-8)。
本项目进行了校验、逐谱插值与归一化、类别过滤和按图建模评价，性能来自本项目实验。完整署名、原始计数冲突和修改声明见[数据说明](%E6%95%B0%E6%8D%AE%E6%9D%A5%E6%BA%90%E4%B8%8E%E7%A0%94%E7%A9%B6%E8%BE%B9%E7%95%8C.md)与[数据许可](DATA_LICENSE.md)。

## 文件完整性

原始提交包的逐文件清单保存在[artifacts/original_submission_manifest.json](artifacts/original_submission_manifest.json)，它对应Release原ZIP；仓库首页新增了GitHub导航，原运行说明保存在docs。完整ZIP还含全部省略二进制文件的清单和SHA。

```text
e9b0cbcfd9546710098d02e593e0cc0f601d73aac01b71566fa585d383085a30  raman-coursework-submission.zip
f86fe95c92a4a55edb07d7424964126f2ad4cbb8185c7d6ded4cee285159ff71  raman-coursework-full.zip
```
