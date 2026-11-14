## architecture_comparison

模型比较。点及误差棒为3外折均值±折间样本标准差，灰色短线为各折值。神经网络每折先平均3种子；传统基线每折1次。误差棒不是置信区间，3折也不是独立物理标本。源数据：condition_fold_means.csv、condition_summary.csv。

## augmentation_loss_paired

配对消融，横轴为变体减参照的宏平均F1差值，单位为百分点。参照：同折同训练种子的默认增强、加权损失CNN；每折先平均3个配对种子的差值。点/棒为3折均值±折间样本标准差，灰线为每折差值；不作显著性或单一机制因果声明。源数据：paired_ablation_runs.csv、paired_ablation_fold_means.csv、paired_ablation_summary.csv。

## preprocessing_paired

配对消融，横轴为变体减参照的宏平均F1差值，单位为百分点。参照：同折同seed101、关闭增强的默认预处理CNN；所有变体关闭增强，排除固定绝对增强噪声与L2尺度的混杂。每折仅1训练种子。点/棒为3折均值±折间样本标准差，灰线为每折差值；不作显著性或单一机制因果声明。源数据：paired_ablation_runs.csv、paired_ablation_fold_means.csv、paired_ablation_summary.csv。

## learning_curve_maps

嵌套按图学习曲线。每个点为同一外折中3种子均值，三条线分别为外折，不将其当作独立物理样品。只改变内训练图子集，内验证/外评价图固定；100%复用核心CNN。横轴为真实图比例或真实像素数，名义60%/80%不等于像素比例。各折曲线没有折间误差棒，因为没有把不同横坐标折叠到同一个点。源数据：fold_train_sizes.csv、learning_curve_fold_means.csv。

## learning_curve_pixels

嵌套按图学习曲线。每个点为同一外折中3种子均值，三条线分别为外折，不将其当作独立物理样品。只改变内训练图子集，内验证/外评价图固定；100%复用核心CNN。横轴为真实图比例或真实像素数，名义60%/80%不等于像素比例。各折曲线没有折间误差棒，因为没有把不同横坐标折叠到同一个点。源数据：fold_train_sizes.csv、learning_curve_fold_means.csv。

## noise_robustness

模拟高斯扰动敏感性。0为原外折输入；其余标准差在既有SNV输入空间加噪且不再次SNV。同折、同噪声水平的模型与种子使用同一噪声实现。每折先平均3种子，再报告3折均值±折间样本标准差。每水平仅1噪声实现，不代表真实仪器信噪比、外部验证或真实噪声机制。源数据：noise_run_metrics.csv、noise_fold_means.csv、noise_summary.csv。

## calibration_ece

未进行后验校准的softmax置信度误差。ECE使用10个等宽置信度区间，每折先平均3种子，再报告3折均值±折间样本标准差；越低表示该分箱定义下平均置信度与正确率越接近。NLL和多类Brier另列校准表，Brier按类别求和后对像素平均，未再除类别数。传统基线无概率输出，因此不编造其校准指标。源数据：calibration_summary.csv。

## calibration_reliability

可靠性图固定展示预先指定的seed101，汇集3个互斥外折的逐像素预测；每个像素恰好出现一次，点按像素加权，空箱保留在CSV而不画为零。该图仅作诊断，不含折间误差棒，不替代主表的折优先统计，不用于挑选种子或选择后验温度。源数据：reliability_seed101_bins.csv，内含每箱数量。