"""校验并汇总冻结的12类开发研究；不加载模型、不重新评价原六张测试图。

用法：python extension_study/src/summarize_study.py --root .
可选 --allow-partial 只生成明确标识的进度报告；缺种子的外折不进入汇总。
可选 --qa-tool-dir 指向科研绘图审计脚本目录，运行PDF字形及碰撞审计。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support, recall_score


PROJECT = Path(__file__).resolve().parents[2]
METRICS = ["accuracy", "balanced_accuracy", "macro_f1_present", "map_mean_accuracy",
           "map_equal_weight_macro_f1", "ece_10_bins", "negative_log_likelihood", "multiclass_brier"]
CORE = ["cnn", "mlp", "resnet"]
DISPLAY = {"cnn": "1D-CNN", "mlp": "MLP", "resnet": "1D-ResNet",
           "cosine_centroid": "Cosine centroid", "pca64_linear_svm": "PCA64 + linear SVM",
           "random_forest_300": "Random forest (300)", "noaug": "No augmentation",
           "unweighted": "Unweighted loss", "no_sg": "Without SG", "no_baseline": "Without baseline",
           "l2": "L2 normalization", "quadratic": "Quadratic baseline"}
SHORT = {"acrylonitrile butadiene styrene": "ABS", "blank": "Blank", "nylon": "PA",
         "poly (methyl methacrylate)": "PMMA", "polyester": "PEs", "polyethylene": "PE",
         "polyethylene terephthalate": "PET", "polyoxymethylene": "POM", "polypropylene": "PP",
         "polystyrene": "PS", "polytetrafluoroethylene": "PTFE", "polyvinyl chloride": "PVC"}
COLORS = {"cnn": "#246A9A", "mlp": "#C18A31", "resnet": "#935D85"}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def score_predictions(y, pred, maps):
    present = np.unique(y)
    unique, inverse, counts = np.unique(maps, return_inverse=True, return_counts=True)
    return dict(accuracy=float(np.mean(y == pred)),
                balanced_accuracy=float(recall_score(y, pred, labels=present, average="macro", zero_division=0)),
                macro_f1_present=float(f1_score(y, pred, labels=present, average="macro", zero_division=0)),
                map_mean_accuracy=float(np.mean([np.mean(y[maps == g] == pred[maps == g]) for g in unique])),
                map_equal_weight_macro_f1=float(f1_score(y, pred, labels=present, average="macro",
                                                       sample_weight=1.0 / counts[inverse], zero_division=0)),
                evaluated_classes=len(present), pixels=len(y), maps=len(unique))


def reliability_bins(y, probabilities):
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == y
    bins = []
    for k in range(10):
        selected = (confidence >= k / 10) & ((confidence < (k + 1) / 10) if k < 9 else (confidence <= 1))
        n = int(selected.sum())
        bins.append(dict(bin=k, lower=k / 10, upper=(k + 1) / 10, count=n,
                         mean_confidence=float(confidence[selected].mean()) if n else np.nan,
                         observed_accuracy=float(correct[selected].mean()) if n else np.nan))
    return bins


def calibration(y, probabilities):
    bins = reliability_bins(y, probabilities)
    ece = sum(row["count"] / len(y) * abs(row["observed_accuracy"] - row["mean_confidence"])
              for row in bins if row["count"])
    target = np.eye(probabilities.shape[1], dtype=np.float32)[y]
    return dict(ece_10_bins=float(ece),
                negative_log_likelihood=float(-np.log(np.clip(probabilities[np.arange(len(y)), y], 1e-8, 1)).mean()),
                multiclass_brier=float(np.sum((probabilities - target) ** 2, axis=1).mean()))


def assert_metrics(observed, recomputed, context):
    for key, value in recomputed.items():
        require(key in observed and np.isfinite(observed[key]), f"{context}: 缺失或非有限指标 {key}")
        require(np.isclose(float(observed[key]), value, atol=3e-6, rtol=3e-6),
                f"{context}: {key} 与预测重算不一致 ({observed[key]} vs {value})")


def condition(config):
    family = config["family"]
    if family in ("architecture", "baseline"):
        return family + ":" + config["model"]
    key = {"preprocessing": "variant", "learning_curve": "map_level"}.get(family, "model")
    return family + ":" + str(config[key])


def load_runs(root, allow_partial):
    study = root / "extension_study"
    splits = read_json(study / "protocol/splits.json")
    plan = read_json(study / "protocol/experiment_plan.json")
    hashes = dict(protocol_sha256=sha256(study / "protocol/splits.json"),
                  plan_sha256=sha256(study / "protocol/experiment_plan.json"),
                  data_sha256=sha256(study / "data/study_data.npz"))
    require(plan["split_sha256"] == hashes["protocol_sha256"], "计划中的划分哈希不符")
    require(plan.get("original_test_reuse") is False and splits.get("original_test_reused") is False,
            "协议必须排除原六张测试图")
    labels = [str(label) for label in splits["labels"]]
    require(labels == plan["task_classes"], "计划与划分类别顺序不一致")
    fold_ids = [int(f["fold_id"]) for f in splits["folds"]]
    require(len(fold_ids) == 3 and len(labels) == 12, "此脚本要求冻结的3外折、12类协议")
    audit = read_json(study / "data/study_data_audit.json")
    require(audit["study_dataset_sha256"] == hashes["data_sha256"] and
            audit["protocol_sha256"] == hashes["protocol_sha256"], "派生数据审计哈希不符")
    with np.load(study / "data/study_data.npz", allow_pickle=False) as data:
        original = data["original_indices"].astype(np.int64)
        truth = pd.DataFrame(dict(original_index=original, true_label=data["y"].astype(str),
                                  map_id=data["groups"].astype(str), pixel_index=data["pixel_index"])).set_index("original_index")
    require(truth.index.is_unique, "派生数据含重复原始行号")
    require(set(truth.map_id).isdisjoint(splits["original_test_maps_excluded"]), "派生数据包含原始测试图")
    union = [i for f in splits["folds"] for i in f["outer_eval_indices"]]
    require(len(set(union)) == len(union) and set(union) == set(original), "外折预测总体必须互斥且覆盖派生数据")
    expected = list(plan["configs"]) + [dict(run_id=f"baseline_{name}_f{fold}", model=name,
                fold_id=fold, family="baseline", seed=101) for fold in fold_ids for name in plan["baselines"]]
    require(len({c["run_id"] for c in expected}) == len(expected), "计划存在重复run_id")
    for config in expected:
        if config["family"] == "preprocessing":
            require(config["augment"] is False and config["seed"] == 101,
                    "预处理必须关闭增强并使用匹配seed101；请使用修订后的冻结计划")
    folds = {int(f["fold_id"]): f for f in splits["folds"]}
    records, map_rows, class_rows, noise_rows, pooled, missing = [], [], [], [], {}, []
    for config in expected:
        run_id = config["run_id"]
        directory = study / "results/runs" / run_id
        metadata_path = directory / "metadata.json"
        if not metadata_path.exists() or read_json(metadata_path).get("status") != "finished":
            missing.append(run_id)
            continue
        metadata = read_json(metadata_path)
        require(all(metadata.get(key) == value for key, value in hashes.items()), f"{run_id}: 文件哈希不符")
        require(all(metadata["config"].get(key) == value for key, value in config.items()), f"{run_id}: 配置与冻结计划不符")
        require(metadata.get("labels") == labels and metadata.get("original_test_used") is False,
                f"{run_id}: 类别或原测试图排除声明不符")
        neural = config["family"] != "baseline"
        required = ["outer_predictions.csv", "outer_metrics.json", "outer_map_metrics.csv"]
        if neural:
            required += ["outer_probabilities.npz", "history.csv"]
        if config["family"] == "architecture":
            required += ["noise_robustness.csv"]
        require(all((directory / name).exists() for name in required), f"{run_id}: 已完成记录缺少结果文件")
        prediction = pd.read_csv(directory / "outer_predictions.csv")
        indices = prediction.original_index.to_numpy(dtype=np.int64)
        fold = folds[int(config["fold_id"])]
        require(np.array_equal(indices, np.asarray(fold["outer_eval_indices"])), f"{run_id}: 外折索引或行序不符")
        expected_truth = truth.loc[indices]
        for key in ["true_label", "map_id", "pixel_index"]:
            require(np.array_equal(prediction[key].to_numpy(), expected_truth[key].to_numpy()), f"{run_id}: {key}不符")
        require(set(prediction.predicted_label).issubset(labels), f"{run_id}: 预测标签超出任务类别")
        require(set(prediction.true_label) == set(labels), f"{run_id}: 外折没有覆盖全部12类")
        y = np.asarray([labels.index(v) for v in prediction.true_label])
        pred = np.asarray([labels.index(v) for v in prediction.predicted_label])
        groups = prediction.map_id.to_numpy()
        scores = read_json(directory / "outer_metrics.json")
        assert_metrics(scores, score_predictions(y, pred, groups), run_id)
        assert_metrics(metadata["outer_metrics"], scores, run_id + "/metadata")
        require(np.isfinite(metadata["runtime_seconds"]) and metadata["runtime_seconds"] >= 0,
                f"{run_id}: 运行时长异常")
        if neural:
            with np.load(directory / "outer_probabilities.npz", allow_pickle=False) as archive:
                probabilities = archive["probabilities"]
                require(np.array_equal(archive["original_indices"], indices), f"{run_id}: 概率行序不符")
            require(probabilities.shape == (len(indices), len(labels)) and np.isfinite(probabilities).all(), f"{run_id}: 概率形状/有限值异常")
            require((probabilities >= 0).all() and (probabilities <= 1).all() and
                    np.allclose(probabilities.sum(axis=1), 1, atol=2e-5), f"{run_id}: 非法概率")
            require(np.array_equal(probabilities.argmax(axis=1), pred), f"{run_id}: 概率与类别预测不符")
            require(np.allclose(prediction.confidence, probabilities.max(axis=1), atol=2e-6), f"{run_id}: 置信度不符")
            assert_metrics(scores, calibration(y, probabilities), run_id)
            history = pd.read_csv(directory / "history.csv")
            require(len(history) == metadata["epochs_completed"] and
                    int(metadata["best_epoch"]) in set(history.epoch), f"{run_id}: 训练历史与最佳epoch不符")
            if config["family"] == "architecture" and config["seed"] == 101:
                pooled.setdefault(config["model"], []).append((indices, y, probabilities))
        reference = dict(run_id=run_id, condition=condition(config), family=config["family"],
                         model=config["model"], fold_id=int(config["fold_id"]), seed=int(config["seed"]))
        train_size = next(c for c in fold["learning_curve"] if c["level"] == config.get("map_level", "maps100"))
        require(metadata["train_maps"] == train_size["map_count"] and metadata["train_pixels"] == train_size["pixels"],
                f"{run_id}: 训练规模与冻结协议不符")
        records.append(dict(reference, **scores, train_maps=metadata["train_maps"], train_pixels=metadata["train_pixels"],
                            parameters=metadata.get("parameters", np.nan), runtime_seconds=metadata["runtime_seconds"],
                            best_epoch=metadata.get("best_epoch", np.nan),
                            best_inner_val_macro_f1=metadata.get("best_inner_val_macro_f1", np.nan)))
        per_map = pd.read_csv(directory / "outer_map_metrics.csv").set_index("map_id")
        require(set(per_map.index) == set(groups) and per_map.index.is_unique, f"{run_id}: 逐图结果不符")
        for group in sorted(set(groups)):
            mask = groups == group
            values = score_predictions(y[mask], pred[mask], groups[mask])
            assert_metrics(per_map.loc[group].to_dict(), values, f"{run_id}/{group}")
            map_rows.append(dict(reference, map_id=group, **values))
        precision, recall, f1, support = precision_recall_fscore_support(y, pred, labels=np.arange(len(labels)), zero_division=0)
        for j, label in enumerate(labels):
            class_rows.append(dict(reference, label=label, precision=precision[j], recall=recall[j], f1=f1[j], support=int(support[j])))
        if config["family"] == "architecture":
            noise_rows.append(dict(reference, noise_std=0.0, **scores))
            noisy = pd.read_csv(directory / "noise_robustness.csv")
            require(len(noisy) == len(plan["noise_levels"]) and
                    np.allclose(sorted(noisy.noise_std), sorted(plan["noise_levels"])), f"{run_id}: 噪声水平不符")
            for values in noisy.to_dict("records"):
                require(all(np.isfinite(values[key]) for key in METRICS), f"{run_id}: 噪声指标非有限")
                require(all(0 <= values[key] <= 1 for key in METRICS[:6]) and
                        values["negative_log_likelihood"] >= 0 and 0 <= values["multiclass_brier"] <= 2.00001,
                        f"{run_id}: 噪声指标超出定义范围")
                require(all(values[key] == scores[key] for key in ["pixels", "maps", "evaluated_classes"]),
                        f"{run_id}: 噪声实验评价总体不符")
                noise_rows.append(dict(reference, **values))
    require(allow_partial or not missing, f"尚有{len(missing)}/{len(expected)}个run未完成；需要进度报告时使用--allow-partial")
    return dict(study=study, splits=splits, plan=plan, hashes=hashes, expected=expected, missing=missing,
                runs=pd.DataFrame(records), maps=pd.DataFrame(map_rows), classes=pd.DataFrame(class_rows),
                noise=pd.DataFrame(noise_rows), pooled=pooled)


def hierarchical(frame, expected, keys, value_columns):
    """完整的同条件同折种子先平均；再按三个外折计算均值及样本标准差。"""
    if frame.empty:
        return pd.DataFrame(), pd.DataFrame()
    available = [key for key in value_columns if key in frame]
    fold_rows = []
    for grouping, part in frame.groupby(keys + ["fold_id"], dropna=False, sort=True):
        grouping = grouping if isinstance(grouping, tuple) else (grouping,)
        identity = dict(zip(keys + ["fold_id"], grouping))
        cond = identity["condition"]
        wanted = {int(c["seed"]) for c in expected if condition(c) == cond and int(c["fold_id"]) == identity["fold_id"]}
        require(not part.seed.duplicated().any(), f"重复的折/种子记录: {identity}")
        complete = set(part.seed.astype(int)) == wanted
        fold_rows.append(dict(identity, n_seeds=len(part), expected_seeds=len(wanted), seed_complete=complete,
                              **{key: float(part[key].mean()) if complete and part[key].notna().all() else np.nan for key in available}))
    fold_frame = pd.DataFrame(fold_rows)
    summary_rows = []
    for grouping, part in fold_frame.groupby(keys, dropna=False, sort=True):
        grouping = grouping if isinstance(grouping, tuple) else (grouping,)
        identity = dict(zip(keys, grouping))
        complete = len(part) == 3 and part.seed_complete.all()
        values = {}
        for key in available:
            valid = complete and part[key].notna().all()
            values[key + "_mean"] = float(part[key].mean()) if valid else np.nan
            values[key + "_fold_sd"] = float(part[key].std(ddof=1)) if valid else np.nan
        summary_rows.append(dict(identity, n_complete_folds=int(part.seed_complete.sum()), complete=bool(complete), **values))
    return fold_frame, pd.DataFrame(summary_rows)


def paired_ablations(runs):
    rows = []
    if runs.empty:
        return pd.DataFrame()
    lookup = runs.set_index("run_id")
    for row in runs.itertuples(index=False):
        if row.family not in ("noaug", "unweighted", "preprocessing"):
            continue
        reference_id = (f"noaug_cnn_f{row.fold_id}_s101" if row.family == "preprocessing"
                        else f"core_cnn_f{row.fold_id}_s{row.seed}")
        if reference_id not in lookup.index:
            continue
        reference = lookup.loc[reference_id]
        values = {key: getattr(row, key) - reference[key] for key in METRICS if key in runs}
        rows.append(dict(condition=row.condition, family=row.family, fold_id=row.fold_id, seed=row.seed,
                         run_id=row.run_id, reference_run_id=reference_id, **values))
    return pd.DataFrame(rows)


class Figures:
    """独立单面板、白底、固定色板；每幅图提供可审查数据与图注。"""
    def __init__(self, directory, dpi, partial):
        self.directory, self.dpi, self.partial = directory, dpi, partial
        self.directory.mkdir(parents=True, exist_ok=True)
        self.captions = []
        plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
                             "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 12,
                             "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
                             "axes.spines.right": False, "axes.spines.top": False, "axes.linewidth": 0.8,
                             "pdf.fonttype": 42, "svg.fonttype": "none", "legend.frameon": False,
                             "savefig.facecolor": "white"})

    def new(self, title, ylabel="Outer-fold macro-F1"):
        fig, ax = plt.subplots(figsize=(7.2, 4.8), layout="constrained")
        ax.set_title(("PARTIAL: " if self.partial else "") + title, loc="left", pad=12)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#E5E7EB", linewidth=0.7)
        ax.set_axisbelow(True)
        return fig, ax

    def save(self, fig, name, caption):
        fig.canvas.draw()
        # 每幅图只有一个科学坐标轴，故多面板对齐审计不适用。
        require(len(fig.axes) == 1, "独立图意外包含多个坐标轴，必须加入多面板对齐审计")
        stem = self.directory / name
        fig.savefig(stem.with_suffix(".png"), dpi=self.dpi, bbox_inches="tight")
        fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
        fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
        write_json(stem.with_suffix(".alignment.json"), dict(status="not_applicable", reason="Single scientific axes", axes_count=1))
        plt.close(fig)
        self.captions.append((name, caption))


def plot_results(bundle, tables, directory, dpi):
    plots = Figures(directory, dpi, bool(bundle["missing"]))
    summary, folds = tables["condition_summary"], tables["condition_fold_means"]
    if summary.empty:
        return plots
    complete = summary[summary.complete]
    order = ["baseline:" + m for m in bundle["plan"]["baselines"]] + ["architecture:" + m for m in CORE]
    main = complete[complete.condition.isin(order)].set_index("condition").reindex(order).dropna(subset=["macro_f1_present_mean"])
    if not main.empty:
        fig, ax = plots.new("Model comparison: 12-class development study")
        ypos = np.arange(len(main))
        ax.errorbar(main.macro_f1_present_mean, ypos, xerr=main.macro_f1_present_fold_sd,
                    fmt="o", color="#246A9A", capsize=4, markersize=6, linestyle="none")
        for i, name in enumerate(main.index):
            values = folds[(folds.condition == name) & folds.seed_complete].macro_f1_present
            ax.scatter(values, np.full(len(values), i) + np.linspace(-0.09, 0.09, len(values)),
                       color="#8A8F96", marker="|", s=65, zorder=3)
        ax.set_yticks(ypos, [DISPLAY.get(s.split(":")[1], s) for s in main.index])
        ax.set_ylabel(""); ax.set_xlabel("Macro-F1 (mean and between-fold SD)"); ax.set_xlim(0, 1); ax.invert_yaxis()
        plots.save(fig, "architecture_comparison", "模型比较。点及误差棒为3外折均值±折间样本标准差，灰色短线为各折值。神经网络每折先平均3种子；传统基线每折1次。误差棒不是置信区间，3折也不是独立物理标本。源数据：condition_fold_means.csv、condition_summary.csv。")
    deltas, delta_folds = tables["paired_ablation_summary"], tables["paired_ablation_fold_means"]
    if not deltas.empty:
        for families, name, title in [(["noaug", "unweighted"], "augmentation_loss_paired", "Matched augmentation and loss contrasts"),
                                      (["preprocessing"], "preprocessing_paired", "Matched preprocessing contrasts; no augmentation")]:
            group = deltas[deltas.complete & deltas.condition.str.split(":").str[0].isin(families)]
            if group.empty:
                continue
            fig, ax = plots.new(title, "")
            ypos = np.arange(len(group))
            ax.axvline(0, color="#515961", linewidth=1, linestyle="--")
            ax.errorbar(100 * group.macro_f1_present_mean, ypos, xerr=100 * group.macro_f1_present_fold_sd,
                        fmt="o", color="#246A9A", capsize=4, markersize=6)
            for i, cond in enumerate(group.condition):
                values = delta_folds[(delta_folds.condition == cond) & delta_folds.seed_complete].macro_f1_present
                ax.scatter(100 * values, np.full(len(values), i) + np.linspace(-0.09, 0.09, len(values)), color="#8A8F96", marker="|", s=65)
            ax.set_yticks(ypos, [DISPLAY.get(s.split(":")[1], DISPLAY.get(s.split(":")[0], s)) for s in group.condition])
            if name == "augmentation_loss_paired":
                ax.set_yticklabels([DISPLAY[s.split(":")[0]] for s in group.condition])
            ax.set_xlabel("Change in macro-F1 (percentage points)"); ax.invert_yaxis()
            reference = "同折同seed101、关闭增强的默认预处理CNN；所有变体关闭增强，排除固定绝对增强噪声与L2尺度的混杂。每折仅1训练种子" if name == "preprocessing_paired" else "同折同训练种子的默认增强、加权损失CNN；每折先平均3个配对种子的差值"
            plots.save(fig, name, f"配对消融，横轴为变体减参照的宏平均F1差值，单位为百分点。参照：{reference}。点/棒为3折均值±折间样本标准差，灰线为每折差值；不作显著性或单一机制因果声明。源数据：paired_ablation_runs.csv、paired_ablation_fold_means.csv、paired_ablation_summary.csv。")
    curve = tables["learning_curve_fold_means"]
    if not curve.empty:
        for x, name, xlabel in [("actual_map_fraction", "learning_curve_maps", "Actual fraction of inner-training maps"),
                                ("pixels", "learning_curve_pixels", "Inner-training pixel spectra")]:
            fig, ax = plots.new("CNN learning curve: nested map subsets")
            for i, (fold_id, values) in enumerate(curve.groupby("fold_id")):
                values = values[values.seed_complete].sort_values(x)
                if values.empty:
                    continue
                ax.plot(values[x], values.macro_f1_present, marker=["o", "s", "^"][i],
                        color=["#246A9A", "#C18A31", "#935D85"][i], label=f"Fold {fold_id}")
            ax.set_xlabel(xlabel); ax.set_ylim(0, 1); ax.legend(loc="best")
            plots.save(fig, name, "嵌套按图学习曲线。每个点为同一外折中3种子均值，三条线分别为外折，不将其当作独立物理样品。只改变内训练图子集，内验证/外评价图固定；100%复用核心CNN。横轴为真实图比例或真实像素数，名义60%/80%不等于像素比例。各折曲线没有折间误差棒，因为没有把不同横坐标折叠到同一个点。源数据：fold_train_sizes.csv、learning_curve_fold_means.csv。")
    noise = tables["noise_summary"]
    if not noise.empty and noise.complete.any():
        fig, ax = plots.new("Gaussian perturbation sensitivity in SNV space")
        for model in CORE:
            group = noise[(noise.condition == "architecture:" + model) & noise.complete].sort_values("noise_std")
            if group.empty:
                continue
            ax.errorbar(group.noise_std, group.macro_f1_present_mean, yerr=group.macro_f1_present_fold_sd,
                        marker={"cnn": "o", "mlp": "s", "resnet": "^"}[model], color=COLORS[model],
                        capsize=3, label=DISPLAY[model])
        ax.set_xlabel("Added Gaussian noise SD (preprocessed SNV units)"); ax.set_ylim(0, 1); ax.legend(loc="best")
        plots.save(fig, "noise_robustness", "模拟高斯扰动敏感性。0为原外折输入；其余标准差在既有SNV输入空间加噪且不再次SNV。同折、同噪声水平的模型与种子使用同一噪声实现。每折先平均3种子，再报告3折均值±折间样本标准差。每水平仅1噪声实现，不代表真实仪器信噪比、外部验证或真实噪声机制。源数据：noise_run_metrics.csv、noise_fold_means.csv、noise_summary.csv。")
    neural = complete[complete.condition.isin(["architecture:" + m for m in CORE])].dropna(subset=["ece_10_bins_mean"])
    if not neural.empty:
        fig, ax = plots.new("Calibration error of uncalibrated softmax", "ECE (10 equal-width confidence bins)")
        x = np.arange(len(neural))
        ax.errorbar(x, neural.ece_10_bins_mean, yerr=neural.ece_10_bins_fold_sd, fmt="o", capsize=4, color="#246A9A")
        ax.set_xticks(x, [DISPLAY[s.split(":")[1]] for s in neural.condition]); ax.set_ylim(bottom=0)
        plots.save(fig, "calibration_ece", "未进行后验校准的softmax置信度误差。ECE使用10个等宽置信度区间，每折先平均3种子，再报告3折均值±折间样本标准差；越低表示该分箱定义下平均置信度与正确率越接近。NLL和多类Brier另列校准表，Brier按类别求和后对像素平均，未再除类别数。传统基线无概率输出，因此不编造其校准指标。源数据：calibration_summary.csv。")
    reliability = tables["reliability_seed101_bins"]
    if not reliability.empty:
        fig, ax = plots.new("Reliability: preselected seed 101, pooled outer folds", "Observed accuracy in bin")
        ax.plot([0, 1], [0, 1], color="#515961", linestyle="--", linewidth=1, label="Ideal calibration")
        for model, group in reliability.groupby("model", sort=False):
            group = group[group["count"] > 0]
            ax.plot(group.mean_confidence, group.observed_accuracy, marker={"cnn": "o", "mlp": "s", "resnet": "^"}[model],
                    color=COLORS[model], label=DISPLAY[model])
        ax.set_xlabel("Mean confidence in nonempty bin"); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.legend(loc="best")
        plots.save(fig, "calibration_reliability", "可靠性图固定展示预先指定的seed101，汇集3个互斥外折的逐像素预测；每个像素恰好出现一次，点按像素加权，空箱保留在CSV而不画为零。该图仅作诊断，不含折间误差棒，不替代主表的折优先统计，不用于挑选种子或选择后验温度。源数据：reliability_seed101_bins.csv，内含每箱数量。")
    return plots


def md_table(frame, columns):
    if frame.empty:
        return "目前没有完整三折的可报告结果。"
    lines = ["| " + " | ".join(columns.values()) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in frame.to_dict("records"):
        lines.append("| " + " | ".join(str(row.get(key, "")) for key in columns) + " |")
    return "\n".join(lines)


def metric_table(frame, keys):
    rows = []
    for item in frame.to_dict("records"):
        row = dict(condition=item["condition"])
        for key in keys:
            value, sd = item.get(key + "_mean", np.nan), item.get(key + "_fold_sd", np.nan)
            row[key] = f"{value:.4f} ± {sd:.4f}" if np.isfinite(value) else "未完成/无概率输出"
        rows.append(row)
    return pd.DataFrame(rows)


def write_report(root, bundle, tables, plots, qa):
    splits, plan = bundle["splits"], bundle["plan"]
    complete = not bundle["missing"]
    text = ["# 拉曼塑料识别：追加12类开发研究", "",
            ("本报告全部计划任务已完成，数值来自保存的外折预测与概率，并经重新计算核对。" if complete else
             f"**进度报告：已完成{len(bundle['runs'])}/{len(bundle['expected'])}个计划任务，尚缺{len(bundle['missing'])}个。以下只列完整三折、各折种子齐全的条件；未完成值不填补，不作最终排名或结论。**"),
            "", "## 两阶段证据关系", "",
            "第一阶段固定六张测试图、15类任务已经完成。其结果用于发现跨采集图识别的困难，不能再用这六张图挑选追加研究模型。第二阶段严格排除这些原测试图，只在原28张开发图上研究共同12类。两个阶段的类别、评价图和研究目的不同，不能把分数直接相减、排在一张性能榜上，或将第二阶段高分当作原15类难点已经解决。", ""]
    historical = root / "results/final_test/model_comparison.csv"
    if historical.exists():
        old = pd.read_csv(historical)
        svm = old[old.model == "pca64_linear_svm"]
        cnn = old[old.model.isin(["cnn_aug_seed42", "cnn_aug_seed43", "cnn_aug_seed44"])]
        if len(svm) == 1 and len(cnn) == 3:
            text += [f"作为历史背景，第一阶段SVM测试准确率{svm.iloc[0].accuracy:.2%}、15类宏平均F1 {svm.iloc[0].macro_f1_present:.4f}；增强CNN三种子平均准确率{cnn.accuracy.mean():.2%}、宏平均F1 {cnn.macro_f1_present.mean():.4f}（种子标准差{cnn.macro_f1_present.std(ddof=1):.4f}）。这支持‘CNN尚未稳定胜出’，而非深度模型必优。该处仅读取历史汇总表，没有新增原测试预测。", ""]
    excluded = "、".join(f"{v['label']}（开发图覆盖{v['development_map_coverage']}张，{v['development_pixels']}条）" for v in splits["excluded_classes"])
    text += ["## 数据、划分与预设方法", "",
             f"开发总体{splits['development_population_pixels']:,}条谱，按‘类别在至少3张开发图出现’规则，排除{excluded}，保留{splits['eligible_population_pixels']:,}条、12类（含blank）。排除是为了各外折均有真值支持，明确缩小了任务范围；并未验证被排除三类的新模型表现。",
             "", "三外折按map互斥，外评价并集覆盖共同类开发谱一次。每个外训练集合再划分内训练与5张内验证图，全部角色均覆盖12类。神经网络仅按内验证宏平均F1选择最佳epoch，SVM仅按内验证选择C，随后直接评价外折；不对全外训练集重拟合，不用外评价选模型、轮次、种子或预处理。", "",
             f"预设{len(plan['configs'])}次神经拟合与{len(plan['baselines']) * 3}组基线。架构比较CNN、MLP和ResNet，各折种子101/202/303；另做增强、损失和嵌套图学习曲线。预处理4变体只用seed101，全部关闭增强，与同折关闭增强的默认CNN匹配，避免L2尺度与固定绝对增强噪声强度混杂。",
             "", "统计层级：每条件每外折先对预定种子平均，再计算3折均值及折间样本标准差（ddof=1）。不把3折×3种子视为9个独立测试样本；折训练集合重叠，因此折间标准差只是此协议下的描述性变化，不是独立标本置信区间。逐图与逐类结果用于诊断，不据此继续调参。", ""]
    sizes = tables["fold_train_sizes"]
    text += [md_table(sizes, {"fold_id": "外折", "level": "名义图水平", "map_count": "实际训练图", "pixels": "实际训练像素", "actual_map_fraction": "实际图比例"}), "",
             "## 架构、传统基线与校准", ""]
    summary = tables["condition_summary"]
    selected = summary[summary.complete & summary.condition.str.startswith(("architecture:", "baseline:"))] if not summary.empty else pd.DataFrame()
    text += [md_table(metric_table(selected, ["accuracy", "macro_f1_present", "map_equal_weight_macro_f1"]),
                      {"condition": "条件", "accuracy": "像素准确率", "macro_f1_present": "12类宏F1", "map_equal_weight_macro_f1": "图等权宏F1"}), ""]
    if complete and not selected.empty:
        best = selected.loc[selected.macro_f1_present_mean.idxmax()]
        text += [f"在这个预设协议中，{best.condition}的3折平均宏F1最高，为{best.macro_f1_present_mean:.4f} ± {best.macro_f1_present_fold_sd:.4f}。这是描述性比较；未进行显著性检验，不证明在未知物理标本、仪器或被排除类别上普遍占优。", ""]
    calibrated = selected[selected.condition.str.startswith("architecture:")] if not selected.empty else pd.DataFrame()
    text += [md_table(metric_table(calibrated, ["ece_10_bins", "negative_log_likelihood", "multiclass_brier"]),
                      {"condition": "条件", "ece_10_bins": "ECE，10箱", "negative_log_likelihood": "NLL", "multiclass_brier": "多类Brier"}),
             "", "ECE、NLL及Brier均越低越好；ECE依赖分箱，Brier按类别求和后再对像素平均，未除类别数。这里只报告未做后验校准的softmax。传统基线没有保存概率，校准表留空。可靠性图预先固定seed101，拼接互斥外折预测、按像素加权；它与主表的折优先平均是不同展示口径，不将汇集图误当作9个独立重复。", "",
             "## 配对消融与学习曲线", ""]
    paired = tables["paired_ablation_summary"]
    valid_paired = paired[paired.complete] if not paired.empty else pd.DataFrame()
    text += ["下表是变体减参照的差值，数值使用F1的0–1单位；图中换算为百分点。增强/损失以同折同seed的核心CNN为参照。预处理以同折seed101、关闭增强的默认CNN为参照，只支持单训练种子下3折的匹配结果。", "",
             md_table(metric_table(valid_paired, ["macro_f1_present", "accuracy"]), {"condition": "变体", "macro_f1_present": "宏F1配对差值", "accuracy": "准确率配对差值"}), "",
             "训练图学习曲线使用内训练图的嵌套子集，各水平覆盖12类；真实图比例受向上取整影响，像素数与图比例也不相同。100%复用核心CNN，不算额外拟合。图中保留各折真实横坐标，不把名义60%称为保留60%像素。", "",
             "## 扰动、逐图与逐类诊断", "",
             "加噪实验在已预处理的SNV空间加入零均值高斯扰动，标准差为0.02/0.05/0.10/0.20，加噪后不再SNV；同折同水平使用相同噪声实现。它测量模拟扰动敏感性，并非测量真实仪器SNR、外部数据集性能或噪声因果。各水平仅一次噪声实现，不能据此证明某种真实采集噪声导致第一阶段错误。", ""]
    classes = tables["class_summary"]
    if complete and not classes.empty:
        core = classes[(classes.condition == "architecture:cnn") & classes.complete].sort_values("f1_mean").head(3)
        details = "；".join(f"{SHORT.get(r.label, r.label)}：F1 {r.f1_mean:.4f} ± {r.f1_fold_sd:.4f}" for r in core.itertuples(index=False))
        text += ["核心CNN的较弱类别（事先固定汇报CNN，不据此挑模型）：" + details + "。完整类别表保留全部类别，以上仅为误差定位。", ""]
        maps = tables["map_seed_means"]
        weak = maps[maps.condition == "architecture:cnn"].sort_values("accuracy").head(3)
        text += ["核心CNN的较弱外评价图（每图先平均3种子）：" + "；".join(f"{r.map_id}：准确率{r.accuracy:.2%}" for r in weak.itertuples(index=False)) + "。这里的图值属于各自外折，不能把单图差异直接归因于唯一噪声或标签问题。", ""]
    text += ["## 证据边界与结论", "",
             "34张公开采集图并不等同于34个可追踪的独立物理标本，跨图可能共享原始材料；像素之间相关，不能以25,541条谱数构造极窄的独立样本区间。任务保留原库PET与PEs标签层级，含blank但不等于未知类拒识，也没有混合物定量、颗粒计数或二维分割结果。原CSV标签与元数据计数的冲突仍保留在数据审计中。", "",
             ("第二阶段已完成预设规模的模型、消融、学习曲线与扰动校准评价。其作用是在共同12类开发图上检验方法选择的稳定性和误差结构；第一阶段15类六图测试的原始结论保持不变。后续若要宣称独立材料或跨仪器应用，需要采集可追踪的新样品、明确物理标本分组并锁定外部评价集。" if complete else
              "尚未完成全部预设任务，因此目前不作最终研究结论。缺失run清单见run_validation.json；补全相同冻结计划后重新运行此脚本，禁止依据已有外折分数修改方案。"),
             "", "## 可复现与图表说明", "",
             "输入来自冻结协议、训练元数据、外折预测/概率及基线结果；本脚本只读取结果，不加载权重训练，也不读取原测试图作新预测。summary目录保留run级、折级、跨折汇总、配对差值、完整逐图逐类指标及可靠性分箱计数。PNG以高分辨率导出，同时提供可编辑文本PDF。每图一个科学坐标轴，多面板对齐不适用。", "",
             f"计划SHA256：`{bundle['hashes']['plan_sha256']}`；划分SHA256：`{bundle['hashes']['protocol_sha256']}`；数据SHA256：`{bundle['hashes']['data_sha256']}`。", "",
             f"图形自动审计状态：{qa['status']}。详见summary/figure_qa.json；自动几何检查不能替代最终显示尺寸的目视检查。", "",
             "数据署名：Smith等RaMPI，Borealis，[DOI 10.5683/SP3/8UQQQN](https://doi.org/10.5683/SP3/8UQQQN)，CC BY 4.0；相关论文Hogan等，Scientific Data 13,765(2026)，[DOI 10.1038/s41597-026-07103-8](https://doi.org/10.1038/s41597-026-07103-8)。本作业进行了数据校验、插值、逐谱预处理、共同类过滤和按图建模评价。姓名、学号与实际贡献在最终汇报中按本人实做填写。", ""]
    for name, caption in plots.captions:
        text += [f"### {name}", "", f"![{name}](results/summary/figures/{name}.png)", "", caption, ""]
    (bundle["study"] / "extension_report.md").write_text("\n".join(text), encoding="utf-8")


def audit_figures(source, figures, tool_dir):
    if tool_dir is None:
        return dict(status="未运行外置PDF几何审计；请在交付前运行并目视检查", checks=[])
    checks = []
    commands = [[sys.executable, str(tool_dir / "validate_figure.py"), str(source), "--json"]]
    for pdf in sorted(figures.glob("*.pdf")):
        commands += [[sys.executable, str(tool_dir / "audit_pdf_text.py"), str(pdf), "--min-pt", "5", "--json"],
                     [sys.executable, str(tool_dir / "audit_figure_collisions.py"), str(pdf),
                      "--json-out", str(pdf.with_suffix(".collision-audit.json"))]]
    for command in commands:
        result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
        checks.append(dict(tool=Path(command[1]).name, target=Path(command[2]).name,
                           returncode=result.returncode, stdout=result.stdout, stderr=result.stderr))
    passed = all(c["returncode"] == 0 for c in checks)
    return dict(status="自动审计通过，仍需目视检查警告" if passed else "自动审计存在失败或依赖缺失，需修复后交付",
                reviewed_source_warnings={
                    "PNG_without_TIFF": "本课程交付为高分辨率PNG预览及PDF/SVG矢量图，不要求期刊TIFF上传格式。",
                    "dynamic_DPI": "PNG的dpi来自CLI，默认400；入口强制至少300。",
                    "dropna": "仅剔除没有完整三折指标的绘图条件或无概率基线；缺失run及完整性标志保存在CSV和run_validation.json，未删除原始像素。",
                    "log_guard": "NLL仅对np.clip(p_true, 1e-8, 1)取对数，已有显式正数下界。",
                    "single_axes": "每幅图只有一个科学坐标轴，保存前显式断言，故多面板对齐不适用。"}, checks=checks)


def main(args):
    require(args.dpi >= 300, "导出分辨率至少300dpi")
    root = args.root.resolve()
    bundle = load_runs(root, args.allow_partial)
    out = bundle["study"] / "results/summary"
    out.mkdir(parents=True, exist_ok=True)
    runs, expected = bundle["runs"], bundle["expected"]
    values = METRICS + ["train_maps", "train_pixels", "parameters", "runtime_seconds"]
    folds, summary = hierarchical(runs, expected, ["condition"], values)
    paired = paired_ablations(runs)
    delta_folds, delta_summary = hierarchical(paired, expected, ["condition"], METRICS)
    noise_folds, noise_summary = hierarchical(bundle["noise"], expected, ["condition", "noise_std"], METRICS)
    class_folds, class_summary = hierarchical(bundle["classes"], expected, ["condition", "label"], ["precision", "recall", "f1", "support"])
    map_means, _ = hierarchical(bundle["maps"], expected, ["condition", "map_id"], METRICS)
    sizes = pd.DataFrame([dict(fold_id=f["fold_id"], **{k: c[k] for k in ["level", "map_count", "pixels", "actual_map_fraction", "target_fraction"]})
                          for f in bundle["splits"]["folds"] for c in f["learning_curve"]])
    curve = pd.DataFrame()
    if not folds.empty:
        curve = folds[folds.condition.isin(["architecture:cnn", "learning_curve:maps60", "learning_curve:maps80"])].copy()
        curve["level"] = curve.condition.map({"architecture:cnn": "maps100", "learning_curve:maps60": "maps60", "learning_curve:maps80": "maps80"})
        curve = curve.merge(sizes, on=["fold_id", "level"], validate="one_to_one")
    bins = []
    for model, arrays in bundle["pooled"].items():
        if len(arrays) != 3:
            continue
        indices = np.concatenate([a[0] for a in arrays])
        require(len(set(indices)) == len(indices), "可靠性图汇集外折重复计入像素")
        for row in reliability_bins(np.concatenate([a[1] for a in arrays]), np.concatenate([a[2] for a in arrays])):
            bins.append(dict(model=model, seed=101, weighting="pooled_outer_pixels", **row))
    tables = dict(run_metrics=runs, condition_fold_means=folds, condition_summary=summary,
                  paired_ablation_runs=paired, paired_ablation_fold_means=delta_folds, paired_ablation_summary=delta_summary,
                  noise_run_metrics=bundle["noise"], noise_fold_means=noise_folds, noise_summary=noise_summary,
                  class_run_metrics=bundle["classes"], class_fold_means=class_folds, class_summary=class_summary,
                  map_run_metrics=bundle["maps"], map_seed_means=map_means, fold_train_sizes=sizes,
                  learning_curve_fold_means=curve, reliability_seed101_bins=pd.DataFrame(bins),
                  calibration_summary=summary[[c for c in summary.columns if c in ["condition", "complete", "n_complete_folds"] or
                                                c.startswith(("ece_", "negative_log_", "multiclass_brier"))]] if not summary.empty else pd.DataFrame())
    for name, frame in tables.items():
        frame.to_csv(out / (name + ".csv"), index=False, encoding="utf-8-sig")
    write_json(out / "run_validation.json", dict(status="complete" if not bundle["missing"] else "partial",
                expected_runs=len(expected), validated_runs=len(runs), missing_runs=bundle["missing"],
                original_test_reused=False, hashes=bundle["hashes"],
                metric_recalculation="clean predictions and probabilities checked against saved metrics",
                aggregation="mean of planned seeds within each fold, then mean and sample SD across all three folds"))
    plots = plot_results(bundle, tables, out / "figures", args.dpi)
    (out / "figure_captions.md").write_text("\n\n".join(f"## {name}\n\n{caption}" for name, caption in plots.captions), encoding="utf-8")
    qa = audit_figures(Path(__file__).resolve(), out / "figures", args.qa_tool_dir)
    write_json(out / "figure_qa.json", qa)
    write_report(root, bundle, tables, plots, qa)
    print(json.dumps(dict(status="complete" if not bundle["missing"] else "partial", validated_runs=len(runs),
                         expected_runs=len(expected), figures=len(plots.captions), output=str(out), figure_qa=qa["status"]), ensure_ascii=False))
    if args.qa_tool_dir and any(c["returncode"] != 0 for c in qa["checks"]):
        raise SystemExit("图形审计未通过，报告中已明确标识；请修复后重新运行。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--dpi", type=int, default=400)
    parser.add_argument("--qa-tool-dir", type=Path)
    main(parser.parse_args())
