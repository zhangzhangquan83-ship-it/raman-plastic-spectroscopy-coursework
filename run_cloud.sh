#!/usr/bin/env bash
# 云端离线计算批次；不会启动Web服务或操作云实例开关机。
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export PYTHONDONTWRITEBYTECODE=1
if [[ -f data/processed/dataset.npz ]]; then
    python src/prepare_data.py --reuse-processed
else
    python src/prepare_data.py
fi
python src/make_split.py
python src/sanity_check.py --device cuda
python src/train_baselines.py
for seed in 42 43 44; do
    python src/train_cnn.py --device cuda --epochs 80 --batch-size 256 --seed "$seed" --augment --cpu-threads 8
done
python src/train_cnn.py --device cuda --epochs 80 --batch-size 256 --seed 42 --no-augment --cpu-threads 8
python src/evaluate.py --device cuda
python src/build_report.py
printf '%s\n' '全部正式实验和独立测试完成。请备份results及扩大版实验结果.md。'
