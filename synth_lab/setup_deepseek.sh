#!/bin/bash
# setup_deepseek.sh

# === エラーハンドリング ===
set -e

# === モジュール初期化とロード ===
module purge

# module load miniconda/24.7.1-py312
module load cuda/12.4
module load cudnn/9.6.0
module load nccl/2.20.5

# === 状態確認 ===
echo "✅ Environment setup complete"
echo "Conda env: $(conda info --envs | grep '*' | awk '{print $1}')"
echo "Loaded modules:"
module list

conda activate deepseek_v3

export CUDA_HOME=/home/appli/cuda/12.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export BITSANDBYTES_CUDA_HOME="$CUDA_HOME"
export BITSANDBYTES_CUDA_VERSION=124

