#!/bin/bash

#SBATCH --job-name=4-Synth_Qwen3_32b_from_DeepMath_69_4     # ジョブ名
#SBATCH --partition=P07                                     # パーティション
#SBATCH --nodelist=osk-gpu69                                # 指定ノード
#SBATCH --nodes=1                                           # ノード数
#SBATCH --ntasks=1                                          # MPI 等で使うタスク数（単一プロセスなら1）
#SBATCH --cpus-per-task=64                                  # 各タスクあたりの CPU スレッド数
#SBATCH --gpus-per-node=4                                   # ノードあたりの GPU 数 (#SBATCH --gres=gpu:1 と同義)
#SBATCH --mem=512G                                          # ノード全体のメモリ（必要に応じて調整）
#SBATCH --time=24:00:00                                     # 最大実行時間
#SBATCH --output=%x_%j.log                                  # 出力ログ（%x=ジョブ名, %j=ジョブID）

# === モジュール初期化とロード ===
module purge

module load miniconda/24.7.1-py312
module load cuda/12.4
module load cudnn/9.6.0
module load nccl/2.20.5


# === 状態確認 ===
echo "✅ Environment setup complete"
echo "Conda env: $(conda info --envs | grep '*' | awk '{print $1}')"
echo "Loaded modules:"
module list

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Conda 環境のアクティベート
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# conda activate を使う前にフックを読み込む
# 1) which conda で場所を掴み、そこから etc/profile.d/conda.sh を source
CONDA_BIN=$(which conda)
if [ -z "$CONDA_BIN" ]; then
  echo "❌ conda コマンドが見つかりません. モジュールのロードを確認してください."
  exit 1
fi
CONDA_BASE=$(dirname $(dirname "$CONDA_BIN"))
source "$CONDA_BASE/etc/profile.d/conda.sh"
echo "✅ Sourced conda.sh from $CONDA_BASE"

conda activate /home/Competition2025/P07/shareP07/share_env_KN/Synth_gen_vllm
echo "✅ Sourced activate /home/Competition2025/P07/shareP07/share_env_KN/Synth_gen_vllm"

export CUDA_HOME=/home/appli/cuda/12.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

# === 実行コマンド ===
echo "▶︎ Starting inference script at $(date)"
srun python gen_synth_data_qwen3-32B_from_DeepMath-103K.py \
  --input-jsonl /home/Competition2025/P07/shareP07/K_Nishizawa/synth_lab/SeedData/DeepMath103k/0000.jsonl \
  --cache-path  /home/Competition2025/P07/shareP07/K_Nishizawa/synth_lab/processed_cache/DeepMath103k/0000.log \
  --batch-size  4
echo "▶︎ Finished at $(date)"