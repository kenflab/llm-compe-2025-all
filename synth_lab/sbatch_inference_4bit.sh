#!/bin/bash

#SBATCH --job-name=DS-P_V2-671B           # ジョブ名
#SBATCH --partition=P07                   # パーティション
#SBATCH --nodelist=osk-gpu70              # 指定ノード
#SBATCH --nodes=1                         # ノード数
#SBATCH --ntasks=1                        # MPI 等で使うタスク数（単一プロセスなら1）
#SBATCH --cpus-per-task=200               # 各タスクあたりの CPU スレッド数
#SBATCH --gpus-per-node=8                 # ノードあたりの GPU 数 (#SBATCH --gres=gpu:1 と同義)
#SBATCH --mem=1300G                       # ノード全体のメモリ（必要に応じて調整）
#SBATCH --time=08:00:00                   # 最大実行時間
#SBATCH --output=%x_%j.log                # 出力ログ（%x=ジョブ名, %j=ジョブID）

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

conda activate deepseek_v3
echo "✅ Sourced activate deepseek_v3"


# === 実行コマンド ===
echo "▶︎ Starting inference script at $(date)"
srun python /home/Competition2025/P07/shareP07/K_Nishizawa/synth_lab/inference_4bit.py
echo "▶︎ Finished at $(date)"