#!/bin/bash
#SBATCH --job-name=pairwise_comparison
#SBATCH --partition=P07
#SBATCH --nodelist=osk-gpu69
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/home/Competition2025/P07/自分の個人フォルダ名（例：P07U021）/step2-evaluation-pairwise/logs/%x-%j.out
#SBATCH --error=/home/Competition2025/P07/自分の個人フォルダ名（例：P07U021）/step2-evaluation-pairwise/logs/%x-%j.err

#--- モジュール & Conda --------------------------------------------
module purge
module load cuda/12.6 miniconda/24.7.1-py312
module load cudnn/9.6.0
module load nccl/2.24.3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /home/Competition2025/P07/shareP07/share_env_llmbench/llmbench

# --- load from config.yaml (requires yq) ---
CFG_PATH=/home/Competition2025/P07/自分の個人フォルダ名（例：P07U021）/step2-evaluation-pairwise/conf/config_pairwise.yaml

# Hugging Face 認証
export HF_TOKEN=$(yq e '.hf_token' $CFG_PATH)
export HF_HOME=${SLURM_TMPDIR:-$HOME}/.hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export HUGGINGFACE_HUB_TOKEN=$HF_TOKEN
mkdir -p "$HF_HOME"
echo "HF cache dir : $HF_HOME"                   # デバッグ用

#--- GPU 監視 -------------------------------------------------------
nvidia-smi -i 0,1,2,3,4,5,6,7 -l 3 > nvidia-smi.log &
pid_nvsmi=$!

# vLLM serve 用パラメータ
VLLM_MODEL=$(yq e '.vllm_serve.model' $CFG_PATH)
TP_SIZE=$(yq e '.vllm_serve.tensor_parallel_size' $CFG_PATH)
RPARSER=$(yq e '.vllm_serve.reasoning_parser' $CFG_PATH)
ROPE_SCALE=$(yq e '.vllm_serve.rope_scaling' $CFG_PATH)
MAX_LEN=$(yq e '.vllm_serve.max_model_len' $CFG_PATH)
GPU_UTIL=$(yq e '.vllm_serve.gpu_memory_utilization' $CFG_PATH)

echo "TP_SIZE=$TP_SIZE"
echo "HF_TOKEN    = ${HF_TOKEN:0:8}... (truncated)"
echo "VLLM_MODEL  = $VLLM_MODEL"

#--- vLLM 起動（2GPU for pairwise comparison）--------------------
vllm serve "$VLLM_MODEL" \
  --tensor-parallel-size "$TP_SIZE" \
  --reasoning-parser "$RPARSER" \
  --rope-scaling "$ROPE_SCALE" \
  --max-model-len "$MAX_LEN" \
  --gpu-memory-utilization "$GPU_UTIL" \
  > vllm_pairwise_comparison.log 2>&1 &
pid_vllm=$!

#--- ヘルスチェック -------------------------------------------------
until curl -s http://127.0.0.1:8000/health >/dev/null; do
  echo "$(date +%T) vLLM starting …"
  sleep 10
done
echo "vLLM READY"

#--- ペアワイズ比較実行 ---------------------------------------------
python predict_pairwise.py > predict_pairwise_comparison.log 2>&1

#--- 後片付け -------------------------------------------------------
kill $pid_vllm
kill $pid_nvsmi
wait