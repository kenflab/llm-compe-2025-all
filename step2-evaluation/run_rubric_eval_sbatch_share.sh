#!/bin/bash
#SBATCH --job-name=rubric_eval
#SBATCH --partition=P07
#SBATCH --nodelist=osk-gpu69
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=09:00:00
#SBATCH --output=/home/Competition2025/P07/YOUR_FOLDER/step2-evaluation/logs/%x-%j.out
#SBATCH --error=/home/Competition2025/P07/YOUR_FOLDER/step2-evaluation/logs/%x-%j.err

#--- Module & Conda --------------------------------------------
module purge
module load cuda/12.6 miniconda/24.7.1-py312
module load cudnn/9.6.0
module load nccl/2.24.3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /home/Competition2025/P07/shareP07/share_env_llmbench/llmbench

# --- Load from config.yaml -------------------------------------
CFG_PATH=/home/Competition2025/P07/YOUR_FOLDER/step2-evaluation/conf/config.yaml

# --- Hugging Face Authentication ------------------------------------------
export HF_TOKEN=$(yq e '.hf_token' $CFG_PATH)
export HF_HOME=${SLURM_TMPDIR:-$HOME}/.hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export HUGGINGFACE_HUB_TOKEN=$HF_TOKEN
mkdir -p "$HF_HOME"
echo "HF cache dir : $HF_HOME"

# Extract dataset information
INPUT_DATASET=$(yq e '.dataset' $CFG_PATH)
OUTPUT_DATASET=$(yq e '.hf_hub_repo' $CFG_PATH)

echo "===================="
echo "Rubric Evaluation Start"
echo "Input Dataset: $INPUT_DATASET"
echo "Output Dataset: $OUTPUT_DATASET"
echo "Time: $(date)"
echo "===================="

#--- GPU Monitoring -------------------------------------------------------
nvidia-smi -i 0,1 -l 10 > nvidia-smi.log &
pid_nvsmi=$!

# vLLM serve parameters
VLLM_MODEL=$(yq e '.vllm_serve.model' $CFG_PATH)
TP_SIZE=$(yq e '.vllm_serve.tensor_parallel_size' $CFG_PATH)
MAX_LEN=$(yq e '.vllm_serve.max_model_len' $CFG_PATH)
GPU_UTIL=$(yq e '.vllm_serve.gpu_memory_utilization' $CFG_PATH)

echo "Starting rubric evaluation..."
echo "VLLM_MODEL  = $VLLM_MODEL"
echo "TP_SIZE=$TP_SIZE"

#--- Start vLLM Server -----------------------------------------------
vllm serve "$VLLM_MODEL" \
  --tensor-parallel-size "$TP_SIZE" \
  --max-model-len "$MAX_LEN" \
  --gpu-memory-utilization "$GPU_UTIL" \
  > vllm.log 2>&1 &
pid_vllm=$!

#--- Health Check -------------------------------------------------
echo "Waiting for vLLM server to start..."
max_wait=300
waited=0
until curl -s http://127.0.0.1:8000/health >/dev/null; do
  if [ $waited -ge $max_wait ]; then
    echo "ERROR: vLLM server failed to start after ${max_wait}s"
    kill $pid_vllm $pid_nvsmi
    exit 1
  fi
  echo "$(date +%T) vLLM starting... ($waited/$max_wait)"
  sleep 10
  waited=$((waited + 10))
done
echo "vLLM server is ready!"

#--- Run Evaluation -----------------------------------------------------------
echo "Starting rubric evaluation on LogP-evaluated dataset..."
python predict.py > evaluation.log 2>&1
eval_status=$?

#--- Cleanup -------------------------------------------------------
echo "Shutting down services..."
kill $pid_vllm
kill $pid_nvsmi
wait

echo "===================="
echo "Rubric Evaluation Complete"
echo "Status: $eval_status"
echo "Time: $(date)"
echo "===================="

# Show results summary
if [ $eval_status -eq 0 ]; then
    echo ""
    echo "=== Results Summary ==="
    if [ -f "outputs/evaluation_statistics.json" ]; then
        echo "Statistics:"
        cat outputs/evaluation_statistics.json | python -m json.tool
    fi
    echo ""
    echo "Output files:"
    ls -la outputs/evaluation_results.jsonl 2>/dev/null || echo "No output files found"
    echo ""
    echo "Dataset updated: $OUTPUT_DATASET"
fi

exit $eval_status