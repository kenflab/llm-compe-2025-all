#!/bin/bash
#SBATCH --job-name=step2_eval
#SBATCH --partition=P07
#SBATCH --nodelist=osk-gpu69
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=6:00:00
#SBATCH --output=/home/Competition2025/P07/YOUR_FOLDER/step2-evaluation/logs/%x-%j.out
#SBATCH --error=/home/Competition2025/P07/YOUR_FOLDER/step2-evaluation/logs/%x-%j.err

#--- Module & Conda --------------------------------------------
module purge
module load cuda/12.6 miniconda/24.7.1-py312
module load cudnn/9.6.0
module load nccl/2.24.3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /home/Competition2025/P07/shareP07/share_env_llmbench/llmbench

# Configuration files
LOGP_CFG_PATH=/home/Competition2025/P07/YOUR_FOLDER/step2-evaluation/conf/config_logp_eval.yaml
RUBRIC_CFG_PATH=/home/Competition2025/P07/YOUR_FOLDER/step2-evaluation/conf/config.yaml

# Set Hugging Face token
export HF_TOKEN=$(yq e '.hf_token' $LOGP_CFG_PATH)
export HF_HOME=${SLURM_TMPDIR:-$HOME}/.hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export HUGGINGFACE_HUB_TOKEN=$HF_TOKEN
mkdir -p "$HF_HOME"

# Extract configuration information
LOGP_MODEL=$(yq e '.model' $LOGP_CFG_PATH)
LOGP_INPUT_DATASET=$(yq e '.dataset' $LOGP_CFG_PATH)
LOGP_OUTPUT_DATASET=$(yq e '.hf_hub_repo' $LOGP_CFG_PATH)

RUBRIC_MODEL=$(yq e '.vllm_serve.model' $RUBRIC_CFG_PATH)
RUBRIC_INPUT_DATASET=$(yq e '.dataset' $RUBRIC_CFG_PATH)
RUBRIC_OUTPUT_DATASET=$(yq e '.hf_hub_repo' $RUBRIC_CFG_PATH)

echo "=================================================="
echo "Complete Evaluation Pipeline Start"
echo "Time: $(date)"
echo "=================================================="
echo "Step 1: LogP Evaluation"
echo "  Config: $LOGP_CFG_PATH"
echo "  Model: $LOGP_MODEL"
echo "  Input Dataset: $LOGP_INPUT_DATASET"
echo "  Output Dataset: $LOGP_OUTPUT_DATASET"
echo ""
echo "Step 2: Rubric Evaluation"
echo "  Config: $RUBRIC_CFG_PATH"
echo "  Model: $RUBRIC_MODEL"
echo "  Input Dataset: $RUBRIC_INPUT_DATASET"
echo "  Output Dataset: $RUBRIC_OUTPUT_DATASET"
echo "=================================================="

#--- GPU Monitoring -------------------------------------------------------
nvidia-smi -i 0,1 -l 10 > nvidia-smi.log &
pid_nvsmi=$!

# Function to cleanup processes
cleanup() {
    echo "Cleaning up processes..."
    if [ -n "$pid_vllm" ]; then
        kill $pid_vllm 2>/dev/null
    fi
    kill $pid_nvsmi 2>/dev/null
    wait
}

# Set trap for cleanup on exit
trap cleanup EXIT

#=============================================================================
# STEP 1: LogP Evaluation using HuggingFace Direct Inference
#=============================================================================
echo ""
echo "=== STEP 1: LogP Evaluation ==="
echo "Using HuggingFace Transformers directly for LogP evaluation"
echo "No vLLM server required - direct model inference"

# Create step1 output directory
mkdir -p outputs/step1_logp

# Run LogP evaluation
echo "Starting LogP evaluation with HF direct inference..."
python run_logp_evaluation.py > outputs/step1_logp/evaluation_logp.log 2>&1
logp_status=$?

if [ $logp_status -ne 0 ]; then
    echo "ERROR: LogP evaluation failed with status $logp_status"
    echo "Check outputs/step1_logp/evaluation_logp.log for details"
    exit $logp_status
fi

echo "LogP evaluation completed successfully!"

# Show LogP results summary
if [ -f "outputs/evaluation_stats.json" ]; then
    echo ""
    echo "=== LogP Evaluation Results ==="
    cat outputs/evaluation_stats.json | python -m json.tool
    
    # Move LogP results to step1 directory
    mv outputs/evaluation_stats.json outputs/step1_logp/
    mv outputs/logp_evaluated_hf_direct_*.jsonl outputs/step1_logp/ 2>/dev/null || true
fi

# Verify LogP dataset was uploaded to HuggingFace
echo ""
echo "Verifying LogP dataset upload..."
echo "LogP dataset should be available at: $LOGP_OUTPUT_DATASET"

# Wait a bit for HuggingFace to process the upload
echo "Waiting 30 seconds for HuggingFace processing..."
sleep 30

#=============================================================================
# STEP 2: Rubric Evaluation using vLLM
#=============================================================================
echo ""
echo "=== STEP 2: Rubric Evaluation ==="
echo "Starting vLLM server for rubric evaluation..."

# vLLM serve parameters for rubric evaluation
TP_SIZE=$(yq e '.vllm_serve.tensor_parallel_size' $RUBRIC_CFG_PATH)
MAX_LEN=$(yq e '.vllm_serve.max_model_len' $RUBRIC_CFG_PATH)
GPU_UTIL=$(yq e '.vllm_serve.gpu_memory_utilization' $RUBRIC_CFG_PATH)

echo "VLLM_MODEL = $RUBRIC_MODEL"
echo "TP_SIZE = $TP_SIZE"

# Create step2 output directory
mkdir -p outputs/step2_rubric

#--- Start vLLM Server for Rubric Evaluation -----------------------------------------------
vllm serve "$RUBRIC_MODEL" \
  --tensor-parallel-size "$TP_SIZE" \
  --max-model-len "$MAX_LEN" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --port 8000 \
  > outputs/step2_rubric/vllm.log 2>&1 &
pid_vllm=$!

#--- Health Check -------------------------------------------------
echo "Waiting for vLLM server to start..."
max_wait=300
waited=0
until curl -s http://127.0.0.1:8000/health >/dev/null; do
  if [ $waited -ge $max_wait ]; then
    echo "ERROR: vLLM server failed to start after ${max_wait}s"
    echo "Check outputs/step2_rubric/vllm.log for details"
    exit 1
  fi
  echo "$(date +%T) vLLM starting... ($waited/$max_wait)"
  sleep 10
  waited=$((waited + 10))
done
echo "vLLM server is ready!"

#--- Run Rubric Evaluation -----------------------------------------------------------
echo "Starting rubric evaluation on LogP-evaluated dataset..."
python predict.py > outputs/step2_rubric/evaluation.log 2>&1
rubric_status=$?

# Stop vLLM server
echo "Shutting down vLLM server..."
kill $pid_vllm 2>/dev/null
pid_vllm=""

if [ $rubric_status -ne 0 ]; then
    echo "ERROR: Rubric evaluation failed with status $rubric_status"
    echo "Check outputs/step2_rubric/evaluation.log for details"
    exit $rubric_status
fi

echo "Rubric evaluation completed successfully!"

# Show rubric results summary
if [ -f "outputs/evaluation_statistics.json" ]; then
    echo ""
    echo "=== Rubric Evaluation Results ==="
    cat outputs/evaluation_statistics.json | python -m json.tool
    
    # Move rubric results to step2 directory
    mv outputs/evaluation_statistics.json outputs/step2_rubric/
    mv outputs/evaluation_results.jsonl outputs/step2_rubric/ 2>/dev/null || true
    mv outputs/intermediate_results.jsonl outputs/step2_rubric/ 2>/dev/null || true
    mv outputs/debug_responses outputs/step2_rubric/ 2>/dev/null || true
fi

#=============================================================================
# PIPELINE COMPLETION
#=============================================================================
echo ""
echo "=================================================="
echo "Complete Evaluation Pipeline Finished"
echo "Time: $(date)"
echo "=================================================="
echo "Status Summary:"
echo "  LogP Evaluation: SUCCESS"
echo "  Rubric Evaluation: SUCCESS"
echo ""
echo "Output Datasets:"
echo "  LogP Dataset: $LOGP_OUTPUT_DATASET"
echo "  Final Dataset: $RUBRIC_OUTPUT_DATASET"
echo ""
echo "Local Output Files:"
echo "  Step 1 (LogP): outputs/step1_logp/"
echo "  Step 2 (Rubric): outputs/step2_rubric/"
echo ""

# Show final directory structure
echo "=== Output Directory Structure ==="
find outputs -type f -name "*.log" -o -name "*.json" -o -name "*.jsonl" | sort

echo ""
echo "Pipeline completed successfully!"
echo "Final dataset with both LogP and Rubric scores: $RUBRIC_OUTPUT_DATASET"

exit 0