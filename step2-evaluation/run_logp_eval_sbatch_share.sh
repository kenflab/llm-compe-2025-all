#!/bin/bash
#SBATCH --job-name=logp_eval
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

# Configuration file
CFG_PATH=/home/Competition2025/P07/YOUR_FOLDER/step2-evaluation/conf/config_logp_eval.yaml

# Set Hugging Face token
export HF_TOKEN=$(yq e '.hf_token' $CFG_PATH)
export HF_HOME=${SLURM_TMPDIR:-$HOME}/.hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export HUGGINGFACE_HUB_TOKEN=$HF_TOKEN
mkdir -p "$HF_HOME"

# Extract model for verification
MODEL=$(yq e '.model' $CFG_PATH)

echo "===================="
echo "LogP Evaluation Start"
echo "Config: $CFG_PATH"
echo "Provider: hf_direct"
echo "Model: $MODEL"
echo "Time: $(date)"
echo "===================="

#--- GPU Monitoring -------------------------------------------------------
nvidia-smi -i 0,1 -l 10 > nvidia-smi.log &
pid_nvsmi=$!

echo "=== HuggingFace Direct Mode ==="
echo "Using HuggingFace Transformers directly for LogP evaluation"
echo "No vLLM server required - direct model inference"

# Run evaluation
echo "Starting LogP evaluation with HF direct inference..."
python run_logp_evaluation.py > evaluation_logp.log 2>&1
eval_status=$?

# Cleanup
echo "Cleaning up..."
kill $pid_nvsmi 2>/dev/null
wait

echo "===================="
echo "Evaluation Complete"
echo "Provider: hf_direct"
echo "Status: $eval_status"
echo "Time: $(date)"
echo "===================="

# Show results summary
if [ $eval_status -eq 0 ]; then
    echo ""
    echo "=== Results Summary ==="
    if [ -f "outputs/evaluation_stats.json" ]; then
        echo "Statistics:"
        cat outputs/evaluation_stats.json | python -m json.tool
    fi
    echo ""
    echo "Output files:"
    ls -la outputs/logp_evaluated_hf_direct_*.jsonl 2>/dev/null || echo "No output files found"
fi

exit $eval_status