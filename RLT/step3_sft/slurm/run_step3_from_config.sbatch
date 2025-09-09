#!/bin/bash
#SBATCH --partition=P07
export ADDR2LINE=${ADDR2LINE:-addr2line}
#!/bin/bash
#SBATCH --job-name=step3_from_cfg
#SBATCH --partition=P07
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=step3_from_cfg_%j.log
set -eEuo pipefail

# --- 環境（既存sbatchと揃える） ---
module reset
module load nccl/2.22.3
module load hpcx/2.18.1-gcc-cuda12/hpcx-mt
module load miniconda/24.7.1-py311
source /home/appli/miniconda3/24.7.1-py311/etc/profile.d/conda.sh

# ログインノード負荷対策
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export TORCH_COMPILE_DISABLE=1
export NVTE_FUSED_ATTN=0
export NCCL_IB_DISABLE=1

CFG="${STEP3_CONFIG:-$HOME/repos/RLT/step3_sft/conf/step3_config.yaml}"
export CFG

# --- YAML 読み込み → export 生成 ---
python - <<'PY' > /tmp/step3_env.$$
import os, sys
try:
    import yaml
except Exception:
    print("[FATAL] PyYAML が見つかりません。共有環境で実行してください。", file=sys.stderr)
    sys.exit(3)

cfg_path = os.environ["CFG"]
with open(cfg_path, "r") as f:
    C = yaml.safe_load(f)

def setdefault(k, v):
    if os.environ.get(k) is None:
        os.environ[k] = "" if v is None else str(v)

# モデル/データ
setdefault("MODEL", (C.get("model") or ""))
hf = C.get("hf_dataset") or {}
setdefault("HF_DATASET_ID",    hf.get("id") or "")
setdefault("HF_DATASET_SPLIT", hf.get("split") or "train")
setdefault("HF_DATASET_NAME",  ("" if hf.get("name") in (None,"null") else hf.get("name")))

# 学習
tr = C.get("training") or {}
setdefault("HEAD_N",  tr.get("head_n") or "0")
setdefault("EPOCHS",  tr.get("epochs") or "1")
setdefault("MBS",     tr.get("micro_batch_size") or "4")
setdefault("LR",      tr.get("lr") or "1e-5")
setdefault("RESUME",  os.environ.get("RESUME","disable"))

# W&B
wb = C.get("wandb") or {}
setdefault("WANDB_MODE",     wb.get("mode") or "online")
setdefault("WANDB_ENTITY",   wb.get("entity") or "llm-compe-2025-kato")
setdefault("WANDB_PROJECT",  wb.get("project") or "step3_sft")
run_name = wb.get("run_name") or f"step3-{os.environ.get('MODEL','model').split('/')[-1]}"
setdefault("WANDB_RUN_NAME", run_name)
setdefault("WANDB_GROUP",    wb.get("group") or "")
setdefault("WANDB_NOTES",    wb.get("notes") or "")

# 安全: online なのにキー未設定なら明確に止める
if os.environ.get("WANDB_MODE") == "online" and not os.environ.get("WANDB_API_KEY"):
    print("[FATAL] WANDB_MODE=online ですが WANDB_API_KEY が未設定です。 export WANDB_API_KEY=... を実施してください。", file=sys.stderr)
    sys.exit(4)

# export を吐く
for k in ["MODEL","HF_DATASET_ID","HF_DATASET_SPLIT","HF_DATASET_NAME",
          "HEAD_N","EPOCHS","MBS","LR","RESUME",
          "WANDB_MODE","WANDB_ENTITY","WANDB_PROJECT","WANDB_RUN_NAME","WANDB_GROUP","WANDB_NOTES"]:
    v = os.environ.get(k,"")
    print(f'export {k}="{v}"')
PY

source /tmp/step3_env.$$
rm -f /tmp/step3_env.$$

# 共有環境パス（既存 sbatch と揃える）
export SHARE_ENV="/home/Competition2025/P07/shareP07/share_env"

# ===== 実体の学習スクリプトを“この割当内”で実行（投げない） =====
bash "$ROOT/slurm/sft_step3_messages.sbatch"
# ============================================================

echo "[info] run_step3_from_config.sbatch done."
