# step3_sft
- `bin/step3` : 学習ジョブ投入ラッパ
- `slurm/sft_step3_messages.sbatch` : 実体（FSDP SFT）
- `src/llm_bridge_prod` : 参照ソース（各自ディレクトリ優先）
- `configs/` : 実行設定（llm_bridge_prod の YAML）
- `env/` : 環境再現メモ

## 事前
export SHARE_ENV=/path/to/share_env
export CONDA_SH=/path/to/conda.sh
# 任意: export WANDB_API_KEY=...

## 実行例
# 初回だけ HOME に配布
cp step3_sft/bin/step3 ~/bin/
cp step3_sft/slurm/sft_step3_messages.sbatch ~/slurm_jobs/
chmod +x ~/bin/step3

# スモーク
step3 --model Qwen/Qwen3-0.6B --head 512 --time 00:40:00 -c 2 --mbs 2 --wandb online
