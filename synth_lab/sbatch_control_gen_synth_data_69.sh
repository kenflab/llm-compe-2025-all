#!/usr/bin/env bash
#SBATCH --job-name=Con_gen_synth_DS-P_V2-671B_N69     # ジョブ名
#SBATCH --partition=P07                               # パーティション
#SBATCH --nodelist=osk-gpu69                          # 指定ノード
#SBATCH --nodes=1                                     # ノード数
#SBATCH --ntasks=1                                    # 単一プロセスなら1
#SBATCH --cpus-per-task=2                             # 各タスクあたりの CPU スレッド数
#SBATCH --gpus-per-node=0                             # Slurm 管理上は --gres=gpu:1 と同義
#SBATCH --mem=4G                                      # ノード全体のメモリ
#SBATCH --time=72:00:00                               # 最大実行時間
#SBATCH --output=%x_%j.log                            # 出力ログ

echo " Start "

set -euo pipefail
trap 'echo " Finish "' EXIT

#─── ユーザー設定 ───────────────────────────────────────────
PARTITION="P07"
USER_TARGET="kan.hatakeyama"
SLEEP_INTERVAL=60
SCANCEL_SH="/home/Competition2025/P07/shareP07/scripts/scancel.sh"
SBATCH_SCRIPT="/home/Competition2025/P07/shareP07/K_Nishizawa/synth_lab/sbatch_gen_synth_data_qwen3_32b_from_DeepMath_69_4.sh"
RUN_FLAG="/home/Competition2025/P07/shareP07/share_gen_synth-data_flag/node_69"
#──────────────────────────────────────────────────────────

launched_job=""

while true; do
  # ── 0) RUN_FLAG をチェック ─────────────────────────────────
  # ファイルが無ければ「1（稼働可）」扱い
  flag=$(<"$RUN_FLAG" 2>/dev/null || echo "1")
  if [[ "$flag" == "0" ]]; then
    if [[ -n "$launched_job" ]]; then
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] RUN_FLAG=0 detected → cancelling own job $launched_job"
      "$SCANCEL_SH" "$launched_job"
      launched_job=""
    fi
    sleep "$SLEEP_INTERVAL"
    continue
  fi

  # ── 1) USER_TARGET の全ジョブをキャンセル ───────────────────────
  mapfile -t all_jobs < <(squeue -h -p "$PARTITION" -u "$USER_TARGET" -o "%A")
  if (( ${#all_jobs[@]} )); then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Cancelling all jobs for $USER_TARGET: ${all_jobs[*]}"
    for jid in "${all_jobs[@]}"; do
      echo " → cancelling $jid"
      "$SCANCEL_SH" "$jid"
      sleep 15
    done
  fi

  # ── 2) 自身ジョブのステータス確認＆投入判定 ─────────────────
  if [[ -n "$launched_job" ]]; then
    # running or pending?
    status=$(squeue -h -j "$launched_job" -o "%T")
    if [[ -n "$status" ]]; then
      echo "[$(date '+%H:%M:%S')] Own job $launched_job is still in state: $status"
    else
      echo "[$(date '+%H:%M:%S')] Own job $launched_job not found. Will submit new job."
      launched_job=""
    fi
  fi

  if [[ -z "$launched_job" ]]; then
    out=$(sbatch "$SBATCH_SCRIPT")
    launched_job=$(awk '{print $4}' <<<"$out")
    echo "[$(date '+%H:%M:%S')] Launched new synthesis job: $launched_job"
  fi

  sleep "$SLEEP_INTERVAL"
done
