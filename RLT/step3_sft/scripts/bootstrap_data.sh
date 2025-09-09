#!/usr/bin/env bash
set -euo pipefail
dst="$HOME/data_step3/gsm8k"
mkdir -p "$dst"

# 共有の tiny セットがあればそれを配る（なければメッセージを出して終了）
src="/home/Competition2025/P07/shareP07/step3_dist/data_step3/gsm8k"
if compgen -G "$src/*.parquet" > /dev/null; then
  cp -n "$src"/*.parquet "$dst/"
  echo "[OK] tiny dataset copied to $dst"
  exit 0
fi

echo "[FATAL] tiny dataset not found at: $src"
echo " - Step2 の Parquet を \$HOME/data/gsm8k/{train,test}.parquet に置くか"
echo " - #step3-support でデータ受け取り方法を確認してください。"
exit 2
