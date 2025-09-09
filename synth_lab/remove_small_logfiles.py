import os

# ====== 設定 ======
target_dir = "/home/Competition2025/P07/shareP07/K_Nishizawa/synth_lab/logs"  # 対象ディレクトリに置き換えてください
# keyword = "6-Synth_gen_DS-P_V2-671B"
# keyword = "8-Synth_gen_DS-P_V2-671B"
# max_lines = 215
keyword = "DeepSeek-Prover-V2-671B_4bit"
max_lines = 35
# =================

deleted_files = []

for root, dirs, files in os.walk(target_dir):
    for file in files:
        if keyword in file and file.endswith(".log"):
            file_path = os.path.join(root, file)
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    line_count = sum(1 for _ in f)
                if line_count <= max_lines:
                    os.remove(file_path)
                    deleted_files.append((file_path, line_count))
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

# 結果表示
if deleted_files:
    print("削除したファイル:")
    for path, lines in deleted_files:
        print(f"{path} ({lines} 行)")
else:
    print("削除対象のファイルはありませんでした。")
