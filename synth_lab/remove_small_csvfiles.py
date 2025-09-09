import os

# ====== 設定 ======
target_dir = "/home/Competition2025/P07/shareP07/K_Nishizawa/synth_lab/university_math_out"  # 対象ディレクトリに置き換えてください
keyword = "university_math"  # ファイル名に含まれる文字列
max_chars = 5                # 削除対象となる最大行数
# =================

deleted_files = []

for root, dirs, files in os.walk(target_dir):
    for file in files:
        if keyword in file and (file.lower().endswith(".json") or file.lower().endswith(".jsonl")):
            file_path = os.path.join(root, file)
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                char_count = len(content)
                if char_count <= max_chars:
                    os.remove(file_path)
                    deleted_files.append((file_path, char_count))
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

# 結果表示
if deleted_files:
    print("削除したJSON/JSONLファイル:")
    for path, chars in deleted_files:
        print(f"{path} ({chars} 文字)")
else:
    print("削除対象のJSON/JSONLファイルはありませんでした。")