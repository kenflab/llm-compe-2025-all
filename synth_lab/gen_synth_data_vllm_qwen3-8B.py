import os
from datetime import datetime
from vllm import SamplingParams, LLM
import json
import random
import re
import sys
from transformers import AutoTokenizer
args = sys.argv
import time

#Job停止の機能はコメントアウト
# コマンドライン引数からジョブIDを取得（複数プロセスでの並列実行管理用）
# job_id = args[1]
# flag_file_path = f"flags/{job_id}.txt"

# def load_flag():
#     """
#     フラグファイルを読み込んで、処理を継続するかどうかを判定する関数
#     外部からプロセスを停止させるための仕組み
#     """
#     with open(flag_file_path, "r") as f:
#         flag = f.read().strip()

#     print("flag: ", flag)
#     print("flag==1: ", flag == "1")

#     return flag == "1"

# # フラグファイルに"1"を書き込んで処理開始を記録
# with open(flag_file_path, "w") as f:
#     f.write("1")

# 大学数学の分野名を定義（論理推論や問題解決に関連する分野を中心に選択）
math_genre_text = """Linear Algebra
Group Theory
Abstract Algebra
Real Analysis
Complex Analysis
Topology
Differential Equations
Number Theory
Combinatorics
Graph Theory
Probability Theory
Statistics
Calculus
Vector Calculus
Functional Analysis
Measure Theory
Set Theory
Logic
Discrete Mathematics
Optimization
Numerical Analysis
Differential Geometry
Algebraic Geometry
Homological Algebra
Ring Theory
Field Theory
Galois Theory
Category Theory
Mathematical Logic
Model Theory
Computational Complexity
Algorithm Analysis
Information Theory
Game Theory
Operations Research
Mathematical Physics
Quantum Mechanics
Relativity Theory
Fluid Dynamics
Partial Differential Equations
Ordinary Differential Equations
Fourier Analysis
Harmonic Analysis
Spectral Theory
Operator Theory
Banach Spaces
Hilbert Spaces
Metric Spaces
Normed Spaces
Inner Product Spaces"""

# 改行で分割してリスト化し、空文字列を除去
math_genre_list = math_genre_text.split("\n")
math_genre_list = [i for i in math_genre_list if i != ""]

def get_longest_phrase_length(text):
    """
    テキスト内の最長フレーズの長さを測定する関数
    異常に長いフレーズ（壊れた出力）を検出するために使用
    """
    # 区切り文字として、スペース、カンマ、句読点、改行を指定
    delimiters = r'[ ,.!?;\n]'
    # テキストを区切り文字で分割
    try:
        phrases = re.split(delimiters, text)
        # 最大のフレーズの長さを取得
        max_length = max(len(phrase) for phrase in phrases)
    except:
        max_length = 9999
    return max_length

def is_abnormal_text(text, threshold=40):
    """
    テキストが異常かどうかを判定する関数
    単語数と句読点の比率で異常なテキストを検出
    """
    words = text.split()
    word_count = len(words)
    # 複数の区切り文字をカウント
    period_count = text.count('.') + text.count(',') + text.count(';') + text.count('!') + text.count('?')
    ratio = word_count / period_count if period_count > 0 else word_count
    return ratio > threshold

# パラメータ設定
n_turns = 1  # 1ターンのみ生成
batch_size = 30  # 一度に30個の会話を並列処理
out_dir = "university_math_out"  # 出力ディレクトリ

# プロセスIDとタイムスタンプでユニークなシードを生成
pid = os.getpid()
seed = int(pid) + int(datetime.now().timestamp())
print("seed: ", seed)
random.seed(seed)

# 出力ディレクトリを作成
os.system(f"mkdir -p {out_dir}")

# 出力ファイル名を生成（タイムスタンプとランダム数を含める）
current_time_no_symbols = datetime.now().strftime(
    "%Y-%m-%d %H:%M:%S").replace("-", "").replace(":", "").replace(" ", "")
out_path = f"{out_dir}/university_math_{current_time_no_symbols}_{random.randint(0,10000)}.jsonl"

# LLMモデルの初期化
print("LLMモデルの初期化:数分かかります。")

model_name = "Qwen/Qwen3-8B"  # Qwen3-8B
tensor_parallel_size = 2
try:
    llm = LLM(model=model_name, trust_remote_code=True,
            max_model_len=2048,  # 最大入力長を2048トークンに制限
            tensor_parallel_size=tensor_parallel_size,
            )
    print("vllm OK")
except:
    # Vllmのバージョンが古いと，transformersを使う．直接 LLM コンストラクタに渡す
    print("vllmのバージョンが古いため，transformersを使います")
    llm = LLM(
        model="Qwen/Qwen3-8B",
        trust_remote_code=True,
        model_impl="transformers",      # Transformers バックエンドを強制
        max_model_len=2048,
        tensor_parallel_size=2,
    )

# トークナイザーを初期化
tokenizer = AutoTokenizer.from_pretrained(model_name)
print("tokenizerロード完了")

def llm_gen(llm, prompt_list, temperature=0.7, top_p=0.8, top_k=20, min_p=0):
    """
    LLMを使用してバッチ処理で複数のプロンプトを同時に生成する関数
    
    Args:
        llm: 初期化されたLLMオブジェクト
        prompt_list: 生成したいプロンプトのリスト
        temperature: 生成の創造性を制御（0.7は創造性と一貫性のバランス）
        top_k: 上位k個のトークンのみから選択
        top_p: 上位p%のトークンのみから選択
        min_p: 最小確率
    
    Returns:
        生成されたテキストのリスト
    """
    outputs = llm.generate(
        prompt_list,
        sampling_params=SamplingParams(
            temperature=temperature,
            max_tokens=16384,  
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
        )
    )
    return [i.outputs[0].text.strip() for i in outputs]

print("モデルロード完了")
# %%
def question_to_prompt(question, role, history=[], enable_thinking=False):
    """
    tokenizer.apply_chat_templateを使ってチャットフォーマットでプロンプトを生成する関数
    
    Args:
        question: 質問内容
        role: システムロール（今回は数学の専門家として設定）
        history: 会話履歴（1ターンのみなので空）
    
    Returns:
        フォーマットされたプロンプト文字列
    """
    # メッセージリストを構築
    messages = []
    
    # 会話履歴がある場合は追加
    for q, a in history:
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": a})
    
    # 現在の質問を追加
    messages.append({"role": "user", "content": role + "\n" + question})
    
    # tokenizer.apply_chat_templateを使ってフォーマット
    prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    return prompt

# %%

# メイン処理ループ
while True:
    print("生成ループスタート")

    #Job停止の機能はコメントアウト
    # # フラグをチェックして処理を継続するかどうかを判定
    # flag = load_flag()
    # if flag:
    #     print("flag is true. continue processing")
    # else:
    #     print("flag!=1. finish data processing ")
    #     raise ValueError("finish!")
    
    # シードを更新（毎回異なる結果を得るため）
    seed = int(pid) + int(datetime.now().timestamp())
    print("seed: ", seed)
    random.seed(seed)
    
    # 並列会話の初期化（batch_size個の会話を同時に処理）
    parallel_conversations = [{"qid": i, "conversations": []} for i in range(batch_size)]
    
    # 1ターンのみ処理（n_turns=1）
    for turn_id in range(n_turns):
        # フラグを再チェック
        # flag = load_flag()
        # if flag:
        #     print("flag is true. continue processing")
        # else:
        #     print("flag!=1. finish data processing ")
        #     raise ValueError("finish!")
        
        print("turn_id", turn_id)
        
        # 質問生成（数学の分野に関する問題やクイズを生成）
        prompt_list = []
        for qid in range(len(parallel_conversations)):
            # ランダムに数学の分野を選択
            genre = random.choice(math_genre_list)
            # 数学の専門家としてのロールを設定
            role = f"You are a mathematics professor specializing in {genre}. You are knowledgeable, precise, and passionate about teaching mathematics."
            # 分野に関する問題やクイズを生成するよう指示
            command = f"Create one challenging problem or quiz related to {genre}. Provide only the question or instruction, nothing else."
            prompt_list.append(question_to_prompt(command, role))
        
        print("Generated prompts (first 3):", prompt_list[:3])
        question_list = llm_gen(llm, prompt_list)
        
        # 回答生成（数学の問題に対する解答を生成）
        prompt_list = []
        for qid in range(len(parallel_conversations)):
            # 数学の専門家としてのロールを設定
            role = f"You are a mathematics professor. You are knowledgeable, precise, and passionate about teaching mathematics."
            # 段階的に解答するよう指示
            command = f"Answer the following question in English. First, outline your approach to solving this problem. Then provide a step-by-step solution."
            prompt_list.append(question_to_prompt(question_list[qid], role, parallel_conversations[qid]["conversations"], enable_thinking=True))
        
        # 回答を生成
        answer_list = llm_gen(llm, prompt_list)
        
        # 会話履歴に質問と回答を追加
        for qid in range(len(parallel_conversations)):
            parallel_conversations[qid]["conversations"].append((question_list[qid], answer_list[qid]))

    # データの書き出し処理
    for record in parallel_conversations:
        conversation_list = []
        text = ""
        remove_flag = False
        
        # 各会話の品質チェック
        for q, a in record["conversations"]:
            # 長すぎるフレーズがある場合は除外（壊れた出力の可能性）
            if get_longest_phrase_length(q) > 100 or get_longest_phrase_length(a) > 100:
                remove_flag = True
                break
            
            # 異常なテキスト（句読点が少なすぎるなど）の場合は除外
            if is_abnormal_text(q) or is_abnormal_text(a):
                remove_flag = True
                break

            # テキスト形式で会話を記録
            text += f"""user: {q} assistant: {a}\n"""
            conversation_list.append({"role": "user", "content": q})
            conversation_list.append({"role": "assistant", "content": a})
        
        # テキストの前後の空白を除去
        text = text.strip()
        #record["text"] = text
        record["user"] = q
        record["assistant"] = a
        
        # 空のテキストや品質チェックで除外された場合はスキップ
        if text == "" or remove_flag:
            continue

        # 会話履歴を削除（最終的なレコードには不要）
        record.pop("conversations")

        # JSONL形式でファイルに保存（日本語文字も正しく保存）
        with open(out_path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n") 