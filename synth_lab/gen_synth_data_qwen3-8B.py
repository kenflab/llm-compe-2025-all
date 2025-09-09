#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BF16 → 4bit 量子化＋推論スクリプト
"""
# python gen_synth_data_deepseek_prover671b.py [GPU Num Int. 8 or 6] 

from pathlib import Path

model_name = "Qwen/Qwen3-8B"  # Qwen3-8B
GPU_LIMIT_GIB = 70
out_dir = "university_math_out"  # 出力ディレクトリ

import os, sys, re
# ─── 出力を行バッファリングモードに ────────────────────────────────────
if hasattr(sys.stdout, "reconfigure"):
    # Python3.7+ ならこれで OK
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

# Pillow などが使うテンポラリ (ImportError 回避用)
os.environ["TMPDIR"] = "/dev/shm/pillow_cache"
os.makedirs(os.environ["TMPDIR"], exist_ok=True)

# 0) Triton キャッシュ先を exec 可能な場所に移す
os.environ["TRITON_CACHE_DIR"] = "/dev/shm/triton_cache"
os.makedirs(os.environ["TRITON_CACHE_DIR"], exist_ok=True)

import threading, time, subprocess, logging
from datetime import datetime
import json
import traceback
import random

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
    set_seed,
    logging as hf_logging,
)
from accelerate import dispatch_model

# ─── Python ログ設定 ─────────────────────────────────────────────────────
# 現在時刻を「YYYYMMDD_HHMM」の形式で取得
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
job_id = os.getenv("SLURM_JOB_ID")                    # Slurm 内なら数値文字列

# Slurm で走っていればその JobID、そうでなければ従来の timestamp を使う
run_tag = job_id if job_id is not None else timestamp

# ── ログファイル名 ────────────────────────────────────────
LOG_FILE      = f"DeepSeek-Prover-V2-671B_4bit_{run_tag}.log"
VRAM_LOG_FILE = f"vram_usage_{run_tag}.csv"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
# Transformers ログを INFO レベル以上で出力
hf_logging.set_verbosity_debug()
logging.info("▶︎ Logging initialized; output to %s", LOG_FILE)

# ────────────────────────────────────────────────────────────────────────────
# VRAM モニタリング用ユーティリティ
# ────────────────────────────────────────────────────────────────────────────

# デバイスごとの最大使用量を保持
max_vram_usage = []
# スレッド終了フラグ
_monitor_stop = threading.Event()

def get_gpu_memory():
    """nvidia-smi で各 GPU の使用メモリ (MiB) をリストで返す"""
    out = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'], 
        encoding='utf-8'
    )
    return [int(x) for x in out.strip().splitlines()]

def _monitor_vram(log_path: str, interval: float = 5.0):
    """5 秒ごとに VRAM 使用量を取得してログ＆最大値を更新"""
    global max_vram_usage
    # 初回にデバイス数を決定して max_vram_usage を初期化
    usage = get_gpu_memory()                            # [MiB]
    usage_gib = [u / 1024 for u in usage]               # [GiB]
    max_vram_usage = usage_gib
    with open(log_path, 'w') as f:
        f.write("timestamp," + ",".join(f"GPU({i}) [GiB]" for i in range(len(usage))) + "\n")
        while not _monitor_stop.is_set():
            usage = get_gpu_memory()                            # [MiB]
            usage_gib = [u / 1024 for u in usage]               # [GiB]
            # 最大値を更新
            for i, u in enumerate(usage_gib):
                if u > max_vram_usage[i]:
                    max_vram_usage[i] = u
            # ログ出力（GiB 表示）
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            f.write(ts + "," + ",".join(f"{u:.2f}" for u in usage_gib) + "\n")
            f.flush()
            time.sleep(interval)
# ────────────────────────────────────────────────────────────────────────────

def _patched_prepare_inputs_for_generation(self, input_ids, **model_kwargs):
    # 1) まずオリジナルを呼ぶ
    outputs = _orig_prep(self, input_ids, **model_kwargs)

    # 2) attention_mask があれば長さのズレをチェック
    attn = outputs.get("attention_mask", None)
    pkvs = model_kwargs.get("past_key_values", None)
    if attn is not None and pkvs is not None:
        expected = pkvs.get_max_length()  # ここは既に get_seq_length にエイリアス済み
        actual   = attn.size(-1)
        if actual != expected:
            # 差分分だけ「1」のマスクを末尾に足す
            pad_len = expected - actual
            pad = torch.ones(
                attn.shape[:-1] + (pad_len,),
                dtype=attn.dtype,
                device=attn.device,
            )
            outputs["attention_mask"] = torch.cat([attn, pad], dim=-1)
    return outputs

math_fields = [
    "General mathematics", "History and biography of mathematics", "Mathematical logic", "Set theory", "Model theory",
    "Proof theory", "Computability theory", "Combinatorics", "Graph theory", "Design theory",
    "Enumerative combinatorics", "Order theory", "Lattice theory", "Universal algebra", "Number theory",
    "Elementary number theory", "Diophantine equations", "Analytic number theory", "Algebraic number theory", "Transcendental number theory",
    "Field theory", "Galois theory", "Polynomials", "Commutative algebra", "Ideal theory",
    "Algebraic geometry", "Singularity theory", "Linear algebra", "Matrix theory", "Associative algebras",
    "Noncommutative algebra", "Lie algebras", "Jordan algebras", "Hopf algebras", "Category theory",
    "Homological algebra", "Algebraic K-theory", "Group theory", "Finite groups", "Abelian groups",
    "Topological groups", "Lie groups", "Representation theory of groups", "Real analysis", "Functions of one real variable",
    "Functions of several real variables", "Measure theory", "Integration theory", "Complex analysis", "Geometric function theory",
    "Several complex variables", "Potential theory", "Special functions", "Ordinary differential equations", "Boundary value problems",
    "Partial differential equations", "Elliptic PDE", "Parabolic PDE", "Hyperbolic PDE", "Dynamical systems",
    "Ergodic theory", "Chaos theory", "Difference equations", "Functional equations", "Sequences and series",
    "Summability theory", "Approximation theory", "Interpolation theory", "Fourier analysis", "Harmonic analysis",
    "Wavelets", "Integral transforms", "Operator theory", "Functional analysis", "Banach spaces",
    "Hilbert spaces", "Spectral theory", "Calculus of variations", "Optimal control", "Optimization",
    "Geometry", "Projective geometry", "Convex geometry", "Discrete geometry", "Differential geometry",
    "Riemannian geometry", "Symplectic geometry", "Topology", "General topology", "Algebraic topology",
    "Knot theory", "Manifolds", "Global analysis", "Probability theory", "Stochastic processes",
    "Statistics", "Numerical analysis", "Algorithms", "Computational complexity", "Cryptography",
    "Classical mechanics", "Fluid dynamics", "Elasticity", "Thermodynamics", "Electromagnetism",
    "Quantum mechanics", "Statistical mechanics", "Relativity", "Astronomy and astrophysics", "Geophysics",
    "Operations research", "Game theory", "Economic mathematics", "Biomathematics", "Epidemiology modeling",
    "Systems theory", "Control theory", "Information theory", "Coding theory", "Mathematics education" ]
math_personalities = [
    "passionate", "enthusiastic", "curious", "creative", "imaginative", "innovative", "inventive",
    "resourceful", "ambitious", "driven", "motivated", "dedicated", "diligent", "hardworking",
    "persistent", "perseverant", "resilient", "tenacious", "patient", "calm", "composed", "stoic",
    "focused", "determined", "disciplined", "organized", "systematic", "methodical", "structured",
    "efficient", "meticulous", "detail-oriented", "accurate", "precise", "rigorous", "thorough",
    "analytic", "logical", "rational", "critical", "skeptical", "questioning", "probing", "observant",
    "insightful", "intuitive", "abstract-minded", "conceptual", "symbolic-thinking", "pattern-seeking",
    "spatial-thinking", "visual-thinking", "geometric-thinking", "experimental", "empirical",
    "data-driven", "evidence-based", "probabilistic", "statistical", "algorithmic", "computational",
    "axiomatic", "proof-driven", "deductive", "inductive", "heuristic", "optimistic", "pessimistic",
    "realistic", "pragmatic", "philosophical", "contemplative", "reflective", "self-critical",
    "perfectionistic", "risk-averse", "risk-taking", "open-minded", "versatile", "flexible", "adaptive",
    "collaborative", "team-oriented", "cooperative", "supportive", "mentoring", "introverted",
    "extroverted", "communicative", "articulate", "didactic", "pedagogical", "eloquent", "persuasive",
    "leadership-minded", "visionary", "strategic", "big-picture thinker", "holistic",
    "interdisciplinary", "multidisciplinary", "cross-cultural", "international", "networking",
    "entrepreneurial", "ethical", "integrity-driven", "humble", "modest", "confident", "self-aware",
    "competitive", "independent", "autonomous", "self-directed", "goal-oriented", "deadline-oriented",
    "results-driven", "applied-oriented", "theory-oriented", "pure-math-minded", "foundational",
    "practical", "solution-focused", "concept-exploring", "knowledge-seeking", "ingenious", "original",
    "nonconformist", "maverick", "iconoclastic", "discoursive", "debate-loving", "collaborative-spirit",
    "pedantic", "precocious", "metatheoretic", "symbol-manipulating", "innovation-driven",
    "curiosity-driven", "detail-loving", "theorem-proving", "algorithm-designing", "notation-obsessed",
    "formalistic", "constructive", "discrete-minded", "continuous-minded", "model-building",
    "simulation-oriented", "numerical-oriented", "editorial", "reviewer-minded", "grant-seeking",
    "lab-leading", "conference-enthusiast", "paper-writing", "problem-posing", "problem-solving",
    "proof-checking", "error-detecting", "precision-obsessed", "symmetry-loving", "invariant-seeking",
    "dimension-thinking", "axiom-hunting", "notation-inventing" ]
academic_roles = ["Professor", "Associate Professor", "Assistant Professor", "Lecturer", "Instructor",
                  "Postdoctoral Researcher", "Research Fellow", "Research Scientist", "PhD Candidate", "Teaching Assistant"]
math_tasks = [
    "Bachelor's Graduation Exam Proof Problem", "Master's Comprehensive Exam Proof Problem", "PhD Qualifying Exam Proof Problem",
    "Doctoral Dissertation Defense Proof Problem", "Graduate-Level Research Proof Challenge", "Research-Level Proof Challenge",
    "Olympiad-Level Proof Problem", "Putnam Competition Proof Problem", "IMO Shortlist Proof Problem",
    "Frontier Research Proof Challenge", "Advanced Functional Analysis Proof Problem", "Algebraic Topology Proof Challenge",
    "Quantum Field Theory Mathematical Proof Problem", "Langlands Program Theorem Proof Problem", "Undergraduate Calculus Calculation Problem",
    "Linear Algebra Calculation Problem", "Number Theory Calculation Problem", "Probability and Statistics Calculation Problem",
    "Numerical Analysis Calculation Problem", "Complex Analysis Calculation Problem",
    "Calculus Multiple-Choice Calculation Problem (4 choices)", "Linear Algebra Multiple-Choice Calculation Problem (4 choices)",
    "Number Theory Multiple-Choice Calculation Problem (4 choices)", "Probability and Statistics Multiple-Choice Calculation Problem (5 choices)",
    "Numerical Analysis Multiple-Choice Calculation Problem (5 choices)" ]

# 出力ディレクトリを作成
os.system(f"mkdir -p {out_dir}")

# 出力ファイル名を生成（タイムスタンプとランダム数を含める）
current_time_no_symbols = datetime.now().strftime(
    "%Y-%m-%d %H:%M:%S").replace("-", "").replace(":", "").replace(" ", "")
out_path = f"{out_dir}/university_math_{current_time_no_symbols}_{random.randint(0,10000)}.jsonl"
# ファイルがなければ新規作成しておく（空作成だけ）
if not os.path.exists(out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        pass

def main():
    logging.info("▶︎ Script start")
    logging.info("[1/6] Initializing various settings")

    # ───── VRAM モニタリングスレッド開始 ─────
    monitor_thread = threading.Thread(target=_monitor_vram, args=(VRAM_LOG_FILE, 5.0), daemon=True)
    monitor_thread.start()
    logging.info("▶︎ VRAM monitoring started; logging to %s", VRAM_LOG_FILE)

    # ───── GPU 環境チェック ─────
    NUM_GPUS = torch.cuda.device_count()
    # if NUM_GPUS is not SET_NUM_GPUS:
    #     sys.exit(f"❌ Error: SET_NUM_GPUS:{SET_NUM_GPUS} is not actual GPU num {NUM_GPUS}.")
    logging.info(f"✅ Detected {NUM_GPUS} GPU(s)")

    # ───── 乱数シード ─────
    # プロセスIDとタイムスタンプでユニークなシードを生成
    pid = os.getpid()
    seed = int(pid) + int(datetime.now().timestamp())
    random.seed(seed)
    set_seed(seed)
    logging.info(f"Setiing seed : {seed}" )

    logging.info("[2/6] Loading config …" )
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

    # ───── 3) max_memory マップ作成 ─────
    max_mem_map = {i: f"{GPU_LIMIT_GIB}GiB" for i in range(NUM_GPUS)}
    # max_mem_map["cpu"] = "64GiB"
    logging.info(f"• max_memory map: {max_mem_map}")

   # 3) Build empty model + balanced dispatch
    # ───── 4) モデル読み込み ─────
    logging.info("[4/6] Loading 4bit-quantized model onto GPUs …")
    model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=config,
            torch_dtype=torch.bfloat16,      
            trust_remote_code=True,
            device_map="auto",
            max_memory=max_mem_map,
            low_cpu_mem_usage=True,
        )

    # デバイスマップ確認（"cpu" がないことをチェック）
    print(model.hf_device_map)

    # Tie weights & eval mode
    model.tie_weights()
    model.eval()

    # モデルインスタンスのクラスを取得
    model_cls = model.__class__
    # 元の prepare_inputs_for_generation を保存
    _orig_prep = model_cls.prepare_inputs_for_generation

    def _patched_prepare_inputs_for_generation(self, input_ids, **model_kwargs):
        # 1) まずオリジナルを呼ぶ
        outputs = _orig_prep(self, input_ids, **model_kwargs)

        # 2) attention_mask があれば長さのズレをチェック
        attn = outputs.get("attention_mask", None)
        pkvs = model_kwargs.get("past_key_values", None)
        if attn is not None and pkvs is not None:
            expected = pkvs.get_max_length()  # ここは既に get_seq_length にエイリアス済み
            actual   = attn.size(-1)
            if actual != expected:
                # 差分分だけ「1」のマスクを末尾に足す
                pad_len = expected - actual
                pad = torch.ones(
                    attn.shape[:-1] + (pad_len,),
                    dtype=attn.dtype,
                    device=attn.device,
                )
                outputs["attention_mask"] = torch.cat([attn, pad], dim=-1)
        return outputs

    # 3) パッチを当てる
    model_cls.prepare_inputs_for_generation = _patched_prepare_inputs_for_generation
    # logging.info("[INFO] Patched prepare_inputs_for_generation to auto-pad attention_mask")

    # sys.modules にロード済みの modeling_deepseek モジュールを探す
    for name, module in sys.modules.items():
        if name.endswith("modeling_deepseek"):
            if hasattr(module, "DynamicCache"):
                cls = module.DynamicCache
                # get_max_length がなければ alias として追加
                if not hasattr(cls, "get_max_length"):
                    cls.get_max_length = cls.get_seq_length
                logging.info(f"[INFO] Patched DynamicCache.get_max_length in module {name}")
            break
    logging.info("✅ Model loaded")

    # ───── 6) トークナイザー読み込み ─────
    logging.info("[5/6] Loading tokenizer …")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True
    )
    logging.info("✅ Tokenizer loaded")

    # ───── 7) GenerationConfig 設定 ─────
    logging.info("[6/6] Loading GenerationConfig …")
    gcfg = GenerationConfig(max_new_tokens=2048, temperature=0.7, top_p=0.9, repetition_penalty=1.1, do_sample=True)
    logging.info(f"✅ GenerationConfig set:\n{gcfg}")

    # ───── 8) 推論用プロンプト準備 ─────
    logging.info("------------------------------------------------------")
    logging.info("      Setup complete. Beginning generation.       ") 
    
    for iteration in range(500):
        logging.info(f"===== Iteration {iteration} =====")

        choiced_personalitie = random.choice(math_personalities)
        choiced_role = random.choice(academic_roles)
        choiced_math_field = random.choice(math_fields)
        choiced_task = random.choice(math_tasks)
        prompt_templates = [
            (
                f"You are a {choiced_personalitie} mathematics {choiced_role}. "
                f"In the area of {choiced_math_field}, please create five different problems related to {choiced_task}. "
                "Ensure that all problems and solutions are mathematically valid and verifiable—avoid any hallucination or unsubstantiated content. "
                "High-difficulty problems are preferred, but do not include low-confidence or speculative material. "
                "Please provide an **extensive and well-articulated Chain of Thought** for each item. "      # ①
                "Your output must follow the format:\n"
                "[\n"
                "  {'Problem': <generate>, 'Chain of Thought (CoT)': <generate>, 'Answer': <generate>},\n"
                "  ...  # total 5 dictionaries\n"
                "]"
            ),
            (
                f"As a {choiced_personalitie} {choiced_role} specializing in mathematics, "
                f"you are working in the domain of {choiced_math_field}. "
                f"Generate five distinct problems that would be relevant for {choiced_task}. "
                "All generated content must be accurate and checkable—avoid hallucinations or unsupported statements. "
                "While high-difficulty problems are desirable, ensure the outputs remain precise and reliable. "
                "Since reviewers value transparency, **include a thorough, step-by-step Chain of Thought**. "  # ②
                "Please return the result as a list of five dictionaries in this structure:\n"
                "[\n"
                "  {'Problem': <generate>, 'Chain of Thought (CoT)': <generate>, 'Answer': <generate>},\n"
                "  ...  # total 5 items\n"
                "]"
            ),
            (
                f"You are a {choiced_personalitie} mathematical {choiced_role}. "
                f"Within the field of {choiced_math_field}, design five unique problems suitable for {choiced_task}. "
                "Make sure every problem and its solution are logically sound and verifiable—do not hallucinate or invent facts. "
                "High-difficulty challenges are encouraged, but avoid including any content of questionable accuracy. "
                "Kindly **aim for a meticulous and richly detailed Chain of Thought**, as longer explanations are appreciated. "  # ③
                "Return the result in the format of a list containing five items:\n"
                "[\n"
                "  {'Problem': <generate>, 'Chain of Thought (CoT)': <generate>, 'Answer': <generate>},\n"
                "  ...  # 5 entries\n"
                "]"
            )
        ]


        # ランダムにプロンプトを選択
        prompt = random.choice(prompt_templates)
        logging.info(f"Prompt is:\n{prompt}")

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # ───── 9) 推論実行 ─────
        logging.info("[5/5] Generating… (no cache)")
        with torch.no_grad():
            outputs = model.generate(**inputs, generation_config=gcfg, use_cache=False)
        result = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        logging.info("✅ Generation complete")

        # ───── 10) 結果を文字列化して保存 ─────
        # dict で返ってきても、list で返ってきても、必ず一つの文字列にする
        result_str = result if isinstance(result, str) else str(result)
        logging.info("\n[5/5] === OUTPUT ===")
        logging.info(result_str)

        # JSONL形式でファイルに保存
        record = {
            "prompt": prompt,
            "result": result_str
        }
        with open(out_path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()

