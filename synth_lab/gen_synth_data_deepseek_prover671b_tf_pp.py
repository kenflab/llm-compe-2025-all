#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BF16 → 4bit 量子化＋推論スクリプト
"""
# python gen_synth_data_deepseek_prover671b.py [GPU Num Int. 8 or 6] 

from pathlib import Path

BATCH_SIZE = 2  # GPU使用率を上げるために複数プロンプトを一括処理
model_dir = Path("/home/Competition2025/P07/shareP07/share_model/quantized_prover_v2_4bit")
hf_repo = "deepseek-ai/DeepSeek-Prover-V2-671B"
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
from generate_prompt import generate_prompt


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

    # ─── 引数から GPU 数を取得 ──────────────────────────────────────────
    if len(sys.argv) < 2:
        logging.info(f"Usage: {sys.argv[0]} <Setting SET_NUM_GPUS (6 or 8)>")
        sys.exit(1)

    try:
        SET_NUM_GPUS = int(sys.argv[1])
        logging.info(f"Set SET_NUM_GPUS is {SET_NUM_GPUS} .")
        if SET_NUM_GPUS not in (6, 8):
            raise ValueError()
    except ValueError:
        logging.info("Error: <SET_NUM_GPUS> must be integer 6 or 8")
        sys.exit(1)

    # ───── GPU 環境チェック ─────
    NUM_GPUS = torch.cuda.device_count()
    if NUM_GPUS is not SET_NUM_GPUS:
        sys.exit(f"❌ Error: SET_NUM_GPUS:{SET_NUM_GPUS} is not actual GPU num {NUM_GPUS}.")
    logging.info(f"✅ Detected {NUM_GPUS} GPU(s)")

    # ─── GPU 数に応じてマップとメモリ上限を切り替え ───────────────────────
    if SET_NUM_GPUS == 8:
        GPU_LIMIT_GIB = 75
        map_path = Path("/home/Competition2025/P07/shareP07/K_Nishizawa/synth_lab/custom_device_map_8GPU.json")
    else:  # 6 GPUs
        GPU_LIMIT_GIB = 78
        map_path = Path("/home/Competition2025/P07/shareP07/K_Nishizawa/synth_lab/custom_device_map_6GPU.json")

    # ───── パス設定 ─────
    if not model_dir.exists():
        sys.exit(f"❌ Quantized weights not found at: {model_dir}")

    # ───── 乱数シード ─────
    # プロセスIDとタイムスタンプでユニークなシードを生成
    pid = os.getpid()
    seed = int(pid) + int(datetime.now().timestamp())
    random.seed(seed)
    set_seed(seed)
    logging.info(f"Setiing seed : {seed}" )

    logging.info("[2/6] Loading config …" )
    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)

    # ───── 3) max_memory マップ作成 ─────
    max_mem_map = {i: f"{GPU_LIMIT_GIB}GiB" for i in range(NUM_GPUS)}
    # max_mem_map["cpu"] = "64GiB"
    logging.info(f"max_memory map: {max_mem_map}")

    # ───── カスタムマップをファイルからロード ─────
    logging.info("[3/6] Building the custom device_map")
    if not map_path.exists():
        sys.exit(f"❌ custom device map not found: {map_path}")
    with open(map_path, "r", encoding="utf-8") as mf:
        custom_map = json.load(mf)
    logging.info(f"✅ Loaded custom_map from {map_path}")
    logging.debug(f"custom_map : {custom_map}")

    # # カスタムマップ作成
    # logging.info("[3/6] Building the custom device_map")
    # num_layers = config.num_hidden_layers
    # # モデル読み込み前でも、config から層数は取得できる
    # num_layers = config.num_hidden_layers
    # # 各レイヤーを環状に割り当て
    # custom_map = {
    #     "model.embed_tokens": "cuda:0",                                          # 埋め込み層
    #     **{f"model.layers.{i}": f"cuda:{i % NUM_GPUS}" for i in range(num_layers)},  # 各中間層
    #     "model.norm":         "cuda:0",                                          # 最終 LayerNorm
    #     "lm_head":            "cuda:0",                                          # 出力ヘッド
    # }
    # logging.info(f"custom_map : {custom_map}")
    # # ───── 追加︰custom_map をファイルに保存 ─────
    # custom_map_path = "./custom_device_map.json"
    # with open(custom_map_path, "w", encoding="utf-8") as map_f:
    #     json.dump(custom_map, map_f, ensure_ascii=False, indent=2)
    # logging.info(f"✅ Saved custom_map to {custom_map_path}")

    # ───── 4) モデル読み込み ─────
    logging.info("[4/6] Loading 4bit-quantized model onto GPUs …")
    model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            config=config,
            torch_dtype=torch.bfloat16,      
            trust_remote_code=True,
            local_files_only=True,
            device_map=custom_map,
            max_memory=max_mem_map,
            low_cpu_mem_usage=True,
        )

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
        hf_repo,
        local_files_only=True,
        use_fast=True
    )
    logging.info("✅ Tokenizer loaded")

    # ───── 7) GenerationConfig 設定 ─────
    logging.info("[6/6] Loading GenerationConfig …")
    gcfg = GenerationConfig(max_new_tokens=1024, temperature=0.7, top_p=0.9, repetition_penalty=1.1, do_sample=True)
    logging.info(f"✅ GenerationConfig set:\n{gcfg}")

    # ───── 8) 推論用プロンプト準備 ─────
    logging.info("------------------------------------------------------")
    logging.info("      Setup complete. Beginning generation.       ") 
    
    for iteration in range(500):
        logging.info(f"===== Iteration {iteration} =====")

        # 4つのプロンプトをまとめて作る
        prompts = []
        for _ in range(BATCH_SIZE):
            prompt_templates = generate_prompt()                
            prompts.append(prompt_templates)

        logging.info("Prompts:\n" + "\n\n".join(prompts))

        device = torch.device("cuda:0")
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)

        logging.info("[5/5] Generating… (no cache)")
        with torch.no_grad():
            outputs = model.generate(**inputs, generation_config=gcfg, use_cache=False)

        results = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for i, (prompt, result) in enumerate(zip(prompts, results)):
            logging.info(f"\n[5/5] === OUTPUT #{i+1} ===")
            result_str = result if isinstance(result, str) else str(result)
            logging.info(result_str)

            record = {
                "prompt": prompt,
                "result": result_str
            }
            with open(out_path, "a") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()

