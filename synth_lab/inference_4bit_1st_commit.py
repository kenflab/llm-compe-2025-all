#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BF16 → 4bit 量子化＋推論スクリプト
前提：inference/fp8_cast_bf16.py で作成した bf16_weights フォルダがあること
"""
import os, sys, re
# ─── 出力を行バッファリングモードに ────────────────────────────────────
if hasattr(sys.stdout, "reconfigure"):
    # Python3.7+ ならこれで OK
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

import json
import traceback
from pathlib import Path

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

# ─── ロギングレベル ───────────────────────────────────────────────────
hf_logging.set_verbosity_debug()

# 0) Triton キャッシュ先を exec 可能な場所に移す
os.environ["TRITON_CACHE_DIR"] = "/dev/shm/triton_cache"
os.makedirs(os.environ["TRITON_CACHE_DIR"], exist_ok=True)

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

# ─── インデックスファイル自動修正（必要なら有効化） ────────────────────────
# def fix_index_json(bf16_dir: str):
#     idx_path = os.path.join(bf16_dir, "model.safetensors.index.json")
#     text = open(idx_path, "r").read()
#     text = text = re.sub(r",\s*}", "}", text)
#     data = json.loads(text)
#     md = data.get("metadata", {})
#     md["format"] = "pt"
#     data["metadata"] = md
#     with open(idx_path, "w") as f:
#         json.dump(data, f, indent=2)
#     print(f"[INFO] Fixed index JSON: {idx_path}")

def main():
    print("▶︎ Script start")

    # ───── GPU 環境チェック ─────
    NUM_GPUS = torch.cuda.device_count()
    if NUM_GPUS < 1:
        sys.exit("❌ Error: At least 1 GPU is required but none found.")
    print(f"✅ Detected {NUM_GPUS} GPU(s)")

    # ───── パス設定 ─────
    hf_repo = "deepseek-ai/DeepSeek-Prover-V2-671B"
    cpu_weights_dir = Path(
        "/home/Competition2025/P07/shareP07/K_Nishizawa/"
        "synth_lab/quantized_prover_v2_4bit/4bit_cpu_weights"
    )
    if not cpu_weights_dir.exists():
        sys.exit(f"❌ Quantized weights not found at: {cpu_weights_dir}")

    # ───── 乱数シード ─────
    set_seed(42)

    # # 1) Config 読み込み
    # qb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_compute_dtype=torch.bfloat16,
    # )

    print("[1/6] Loading config …" )
    config = AutoConfig.from_pretrained(cpu_weights_dir, trust_remote_code=True)
    # config.quantization_config = qb_config.__dict__  # 辞書として埋め込む  # 正しく埋め込む
    # print("✅ Config Load & Update \n ", config)
    # if hasattr(config, "quantization_config"):
    #     delattr(config, "quantization_config")
    #     print("    • Removed embedded quantization_config" )

    # # 2) BitsAndBytesConfig 準備
    # quant_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_compute_dtype=torch.bfloat16,
    # )
    # print("    • Prepared BitsAndBytesConfig" )

    # ───── 3) max_memory マップ作成 ─────
    GPU_LIMIT_GIB = 70
    max_mem_map = {i: f"{GPU_LIMIT_GIB}GiB" for i in range(NUM_GPUS)}
    max_mem_map["cpu"] = "64GiB"
    print(f"    • max_memory map: {max_mem_map}")

    print("カスタム device_map を作成")
    num_layers = config.num_hidden_layers
    # 3) 層数／カスタムマップ作成
    # モデル読み込み前でも、config から層数は取得できる
    num_layers = config.num_hidden_layers
    # 各レイヤーを環状に割り当て
    custom_map = {
        "model.embed_tokens": "cuda:0",                                          # 埋め込み層
        **{f"model.layers.{i}": f"cuda:{i % NUM_GPUS}" for i in range(num_layers)},  # 各中間層
        "model.norm":         "cuda:0",                                          # 最終 LayerNorm
        "lm_head":            "cuda:0",                                          # 出力ヘッド
    }
    print("custom_map : ", custom_map)

    # ───── 4) モデル読み込み ─────
    print("[2/5] Loading 4bit-quantized model onto GPUs …")
    # model = AutoModelForCausalLM.from_pretrained(
    #     cpu_weights_dir,
    #     config=config,                 # ここにはもう古い設定がない
    #     quantization_config=quant_config,
    #     device_map=custom_map,
    #     max_memory=max_mem_map,
    #     torch_dtype=torch.bfloat16,
    #     low_cpu_mem_usage=True,
    #     trust_remote_code=True,
    # )
    model = AutoModelForCausalLM.from_pretrained(
            cpu_weights_dir,
            config=config,
            torch_dtype=torch.bfloat16,      
            trust_remote_code=True,
            local_files_only=True,
            device_map=custom_map,
            max_memory=max_mem_map,
            low_cpu_mem_usage=True,
        )
    print("✅ Model loaded")

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
    print("[INFO] Patched prepare_inputs_for_generation to auto-pad attention_mask")

    # sys.modules にロード済みの modeling_deepseek モジュールを探す
    for name, module in sys.modules.items():
        if name.endswith("modeling_deepseek"):
            if hasattr(module, "DynamicCache"):
                cls = module.DynamicCache
                # get_max_length がなければ alias として追加
                if not hasattr(cls, "get_max_length"):
                    cls.get_max_length = cls.get_seq_length
                print(f"[INFO] Patched DynamicCache.get_max_length in module {name}")
            break

    # ───── 6) トークナイザー読み込み ─────
    print("[3/5] Loading tokenizer …")
    tokenizer = AutoTokenizer.from_pretrained(
        hf_repo,
        local_files_only=True,
        use_fast=True
    )
    print("✅ Tokenizer loaded")

    # ───── 7) GenerationConfig 設定 ─────
    gcfg = GenerationConfig(max_new_tokens=2048, temperature=0.7, top_p=0.9, repetition_penalty=1.1, do_sample=True)
    print("✅ GenerationConfig set  \n", gcfg)

    # ───── 8) 推論用プロンプト準備 ─────
    prompt = ' ## Please provide as many mathematical proofs at a high level as possible. '
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # ───── 9) 推論実行 ─────
    print("[5/5] Generating… (no cache)")
    with torch.no_grad():
        outputs = model.generate(**inputs, generation_config=gcfg, use_cache=False)
    result = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    print("✅ Generation complete")

    # ───── 10) 結果表示 ─────
    print("\n[5/5] === OUTPUT ===")
    print(result)

    # ───── 11) モデル & トークナイザー保存 ─────
    SAVE_DIR = "/home/Competition2025/P07/shareP07/K_Nishizawa/synth_lab/quantized_prover_v2_4bit"
    model.save_pretrained(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)
    print(f"✅ 4bit モデル & tokenizer saved to: {SAVE_DIR}")

if __name__ == "__main__":
    main()





"""
BF16 → 4bit 量子化＋推論スクリプト
前提：inference/fp8_cast_bf16.py で作成した bf16_weights フォルダがあること

import sys, os
# ─── インデックスファイル自動修正 ────────────────────────────────────
def fix_index_json(bf16_dir: str):
    idx_path = os.path.join(bf16_dir, "model.safetensors.index.json")
    text = open(idx_path, "r").read()
    # JSON の末尾カンマを削除（invalid trailing commas）
    text = re.sub(r",\s*}", "}", text)
    data = json.loads(text)
    md = data.get("metadata", {})
    md["format"] = "pt"          # 必ず format フィールドをセット
    data["metadata"] = md
    with open(idx_path, "w") as f:
        json.dump(data, f, indent=2)
    print("[INFO] Fixed index JSON:", idx_path)
# ────────────────────────────────────────────────────────────────────


import re
import json
import traceback
import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
    logging,
    set_seed,
)
from accelerate import dispatch_model
from pathlib import Path

logging.set_verbosity_debug()

# ───── 環境チェック ─────
NUM_GPUS = torch.cuda.device_count()
if NUM_GPUS < 8:
    sys.exit(f"Need 8 GPUs, found {NUM_GPUS}")

# ───── GPU メモリ制限設定 ─────
GPU_LIMIT = 70  # 各GPUあたり70GiB上限
max_mem = {f"cuda:{i}": f"{GPU_LIMIT}GiB" for i in range(NUM_GPUS)}

def main():
    print("▶︎ Script start")
    # — 設定 —
    hf_repo = "deepseek-ai/DeepSeek-Prover-V2-671B"
    # quant化後重みディレクトリ
    cpu_save = Path(
        "/home/Competition2025/P07/shareP07/K_Nishizawa/"
        "synth_lab/quantized_prover_v2_4bit/4bit_cpu_weights"
    )
    if not cpu_save.exists():
        sys.exit(f"Quantized weights not found: {cpu_save}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    set_seed(42)

    # 0) index.json 修正
    # fix_index_json(bf16_dir)

    # 1) カスタム Config をリポジトリから取得
    print(f"[1/5] Loading config from HF repo {hf_repo} …")
    config = AutoConfig.from_pretrained(
        hf_repo,
        trust_remote_code=True,
    )

    # 2) 古い quantization_config をクリア（念のため）
    if hasattr(config, "quantization_config"):
        config.quantization_config = None

    # 3) BitsAndBytes の 4bit 設定
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",         # nf4 or fp4
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    print("カスタム device_map を作成")
    num_layers = config.num_hidden_layers
    # 3) 層数／カスタムマップ作成
    # モデル読み込み前でも、config から層数は取得できる
    num_layers = config.num_hidden_layers
    # 各レイヤーを環状に割り当て
    custom_map = {
        "model.embed_tokens": "cuda:0",                                          # 埋め込み層
        **{f"model.layers.{i}": f"cuda:{i % NUM_GPUS}" for i in range(num_layers)},  # 各中間層
        "model.norm":         "cuda:0",                                          # 最終 LayerNorm
        "lm_head":            "cuda:0",                                          # 出力ヘッド
    }
    print("custom_map : ", custom_map)

    # 4) max_memory マップ
    max_mem_map = {f"cuda:{i}": f"{GPU_LIMIT}GiB" for i in range(NUM_GPUS)}
    max_mem_map["cpu"] = "64GiB"

    # 1) モデル読み込み
    print("[1/3] Loading 4bit-quantized model …")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            cpu_save,
            config=config,
            quantization_config=quant_config,
            trust_remote_code=True,
            local_files_only=True,
            device_map=custom_map,
            max_memory=max_mem,
        )
    except Exception:
        print("❌ Failed to load model:\n" + traceback.format_exc())
        sys.exit(1)        

    # 5) トークナイザー読み込み
    print("[3/5] Loading tokenizer …")
    tokenizer = AutoTokenizer.from_pretrained(
        hf_repo,
        local_files_only=True,       # ← 追加
        use_fast=True
    )

    # 6) 推論パラメータ設定
    gen_config = GenerationConfig(
        max_new_tokens=128,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
        do_sample=True
    )

    # 7) プロンプト準備
    prompt = (
        "次の日本語の文章を英語に翻訳してください：\n"
        "「機械学習モデルの量子化によって、推論時のメモリ使用量を削減できます。」"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # 8) 推論実行
    print("[4/5] Generating …")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            generation_config=gen_config
        )

    # 9) 結果表示
    result = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    print("\n[5/5] === OUTPUT ===")
    print(result)

    print("モデルを保存")

    SAVE_DIR = "/home/Competition2025/P07/shareP07/K_Nishizawa/synth_lab/quantized_prover_v2_4bit"
    # モデル本体を保存
    model.save_pretrained(SAVE_DIR)
    # トークナイザーも一緒に保存
    tokenizer.save_pretrained(SAVE_DIR)
    print(f"4bit モデルを保存しました: {SAVE_DIR}")

if __name__ == "__main__":
    main()

"""