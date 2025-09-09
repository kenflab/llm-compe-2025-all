from dataclasses import dataclass
import os
import json
import asyncio
import re
from pathlib import Path
from datasets import load_dataset, Dataset, DatasetDict
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio
from inference._configs import Config

# 固定のシステムプロンプト
SYSTEM_PROMPT = (
    "Your role as an assistant involves providing precise and accurate solutions before providing detailed explanations with your full work showing your systematic thinking process leading to each solution. "
    "Your explanations should show how you engaged in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop well-considered thinking process. "
    "Please structure your response into two main sections: Solution and Explanation. "
    "In the Solution section, present your well-thought solution that accurately answers the question. The solution should remain a logical, accurate, concise expression style and detail necessary step needed to reach the conclusion, formatted as follows: <|begin_of_solution|> {final formatted, precise, and clear solution} <|end_of_solution|>. "
    "In the Explanation section, comprehensively detail your reasoning process using the specified format: <|begin_of_explanation|> {explanation with steps separated with '\\n\\n'} <|end_of_explanation|> Each step should show detailed considerations leading to your solutions such as analisying questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps. "
)

def extract_qa_from_conversations(item):
    """
    item is expected to have a 'conversations' list of dicts with 'from' and 'value'.
    Extract the first user question and the solution part of the assistant answer.
    """
    question_text = ""
    answer_text = ""
    convs = item.get("conversations", [])
    for turn in convs:
        if turn.get("from") == "user" and not question_text:
            question_text = turn.get("value", "").strip()
        elif turn.get("from") == "assistant" and not answer_text:
            assistant_value = turn.get("value", "")
            # Extract between <|begin_of_solution|> and <|end_of_solution|>
            m = re.search(r"<\|begin_of_solution\|>(.*?)<\|end_of_solution\|>", assistant_value, re.DOTALL)
            if m:
                answer_text = m.group(1).strip()
            else:
                answer_text = assistant_value.strip()
    return question_text, answer_text

def build_messages(question_text, answer_text):
    # ユーザーには質問のみ
    user_content = question_text
    
    # アシスタントには解答（タグ付き）+ 空行 + 説明開始タグ
    tagged_solution = (
        "<|begin_of_solution|>\n"
        f"{answer_text}\n"
        "<|end_of_solution|>"
    )
    
    assistant_content = (
        f"{tagged_solution}\n\n"
        "<|begin_of_explanation|>"
    )

    return [
        {"role": "system",  "content": SYSTEM_PROMPT},
        {"role": "user",    "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]

def validate_and_extract_explanation(content):
    """
    推論結果を検証し、Chain of Thoughtを抽出する
    
    Returns:
        tuple: (is_valid: bool, explanation_content: str)
        - is_valid: 無効なタグが含まれておらず、適切な終了タグで終わっている場合True
        - explanation_content: 抽出された説明内容（<think>タグと終了タグを除去）
    """
    # 無効なタグが含まれているかチェック
    invalid_tags = [
        "<|begin_of_solution|>",
        "<|end_of_solution|>", 
        "<|begin_of_explanation|>"
    ]
    
    has_invalid_tags = any(tag in content for tag in invalid_tags)
    if has_invalid_tags:
        return False, ""
    
    # <think>タグを除去
    cleaned_content = content
    if cleaned_content.startswith("<think>"):
        # <think>から</think>までを除去、または<think>タグのみを除去
        think_end = cleaned_content.find("</think>")
        if think_end != -1:
            cleaned_content = cleaned_content[think_end + 8:].strip()  # "</think>"の長さ分をスキップ
        else:
            cleaned_content = cleaned_content[7:].strip()  # "<think>"の長さ分をスキップ
    
    # <|end_of_explanation|>タグで終わっているかチェック
    end_tag = "<|end_of_explanation|>"
    if not cleaned_content.strip().endswith(end_tag):
        # 終了タグで終わっていない場合は無効として扱う
        return False, ""
    
    # 終了タグを除去して説明内容を抽出
    explanation_content = cleaned_content[:cleaned_content.rfind(end_tag)].strip()
    
    return True, explanation_content

async def attempt_question(args, entry):
    """
    entry: dict with keys depending on dataset type.
    For new format: should contain 'question_text', 'answer_text', 'id'
    """
    unique_id = entry.get("id")
    question_text = entry.get("question_text")
    answer_text = entry.get("answer_text")

    if question_text is None or answer_text is None:
        # fallback: if legacy format, try to reconstruct similar to original behavior
        return None

    messages = build_messages(question_text, answer_text)
    try:
        response = await client.chat.completions.create(
            model=args.model,
            temperature=args.temperature,
            max_completion_tokens=args.max_completion_tokens,
            messages=messages,
            stream=False,
        )
        content = response.choices[0].message.content or ""
        if response.choices[0].finish_reason == "length":
            # truncated; we still can try to parse but mark for attention
            pass
        usage = json.loads(response.usage.model_dump_json())
        if getattr(args, "reasoning", False):
            # Some backends may supply reasoning_content
            rc = getattr(response.choices[0].message, "reasoning_content", None)
            if rc is not None:
                usage["reasoning_content"] = rc
    except Exception as e:
        print(f"Error for id={unique_id}: {e}")
        return None

    # 推論結果を検証し、Chain of Thoughtを抽出
    is_valid, chain_of_thought = validate_and_extract_explanation(content)

    return {
        "id": unique_id,
        "question_text": question_text,
        "answer_text": answer_text,
        "full_response": content,
        "chain_of_thought": chain_of_thought,
        "is_valid": is_valid,
        "usage": usage,
    }

async def attempt_all(args, entries):
    semaphore = asyncio.Semaphore(args.num_workers)
    async def bound_func(entry):
        async with semaphore:
            return await attempt_question(args, entry)
    tasks = [bound_func(e) for e in entries]
    results = await tqdm_asyncio.gather(*tasks)
    return results

def prepare_entries_from_hf_dataset(dataset):
    raw = list(dataset)  # list of dicts
    entries = []
    for idx, item in enumerate(raw):
        # Unique id: prefer existing 'id', else positional
        unique_id = item.get("id", str(idx))
        question_text, answer_text = extract_qa_from_conversations(item)
        if not question_text or not answer_text:
            # skip malformed
            continue
        entries.append({
            "id": unique_id,
            "question_text": question_text,
            "answer_text": answer_text,
        })
    return entries

def load_existing_output(output_path):
    existing = {}
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    existing[obj["id"]] = obj
                except:
                    continue
    return existing

def count_tokens(text):
    """
    テキストのトークン数を概算で計算
    日本語と英語の混在テキストに対応した簡易計算
    """
    if not text:
        return 0
    
    # 簡易的なトークン数計算（GPT系モデルの概算）
    # 英語: 約4文字で1トークン、日本語: 約1文字で1トークン
    import re
    
    # 日本語文字の検出（ひらがな、カタカナ、漢字）
    japanese_chars = re.findall(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]', text)
    # 英語・数字・記号
    other_chars = re.sub(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]', '', text)
    
    # トークン数の概算
    japanese_tokens = len(japanese_chars)
    other_tokens = len(other_chars.split())  # 単語数で概算
    
    return japanese_tokens + other_tokens

def calculate_validation_statistics(records):
    """検証統計情報を計算"""
    if not records:
        return {}
    
    total_samples = len(records)
    valid_explanations = sum(1 for r in records if r["chain_of_thought"].strip())
    
    # トークン数統計
    token_counts = [count_tokens(r["chain_of_thought"]) for r in records if r["chain_of_thought"].strip()]
    
    stats = {
        "total_samples": total_samples,
        "valid_explanations": valid_explanations,
        "success_rate": (valid_explanations / total_samples * 100) if total_samples > 0 else 0,
    }
    
    if token_counts:
        stats.update({
            "min_tokens": min(token_counts),
            "max_tokens": max(token_counts),
            "avg_tokens": sum(token_counts) / len(token_counts),
        })
        
        # 分布情報
        ranges = [
            (0, 100),
            (101, 500), 
            (501, 1000),
            (1001, 2000),
            (2001, 5000),
            (5001, float('inf'))
        ]
        
        distribution = {}
        for min_val, max_val in ranges:
            if max_val == float('inf'):
                range_name = f"{min_val}+"
                count = sum(1 for tc in token_counts if tc >= min_val)
            else:
                range_name = f"{min_val}-{max_val}"
                count = sum(1 for tc in token_counts if min_val <= tc <= max_val)
            
            percentage = (count / len(token_counts)) * 100 if token_counts else 0
            distribution[range_name] = {"count": count, "percentage": percentage}
        
        stats["distribution"] = distribution
    
    return stats

def create_readme_content(stats, model_name):
    """HFデータセット用のREADME内容を生成"""
    
    # 分布情報の整形
    distribution_text = ""
    if "distribution" in stats:
        for range_name, info in stats["distribution"].items():
            distribution_text += f"  - {range_name}トークン: {info['count']}件 ({info['percentage']:.1f}%)\n"
    
    return f"""---
language:
- ja
- en
license: mit
task_categories:
- text-generation
tags:
- explanation-generation
- chain-of-thought
- reasoning
size_categories:
- n<1K
---

# Chain of Thought生成データセット

このデータセットは、問題と解答から説明（Chain of Thought）を生成したデータセットです。

## 概要
- **処理したサンプル数**: {stats.get('total_samples', 0)}
- **有効な説明生成数**: {stats.get('valid_explanations', 0)}
- **生成成功率**: {stats.get('success_rate', 0):.2f}%
- **使用モデル**: {model_name}

## トークン数統計
- **最小トークン数**: {stats.get('min_tokens', 0)}
- **最大トークン数**: {stats.get('max_tokens', 0)}
- **平均トークン数**: {stats.get('avg_tokens', 0):.1f}

### トークン数分布
{distribution_text}

## データセット構造
- `system_prompt`: モデルに送信されたシステムプロンプト
- `question_text`: 元の問題文
- `answer_text`: 問題の解答
- `chain_of_thought`: 生成された説明・推論過程

## プロンプト構造
- **システム**: 解答と説明の両方を生成するよう指示
- **ユーザー**: 問題文のみ
- **アシスタント**: タグ付き解答 + 空行 + 説明開始タグ
- **モデルの継続**: 説明を完成させ、適切なタグで終了

## 品質管理
以下の条件を満たさない推論結果はリトライ対象となり、最終的にデータセットから除外されます：
1. `<|begin_of_solution|>`、`<|end_of_solution|>`、`<|begin_of_explanation|>`タグが含まれている
2. `<|end_of_explanation|>`で終わっていない
3. 空の説明内容

## データ処理
- **<think>タグ**: Qwen3モデルなどが使用する内部推論タグは自動的に除去されます
- **無効なタグ**: 解答や説明の開始タグが含まれる場合は自動的に除外されます
- **リトライ機能**: 失敗したサンプルは指定回数まで自動的に再試行されます

## システムプロンプト
```
{SYSTEM_PROMPT}
```
"""
    existing = {}
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    existing[obj["id"]] = obj
                except:
                    continue
    return existing

def main(args: Config):
    global client
    client = AsyncOpenAI(
        base_url=args.base_url,
        timeout=86400,
        max_retries=3,
        api_key="fakeapikey",  # replace or override via env if needed
    )

    split = getattr(args, "split", "train")
    if getattr(args, "hf_hub_repo", None):
        # outputs/<sanitized_repo>/ 以下に
        sanitized = args.hf_hub_repo.replace("/", "_")
        base_output = os.path.join("outputs", sanitized)
    else:
        # 従来互換
        sanitized = os.path.basename(args.model).replace("/", "_")
        base_output = getattr(args, "output_dir", os.path.join("outputs", sanitized))
    os.makedirs(base_output, exist_ok=True)

    output_filepath = os.path.join(base_output, f"{sanitized}.jsonl")
    excluded_path   = os.path.join(base_output, f"{sanitized}_excluded.jsonl")
    compact_jsonl   = os.path.join(base_output, f"{sanitized}_compact.jsonl")
    hf_dataset_path = os.path.join(base_output, sanitized)   # HF dataset 保存ディレクトリ

    # Load input dataset
    dataset = load_dataset(args.dataset, split=split)
    # Build entries depending on format
    if "conversations" in dataset.column_names:
        entries = prepare_entries_from_hf_dataset(dataset)
    else:
        raise ValueError("Dataset does not contain 'conversations'; this pipeline expects the bespoke format with conversations.")

    # Apply max_samples cap
    if getattr(args, "max_samples", None):
        entries = entries[: args.max_samples]

    # Load existing to skip
    existing = load_existing_output(output_filepath)
    to_retry = [e for e in entries if e["id"] not in existing]

    # retry logic: only failures are retried up to max_retries times
    max_retries = getattr(args, "max_retries", 2)
    attempt = 0
    while to_retry and attempt <= max_retries:
        if attempt == 0:
            print(f"Running initial inference on {len(to_retry)} entries...")
        else:
            print(f"Retry {attempt}/{max_retries} on {len(to_retry)} failed entries...")
        results = asyncio.run(attempt_all(args, to_retry))

        # merge successes into existing, leave failures in to_retry
        next_retry = []
        for entry, res in zip(to_retry, results):
            if res is None:
                # API error, keep for next round
                next_retry.append(entry)
            elif not res.get("is_valid", False):
                # 無効なタグが含まれている場合、リトライ対象にする
                next_retry.append(entry)
            elif not res.get("chain_of_thought", "").strip() and not res.get("full_response", "").strip():
                # 空のレスポンス、リトライ対象にする
                next_retry.append(entry)
            else:
                # success → record it
                uid = res["id"]
                existing[uid] = {
                    "id": uid,
                    "system_prompt": SYSTEM_PROMPT,
                    "question_text": res["question_text"],
                    "answer_text": res["answer_text"],
                    "chain_of_thought": res["chain_of_thought"],
                    "full_response": res["full_response"],
                    "usage": res["usage"],
                    "model": args.model,
                }
        to_retry = next_retry
        attempt += 1

    if to_retry:
        print(f"Warning: {len(to_retry)} entries failed after {max_retries} retries: "
              f"{[e['id'] for e in to_retry]}")

    # Rewrite the JSONL with all accumulated
    with open(output_filepath, "w", encoding="utf-8") as f:
        for rec in existing.values():
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Build a HuggingFace dataset from accumulated records (only the four required columns + metadata optionally)
    all_records = list(existing.values())
    if not all_records:
        print("No records produced; nothing to save as HF dataset.")
        return

    hf_records = []
    for r in all_records:
        hf_records.append({
            "system_prompt": r["system_prompt"],
            "question_text": r["question_text"],
            "answer_text": r["answer_text"],
            "chain_of_thought": r["chain_of_thought"],
        })

    # Partition
    good_records     = [r for r in hf_records if r["chain_of_thought"].strip()]
    excluded_records = [r for r in hf_records if not r["chain_of_thought"].strip()]

    # Log counts
    print(f"Records to upload:    {len(good_records)}")
    print(f"Records excluded:     {len(excluded_records)}")

    # 統計情報の計算とログ出力
    stats = calculate_validation_statistics(all_records)
    print(f"\n=== 生成統計情報 ===")
    print(f"総処理件数: {stats.get('total_samples', 0)}")
    print(f"有効な説明生成件数: {stats.get('valid_explanations', 0)}")
    print(f"生成成功率: {stats.get('success_rate', 0):.2f}%")
    if "min_tokens" in stats:
        print(f"最小トークン数: {stats['min_tokens']}")
        print(f"最大トークン数: {stats['max_tokens']}")
        print(f"平均トークン数: {stats['avg_tokens']:.1f}")
    print(f"==================\n")

    # Save excluded records separately
    with open(excluded_path, "w", encoding="utf-8") as ef:
        for rec in excluded_records:
            ef.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Excluded records saved to {excluded_path}")

    # Build and save dataset only from good_records
    hf_ds = Dataset.from_list(good_records)
    hf_ds.save_to_disk(hf_dataset_path)
    with open(compact_jsonl, "w", encoding="utf-8") as f:
        for rec in good_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Compact JSONL (good only) saved to {compact_jsonl}")
    print(f"HuggingFace dataset saved to {hf_dataset_path}")

    # ── Push only the dataset (no model files) to HF Hub ─────────────────────────
    if getattr(args, "hf_hub_repo", None):
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        if not token:
            print("WARNING: No HuggingFace token found in HF_TOKEN / HUGGINGFACE_HUB_TOKEN.")
        else:
            # READMEファイル作成
            readme_content = create_readme_content(stats, args.model)
            readme_path = os.path.join(hf_dataset_path, "README.md")
            with open(readme_path, "w", encoding="utf-8") as f:
                f.write(readme_content)
            
            print(f"Pushing {len(good_records)} records to Hugging Face Hub '{args.hf_hub_repo}' ...")
            
            # READMEファイルも含めてアップロード
            hf_ds.push_to_hub(
                repo_id=args.hf_hub_repo,
                token=token,
                private=getattr(args, "hf_hub_private", False),
            )
            
            # READMEファイルを明示的にアップロード
            from huggingface_hub import HfApi
            api = HfApi()
            api.upload_file(
                path_or_fileobj=readme_path,
                path_in_repo="README.md",
                repo_id=args.hf_hub_repo,
                token=token,
                repo_type="dataset"
            )
            
            print("Push to Hub complete (including README).")