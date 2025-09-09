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

def validate_explanation_tags(content):
    """
    推論結果が適切な説明のみを含んでいるかを確認
    以下の場合はhas_explanation_tags=False、explanation_content=""とする：
    1. 説明終了タグがない
    2. 解答の開始タグ・終了タグが含まれている
    3. 説明の開始タグが含まれている
    Returns: (has_valid_tags: bool, explanation_content: str)
    """
    # 説明終了タグが最後に存在するかチェック
    end_tag = "<|end_of_explanation|>"
    has_end_tag = content.strip().endswith(end_tag)
    
    # 無効なタグが含まれているかチェック
    invalid_tags = [
        "<|begin_of_solution|>",
        "<|end_of_solution|>", 
        "<|begin_of_explanation|>"
    ]
    
    has_invalid_tags = any(tag in content for tag in invalid_tags)
    
    # 説明内容の抽出
    if has_end_tag and not has_invalid_tags:
        # 条件を満たす場合のみ説明内容を抽出
        cleaned_content = content.strip()
        
        # <think>タグがある場合は除去
        if cleaned_content.startswith("<think>"):
            cleaned_content = cleaned_content[7:]  # "<think>"の長さ分をスキップ
        
        # 終了タグを除去して純粋な説明内容のみを取得
        explanation_content = cleaned_content[:-len(end_tag)].strip()
        return True, explanation_content
    else:
        # 条件を満たさない場合は空文字を設定
        return False, ""

async def attempt_question(args, entry):
    """
    entry: dict with keys depending on dataset type.
    For new format: should contain 'question_text', 'answer_text', 'id'
    """
    unique_id = entry.get("id")
    question_text = entry.get("question_text")
    answer_text = entry.get("answer_text")

    if question_text is None or answer_text is None:
        return None

    messages = build_messages(question_text, answer_text)
    
    # プロンプトの内容を保存
    user_content = messages[1]["content"]  # ユーザーの内容
    assistant_content = messages[2]["content"]  # アシスタントの内容
    
    try:
        response = await client.chat.completions.create(
            model=args.model,
            temperature=args.temperature,
            max_completion_tokens=args.max_completion_tokens,
            messages=messages,
            stream=False,
        )
        inference_result = response.choices[0].message.content or ""  # 生の推論結果
        usage = json.loads(response.usage.model_dump_json())
        if getattr(args, "reasoning", False):
            rc = getattr(response.choices[0].message, "reasoning_content", None)
            if rc is not None:
                usage["reasoning_content"] = rc
    except Exception as e:
        print(f"Error for id={unique_id}: {e}")
        return None

    # タグ付与の検証
    has_valid_tags, explanation_content = validate_explanation_tags(inference_result)
    
    # トークン数の計算
    token_count = count_tokens(inference_result)

    return {
        "id": unique_id,
        "user_content": user_content,
        "assistant_content": assistant_content,
        "explanation_content": explanation_content,  # 加工済み：純粋な説明内容のみ（条件を満たさない場合は空文字）
        "inference_result": inference_result,  # 生の推論結果（未加工）
        "token_count": token_count,  # 推論結果のトークン数
        "has_explanation_tags": has_valid_tags,  # 全条件を満たすかどうか
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

def prepare_entries_from_hf_dataset(dataset, max_samples=100):
    """データセットから最大max_samples件のエントリを準備"""
    raw = list(dataset)[:max_samples]  # 最初の100件のみ取得
    entries = []
    for idx, item in enumerate(raw):
        unique_id = item.get("id", str(idx))
        question_text, answer_text = extract_qa_from_conversations(item)
        if not question_text or not answer_text:
            continue
        entries.append({
            "id": unique_id,
            "question_text": question_text,
            "answer_text": answer_text,
        })
    return entries

def calculate_and_log_tag_percentage(results):
    """終了タグ付与率を計算してログ出力"""
    total_count = len(results)
    end_tagged_count = sum(1 for r in results if r and r.get("has_explanation_tags", False))
    
    if total_count > 0:
        percentage = (end_tagged_count / total_count) * 100
        print(f"\n=== 説明終了タグ検証結果 ===")
        print(f"総処理件数: {total_count}")
        print(f"終了タグ付与成功件数: {end_tagged_count}")
        print(f"終了タグ付与成功率: {percentage:.2f}%")
        print(f"========================\n")
        return percentage
    else:
        print("処理された結果がありません。")
        return 0.0

def calculate_token_statistics(results):
    """トークン数の統計情報を計算"""
    valid_results = [r for r in results if r is not None]
    if not valid_results:
        return {}
    
    token_counts = [r["token_count"] for r in valid_results]
    
    stats = {
        "min_tokens": min(token_counts),
        "max_tokens": max(token_counts),
        "avg_tokens": sum(token_counts) / len(token_counts),
        "total_samples": len(token_counts)
    }
    
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

def create_readme_content(tag_percentage, total_samples, token_stats):
    """HFデータセット用のREADME内容を生成"""
    
    # 分布情報の整形
    distribution_text = ""
    for range_name, info in token_stats.get("distribution", {}).items():
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
- tag-validation
- chain-of-thought
size_categories:
- n<1K
---

# タグ検証結果

このデータセットは説明生成におけるタグ検証の結果を含んでいます。

## 概要
- **処理したサンプル数**: {total_samples}
- **説明終了タグ成功率**: {tag_percentage:.2f}%
- **検証対象タグ**: `<|end_of_explanation|>`

## トークン数統計
- **最小トークン数**: {token_stats.get('min_tokens', 0)}
- **最大トークン数**: {token_stats.get('max_tokens', 0)} 
- **平均トークン数**: {token_stats.get('avg_tokens', 0):.1f}

### トークン数分布
{distribution_text}

## データセット構造
- `id`: 各サンプルの一意識別子
- `user_content`: モデルにユーザーメッセージとして送信した内容（質問のみ）
- `assistant_content`: モデルにアシスタントメッセージとして送信した内容（タグ付き解答 + 説明開始タグ）
- `explanation_content`: **加工済み内容** - `<think>`タグと終了タグを除去した純粋な説明文（条件を満たさない場合は空文字）
- `inference_result`: **生の推論結果** - モデルの出力をそのまま保存した完全な内容
- `token_count`: 推論結果のトークン数（概算）
- `has_explanation_tags`: 全ての条件を満たすかを示すブール値

## 検証条件
`has_explanation_tags`がFalseになる場合：
1. 説明終了タグ（`<|end_of_explanation|>`）がない
2. 解答の開始タグ（`<|begin_of_solution|>`）または終了タグ（`<|end_of_solution|>`）が含まれている
3. 説明の開始タグ（`<|begin_of_explanation|>`）が含まれている

## データ処理に関する注意事項
- **explanation_content**: `has_explanation_tags`がTrueの場合のみ内容が設定されます
- **inference_result**: 常にモデルの完全な未加工出力が含まれます
- **Qwen3モデル**: 内部推論のために`<think>`タグで応答を開始する場合があります
- **トークン数**: 日本語・英語混在テキストに対応した概算値です

## 目的
このデータセットは、モデルが必要な終了タグで説明を正しく終了でき、かつ不適切なタグが含まれていないかを検証するために作成されました。

## プロンプト構造
- **ユーザー**: 質問文のみ
- **アシスタント**: タグ付き解答 + 空行 + 説明開始タグ
- **モデルの継続**: 説明を完成させ、適切なタグで終了することが期待される
"""

def main(args: Config):
    global client
    client = AsyncOpenAI(
        base_url=args.base_url,
        timeout=86400,
        max_retries=3,
        api_key="fakeapikey",
    )

    # 100件固定で処理
    max_validation_samples = 100
    
    split = getattr(args, "split", "train")
    if getattr(args, "hf_hub_repo", None):
        sanitized = args.hf_hub_repo.replace("/", "_")
        base_output = os.path.join("outputs", f"{sanitized}_tag_validation")
    else:
        sanitized = os.path.basename(args.model).replace("/", "_")
        base_output = os.path.join("outputs", f"{sanitized}_tag_validation")
    
    os.makedirs(base_output, exist_ok=True)

    output_filepath = os.path.join(base_output, f"{sanitized}_validation.jsonl")
    hf_dataset_path = os.path.join(base_output, sanitized)

    # データセット読み込み（100件のみ）
    print(f"Loading dataset: {args.dataset} (first {max_validation_samples} samples)")
    dataset = load_dataset(args.dataset, split=split)
    
    if "conversations" in dataset.column_names:
        entries = prepare_entries_from_hf_dataset(dataset, max_validation_samples)
    else:
        raise ValueError("Dataset does not contain 'conversations'; this pipeline expects the bespoke format with conversations.")

    print(f"Prepared {len(entries)} entries for validation")

    # 推論実行
    print("Starting inference for tag validation...")
    results = asyncio.run(attempt_all(args, entries))
    
    # None結果を除外
    valid_results = [r for r in results if r is not None]
    
    # タグ付与率の計算とログ出力
    tag_percentage = calculate_and_log_tag_percentage(valid_results)
    
    # トークン数統計の計算
    token_stats = calculate_token_statistics(valid_results)
    print(f"\n=== トークン数統計 ===")
    print(f"最小トークン数: {token_stats.get('min_tokens', 0)}")
    print(f"最大トークン数: {token_stats.get('max_tokens', 0)}")
    print(f"平均トークン数: {token_stats.get('avg_tokens', 0):.1f}")
    print(f"===================\n")

    # 結果をJSONL形式で保存
    with open(output_filepath, "w", encoding="utf-8") as f:
        for result in valid_results:
            # 完全な情報を保持
            record = {
                "id": result["id"],
                "user_content": result["user_content"],
                "assistant_content": result["assistant_content"],
                "explanation_content": result["explanation_content"],  # 加工済み説明内容
                "inference_result": result["inference_result"],  # 生の推論結果
                "token_count": result["token_count"],  # トークン数
                "has_explanation_tags": result["has_explanation_tags"],
                "usage": result["usage"],
                "model": args.model,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Validation results saved to {output_filepath}")

    # HuggingFace Dataset作成
    if valid_results:
        hf_records = []
        for r in valid_results:
            hf_records.append({
                "id": r["id"],
                "user_content": r["user_content"],
                "assistant_content": r["assistant_content"],
                "explanation_content": r["explanation_content"],  # 加工済み説明内容
                "inference_result": r["inference_result"],  # 生の推論結果
                "token_count": r["token_count"],  # トークン数
                "has_explanation_tags": r["has_explanation_tags"],
            })

        hf_ds = Dataset.from_list(hf_records)
        hf_ds.save_to_disk(hf_dataset_path)
        print(f"HuggingFace dataset saved to {hf_dataset_path}")

        # HF Hubにアップロード
        if getattr(args, "hf_hub_repo", None):
            token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
            if not token:
                print("WARNING: No HuggingFace token found in HF_TOKEN / HUGGINGFACE_HUB_TOKEN.")
            else:
                # READMEファイル作成
                readme_content = create_readme_content(tag_percentage, len(valid_results), token_stats)
                readme_path = os.path.join(hf_dataset_path, "README.md")
                with open(readme_path, "w", encoding="utf-8") as f:
                    f.write(readme_content)
                
                print(f"Pushing validation results to Hugging Face Hub '{args.hf_hub_repo}' ...")
                
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
    else:
        print("No valid results to save as HF dataset.")