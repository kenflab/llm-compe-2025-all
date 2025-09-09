from dataclasses import dataclass
import os
import json
import asyncio
import re
from pathlib import Path
from datasets import load_dataset, Dataset, DatasetDict
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio
from inference._configs_pairwise import PairwiseConfig

# System prompt for pairwise comparison evaluation
PAIRWISE_SYSTEM_PROMPT = """You are an expert evaluator tasked with comparing two explanations for mathematical problems. Your role is to determine which explanation better helps someone understand how to arrive at the given answer from the question.

You will be given:
1. A question
2. The correct answer
3. Two explanations (Explanation A and Explanation B) that aim to show the reasoning process from question to answer

Evaluate based on these criteria:
- **Clarity**: How clear and understandable is the explanation?
- **Completeness**: Does the explanation cover all necessary steps?
- **Logical flow**: Does the reasoning follow a logical sequence?
- **Accuracy**: Are all steps mathematically correct?
- **Helpfulness**: Would this explanation help someone learn and understand?

Respond in the following format:
WINNER: [A/B/TIE]
REASONING: [Provide a concise explanation of why you chose this winner, highlighting the key differences between the explanations]"""

def extract_answer_from_assistant_content(assistant_content):
    """
    Extract the answer part from assistant_content by removing solution tags.
    Expected format: <|begin_of_solution|>ANSWER<|end_of_solution|>
    """
    # Extract between solution tags
    match = re.search(r"<\|begin_of_solution\|>(.*?)<\|end_of_solution\|>", assistant_content, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        # If no tags found, return the content as is (fallback)
        return assistant_content.strip()

def build_pairwise_messages(question, answer, explanation_a, explanation_b):
    """Build messages for pairwise comparison evaluation"""
    
    user_content = f"""Question: {question}

Correct Answer: {answer}

Explanation A:
{explanation_a}

Explanation B:
{explanation_b}

Please evaluate which explanation better helps understand how to arrive at the answer from the question."""

    return [
        {"role": "system", "content": PAIRWISE_SYSTEM_PROMPT},
        {"role": "user", "content": user_content}
    ]

def parse_evaluation_result(content):
    """Parse the evaluation result to extract winner and reasoning"""
    winner_match = re.search(r"WINNER:\s*([ABT](?:IE)?)", content.upper())
    reasoning_match = re.search(r"REASONING:\s*(.*)", content, re.DOTALL | re.IGNORECASE)
    
    winner = None
    if winner_match:
        winner_text = winner_match.group(1).upper()
        if winner_text in ['A', 'B']:
            winner = winner_text
        elif winner_text in ['T', 'TIE']:
            winner = 'TIE'
    
    reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
    
    return winner, reasoning

async def evaluate_pair(args, comparison_item):
    """Evaluate a single pair of explanations"""
    
    item_id = comparison_item.get("id")
    question = comparison_item.get("question")
    answer = comparison_item.get("answer")
    explanation_a = comparison_item.get("explanation_a")
    explanation_b = comparison_item.get("explanation_b")
    
    messages = build_pairwise_messages(question, answer, explanation_a, explanation_b)
    
    try:
        response = await client.chat.completions.create(
            model=args.model,
            temperature=args.temperature,
            max_completion_tokens=args.max_completion_tokens,
            messages=messages,
            stream=False,
        )
        
        evaluation_result = response.choices[0].message.content or ""
        usage = json.loads(response.usage.model_dump_json())
        
        if getattr(args, "reasoning", False):
            rc = getattr(response.choices[0].message, "reasoning_content", None)
            if rc is not None:
                usage["reasoning_content"] = rc
                
    except Exception as e:
        print(f"Error evaluating pair for id={item_id}: {e}")
        return None
    
    # Parse the evaluation result
    winner, reasoning = parse_evaluation_result(evaluation_result)
    
    return {
        "id": item_id,
        "question": question,
        "answer": answer,
        "explanation_a": explanation_a,
        "explanation_b": explanation_b,
        "winner": winner,
        "reasoning": reasoning,
        "raw_evaluation": evaluation_result,
        "usage": usage,
    }

async def evaluate_all_pairs(args, comparison_items):
    """Evaluate all pairs with controlled concurrency"""
    semaphore = asyncio.Semaphore(args.num_workers)
    
    async def bound_func(item):
        async with semaphore:
            return await evaluate_pair(args, item)
    
    tasks = [bound_func(item) for item in comparison_items]
    results = await tqdm_asyncio.gather(*tasks)
    return results

def prepare_comparison_data(dataset1, dataset2, max_samples=None):
    """
    Prepare comparison data by matching records from two datasets.
    Only include pairs where both records have has_explanation_tags=True.
    """
    
    # Convert datasets to lists
    data1 = list(dataset1)
    data2 = list(dataset2)
    
    # Create dictionaries for quick lookup by id
    data1_dict = {item.get("id"): item for item in data1}
    data2_dict = {item.get("id"): item for item in data2}
    
    # Find matching IDs where both have valid explanation tags
    valid_pairs = []
    skipped_count = 0
    
    for item_id in data1_dict.keys():
        if item_id in data2_dict:
            item1 = data1_dict[item_id]
            item2 = data2_dict[item_id]
            
            # Check if both have valid explanation tags
            if (item1.get("has_explanation_tags", False) and 
                item2.get("has_explanation_tags", False)):
                
                # Extract answer from assistant_content
                answer = extract_answer_from_assistant_content(item1.get("assistant_content", ""))
                
                comparison_item = {
                    "id": item_id,
                    "question": item1.get("user_content", ""),
                    "answer": answer,
                    "explanation_a": item1.get("explanation_content", ""),
                    "explanation_b": item2.get("explanation_content", ""),
                }
                valid_pairs.append(comparison_item)
            else:
                skipped_count += 1
    
    print(f"Found {len(valid_pairs)} valid pairs for comparison")
    print(f"Skipped {skipped_count} pairs due to invalid explanation tags")
    
    # Apply max_samples limit if specified
    if max_samples and max_samples < len(valid_pairs):
        valid_pairs = valid_pairs[:max_samples]
        print(f"Limited to first {max_samples} pairs")
    
    return valid_pairs

def calculate_comparison_statistics(results):
    """Calculate and display comparison statistics"""
    valid_results = [r for r in results if r is not None and r.get("winner") is not None]
    
    if not valid_results:
        print("No valid comparison results found.")
        return {}
    
    total_comparisons = len(valid_results)
    a_wins = sum(1 for r in valid_results if r["winner"] == "A")
    b_wins = sum(1 for r in valid_results if r["winner"] == "B") 
    ties = sum(1 for r in valid_results if r["winner"] == "TIE")
    
    stats = {
        "total_comparisons": total_comparisons,
        "model_a_wins": a_wins,
        "model_b_wins": b_wins,
        "ties": ties,
        "model_a_win_rate": (a_wins / total_comparisons * 100) if total_comparisons > 0 else 0,
        "model_b_win_rate": (b_wins / total_comparisons * 100) if total_comparisons > 0 else 0,
        "tie_rate": (ties / total_comparisons * 100) if total_comparisons > 0 else 0,
    }
    
    return stats

def print_comparison_results(stats, dataset1_name, dataset2_name):
    """Print detailed comparison results to stdout"""
    print("\n" + "="*80)
    print("PAIRWISE COMPARISON RESULTS")
    print("="*80)
    print(f"Dataset A (Model A): {dataset1_name}")
    print(f"Dataset B (Model B): {dataset2_name}")
    print(f"Total Valid Comparisons: {stats['total_comparisons']}")
    print("-"*80)
    print(f"Model A Wins: {stats['model_a_wins']} ({stats['model_a_win_rate']:.1f}%)")
    print(f"Model B Wins: {stats['model_b_wins']} ({stats['model_b_win_rate']:.1f}%)")
    print(f"Ties: {stats['ties']} ({stats['tie_rate']:.1f}%)")
    print("-"*80)
    
    if stats['model_a_wins'] > stats['model_b_wins']:
        print(f"🏆 WINNER: Model A (Dataset: {dataset1_name})")
        print(f"   Margin: +{stats['model_a_wins'] - stats['model_b_wins']} wins ({stats['model_a_win_rate'] - stats['model_b_win_rate']:.1f}% difference)")
    elif stats['model_b_wins'] > stats['model_a_wins']:
        print(f"🏆 WINNER: Model B (Dataset: {dataset2_name})")  
        print(f"   Margin: +{stats['model_b_wins'] - stats['model_a_wins']} wins ({stats['model_b_win_rate'] - stats['model_a_win_rate']:.1f}% difference)")
    else:
        print("🤝 RESULT: Tie between both models")
    
    print("="*80)

def create_comparison_readme(stats, dataset1_name, dataset2_name):
    """Create README content for the comparison results dataset"""
    
    winner_info = ""
    if stats['model_a_wins'] > stats['model_b_wins']:
        winner_info = f"**Winner: Model A (Dataset: {dataset1_name})**\n\nMargin: +{stats['model_a_wins'] - stats['model_b_wins']} wins ({stats['model_a_win_rate'] - stats['model_b_win_rate']:.1f}% difference)"
    elif stats['model_b_wins'] > stats['model_a_wins']:
        winner_info = f"**Winner: Model B (Dataset: {dataset2_name})**\n\nMargin: +{stats['model_b_wins'] - stats['model_a_wins']} wins ({stats['model_b_win_rate'] - stats['model_a_win_rate']:.1f}% difference)"
    else:
        winner_info = "**Result: Tie between both models**"
    
    return f"""---
language:
- ja  
- en
license: mit
task_categories:
- text-generation
tags:
- pairwise-comparison
- explanation-evaluation
- model-comparison
size_categories:
- n<1K
---

# Pairwise Comparison Results

This dataset contains the results of a pairwise comparison between two models' explanation generation capabilities.

## Overview
- **Total Comparisons**: {stats['total_comparisons']}
- **Model A (Dataset)**: {dataset1_name}
- **Model B (Dataset)**: {dataset2_name}

## Results Summary

{winner_info}

### Detailed Statistics
- **Model A Wins**: {stats['model_a_wins']} ({stats['model_a_win_rate']:.1f}%)
- **Model B Wins**: {stats['model_b_wins']} ({stats['model_b_win_rate']:.1f}%)
- **Ties**: {stats['ties']} ({stats['tie_rate']:.1f}%)

## Dataset Structure
- `id`: Unique identifier for each comparison
- `question`: The original question/problem
- `answer`: The correct answer extracted from assistant_content
- `explanation_a`: Explanation from Model A
- `explanation_b`: Explanation from Model B  
- `winner`: Evaluation result (A/B/TIE)
- `reasoning`: Judge model's reasoning for the decision
- `raw_evaluation`: Complete raw output from the judge model

## Evaluation Criteria
Explanations were evaluated based on:
- **Clarity**: How clear and understandable is the explanation?
- **Completeness**: Does the explanation cover all necessary steps?
- **Logical flow**: Does the reasoning follow a logical sequence?
- **Accuracy**: Are all steps mathematically correct?
- **Helpfulness**: Would this explanation help someone learn and understand?

## Methodology
- Only pairs where both models had `has_explanation_tags=True` were evaluated
- Pairs with invalid explanation tags were excluded from comparison
- Evaluation performed by an independent judge model
- Temperature set to 0.0 for consistent evaluation

## Notes
- This comparison focuses specifically on explanation quality for mathematical reasoning tasks
- Results represent the judge model's assessment of explanation quality
- All invalid or incomplete explanations were filtered out before comparison
"""

def main(args: PairwiseConfig):
    global client
    client = AsyncOpenAI(
        base_url=args.base_url,
        timeout=86400,
        max_retries=3,
        api_key="fakeapikey",
    )
    
    # Set up output paths
    sanitized_repo = args.hf_hub_repo.replace("/", "_") if args.hf_hub_repo else "pairwise_comparison"
    base_output = os.path.join(args.output_dir, f"{sanitized_repo}_results")
    os.makedirs(base_output, exist_ok=True)
    
    output_filepath = os.path.join(base_output, f"{sanitized_repo}_comparison.jsonl")
    hf_dataset_path = os.path.join(base_output, sanitized_repo)
    
    # Load datasets
    print(f"Loading Dataset 1: {args.dataset1}")
    dataset1 = load_dataset(args.dataset1, split="train")
    
    print(f"Loading Dataset 2: {args.dataset2}")
    dataset2 = load_dataset(args.dataset2, split="train")
    
    # Prepare comparison data
    print("Preparing comparison pairs...")
    comparison_items = prepare_comparison_data(dataset1, dataset2, args.max_samples)
    
    if not comparison_items:
        print("No valid comparison pairs found. Exiting.")
        return
    
    # Run pairwise evaluation
    print("Starting pairwise evaluation...")
    results = asyncio.run(evaluate_all_pairs(args, comparison_items))
    
    # Filter out None results
    valid_results = [r for r in results if r is not None]
    
    # Calculate and display statistics
    stats = calculate_comparison_statistics(valid_results)
    print_comparison_results(stats, args.dataset1, args.dataset2)
    
    # Save results to JSONL
    with open(output_filepath, "w", encoding="utf-8") as f:
        for result in valid_results:
            record = {
                "id": result["id"],
                "question": result["question"],
                "answer": result["answer"],
                "explanation_a": result["explanation_a"],
                "explanation_b": result["explanation_b"],
                "winner": result["winner"],
                "reasoning": result["reasoning"],
                "raw_evaluation": result["raw_evaluation"],
                "usage": result["usage"],
                "judge_model": args.model,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    print(f"\nComparison results saved to {output_filepath}")
    
    # Create and save HuggingFace dataset
    if valid_results:
        hf_records = []
        for r in valid_results:
            hf_records.append({
                "id": r["id"],
                "question": r["question"],
                "answer": r["answer"],
                "explanation_a": r["explanation_a"],
                "explanation_b": r["explanation_b"],
                "winner": r["winner"],
                "reasoning": r["reasoning"],
                "raw_evaluation": r["raw_evaluation"],
            })
        
        hf_ds = Dataset.from_list(hf_records)
        hf_ds.save_to_disk(hf_dataset_path)
        print(f"HuggingFace dataset saved to {hf_dataset_path}")
        
        # Upload to HF Hub if configured
        if args.hf_hub_repo:
            token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN") or args.hf_token
            if not token:
                print("WARNING: No HuggingFace token found.")
            else:
                # Create README
                readme_content = create_comparison_readme(stats, args.dataset1, args.dataset2)
                readme_path = os.path.join(hf_dataset_path, "README.md")
                with open(readme_path, "w", encoding="utf-8") as f:
                    f.write(readme_content)
                
                print(f"Pushing comparison results to Hugging Face Hub '{args.hf_hub_repo}' ...")
                
                # Upload dataset and README
                hf_ds.push_to_hub(
                    repo_id=args.hf_hub_repo,
                    token=token,
                    private=args.hf_hub_private,
                )
                
                # Upload README explicitly
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