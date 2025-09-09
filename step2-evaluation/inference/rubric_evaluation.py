import os
import json
import asyncio
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datasets import load_dataset, Dataset
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio
from inference._configs import Config
import traceback
import logging
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rubric_evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Rubric evaluation criteria
RUBRIC_CRITERIA = {
    'logical_coherence': {
        'description': 'Logical consistency and flow of reasoning',
        'levels': {
            0: 'No logical flow, contradictory or incoherent reasoning',
            1: 'Minimal logical structure, major gaps in reasoning',
            2: 'Basic logical flow with some inconsistencies',
            3: 'Good logical structure with minor issues',
            4: 'Excellent logical consistency throughout'
        }
    },
    'completeness': {
        'description': 'Coverage of all necessary solution steps',
        'levels': {
            0: 'Major elements missing, incomplete explanation',
            1: 'Several important steps omitted',
            2: 'Most key steps present but some gaps',
            3: 'Nearly complete with minor omissions',
            4: 'Comprehensive coverage of all elements'
        }
    },
    'clarity': {
        'description': 'Clarity and understandability of explanation',
        'levels': {
            0: 'Very difficult to understand, unclear throughout',
            1: 'Generally unclear with confusing sections',
            2: 'Moderately clear with some ambiguity',
            3: 'Clear and easy to follow with minor issues',
            4: 'Exceptionally clear and easy to understand'
        }
    },
    'pedagogical_value': {
        'description': 'Educational value and teaching effectiveness',
        'levels': {
            0: 'No educational value, misleading or incorrect',
            1: 'Limited educational benefit',
            2: 'Moderate educational value',
            3: 'Good teaching approach with learning support',
            4: 'Excellent pedagogical approach with deep insights'
        }
    },
    'efficiency': {
        'description': 'Conciseness and directness of explanation',
        'levels': {
            0: 'Extremely verbose or unnecessarily complex',
            1: 'Significantly inefficient with redundancy',
            2: 'Some unnecessary complexity or repetition',
            3: 'Generally efficient with minor verbosity',
            4: 'Optimally concise while maintaining clarity'
        }
    }
}

def build_rubric_evaluation_prompt(question: str, answer: str, chain_of_thought: str) -> List[Dict[str, str]]:
    """Build the evaluation prompt for rubric assessment as messages"""
    
    # Simplified prompt for better JSON generation
    system_prompt = """You are an expert evaluator. Evaluate the mathematical explanation based on these criteria:
1. logical_coherence: Is the reasoning logically consistent? (0-4)
2. completeness: Are all necessary steps covered? (0-4)
3. clarity: Is the explanation clear and understandable? (0-4)
4. pedagogical_value: Does it have educational value? (0-4)
5. efficiency: Is it concise without losing clarity? (0-4)

Score each from 0 (worst) to 4 (best).

You must respond with ONLY a valid JSON object in this exact format:
{
    "logical_coherence": {"score": <number>, "justification": "<brief reason>"},
    "completeness": {"score": <number>, "justification": "<brief reason>"},
    "clarity": {"score": <number>, "justification": "<brief reason>"},
    "pedagogical_value": {"score": <number>, "justification": "<brief reason>"},
    "efficiency": {"score": <number>, "justification": "<brief reason>"},
    "overall_quality": "<brief overall assessment>"
}"""

    user_prompt = f"""Evaluate this explanation:

QUESTION: {question[:500]}  

CORRECT ANSWER: {answer[:200]}

EXPLANATION TO EVALUATE: {chain_of_thought[:1500]}

Provide your evaluation as a JSON object."""
    
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

def parse_rubric_response(response_text: str, entry_id: str = "") -> Optional[Dict]:
    """Parse the rubric evaluation response with better error handling"""
    try:
        logger.debug(f"Parsing response for entry {entry_id}: {response_text[:200]}...")
        
        # Clean up the response
        response_text = response_text.strip()
        
        # Try multiple extraction patterns
        json_text = response_text
        
        # Pattern 1: JSON in markdown code blocks
        if "```json" in response_text:
            match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
            if match:
                json_text = match.group(1)
        elif "```" in response_text:
            match = re.search(r'```\s*(.*?)\s*```', response_text, re.DOTALL)
            if match:
                json_text = match.group(1)
        
        # Pattern 2: Find JSON object directly
        json_match = re.search(r'\{.*\}', json_text, re.DOTALL)
        if json_match:
            json_text = json_match.group(0)
        
        # Parse JSON
        result = json.loads(json_text)
        
        # Validate structure
        required_keys = ['logical_coherence', 'completeness', 'clarity', 
                        'pedagogical_value', 'efficiency']
        
        for key in required_keys:
            if key not in result:
                logger.warning(f"Missing key '{key}' in response for entry {entry_id}")
                return None
            
            # Handle both dict and direct score formats
            if isinstance(result[key], dict):
                if 'score' not in result[key]:
                    logger.warning(f"Missing 'score' in '{key}' for entry {entry_id}")
                    return None
                score = result[key]['score']
            elif isinstance(result[key], (int, float)):
                # Convert direct score to expected format
                result[key] = {'score': result[key], 'justification': ''}
                score = result[key]['score']
            else:
                logger.warning(f"Invalid format for '{key}' in entry {entry_id}")
                return None
            
            # Ensure score is within range
            if not isinstance(score, (int, float)) or score < 0 or score > 4:
                logger.warning(f"Score out of range for '{key}': {score} in entry {entry_id}")
                # Clamp to valid range
                result[key]['score'] = max(0, min(4, int(score)))
        
        logger.info(f"Successfully parsed rubric for entry {entry_id}")
        return result
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error for entry {entry_id}: {e}")
        logger.debug(f"Failed to parse: {response_text[:500]}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error parsing rubric for entry {entry_id}: {e}")
        logger.debug(f"Response text: {response_text[:500]}")
        return None

def calculate_weighted_score(rubric_result: Dict, weights: Dict[str, float]) -> float:
    """Calculate weighted score from rubric results"""
    total_score = 0.0
    total_weight = 0.0
    
    for criterion, weight in weights.items():
        if criterion in rubric_result and 'score' in rubric_result[criterion]:
            score = rubric_result[criterion]['score']
            # Normalize to 0-1 range (from 0-4 scale)
            normalized_score = score / 4.0
            total_score += normalized_score * weight
            total_weight += weight
    
    if total_weight > 0:
        return total_score / total_weight
    return 0.0

async def evaluate_single_entry(client: AsyncOpenAI, args: Config, entry: Dict, idx: int) -> Dict:
    """Evaluate a single entry with rubric"""
    
    entry_id = entry.get('id', str(idx))
    
    # Extract fields
    question = entry.get('question_text', '')
    answer = entry.get('answer_text', '')
    chain_of_thought = entry.get('chain_of_thought', '')
    
    # Skip if chain_of_thought is empty
    if not chain_of_thought or not chain_of_thought.strip():
        logger.info(f"Skipping entry {entry_id}: empty chain_of_thought")
        return {
            **entry,
            'rubric_score': 0.0
        }
    
    # Build evaluation prompt
    messages = build_rubric_evaluation_prompt(question, answer, chain_of_thought)
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            logger.debug(f"Evaluating entry {entry_id}, attempt {attempt + 1}/{max_retries}")
            
            # Get evaluation from model
            response = await client.chat.completions.create(
                model=args.model,
                temperature=0.0,  # Force deterministic
                max_completion_tokens=args.max_completion_tokens,
                messages=messages,
                stream=False,
            )
            
            response_text = response.choices[0].message.content or ""
            
            if not response_text:
                logger.warning(f"Empty response for entry {entry_id}")
                continue
            
            # Log raw response for debugging
            logger.debug(f"Raw response for {entry_id}: {response_text[:500]}")
            
            # Parse rubric response
            rubric_result = parse_rubric_response(response_text, entry_id)
            
            if rubric_result:
                # Calculate weighted score
                weighted_score = calculate_weighted_score(rubric_result, args.rubric_weights)
                
                logger.info(f"Successfully evaluated entry {entry_id} with weighted score: {weighted_score:.3f}")
                
                return {
                    **entry,
                    'rubric_evaluation': rubric_result,
                    'rubric_score': float(weighted_score)
                }
            else:
                logger.warning(f"Failed to parse rubric for entry {entry_id}, attempt {attempt + 1}")
                
                # Save raw response for debugging
                debug_path = Path(args.output_dir) / "debug_responses"
                debug_path.mkdir(exist_ok=True)
                with open(debug_path / f"{entry_id}_attempt{attempt}.txt", "w") as f:
                    f.write(response_text)
                
                if attempt < max_retries - 1:
                    await asyncio.sleep(2)  # Wait before retry
                
        except Exception as e:
            logger.error(f"Error evaluating entry {entry_id}: {e}")
            logger.error(traceback.format_exc())
            
            if attempt < max_retries - 1:
                await asyncio.sleep(5)  # Wait longer on error
    
    # All attempts failed
    logger.error(f"Failed to evaluate entry {entry_id} after {max_retries} attempts")
    return {
        **entry,
        'rubric_evaluation': None,
        'rubric_score': 0.0
    }

async def evaluate_all_entries(client: AsyncOpenAI, args: Config, entries: List[Dict]) -> List[Dict]:
    """Evaluate all entries with rate limiting"""
    semaphore = asyncio.Semaphore(args.num_workers)
    
    async def bound_evaluate(entry, idx):
        async with semaphore:
            return await evaluate_single_entry(client, args, entry, idx)
    
    tasks = [bound_evaluate(entry, idx) for idx, entry in enumerate(entries)]
    results = await tqdm_asyncio.gather(*tasks, desc="Evaluating entries")
    return results

def calculate_statistics(results: List[Dict]) -> Dict:
    """Calculate evaluation statistics for both rubric and LogP"""
    stats = {
        'total_samples': len(results),
        'evaluated_samples': 0,
        'failed_samples': 0,
        'rubric_stats': {
            'average_scores': {},
            'score_distribution': {}
        },
        'logp_stats': {
            'total_samples_with_logp': 0,
            'valid_logp_samples': 0,
            'avg_logp_score': 0.0,
            'std_logp_score': 0.0,
            'min_logp_score': 0.0,
            'max_logp_score': 0.0
        }
    }
    
    # Collect rubric scores
    criterion_scores = {criterion: [] for criterion in RUBRIC_CRITERIA.keys()}
    weighted_scores = []
    
    # Collect LogP scores
    logp_scores = []
    
    for result in results:
        # Rubric statistics
        if result.get('rubric_evaluation'):
            stats['evaluated_samples'] += 1
            rubric = result['rubric_evaluation']
            
            for criterion in RUBRIC_CRITERIA.keys():
                if criterion in rubric and 'score' in rubric[criterion]:
                    criterion_scores[criterion].append(rubric[criterion]['score'])
            
            if 'rubric_score' in result:
                weighted_scores.append(result['rubric_score'])
        else:
            stats['failed_samples'] += 1
        
        # LogP statistics
        if 'logp_score' in result and result['logp_score'] is not None:
            stats['logp_stats']['total_samples_with_logp'] += 1
            logp_score = result['logp_score']
            if logp_score != float('-inf') and not np.isnan(logp_score):
                logp_scores.append(logp_score)
    
    # Calculate rubric averages
    for criterion, scores in criterion_scores.items():
        if scores:
            stats['rubric_stats']['average_scores'][criterion] = sum(scores) / len(scores)
    
    if weighted_scores:
        stats['rubric_stats']['average_scores']['weighted_total'] = sum(weighted_scores) / len(weighted_scores)
    
    # Rubric score distribution
    for criterion, scores in criterion_scores.items():
        distribution = {i: 0 for i in range(5)}
        for score in scores:
            distribution[int(score)] += 1
        stats['rubric_stats']['score_distribution'][criterion] = distribution
    
    # Calculate LogP statistics
    if logp_scores:
        stats['logp_stats']['valid_logp_samples'] = len(logp_scores)
        stats['logp_stats']['avg_logp_score'] = np.mean(logp_scores)
        stats['logp_stats']['std_logp_score'] = np.std(logp_scores)
        stats['logp_stats']['min_logp_score'] = np.min(logp_scores)
        stats['logp_stats']['max_logp_score'] = np.max(logp_scores)
    
    return stats

def create_readme_content(stats: Dict, model_name: str, source_dataset: str) -> str:
    """Create README content for Hugging Face dataset including both rubric and LogP stats"""
    
    # Rubric scores
    rubric_avg_scores_text = ""
    for criterion, score in stats['rubric_stats']['average_scores'].items():
        rubric_avg_scores_text += f"- **{criterion}**: {score:.2f}\n"
    
    # LogP stats
    logp_stats = stats['logp_stats']
    logp_stats_text = f"""
## LogP Evaluation Statistics
- **Total samples with LogP scores**: {logp_stats['total_samples_with_logp']}
- **Valid LogP samples**: {logp_stats['valid_logp_samples']}
- **Average LogP score**: {logp_stats['avg_logp_score']:.4f}
- **Standard deviation**: {logp_stats['std_logp_score']:.4f}
- **Min LogP score**: {logp_stats['min_logp_score']:.4f}
- **Max LogP score**: {logp_stats['max_logp_score']:.4f}
"""
    
    return f"""---
language:
- en
license: mit
task_categories:
- text-generation
tags:
- rubric-evaluation
- logp-evaluation
- chain-of-thought
- reasoning
- evaluation
---

# Complete Evaluation Dataset (Rubric + LogP)

This dataset contains chain-of-thought explanations evaluated using both comprehensive rubric assessment and LogP evaluation.

## Overview
- **Source Dataset**: {source_dataset}
- **Total Samples**: {stats['total_samples']}
- **Successfully Evaluated (Rubric)**: {stats['evaluated_samples']}
- **Failed Evaluations (Rubric)**: {stats['failed_samples']}
- **Evaluation Model**: {model_name}

## Rubric Evaluation Results
### Average Rubric Scores (0-4 scale)
{rubric_avg_scores_text}

{logp_stats_text}

## Dataset Structure
- `system_prompt`: System prompt used for generation
- `question_text`: Original question
- `answer_text`: Correct answer
- `chain_of_thought`: Generated explanation
- `rubric_evaluation`: Detailed rubric evaluation results
- `rubric_score`: Weighted rubric score (0-1 scale)
- `logp_score`: LogP evaluation score

## Evaluation Methods

### Rubric Evaluation
1. **Logical Coherence**: Logical consistency and flow of reasoning
2. **Completeness**: Coverage of all necessary solution steps
3. **Clarity**: Clarity and understandability of explanation
4. **Pedagogical Value**: Educational value and teaching effectiveness
5. **Efficiency**: Conciseness and directness of explanation

Each criterion is scored on a 0-4 scale, with weighted averaging for final score.

### LogP Evaluation
LogP scores are computed using RLT TeacherKLBasedReward logic on solution parts only.
The evaluation focuses on the probability distribution of generated tokens in the solution sections.

## Evaluation Details
- **Rubric Temperature**: 0.0 (deterministic)
- **Max retries per entry**: 3
- **Parallel workers**: {stats.get('workers', 10)}
- **LogP evaluation**: Based on solution token masking and probability computation
"""

async def main(args: Config):
    """Main evaluation pipeline"""
    
    logger.info("Starting rubric evaluation pipeline")
    logger.info(f"Configuration: {args}")
    
    # Initialize OpenAI client for vLLM
    client = AsyncOpenAI(
        base_url=args.base_url,
        timeout=86400,
        max_retries=3,
        api_key="placeholder",
    )
    
    # Test connection
    try:
        logger.info("Testing vLLM connection...")
        test_response = await client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": "Test"}],
            max_completion_tokens=10,
        )
        logger.info("vLLM connection successful")
    except Exception as e:
        logger.error(f"Failed to connect to vLLM: {e}")
        return
    
    # Load dataset (now expecting LogP-evaluated dataset)
    logger.info(f"Loading LogP-evaluated dataset: {args.dataset}")
    dataset = load_dataset(args.dataset, split=args.split)
    
    # Convert to list of dicts
    entries = list(dataset)
    logger.info(f"Loaded {len(entries)} entries from LogP-evaluated dataset")
    
    # Check if LogP scores exist
    logp_count = sum(1 for entry in entries if 'logp_score' in entry and entry['logp_score'] is not None)
    logger.info(f"Found {logp_count} entries with LogP scores")
    
    # Apply max_samples if specified
    if args.max_samples:
        entries = entries[:args.max_samples]
        logger.info(f"Limited to {args.max_samples} samples")
    
    # Prepare output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info(f"Starting rubric evaluation of {len(entries)} entries...")
    
    # Evaluate all entries
    results = await evaluate_all_entries(client, args, entries)
    
    # Save intermediate results
    intermediate_path = Path(args.output_dir) / "intermediate_results.jsonl"
    with open(intermediate_path, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    logger.info(f"Intermediate results saved to {intermediate_path}")
    
    # Calculate statistics
    stats = calculate_statistics(results)
    stats['workers'] = args.num_workers
    
    # Print statistics
    logger.info("\n=== Evaluation Statistics ===")
    logger.info(f"Total Samples: {stats['total_samples']}")
    logger.info(f"Successfully Evaluated (Rubric): {stats['evaluated_samples']}")
    logger.info(f"Failed Evaluations (Rubric): {stats['failed_samples']}")
    logger.info("\nRubric Average Scores:")
    for criterion, score in stats['rubric_stats']['average_scores'].items():
        logger.info(f"  {criterion}: {score:.3f}")
    
    logger.info("\nLogP Statistics:")
    logp_stats = stats['logp_stats']
    logger.info(f"  Total with LogP: {logp_stats['total_samples_with_logp']}")
    logger.info(f"  Valid LogP: {logp_stats['valid_logp_samples']}")
    logger.info(f"  Avg LogP: {logp_stats['avg_logp_score']:.4f}")
    logger.info(f"  Std LogP: {logp_stats['std_logp_score']:.4f}")
    
    # Save final results
    output_path = Path(args.output_dir) / "evaluation_results.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    logger.info(f"\nFinal results saved to {output_path}")
    
    # Save statistics
    stats_path = Path(args.output_dir) / "evaluation_statistics.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    logger.info(f"Statistics saved to {stats_path}")
    
    # Create Hugging Face dataset
    hf_dataset = Dataset.from_list(results)
    
    # Push to Hugging Face Hub (overwrite)
    if args.hf_hub_repo and args.hf_token:
        logger.info(f"\nOverwriting Hugging Face Hub dataset: {args.hf_hub_repo}")
        
        # Create README with both rubric and LogP statistics
        readme_content = create_readme_content(stats, args.model, args.dataset)
        readme_path = Path(args.output_dir) / "README.md"
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(readme_content)
        
        try:
            # Push dataset (this will overwrite existing dataset)
            hf_dataset.push_to_hub(
                repo_id=args.hf_hub_repo,
                token=args.hf_token,
                private=args.hf_hub_private,
            )
            
            # Upload README
            from huggingface_hub import HfApi
            api = HfApi()
            api.upload_file(
                path_or_fileobj=str(readme_path),
                path_in_repo="README.md",
                repo_id=args.hf_hub_repo,
                token=args.hf_token,
                repo_type="dataset"
            )
            
            logger.info("Dataset successfully overwritten on Hugging Face Hub!")
        except Exception as e:
            logger.error(f"Failed to push to Hugging Face Hub: {e}")
    
    return results