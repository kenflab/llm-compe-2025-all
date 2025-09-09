"""
Main dataset processing script for LogP evaluation using HuggingFace direct inference only
"""

import os
import json
import logging
import torch
from pathlib import Path
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import numpy as np
from typing import Dict, List
from datetime import datetime

from inference.logp_evaluator_core import RLTLogPEvaluator
from inference.hf_direct_logp_evaluator import HFDirectLogPEvaluator

# Setup logging
def setup_logging(log_level: str = "INFO", debug: bool = False):
    level = logging.DEBUG if debug else getattr(logging, log_level.upper())
    
    # Create logs directory
    os.makedirs("logs", exist_ok=True)
    
    # Log file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/logp_eval_{timestamp}.log"
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def evaluate_sample_hf(
    sample: Dict,
    evaluator: RLTLogPEvaluator,
    hf_evaluator: HFDirectLogPEvaluator,
    sample_idx: int,
    config
) -> Dict:
    """Evaluate a single sample using HuggingFace direct inference"""
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"Processing sample {sample_idx} with HF direct inference")
        
        question = sample.get("question_text", "")
        answer = sample.get("answer_text", "")
        chain_of_thought = sample.get("chain_of_thought", "")
        
        # Build chat and get masks using RLT evaluator
        chat_text, token_masks = evaluator.get_student_chat_and_masks(
            question, chain_of_thought, answer
        )
        
        # Use HF direct evaluator for RLT-exact computation
        logp_score = hf_evaluator.evaluate_sample_rlt_compatible(
            chat_text=chat_text,
            token_masks=token_masks,
            temperature=config.temperature,
            unbias_log_probabilities=config.unbias_log_probabilities,
            answer_log_prob_coeff=config.answer_log_prob_coeff[0] if isinstance(config.answer_log_prob_coeff, list) else config.answer_log_prob_coeff,
            normalize_log_prob_fn=config.normalize_log_prob_fn,
            reduction_log_prob_fn=config.reduction_log_prob_fn[0] if isinstance(config.reduction_log_prob_fn, list) else config.reduction_log_prob_fn,
            not_matched_penalty=config.not_matched_penalty
        )
        
        result = {
            **sample,
            "rubric_score": 0.0,
            "logp_score": float(logp_score)
        }
        
        logger.info(f"Sample {sample_idx} evaluated: LogP={logp_score:.4f}")
        return result
        
    except Exception as e:
        logger.error(f"Error evaluating sample {sample_idx}: {e}", exc_info=True)
        return {
            **sample,
            "rubric_score": 0.0,
            "logp_score": float('-inf')
        }

def process_dataset_hf(config, evaluator):
    """Process dataset using HuggingFace direct inference"""
    logger = logging.getLogger(__name__)
    
    # Initialize HF direct evaluator
    logger.info("Initializing HuggingFace direct LogP evaluator")
    hf_evaluator = HFDirectLogPEvaluator(
        model_name=config.model,
        device=config.hf_direct.device,
        torch_dtype=getattr(torch, config.hf_direct.torch_dtype),
        max_memory_per_gpu=config.hf_direct.max_memory_per_gpu
    )
    
    # Load dataset
    logger.info(f"Loading dataset: {config.dataset}")
    dataset = load_dataset(config.dataset, split=config.split)
    
    if config.max_samples:
        dataset = dataset.select(range(min(config.max_samples, len(dataset))))
    
    logger.info(f"Processing {len(dataset)} samples with HF direct inference")
    
    # Process samples
    evaluated_samples = []
    failed_samples = []
    
    for i in tqdm(range(len(dataset)), desc="Evaluating samples"):
        sample = dataset[i]
        
        try:
            result = evaluate_sample_hf(sample, evaluator, hf_evaluator, i, config)
            evaluated_samples.append(result)
        except Exception as e:
            logger.error(f"Sample {i} failed: {e}")
            failed_samples.append(i)
            
            evaluated_samples.append({
                **sample,
                "rubric_score": 0.0,
                "logp_score": float('-inf')
            })
    
    return evaluated_samples, failed_samples

def process_dataset(config):
    """Main processing function"""
    logger = setup_logging(config.log_level, config.debug)
    logger.info("Starting LogP evaluation")
    logger.info(f"Configuration: {config}")
    logger.info(f"Provider: {config.provider}")
    
    # Verify provider
    if config.provider != "hf_direct":
        raise ValueError(f"Only 'hf_direct' provider is supported for LogP evaluation, got: {config.provider}")
    
    # Initialize tokenizer
    logger.info(f"Loading tokenizer: {config.model}")
    tokenizer = AutoTokenizer.from_pretrained(config.model)
    
    # Initialize RLT evaluator
    evaluator = RLTLogPEvaluator(
        tokenizer=tokenizer,
        answer_log_prob_coeff=config.answer_log_prob_coeff,
        normalize_log_prob_fn=config.normalize_log_prob_fn,
        clip_log_prob=config.clip_log_prob,
        reduction_log_prob_fn=config.reduction_log_prob_fn,
        not_matched_penalty=config.not_matched_penalty,
        unbias_log_probabilities=config.unbias_log_probabilities,
        temperature=config.temperature,
        debug=config.debug
    )
    
    # Process using HF direct
    logger.info("Using HuggingFace direct inference for LogP evaluation")
    evaluated_samples, failed_samples = process_dataset_hf(config, evaluator)
    
    # Calculate statistics
    valid_scores = [s["logp_score"] for s in evaluated_samples 
                   if s["logp_score"] != float('-inf')]
    
    stats = {
        "total_samples": len(evaluated_samples),
        "valid_samples": len(valid_scores),
        "failed_samples": len(failed_samples),
        "avg_logp_score": np.mean(valid_scores) if valid_scores else 0,
        "std_logp_score": np.std(valid_scores) if valid_scores else 0,
        "min_logp_score": np.min(valid_scores) if valid_scores else 0,
        "max_logp_score": np.max(valid_scores) if valid_scores else 0,
        "provider": config.provider
    }
    
    logger.info("=== Evaluation Statistics ===")
    for key, value in stats.items():
        logger.info(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
    
    if failed_samples:
        logger.warning(f"Failed sample indices: {failed_samples}")
    
    # Save results
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save to JSONL
    output_file = output_dir / f"logp_evaluated_hf_direct_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for sample in evaluated_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    
    logger.info(f"Results saved to {output_file}")
    
    # Save statistics
    stats_file = output_dir / "evaluation_stats.json"
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)
    
    # Upload to Hugging Face if configured
    if config.hf_hub_repo and config.hf_token:
        logger.info(f"Uploading to Hugging Face: {config.hf_hub_repo}")
        
        try:
            os.environ["HF_TOKEN"] = config.hf_token
            
            hf_dataset = Dataset.from_list(evaluated_samples)
            
            readme_content = f"""---
language:
- en
- ja
license: mit
task_categories:
- text-generation
tags:
- logp-evaluation
- reasoning
- rlt-based
---

# LogP Evaluated Dataset

This dataset contains LogP evaluation scores computed using RLT-based evaluation logic with HuggingFace direct inference.

## Configuration
- Provider: {config.provider}
- Model: {config.model}
- Temperature: {config.temperature}
- Normalization: {config.normalize_log_prob_fn}
- Reduction: {config.reduction_log_prob_fn}
- Coefficients: {config.answer_log_prob_coeff}

## Statistics
- Total samples: {stats['total_samples']}
- Valid samples: {stats['valid_samples']}
- Failed samples: {stats['failed_samples']}
- Average LogP: {stats['avg_logp_score']:.4f}
- Std LogP: {stats['std_logp_score']:.4f}
- Min LogP: {stats['min_logp_score']:.4f}
- Max LogP: {stats['max_logp_score']:.4f}

## Evaluation Method
Uses RLT TeacherKLBasedReward logic for computing masked LogP scores on solution parts only.
Provider: HuggingFace direct inference - provides RLT-exact computation for accurate LogP evaluation.

## Usage
This dataset can be used for:
- Training reward models based on LogP scores
- Evaluating reasoning quality through probability analysis
- Research on mathematical reasoning evaluation metrics
"""
            
            readme_path = output_dir / "README.md"
            with open(readme_path, "w") as f:
                f.write(readme_content)
            
            hf_dataset.push_to_hub(
                repo_id=config.hf_hub_repo,
                token=config.hf_token,
                private=config.hf_hub_private
            )
            
            logger.info(f"Successfully uploaded to {config.hf_hub_repo}")
            
        except Exception as e:
            logger.error(f"Failed to upload to Hugging Face: {e}", exc_info=True)
    
    return evaluated_samples, stats

def main(config):
    """Main entry point"""
    return process_dataset(config)