"""
Direct HuggingFace model inference for RLT-compatible LogP evaluation
This provides the most accurate replication of RLT's computation
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
from typing import List, Tuple
import gc

logger = logging.getLogger(__name__)

class HFDirectLogPEvaluator:
    """
    Direct HuggingFace model inference to exactly replicate RLT's LogP computation
    """
    
    def __init__(
        self, 
        model_name: str, 
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        max_memory_per_gpu: str = "40GB"
    ):
        self.model_name = model_name
        self.device = device
        self.torch_dtype = torch_dtype
        
        logger.info(f"Loading model {model_name} for direct LogP evaluation")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.padding_side = 'left'
        
        # Check if accelerate is available
        try:
            import accelerate
            use_device_map = True
            logger.info("Using accelerate for device mapping")
        except ImportError:
            use_device_map = False
            logger.warning("accelerate not available, using manual device placement")
        
        # Load model with appropriate method
        if use_device_map:
            # Use device_map when accelerate is available
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                device_map="auto",
                max_memory={i: max_memory_per_gpu for i in range(torch.cuda.device_count())},
                trust_remote_code=True,
            )
        else:
            # Fallback to manual device placement
            logger.info(f"Loading model to {device}")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
            )
            # Move to device manually
            self.model = self.model.to(device)
        
        self.model.eval()
        
        logger.info("Model loaded successfully for LogP evaluation")
    
    @torch.no_grad()
    def compute_token_logprobs_rlt_exact(
        self, 
        text: str, 
        temperature: float = 1.0,
        max_chunk_size: int = 2048
    ) -> torch.Tensor:
        """
        Compute token logprobs using RLT's exact method
        
        This replicates the exact computation from:
        teacher_rewards.py: compute_batch_log_probs()
        """
        # Tokenize exactly like RLT
        encoding = self.tokenizer(
            text, 
            return_tensors='pt', 
            padding=True, 
            truncation=True
        ).to(self.model.device)
        
        input_ids = encoding.input_ids
        attention_mask = encoding.get('attention_mask', None)
        
        # Handle long sequences with chunking if needed
        if input_ids.size(1) > max_chunk_size:
            return self._compute_chunked_logprobs(input_ids, attention_mask, temperature, max_chunk_size)
        
        try:
            # Forward pass to get logits (RLT style)
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits[:, :-1, :]  # Remove last position
            labels = input_ids[:, 1:]  # Shift labels (RLT style)
            
            # Apply temperature scaling exactly like RLT
            if temperature != 1.0:
                logits = logits / temperature
            
            # Compute log probabilities exactly like RLT
            single_token_log_probs = []
            for i in range(logits.size(0)):
                # Per-batch computation exactly like RLT
                b_log_probs = F.log_softmax(logits[i], dim=-1)
                b_labels = labels[i].unsqueeze(-1)
                b_token_log_probs = b_log_probs.gather(1, b_labels).squeeze(-1)
                single_token_log_probs.append(b_token_log_probs)
            
            # Stack exactly like RLT
            token_log_probs = torch.stack(single_token_log_probs, dim=0)
            
            # Clean up GPU memory
            del outputs, logits
            torch.cuda.empty_cache()
            
            return token_log_probs
            
        except torch.cuda.OutOfMemoryError:
            logger.warning("OOM during inference, trying with reduced precision")
            # Fallback to chunked computation
            return self._compute_chunked_logprobs(input_ids, attention_mask, temperature, max_chunk_size // 2)
    
    def _compute_chunked_logprobs(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor, 
        temperature: float,
        chunk_size: int
    ) -> torch.Tensor:
        """
        Compute logprobs for long sequences using chunking
        This maintains compatibility with RLT's computation while handling memory constraints
        """
        batch_size, seq_length = input_ids.shape
        all_token_log_probs = []
        
        for start_idx in range(0, seq_length - 1, chunk_size):
            end_idx = min(start_idx + chunk_size + 1, seq_length)  # +1 for label alignment
            
            chunk_input_ids = input_ids[:, start_idx:end_idx]
            chunk_attention_mask = attention_mask[:, start_idx:end_idx] if attention_mask is not None else None
            
            # Compute logits for chunk
            outputs = self.model(input_ids=chunk_input_ids, attention_mask=chunk_attention_mask)
            chunk_logits = outputs.logits[:, :-1, :]
            chunk_labels = chunk_input_ids[:, 1:]
            
            # Apply temperature
            if temperature != 1.0:
                chunk_logits = chunk_logits / temperature
            
            # Compute log probs for chunk (RLT style)
            chunk_token_log_probs = []
            for i in range(chunk_logits.size(0)):
                b_log_probs = F.log_softmax(chunk_logits[i], dim=-1)
                b_labels = chunk_labels[i].unsqueeze(-1)
                b_token_log_probs = b_log_probs.gather(1, b_labels).squeeze(-1)
                chunk_token_log_probs.append(b_token_log_probs)
            
            chunk_result = torch.stack(chunk_token_log_probs, dim=0)
            all_token_log_probs.append(chunk_result)
            
            # Clean up
            del outputs, chunk_logits
            torch.cuda.empty_cache()
        
        # Concatenate all chunks
        return torch.cat(all_token_log_probs, dim=1)
    
    def evaluate_sample_rlt_compatible(
        self, 
        chat_text: str, 
        token_masks, 
        temperature: float = 0.7,
        unbias_log_probabilities: bool = True,
        answer_log_prob_coeff: float = 1.0,
        normalize_log_prob_fn: str = 'exp',
        reduction_log_prob_fn: str = 'mean',
        not_matched_penalty: float = -1.0
    ) -> float:
        """
        Evaluate a sample using RLT-compatible computation
        """
        # Get temperature for unbiasing
        eval_temperature = temperature if unbias_log_probabilities else 1.0
        
        # Compute token logprobs exactly like RLT
        token_log_probs = self.compute_token_logprobs_rlt_exact(
            chat_text, temperature=eval_temperature
        )
        
        # Apply solution mask (only evaluate solution tokens)
        solution_mask = token_masks.solution_mask[:token_log_probs.size(-1)]
        
        if solution_mask.sum() == 0:
            logger.warning("No solution tokens found, returning penalty")
            return not_matched_penalty
        
        # *** 修正: solution_maskを同じデバイスに移動 ***
        solution_mask = solution_mask.to(token_log_probs.device)
        
        # Apply normalization (RLT style)
        if normalize_log_prob_fn == 'exp':
            processed_log_probs = torch.exp(token_log_probs)
        elif normalize_log_prob_fn == 'none':
            processed_log_probs = token_log_probs
        else:
            processed_log_probs = token_log_probs
        
        # Apply reduction (RLT style)
        masked_log_probs = processed_log_probs * solution_mask.float()
        
        if reduction_log_prob_fn == 'mean':
            log_prob_score = masked_log_probs.sum() / solution_mask.sum()
        elif reduction_log_prob_fn == 'sum':
            log_prob_score = masked_log_probs.sum()
        else:
            log_prob_score = masked_log_probs.sum() / solution_mask.sum()
        
        # Apply coefficient
        final_score = log_prob_score * answer_log_prob_coeff
        
        return final_score.item()