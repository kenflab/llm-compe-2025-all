"""
Core LogP evaluation logic directly adapted from RLT's teacher_rewards.py
"""

import logging
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Callable
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class TokenMasks:
    """Store token masks for different parts of the response"""
    full_tokens: List[int]
    solution_mask: torch.Tensor
    thought_mask: torch.Tensor
    attention_mask: torch.Tensor

class RLTLogPEvaluator:
    """
    LogP evaluation using exact logic from RLT's TeacherKLBasedReward
    """
    
    def __init__(
        self,
        tokenizer,
        answer_log_prob_coeff: Union[float, List[float]] = 1.0,
        normalize_log_prob_fn: str = 'exp',
        clip_log_prob: Optional[float] = None,
        reduction_log_prob_fn: Union[str, List[str]] = 'mean',
        not_matched_penalty: float = -1.0,
        unbias_log_probabilities: bool = True,
        temperature: float = 0.7,
        debug: bool = False
    ):
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = 'left'
        
        # Parameters from RLT
        self.answer_log_prob_coeff = answer_log_prob_coeff
        self.not_matched_penalty = not_matched_penalty
        self.unbias_log_probabilities = unbias_log_probabilities
        self.temperature = temperature if unbias_log_probabilities else 1.0
        self.debug = debug
        
        # Initialize normalization and reduction functions (from TeacherReward)
        self.normalize_log_prob_fn = self._make_normalize_fn(
            normalize_log_prob_fn, temp=1, clip_min=-clip_log_prob if clip_log_prob else None
        )
        self.reduction_log_prob_fn, self.log_lp_names = self._make_reduction_fn(
            reduction_log_prob_fn, function_log_name='answer_log_prob'
        )
        
        # Convert coefficients to tensor if needed
        if isinstance(self.answer_log_prob_coeff, (list, tuple)):
            self.answer_log_prob_coeff = torch.tensor(self.answer_log_prob_coeff)
            self.use_answer_log_prob_coeff = True
        else:
            self.use_answer_log_prob_coeff = self.answer_log_prob_coeff > 0
        
        # Tags for parsing (STRATOS format)
        self.thought_start_tag = "<|begin_of_thought|>"
        self.thought_end_tag = "<|end_of_thought|>"
        self.solution_start_tag = "<|begin_of_solution|>"
        self.solution_end_tag = "<|end_of_solution|>"
        
        logger.info(f"Initialized RLTLogPEvaluator with temperature={temperature}, "
                   f"normalize_fn={normalize_log_prob_fn}, reduction_fn={reduction_log_prob_fn}")
    
    def _make_normalize_fn(self, normalize_fn, temp=1, clip_min=None, clip_max=None):
        """Direct copy from TeacherReward._make_normalize_fn"""
        if isinstance(normalize_fn, str):
            normalize_fn = normalize_fn.lower()
        elif isinstance(normalize_fn, Callable):
            return normalize_fn
        
        def apply_clipping(x):
            if clip_min is not None:
                x = torch.clamp(x, min=clip_min)
            if clip_max is not None:
                x = torch.clamp(x, max=clip_max)
            return x
        
        if normalize_fn is None or normalize_fn == 'none':
            def f(x):
                return apply_clipping(x / temp)
        elif normalize_fn == 'exp':
            def f(x):
                return apply_clipping(torch.exp(x / temp))
        elif normalize_fn == 'exp_norm':
            def f(x):
                return apply_clipping(1 - torch.exp(-x / temp))
        else:
            raise NotImplementedError(f"Unknown normalization function: {normalize_fn}")
        
        logger.debug(f"Created normalization function: {normalize_fn}")
        return f
    
    def _make_reduction_fn(self, reduction_fn, function_log_name=None):
        """Direct copy from TeacherReward._make_reduction_fn"""
        if isinstance(reduction_fn, Callable):
            if function_log_name is not None:
                log_names_to_store = [function_log_name + '_custom']
                return reduction_fn, log_names_to_store
            return reduction_fn
        
        def _flatten(seq):
            for i in seq:
                if isinstance(i, (list, tuple)) and not isinstance(i, str):
                    yield from _flatten(i)
                else:
                    yield i
        
        if isinstance(reduction_fn, str):
            ops = [reduction_fn.lower()]
        elif isinstance(reduction_fn, (list, tuple)):
            ops = [op.lower() for op in _flatten(reduction_fn)]
        else:
            raise NotImplementedError(f"Unknown reduction function type: {type(reduction_fn)}")
        
        def f(x, mask):
            out = []
            for op in ops:
                try:
                    if op == 'mean':
                        o = torch.sum(x * mask, dim=-1) / torch.sum(mask, dim=-1)
                    elif op == 'sum':
                        o = torch.sum(x * mask, dim=-1)
                    elif op == 'min':
                        tmp = x.masked_fill(mask == 0, torch.finfo(x.dtype).max)
                        o = torch.min(tmp, dim=-1).values
                    elif op == 'max':
                        tmp = x.masked_fill(mask == 0, torch.finfo(x.dtype).min)
                        o = torch.max(tmp, dim=-1).values
                    elif op == 'median':
                        tmp = x.masked_fill(mask == 0, float('nan'))
                        o = torch.nanmedian(tmp, dim=-1).values
                    else:
                        raise NotImplementedError(f"Unknown reduction op: {op}")
                except Exception as e:
                    logger.error(f"Error in reduction op {op}: {e}")
                    o = torch.full(x.shape[:-1], float('nan'), dtype=x.dtype, device=x.device)
                out.append(o)
            return torch.stack(out, dim=-1)
        
        if function_log_name is not None:
            log_names_to_store = [function_log_name + '_' + op for op in ops]
            logger.debug(f"Created reduction function with ops: {ops}")
            return f, log_names_to_store
        return f
    
    def get_student_chat_and_masks(self, question: str, chain_of_thought: str, answer: str) -> Tuple[str, TokenMasks]:
        """
        Build student chat and get token masks following RLT's approach
        Based on teacher_rewards.py: get_student_chats_and_relevant_num_tokens
        """
        logger.debug("Building student chat format")
        
        # System prompt from RLT
        system_prompt = """Your role as an assistant involves thoroughly exploring questions through a systematic long thinking process before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop well-considered thinking process. Please structure your response into two main sections: Thought and Solution. In the Thought section, detail your reasoning process using the specified format: <|begin_of_thought|> {thought with steps separated with '\\n\\n'} <|end_of_thought|> Each step should include detailed considerations such as analisying questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps. In the Solution section, based on various attempts, explorations, and reflections from the Thought section, systematically present the final solution that you deem correct. The solution should remain a logical, accurate, concise expression style and detail necessary step needed to reach the conclusion, formatted as follows: <|begin_of_solution|> {final formatted, precise, and clear solution} <|end_of_solution|> Now, try to solve the following question through the above guidelines:"""
        
        # Format assistant content with tags
        assistant_content = (
            f"{self.thought_start_tag}\n"
            f"{chain_of_thought}\n"
            f"{self.thought_end_tag}\n\n"
            f"{self.solution_start_tag}\n"
            f"{answer}\n"
            f"{self.solution_end_tag}"
        )
        
        # Build chat messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
            {"role": "assistant", "content": assistant_content}
        ]
        
        # Apply chat template
        chat_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        
        # Tokenize
        encoding = self.tokenizer(chat_text, return_tensors='pt', padding=False, truncation=False)
        tokens = encoding.input_ids[0].tolist()
        
        # Get masks for solution part
        token_masks = self._get_token_masks_for_spans(chat_text, assistant_content, tokens)
        
        logger.debug(f"Chat tokenized: {len(tokens)} tokens, "
                    f"solution mask sum: {token_masks.solution_mask.sum().item()}")
        
        return chat_text, token_masks
    
    def _get_token_masks_for_spans(self, chat_text: str, assistant_content: str, tokens: List[int]) -> TokenMasks:
        """
        Get token masks following RLT's get_mask_for_spans approach
        """
        # Find solution content
        solution_pattern = (
            re.escape(self.solution_start_tag) + r"(.*?)" + re.escape(self.solution_end_tag)
        )
        solution_match = re.search(solution_pattern, assistant_content, re.DOTALL)
        
        match_reward = 0.0
        if not solution_match:
            logger.warning("Solution tags not found, applying penalty")
            match_reward = self.not_matched_penalty
            # Try to recover by adding missing tags
            if self.solution_end_tag not in assistant_content:
                assistant_content += self.solution_end_tag
                solution_match = re.search(solution_pattern, assistant_content, re.DOTALL)
        
        # Extract solution content
        if solution_match:
            solution_content = solution_match.group(1).strip()
            solution_tokens = self.tokenizer.encode(solution_content, add_special_tokens=False)
            
            # Find solution tokens in full sequence
            solution_start_idx, solution_end_idx = self._find_subsequence_indices(tokens, solution_tokens)
        else:
            solution_start_idx, solution_end_idx = -1, -1
        
        # Create masks
        seq_len = len(tokens)
        solution_mask = torch.zeros(seq_len, dtype=torch.bool)
        if solution_start_idx >= 0:
            solution_mask[solution_start_idx:solution_end_idx] = True
        
        # Similarly for thought content
        thought_mask = torch.zeros(seq_len, dtype=torch.bool)
        thought_pattern = (
            re.escape(self.thought_start_tag) + r"(.*?)" + re.escape(self.thought_end_tag)
        )
        thought_match = re.search(thought_pattern, assistant_content, re.DOTALL)
        
        if thought_match:
            thought_content = thought_match.group(1).strip()
            thought_tokens = self.tokenizer.encode(thought_content, add_special_tokens=False)
            thought_start_idx, thought_end_idx = self._find_subsequence_indices(tokens, thought_tokens)
            if thought_start_idx >= 0:
                thought_mask[thought_start_idx:thought_end_idx] = True
        
        # Attention mask (all ones for this case)
        attention_mask = torch.ones(seq_len, dtype=torch.bool)
        
        return TokenMasks(
            full_tokens=tokens,
            solution_mask=solution_mask,
            thought_mask=thought_mask,
            attention_mask=attention_mask
        )
    
    def _find_subsequence_indices(self, tokens: List[int], subseq: List[int]) -> Tuple[int, int]:
        """Find subsequence in token list and return start and end indices"""
        for i in range(len(tokens) - len(subseq) + 1):
            if tokens[i:i+len(subseq)] == subseq:
                return i, i + len(subseq)
        
        # Try approximate match if exact match fails
        if self.debug:
            logger.debug("Exact token match failed, trying approximate match")
        
        best_idx = -1
        best_score = 0
        threshold = 0.8
        
        for i in range(len(tokens) - len(subseq) + 1):
            matches = sum(1 for j in range(len(subseq)) 
                         if i+j < len(tokens) and tokens[i+j] == subseq[j])
            score = matches / len(subseq)
            if score > best_score and score >= threshold:
                best_score = score
                best_idx = i
        
        if best_idx >= 0:
            return best_idx, best_idx + len(subseq)
        
        logger.warning("Could not find token subsequence even with approximate matching")
        return -1, -1
    
    def compute_logp_from_token_logprobs(self, token_logprobs: torch.Tensor, token_masks: TokenMasks) -> float:
        """
        Compute LogP score from token logprobs following RLT's approach
        """
        logger.debug("Computing LogP from token logprobs")
        
        # Apply temperature scaling if needed
        if self.unbias_log_probabilities:
            token_logprobs = token_logprobs / self.temperature
        
        # Adjust mask to match logprobs length (no shift needed for direct logprobs)
        solution_mask = token_masks.solution_mask[:len(token_logprobs)]
        
        # Apply normalization function
        processed_log_probs = self.normalize_log_prob_fn(token_logprobs)
        
        # Apply reduction
        if solution_mask.sum() > 0:
            # Add batch dimension for compatibility
            processed_log_probs = processed_log_probs.unsqueeze(0)
            solution_mask = solution_mask.unsqueeze(0)
            
            log_prob_scores = self.reduction_log_prob_fn(processed_log_probs, solution_mask)
            
            # Handle NaN values
            log_prob_scores = torch.nan_to_num(
                log_prob_scores, nan=self.not_matched_penalty
            )
            
            # Apply coefficients and sum
            if self.use_answer_log_prob_coeff:
                log_prob_reward = (log_prob_scores * self.answer_log_prob_coeff).sum(-1).item()
            else:
                log_prob_reward = log_prob_scores.sum(-1).item()
        else:
            logger.warning("No solution tokens found, returning penalty")
            log_prob_reward = self.not_matched_penalty
        
        logger.debug(f"Computed LogP reward: {log_prob_reward}")
        return log_prob_reward