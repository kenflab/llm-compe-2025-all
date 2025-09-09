from dataclasses import dataclass
from typing import Optional, List, Union

@dataclass
class HFDirectConfig:
    torch_dtype: str = "bfloat16"
    device: str = "cuda"
    max_memory_per_gpu: str = "40GB"
    max_chunk_size: int = 2048
    use_accelerate: bool = False

@dataclass
class LogPEvalConfig:
    dataset: str
    split: str
    provider: str  # "vllm" or "hf_direct"
    base_url: str
    model: str
    max_completion_tokens: int
    reasoning: bool
    num_workers: int
    max_samples: Optional[int]
    temperature: float
    output_dir: str
    hf_hub_repo: str
    hf_hub_private: bool
    hf_token: str
    
    # LogP evaluation parameters from RLT
    answer_log_prob_coeff: Union[float, List[float]]
    normalize_log_prob_fn: str
    clip_log_prob: float
    reduction_log_prob_fn: Union[str, List[str]]
    not_matched_penalty: float
    unbias_log_probabilities: bool
    
    # HF Direct configuration
    hf_direct: HFDirectConfig
    
    # Logging
    debug: bool
    log_level: str