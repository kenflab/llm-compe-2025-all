from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class Config:
    dataset: str
    split: str
    provider: str
    base_url: str
    model: str
    max_completion_tokens: int
    num_workers: int
    max_samples: Optional[int]
    temperature: float
    output_dir: str
    hf_hub_repo: str
    hf_hub_private: bool
    hf_token: str
    rubric_weights: Dict[str, float]