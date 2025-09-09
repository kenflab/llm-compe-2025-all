from dataclasses import dataclass

@dataclass
class PairwiseConfig:
    dataset1: str
    dataset2: str
    provider: str
    base_url: str
    model: str
    max_completion_tokens: int
    reasoning: bool
    num_workers: int
    max_samples: int
    temperature: float
    output_dir: str
    hf_hub_repo: str
    hf_hub_private: bool
    hf_token: str