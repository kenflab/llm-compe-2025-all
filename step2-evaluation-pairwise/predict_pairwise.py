import hydra
from omegaconf import DictConfig
from inference._configs_pairwise import PairwiseConfig

@hydra.main(config_name="config_pairwise", version_base=None, config_path="conf")
def main(cfg: DictConfig):
    # Convert OmegaConf to dataclass
    config = PairwiseConfig(
        dataset1=cfg.dataset1,
        dataset2=cfg.dataset2,
        provider=cfg.provider,
        base_url=cfg.base_url,
        model=cfg.model,
        max_completion_tokens=cfg.max_completion_tokens,
        reasoning=cfg.reasoning,
        num_workers=cfg.num_workers,
        max_samples=cfg.max_samples,
        temperature=cfg.temperature,
        output_dir=cfg.output_dir,
        hf_hub_repo=cfg.hf_hub_repo,
        hf_hub_private=cfg.hf_hub_private,
        hf_token=cfg.hf_token,
    )
    
    if cfg.provider == "vllm":
        from inference import pairwise_comparison
        pairwise_comparison.main(config)

if __name__ == "__main__":
    main()