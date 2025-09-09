#!/usr/bin/env python
"""
Main entry point for LogP evaluation using HuggingFace direct inference only
"""

import hydra
import os
from omegaconf import OmegaConf

@hydra.main(config_name="config_logp_eval", version_base=None, config_path="conf")
def main(cfg):
    # Set environment variables
    if cfg.hf_token:
        os.environ["HF_TOKEN"] = cfg.hf_token
        os.environ["HUGGINGFACE_HUB_TOKEN"] = cfg.hf_token
    
    # Import config class
    from inference.logp_eval_configs import LogPEvalConfig, HFDirectConfig
    
    # Convert Hydra config to dataclass
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    
    # Handle HF direct config
    hf_direct_dict = cfg_dict.pop('hf_direct', {})
    hf_direct_config = HFDirectConfig(**hf_direct_dict)
    
    # Create main config
    config = LogPEvalConfig(**cfg_dict, hf_direct=hf_direct_config)
    
    print(f"Using provider: {config.provider}")
    
    if config.provider != "hf_direct":
        raise ValueError(f"Only 'hf_direct' provider is supported for LogP evaluation, got: {config.provider}")
    
    print("Starting HuggingFace direct inference evaluation...")
    from inference.dataset_logp_processor import main as process_main
    process_main(config)

if __name__ == "__main__":
    main()