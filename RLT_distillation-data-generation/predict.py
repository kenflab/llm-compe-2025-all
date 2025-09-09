import hydra

@hydra.main(config_name="config", version_base=None, config_path="conf")
def main(cfg):
    if cfg.provider == "vllm":
        from inference import vllm_predictions
        vllm_predictions.main(cfg)

if __name__ == "__main__":
    main()
