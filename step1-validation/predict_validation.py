import hydra

@hydra.main(config_name="config_validation", version_base=None, config_path="conf")
def main(cfg):
    if cfg.provider == "vllm":
        from inference import tag_validation_predictions
        tag_validation_predictions.main(cfg)

if __name__ == "__main__":
    main()
