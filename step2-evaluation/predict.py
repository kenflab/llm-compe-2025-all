import hydra
import asyncio

@hydra.main(config_name="config", version_base=None, config_path="conf")
def main(cfg):
    if cfg.provider == "vllm":
        from inference import rubric_evaluation
        asyncio.run(rubric_evaluation.main(cfg))

if __name__ == "__main__":
    main()