import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(
    version_base=None,
    config_name="base",
    config_path="config",
)
def run(config: DictConfig) -> None:
    """.

    Args:
        config (DictConfig): .
    """
    print(OmegaConf.to_yaml(config))

    from extract import ExtractionAnalysis

    analysis = ExtractionAnalysis(config)
    analysis.extract()


if __name__ == "__main__":
    run()
