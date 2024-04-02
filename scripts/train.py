import os

import hydra
import lightning as L
from omegaconf import DictConfig, OmegaConf

from lidardm.core.utils import instantiate_data, instantiate_model, instantiate_trainer

CONFIG_PATH = os.path.join(os.getcwd(), "lidardm", "core", "configs")
CONFIG_NAME = "default.yaml"


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)

    L.seed_everything(cfg.run.seed, workers=True)

    data: L.LightningDataModule = instantiate_data(cfg)
    model: L.LightningModule = instantiate_model(cfg)

    trainer: L.Trainer = instantiate_trainer(cfg)
    trainer.fit(model, datamodule=data)


if __name__ == "__main__":
    main()
