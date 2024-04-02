from typing import Dict

import lightning as L
from torch.utils.data import DataLoader

__all__ = ["DataModule"]


class DataModule(L.LightningDataModule):
    def __init__(self, dataloaders: Dict[str, DataLoader]) -> None:
        super().__init__()
        self.dataloaders = dataloaders

    def train_dataloader(self) -> DataLoader:
        return self.dataloaders["train"]

    def val_dataloader(self) -> DataLoader:
        return self.dataloaders["val"]
