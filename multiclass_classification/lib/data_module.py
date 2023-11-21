import copy
import pathlib

import lightning.pytorch as pl

from torch.utils.data import Dataset, DataLoader, random_split

from lib.datasets import DatasetPhase, SonyDiceDataset
import lib.datasets as datasets_module
import lib.dataloaders as dataloaders_module


class ExampleDataModule(pl.LightningDataModule):
    def __init__(self,
                 no_validation: bool,
                 dataset: dict,
                 dataloader: dict):
        super().__init__()
        self._no_validation = no_validation
        self._dataset_cfg = dataset
        self._dataloader_cfg = dataloader
        self.save_hyperparameters()

    @property
    def train_dataset(self) -> SonyDiceDataset:
        return self._train_dataset

    @property
    def dataloader_cfg(self) -> dict:
        return self._dataloader_cfg

    def setup(self, stage):
        if stage == 'fit':
            self._train_dataset = self._build_dataset(**self.hparams.dataset['train'],
                                                      phase=DatasetPhase.TRAIN)
            if self._no_validation:
                # TrainデータのValとして使いまわす（処理時間短縮のために0.1%のデータを使う）
                val_cfg = copy.deepcopy(self.hparams.dataset['validation'])
                val_cfg['image_dir'] = self.hparams.dataset['train']['image_dir']
                temp_dataset = self._build_dataset(**val_cfg,
                                                   phase=DatasetPhase.VALIDATION)
                val_size = int(len(temp_dataset) * 0.001)
                train_size = len(temp_dataset) - val_size
                _, self._val_dataset = random_split(temp_dataset, [train_size, val_size])
            else:
                self._val_dataset = self._build_dataset(**self.hparams.dataset['validation'],
                                                        phase=DatasetPhase.VALIDATION)
        else:
            self._test_dataset = self._build_dataset(**self.hparams.dataset['test'],
                                                     phase=DatasetPhase.TEST)

    def train_dataloader(self):
        return self._build_dataloader(**self.hparams.dataloader['train'], dataset=self._train_dataset)

    def val_dataloader(self):
        return self._build_dataloader(**self.hparams.dataloader['validation'], dataset=self._val_dataset)

    def test_dataloader(self):
        return self._build_dataloader(**self.hparams.dataloader['test'], dataset=self._test_dataset)

    def predict_dataloader(self):
        return self._build_dataloader(**self.hparams.dataloader['test'], dataset=self._test_dataset)

    def _build_dataset(self,
                       name: str,
                       phase: DatasetPhase,
                       **kwargs: dict
                       ) -> Dataset:
        datasets_class = getattr(datasets_module, name)
        return datasets_class(**kwargs, phase=phase)

    def _build_dataloader(self,
                          name: str,
                          **kwargs: dict
                          ) -> DataLoader:
        dataloaders_class = getattr(dataloaders_module, name)
        return dataloaders_class(**kwargs)
