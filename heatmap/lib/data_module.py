import pathlib
import tempfile

import numpy as np
from sklearn.model_selection import train_test_split

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
            if 'image_npy_path' not in self.hparams.dataset['validation']:
                if self._no_validation:
                    self._train_dataset = self._build_dataset(**self.hparams.dataset['train'], phase=DatasetPhase.TRAIN)
                    # TrainデータのValとして使いまわす（処理時間短縮のために1%のデータを使う）
                    temp_dataset = self._build_dataset(
                        **self.hparams.dataset['train'], phase=DatasetPhase.VALIDATION)
                    val_size = int(len(temp_dataset) * 0.01)
                    train_size = len(temp_dataset) - val_size
                    _, self._val_dataset = random_split(temp_dataset, [train_size, val_size])
                else:
                    with tempfile.TemporaryDirectory() as temp_dir_name:
                        images = np.load(self.hparams.dataset['train']['image_npy_path'])
                        targets = np.load(self.hparams.dataset['train']['target_npy_path'])
                        centers = np.load(self.hparams.dataset['train']['center_npy_path'])
                        center_targets = np.load(self.hparams.dataset['train']['center_target_npy_path'])

                        indecis = np.array(list(range(len(images))))
                        train_indecis, val_indecis = train_test_split(indecis, test_size=0.2, random_state=42)

                        train_images = images[train_indecis]
                        train_targets = targets[train_indecis]
                        train_centers = centers[train_indecis]
                        train_center_targets = center_targets[train_indecis]
                        val_images = images[val_indecis]
                        val_targets = targets[val_indecis]
                        val_centers = centers[val_indecis]
                        val_center_targets = center_targets[val_indecis]

                        temp_dir_path = pathlib.Path(temp_dir_name)
                        np.save(temp_dir_path / 'X_train', train_images)
                        np.save(temp_dir_path / 'y_train', train_targets)
                        np.save(temp_dir_path / 'y_train_center', train_centers)
                        np.save(temp_dir_path / 'y_train_center_target', train_center_targets)
                        np.save(temp_dir_path / 'X_val', val_images)
                        np.save(temp_dir_path / 'y_val', val_targets)
                        np.save(temp_dir_path / 'y_val_center', val_centers)
                        np.save(temp_dir_path / 'y_val_center_target', val_center_targets)
                        self.hparams.dataset['train']['image_npy_path'] = (temp_dir_path / 'X_train.npy').as_posix()
                        self.hparams.dataset['train']['target_npy_path'] = (temp_dir_path / 'y_train.npy').as_posix()
                        self.hparams.dataset['train']['center_npy_path'] = (
                            temp_dir_path / 'y_train_center.npy').as_posix()
                        self.hparams.dataset['train']['center_target_npy_path'] = (
                            temp_dir_path / 'y_train_center_target.npy').as_posix()
                        self.hparams.dataset['validation']['image_npy_path'] = (temp_dir_path / 'X_val.npy').as_posix()
                        self.hparams.dataset['validation']['target_npy_path'] = (temp_dir_path / 'y_val.npy').as_posix()
                        self.hparams.dataset['validation']['center_npy_path'] = (
                            temp_dir_path / 'y_val_center.npy').as_posix()
                        self.hparams.dataset['validation']['center_target_npy_path'] = (
                            temp_dir_path / 'y_val_center_target.npy').as_posix()
                        self._train_dataset = self._build_dataset(
                            **self.hparams.dataset['train'], phase=DatasetPhase.TRAIN)
                        self._val_dataset = self._build_dataset(
                            **self.hparams.dataset['validation'], phase=DatasetPhase.VALIDATION)
            else:
                self._train_dataset = self._build_dataset(**self.hparams.dataset['train'], phase=DatasetPhase.TRAIN)
                self._val_dataset = self._build_dataset(
                    **self.hparams.dataset['validation'], phase=DatasetPhase.VALIDATION)
        else:
            self._test_dataset = self._build_dataset(**self.hparams.dataset['test'], phase=DatasetPhase.TEST)

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
