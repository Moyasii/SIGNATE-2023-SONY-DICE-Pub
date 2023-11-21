from typing import Any, Optional
from collections import OrderedDict

import numpy as np

import torch
import torch.nn as nn

from torchmetrics.classification import (
    MulticlassAUROC,
    MulticlassAveragePrecision,
    MulticlassAccuracy,
)

import lightning.pytorch as pl

import lib.networks as networks_module


class ExampleModule(pl.LightningModule):
    def __init__(self,
                 model: dict,
                 optimizer: dict,
                 scheduler: dict,
                 predict: dict = dict()):
        super().__init__()
        self.save_hyperparameters()
        # https://lightning.ai/docs/pytorch/stable/common/lightning_module.html?highlight=validation_epoch_end
        self.training_step_outputs = []
        self.validation_step_outputs = []

        self.model = self._build_model(**self.hparams.model)
        self.num_classes = self.model._num_classes
        self.metric_accuracy = MulticlassAccuracy(num_classes=self.num_classes)  # Accuracy
        self.metric_auc = MulticlassAUROC(num_classes=self.num_classes, thresholds=None)  # AUC
        self.metric_ap = MulticlassAveragePrecision(num_classes=self.num_classes, thresholds=None)  # AP

        self._set_predict_settings(**self.hparams.predict)

    def _build_model(self, name: str, **kwargs: dict) -> nn.Module:
        model_class = getattr(networks_module, name)
        return model_class(**kwargs)

    def _build_optimizer(self, name: str, **kwargs: dict) -> torch.optim.Optimizer:
        optimizer_class = getattr(torch.optim, name)
        return optimizer_class(self.parameters(), **kwargs)

    def _build_scheduler(self, name: str, **kwargs: dict) -> list:
        epochs = self.trainer.max_epochs
        batch_size = self.trainer.datamodule.dataloader_cfg['train']['batch_size']
        steps_per_epoch = len(self.trainer.datamodule.train_dataset) // batch_size
        if name == 'CosineAnnealingLR':
            interval = 'epoch'
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, **kwargs)
        elif name == 'OneCycleLR':
            interval = 'step'
            scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, **kwargs,
                                                            epochs=epochs, steps_per_epoch=steps_per_epoch)
        else:
            raise ValueError(f'{name} is not supported.')
        return scheduler, interval

    def _set_predict_settings(self, use_tta: bool = False):
        self._use_tta = use_tta

    def configure_optimizers(self) -> list:
        self.optimizer = self._build_optimizer(**self.hparams.optimizer)
        self.scheduler, self.scheduler_interval = self._build_scheduler(**self.hparams.scheduler)
        return [self.optimizer], [{'scheduler': self.scheduler, 'interval': self.scheduler_interval}]

    def forward(self,
                batch: dict,
                force_loss_execute: bool = False,
                ) -> dict:
        return self.model(**batch, force_loss_execute=force_loss_execute)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> dict:
        # forward
        if 'image_path' in batch:
            batch.pop('image_path')

        outputs = self(batch)
        if torch.isnan(outputs['losses']['loss']).any():
            raise ValueError(f'{batch_idx}: train loss is nan)')

        training_step_result = dict(**outputs['losses'])
        self.training_step_outputs.append(training_step_result)
        return training_step_result

    def on_train_epoch_end(self):
        outputs = self.training_step_outputs
        metrics = OrderedDict()
        loss_names = sorted(outputs[0].keys())
        for loss_name in loss_names:
            metrics[f'train/{loss_name}'] = torch.stack([x[loss_name] for x in outputs]).mean()
        self.log_dict(metrics, sync_dist=False)
        self.training_step_outputs.clear()  # free memory

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> dict:
        # forward
        # model.eval()なのでforce_loss_execute=Trueで強制的にロスを算出する
        if 'image_path' in batch:
            batch.pop('image_path')

        outputs = self(batch, force_loss_execute=True)
        if torch.isnan(outputs['losses']['loss']).any():
            raise ValueError(f'{batch_idx}: valid loss is nan)')

        validation_step_result = dict(**outputs, target=batch['target'].detach())
        self.validation_step_outputs.append(validation_step_result)
        return validation_step_result

    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs

        logits = torch.cat([x['logit'] for x in outputs])
        targets = torch.cat([x['target'] for x in outputs])
        predictions = torch.softmax(logits, dim=1)
        target_long = targets.to(torch.long)

        metrics = dict()
        metrics['accuracy'] = self.metric_accuracy(predictions, targets).item()
        metrics['auc'] = self.metric_auc(predictions, targets).item()
        metrics['ap'] = self.metric_ap(predictions, targets).item()

        loss_names = sorted(outputs[0]['losses'].keys())
        for loss_name in loss_names:
            metrics[f'val/{loss_name}'] = torch.stack([x['losses'][loss_name] for x in outputs]).mean()
        self.log_dict(metrics, sync_dist=False)
        self.validation_step_outputs.clear()  # free memory

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = dict()
        if 'image_path' in batch:
            outputs['image_path'] = batch.pop('image_path')
        outputs['prediction'] = self.model.predict(**batch)
        return outputs
