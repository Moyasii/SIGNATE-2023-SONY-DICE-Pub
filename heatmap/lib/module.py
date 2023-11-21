from typing import Any, Optional
from collections import OrderedDict

import numpy as np

import torch
import torch.nn as nn

from torchmetrics.classification import (
    BinaryAUROC,
    BinaryAveragePrecision,
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
        # self.metrics_auc = BinaryAUROC(thresholds=None)  # AUC
        # self.metrics_ap = BinaryAveragePrecision(thresholds=None)  # AP
        self.metrics_mul_accuracy = MulticlassAccuracy(3)

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
        outputs = self(batch, force_loss_execute=True)
        if torch.isnan(outputs['losses']['loss']).any():
            raise ValueError(f'{batch_idx}: valid loss is nan)')

        validation_step_result = dict(**outputs,
                                      targets=batch['target'].detach(),
                                      center_targets=batch['center_target'].detach(),
                                      num_dices_targets=batch['num_dices_target'].detach(),)
        self.validation_step_outputs.append(validation_step_result)
        return validation_step_result

    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs

        heatmaps = torch.cat([x['heatmap'] for x in outputs])
        heatmaps = torch.sigmoid(heatmaps)
        heatmaps = self.model._nms(heatmaps)

        center_targets = torch.cat([x['center_targets'] for x in outputs])
        center_targets = center_targets.to(torch.long)

        num_dices_logits = torch.cat([x['num_dices'] for x in outputs])
        num_dices = torch.softmax(num_dices_logits, dim=1)

        num_dices_targets = torch.cat([x['num_dices_targets'] for x in outputs])
        num_dices_targets = num_dices_targets.to(torch.long)

        metrics = OrderedDict()
        metrics['val/num_dices_acc'] = self.metrics_mul_accuracy(num_dices, num_dices_targets).item()

        predictions = self.model.decode(heatmaps)
        predictions[..., 0] += 1
        targets = torch.cat([x['targets'] for x in outputs]) + 1
        targets = targets.squeeze()
        for score_th in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
            num_correct = 0
            for prediction, target in zip(predictions, targets):
                correct_prediction = prediction[prediction[..., 3] >= score_th]
                pred_dice = torch.sum(correct_prediction[..., 0])
                if pred_dice == target:
                    num_correct += 1
            metrics[f'val/acc-{score_th}'] = num_correct / len(targets)

        heatmaps = heatmaps.view(heatmaps.shape[0], -1)
        # metrics['val/auc'] = self.metrics_auc(heatmaps, center_targets).item()
        # metrics['val/ap'] = self.metrics_ap(heatmaps, center_targets).item()

        loss_names = sorted(outputs[0]['losses'].keys())
        for loss_name in loss_names:
            metrics[f'val/{loss_name}'] = torch.stack([x['losses'][loss_name] for x in outputs]).mean()
        self.log_dict(metrics, sync_dist=False)
        self.validation_step_outputs.clear()  # free memory

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self.model.predict(**batch)
        return outputs
