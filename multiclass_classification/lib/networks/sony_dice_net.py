from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import timm

from .utils import Norm, GeM, BFWithLogitsLoss


class SonyDiceNet(nn.Module):
    def __init__(self,
                 timm_model: dict,
                 loss: dict,
                 num_classes: int = 6,
                 mean: Tuple[float] = (127.5, 127.5, 127.5),
                 std: Tuple[float] = (127.5, 127.5, 127.5),
                 use_gem: bool = False,
                 kernel_size: int = 5):
        super().__init__()
        self._num_classes = num_classes
        self._kernel_size = kernel_size

        # Model Architecture
        self.norm = Norm(mean, std)

        self.backbone = timm.create_model(**timm_model)
        if timm_model['model_name'].startswith('efficientnet') or timm_model['model_name'].startswith('tf_efficientnet'):
            fc_in_features = self.backbone.classifier.in_features
            self.backbone.reset_classifier(0)
            if use_gem:
                self.backbone.global_pool = GeM(p_trainable=True, flatten=True)
        elif timm_model['model_name'].startswith('convnext_'):
            fc_in_features = self.backbone.head.fc.in_features
            self.backbone.head.fc = nn.Identity()
            if use_gem:
                self.backbone.head.global_pool = GeM(p_trainable=True, flatten=False)
        else:
            raise ValueError(f'{timm_model["model_name"]} is not supported.')

        self.fc = nn.Linear(fc_in_features, num_classes)
        self.target_loss = self._build_loss(**loss)

    def _build_loss(self,
                    name: str,
                    **kwargs: dict) -> nn.Module:
        if name == 'BCEWithLogitsLoss':
            if 'pos_weight' in kwargs:
                pos_weight = kwargs.pop('pos_weight')
                kwargs['pos_weight'] = torch.tensor(pos_weight)
            loss = nn.BCEWithLogitsLoss(**kwargs)
        elif name == 'BFWithLogitsLoss':
            loss = BFWithLogitsLoss(**kwargs)
        elif name == 'CrossEntropyLoss':
            loss = nn.CrossEntropyLoss(**kwargs)
        else:
            raise ValueError(f'{name} is not supported.')

        return loss

    def forward(self,
                image: torch.Tensor,
                target: torch.Tensor = None,
                force_loss_execute: bool = False,
                ) -> dict:
        outputs = dict()

        x = self.norm(image)
        x = self.backbone(x)
        logit = self.fc(x).view(-1, self._num_classes)
        outputs['logit'] = logit

        # model.train() で self.training == True になる
        if self.training or force_loss_execute:
            losses = self.loss(logit, target)
            outputs['losses'] = losses

        return outputs

    def loss(self,
             logit: torch.Tensor,
             target: torch.Tensor,
             ) -> dict:
        losses = dict()

        target_loss = self.target_loss(logit, target)
        losses['loss_target'] = target_loss
        losses['loss'] = target_loss
        return losses

    def predict(self,
                image: torch.Tensor,
                h_flip: bool = False,
                ) -> torch.Tensor:
        if not h_flip:
            x = self.norm(image)
            x = self.backbone(x)
        else:
            x = self.norm(image.flip(3))
            x = self.backbone(x)
        logit = self.fc(x).view(-1, self._num_classes)
        return torch.softmax(logit, dim=1)
