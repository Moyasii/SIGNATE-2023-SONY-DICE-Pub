from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import timm

from .utils import Norm, GeM, BFWithLogitsLoss


class SonyDiceNeck(nn.Module):
    def __init__(self,
                 in_channels: list = [32, 56, 120, 208],
                 out_channel: int = 32):
        super().__init__()
        self._in_channels = in_channels
        self._out_channel = out_channel

        self.lateal_conv = nn.ModuleList()
        for i in range(len(in_channels) - 1):
            in_c = in_channels[i]
            out_c = in_channels[i]
            self.lateal_conv.append(nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, stride=1, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            ))
        self.last = nn.Sequential(
            nn.Conv2d(in_channels[0], out_channel, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )

        self.upsample_conv = nn.ModuleList()
        for i in range(len(in_channels) - 1):
            in_c = in_channels[i + 1]
            out_c = in_channels[i + 0]
            self.upsample_conv.append(nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, stride=1, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            ))

    def forward(self, in_feature: Tuple[torch.Tensor]) -> torch.Tensor:
        x = in_feature[-1]
        for i in range(len(self._in_channels) - 1, 0, -1):
            c, h, w = in_feature[i-1].shape[1:]
            x = F.interpolate(x, size=(h, w))
            x = self.upsample_conv[i-1](x)
            lat_x = self.lateal_conv[i-1](in_feature[i-1])
            x = x + lat_x

        x = self.last(x)
        return x


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
        if timm_model['model_name'] == 'resnet18':
            # OLD
            # self.backbone.layer2 = nn.Identity()
            # self.backbone.layer3 = nn.Identity()
            # self.backbone.layer4 = nn.Identity()
            # self.backbone.global_pool = nn.Identity()
            # self.backbone.fc = nn.Identity()
            # self.head = nn.Sequential(
            #     nn.Conv2d(64, 64, 3, stride=1, padding=1),
            #     nn.ReLU(inplace=True),
            #     nn.Conv2d(64, num_classes, kernel_size=1),
            # )
            # NEW
            self.neck = SonyDiceNeck()
            self.head = nn.Sequential(
                nn.Conv2d(64, 64, 3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, num_classes, kernel_size=1),
            )
        elif timm_model['model_name'] == 'tf_efficientnetv2_b2':
            out_channel = 72
            self.neck = SonyDiceNeck(in_channels=[32, 56, 120, 208], out_channel=out_channel)
            self.head = nn.Sequential(
                nn.Conv2d(out_channel, 64, 3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, num_classes, kernel_size=1),
            )
            self.aux_head = nn.Sequential(
                nn.Conv2d(out_channel, 64, 3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                GeM(p_trainable=True),
                nn.Linear(64, 3)
            )
        elif timm_model['model_name'] == 'tf_efficientnetv2_s':
            out_channel = 72
            self.neck = SonyDiceNeck(in_channels=[48, 64, 160, 256], out_channel=out_channel)
            self.head = nn.Sequential(
                nn.Conv2d(out_channel, 64, 3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, num_classes, kernel_size=1),
            )
            self.aux_head = nn.Sequential(
                nn.Conv2d(out_channel, 64, 3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                GeM(p_trainable=True),
                nn.Linear(64, 3)
            )
        elif timm_model['model_name'] == 'tf_efficientnetv2_m':
            out_channel = 96
            self.neck = SonyDiceNeck(in_channels=[48, 80, 176, 512], out_channel=out_channel)
            self.head = nn.Sequential(
                nn.Conv2d(out_channel, 64, 3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, num_classes, kernel_size=1),
            )
            self.aux_head = nn.Sequential(
                nn.Conv2d(out_channel, 64, 3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                GeM(p_trainable=True),
                nn.Linear(64, 3)
            )
        elif timm_model['model_name'] == 'convnext_tiny.fb_in22k':
            out_channel = 96
            self.neck = SonyDiceNeck(in_channels=[96, 192, 384, 768], out_channel=out_channel)
            self.head = nn.Sequential(
                nn.Conv2d(out_channel, 64, 3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, num_classes, kernel_size=1),
            )
            self.aux_head = nn.Sequential(
                nn.Conv2d(out_channel, 64, 3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                GeM(p_trainable=True),
                nn.Linear(64, 3)
            )
        elif timm_model['model_name'] == 'convnext_small.in12k_ft_in1k':
            out_channel = 96
            self.neck = SonyDiceNeck(in_channels=[96, 192, 384, 768], out_channel=out_channel)
            self.head = nn.Sequential(
                nn.Conv2d(out_channel, 64, 3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, num_classes, kernel_size=1),
            )
            self.aux_head = nn.Sequential(
                nn.Conv2d(out_channel, 64, 3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                GeM(p_trainable=True),
                nn.Linear(64, 3)
            )
        elif timm_model['model_name'] == 'swin_tiny_patch4_window7_224.ms_in1k':
            out_channel = 96
            self.neck = SonyDiceNeck(in_channels=[96, 192, 384, 768], out_channel=out_channel)
            self.head = nn.Sequential(
                nn.Conv2d(out_channel, 64, 3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, num_classes, kernel_size=1),
            )
            self.aux_head = nn.Sequential(
                nn.Conv2d(out_channel, 64, 3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                GeM(p_trainable=True),
                nn.Linear(64, 3)
            )
            raise NotImplementedError()
        else:
            raise ValueError(f'{timm_model["model_name"]} is not supported.')

        self.head(self.neck(self.backbone(torch.randn(2, 3, 40, 40))))
        self.target_loss = self._build_loss(**loss)
        self.aux_loss = self._build_loss(name='CrossEntropyLoss', weight=torch.Tensor([1.0, 1.0, 1.5]))

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
                center_target: torch.Tensor = None,
                num_dices_target: torch.Tensor = None,
                force_loss_execute: bool = False,
                ) -> dict:
        outputs = dict()

        x = self.norm(image)
        x = self.backbone(x)
        x = self.neck(x)
        heatmap = self.head(x)
        num_dices = self.aux_head(x)
        outputs['heatmap'] = heatmap
        outputs['num_dices'] = num_dices

        # model.train() で self.training == True になる
        if self.training or force_loss_execute:
            losses = self.loss(heatmap, center_target, num_dices, num_dices_target)
            outputs['losses'] = losses

        return outputs

    def loss(self,
             heatmap: torch.Tensor,
             target: torch.Tensor,
             num_dices: torch.Tensor,
             num_dices_target: torch.Tensor,
             ) -> dict:
        losses = dict()

        target_loss = self.target_loss(heatmap.view(heatmap.shape[0], -1), target)
        aux_loss = self.aux_loss(num_dices, num_dices_target)
        losses['loss_target'] = target_loss
        losses['loss_aux'] = aux_loss
        losses['loss'] = target_loss + (0.01 * aux_loss)
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
        x = self.neck(x)
        heatmap = self.head(x)
        heatmap = heatmap.sigmoid()
        heatmap = self._nms(heatmap)
        num_dices = self.aux_head(x)
        num_dices = torch.softmax(num_dices, dim=1)
        return heatmap, num_dices

    def _nms(self, heatmap):
        padding = (self._kernel_size - 1) // 2
        peak = F.max_pool2d(heatmap, self._kernel_size, stride=1, padding=padding)
        keep = (peak == heatmap).float()
        return heatmap * keep

    def decode(self, heatmap: torch.Tensor, k: int = 3) -> torch.Tensor:
        _, num_classes, feat_h, feat_w = heatmap.shape

        heatmap = heatmap.clone().view(heatmap.size(0), -1)

        # top_kのスコアとインデックスを取得
        scores, indices = torch.topk(heatmap, k, dim=1)

        # インデックスを各次元に対応する形に変換
        class_indices = indices // (feat_h * feat_w)
        h_indices = (indices % (feat_h * feat_w)) // feat_w
        w_indices = indices % feat_w

        # 位置情報を0から1の範囲に正規化
        pred_h = h_indices.float() / feat_h
        pred_w = w_indices.float() / feat_w

        # 結果を結合
        predictions = torch.stack((class_indices.float(), pred_h, pred_w, scores), dim=-1)

        return predictions
