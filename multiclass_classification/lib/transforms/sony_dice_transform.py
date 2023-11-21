from typing import List, Tuple

import cv2
import random
import numpy as np

import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.core.transforms_interface import ImageOnlyTransform

from lib.datasets.dataset_phase import DatasetPhase


class NoiseDA1(ImageOnlyTransform):
    def __init__(self,
                 always_apply: bool = False,
                 mean_limit: List[float] = [0.1, 0.4],
                 sigma_limit: List[float] = [0.0, 0.4],
                 max_prob_limit: List[float] = [0.1, 0.6],
                 min_prob_ratio_limit: float = [1.0, 1.0],
                 p: float = 0.5):
        super().__init__(always_apply=always_apply, p=p)
        self._mean_limit = mean_limit
        self._sigma_limit = sigma_limit
        self._max_prob_limit = max_prob_limit
        self._min_prob_ratio_limit = min_prob_ratio_limit

    def apply(self, img, **params):
        row, col = img.shape
        mean = np.random.uniform(*self._mean_limit)
        sigma = np.random.uniform(*self._sigma_limit)
        max_prob = np.random.uniform(*self._max_prob_limit)
        min_prob_ratio = np.random.uniform(*self._min_prob_ratio_limit)
        min_prob = max_prob * min_prob_ratio
        # normalize
        norm_img = img / 255
        # add noise
        slope = -(max_prob - min_prob)
        intercept = max_prob
        prob = slope * norm_img + intercept
        mask = np.random.rand(row, col) < prob
        gauss = np.random.normal(mean, sigma, (row, col)) * mask
        norm_noisy_img = norm_img + gauss
        norm_noisy_img = np.clip(norm_noisy_img, 0.0, 1.0)
        # de-normalize
        noisy_img = (norm_noisy_img * 255).astype(np.uint8)
        return noisy_img


class NoiseDA2(ImageOnlyTransform):
    def __init__(self,
                 sigma_limit: tuple = (2.0, 100.0),
                 mean_limit: tuple = (0.0, 0.0),
                 always_apply: bool = False,
                 p: float = 0.5):
        super().__init__(always_apply=always_apply, p=p)
        self._sigma_limit = sigma_limit
        self._mean_limit = mean_limit

    def apply(self, img, **params):
        row, col = img.shape

        # ノイズのsigmaをN(0, sigma_scale)からサンプリングする
        sigma = random.uniform(*self._sigma_limit)
        mean = np.random.uniform(*self._mean_limit)
        gauss = np.random.normal(mean, sigma, (row, col))
        white_ratio = np.random.uniform(0.5, 1.0)
        gauss[img == 255] *= white_ratio
        gauss = gauss.astype(np.int64)

        black_prob = np.random.uniform(0.5, 1.0)
        white_prob = np.random.uniform(0.0, 0.5)
        prob = np.ones((row, col), dtype=np.float32)
        prob[img == 0] = black_prob
        prob[img == 255] = white_prob
        mask = np.random.rand(row, col) < prob
        noisy_img = img + (gauss * mask)
        noisy_img = np.clip(noisy_img, 0, 255)

        return noisy_img.astype(np.uint8)


class SonyDiceTransform(A.Compose):
    def __init__(self,
                 phase: DatasetPhase,
                 rotate_cfg: dict = dict(p=0.5, limit=5, interpolation=cv2.INTER_NEAREST,
                                         border_mode=cv2.BORDER_CONSTANT, value=0),
                 affine_cfg: dict = dict(p=0.5, translate_px=2, interpolation=cv2.INTER_NEAREST,
                                         mode=cv2.BORDER_CONSTANT, cval=0),
                 noise1_cfg: dict = dict(p=0.0),
                 noise2_cfg: dict = dict(p=0.0),
                 rotate90_cfg: float = dict(p=0.0),
                 pad_cfg: dict = dict(min_height=14, min_width=14, border_mode=cv2.BORDER_CONSTANT, value=127),
                 resize_cfg: dict = dict(p=1.0, height=42, width=42, interpolation=0),
                 ):
        transform_list = []
        if phase == DatasetPhase.TRAIN:
            transform_list.append(A.Rotate(**rotate_cfg))
            transform_list.append(A.Affine(**affine_cfg))
            transform_list.append(NoiseDA1(**noise1_cfg))
            transform_list.append(NoiseDA2(**noise2_cfg))
            transform_list.append(A.RandomRotate90(**rotate90_cfg))

        transform_list.append(A.PadIfNeeded(**pad_cfg))
        transform_list.append(A.Resize(**resize_cfg))
        transform_list.append(ToTensorV2())
        super().__init__(transforms=transform_list)
