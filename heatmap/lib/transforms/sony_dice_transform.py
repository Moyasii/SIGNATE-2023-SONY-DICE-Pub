from typing import List

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
                 noise1_cfg: dict = dict(p=0.0),
                 noise2_cfg: dict = dict(p=0.0),
                 rotate90_cfg: float = dict(p=0.0),
                 resize_cfg: dict = dict(p=1.0, height=40, width=40, interpolation=0),
                 ):
        transform_list = []
        if phase == DatasetPhase.TRAIN:
            transform_list.append(NoiseDA1(**noise1_cfg))
            transform_list.append(NoiseDA2(**noise2_cfg))
            transform_list.append(A.RandomRotate90(**rotate90_cfg))

        transform_list.append(A.Resize(**resize_cfg))
        transform_list.append(ToTensorV2())

        if phase != DatasetPhase.TEST:
            super().__init__(transforms=transform_list,
                             keypoint_params=A.KeypointParams(format='xy', label_fields=['class_labels']))
        else:
            super().__init__(transforms=transform_list)


class ShiftDA:
    def __init__(self, p: float = 0.5):
        self._p = p

    def apply(self,
              image: np.ndarray,
              centers: np.ndarray):
        if np.random.rand() > self._p:
            return image, centers

        src = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]], np.float32)
        dst = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]], np.float32)

        shift_pixel_x = 0
        shift_pixel_y = 0
        shift_ratio = np.random.uniform(0.0, 1.0)
        left_margin, right_margin, top_margin, bottom_margin = self._calc_margins(image)
        if np.random.rand() > 0.5:
            # 右方向にずらす
            shift_pixel_x = np.random.randint(int(right_margin * shift_ratio), right_margin + 1)
        else:
            # 左方向にずらす
            shift_pixel_x = np.random.randint(-left_margin, -int(left_margin * shift_ratio) + 1)
        if np.random.rand() > 0.5:
            # 下方向にずらす
            shift_pixel_y = np.random.randint(int(bottom_margin * shift_ratio), bottom_margin + 1)
        else:
            # 上方向にずらす
            shift_pixel_y = np.random.randint(-top_margin, -int(top_margin * shift_ratio) + 1)

        dst[:, 0] += shift_pixel_x
        dst[:, 1] += shift_pixel_y
        affine = cv2.getAffineTransform(src, dst)

        height, width = image.shape[:2]
        shifted_image = cv2.warpAffine(image, affine, (width, height))
        shifted_centers = centers + np.array([shift_pixel_x, shift_pixel_y])
        return shifted_image, shifted_centers

    def _calc_margins(self, image: np.ndarray) -> List[int]:
        _, bin_image = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)
        y, x = np.where(bin_image == 255)

        left = np.min(x)
        right = np.max(x)
        top = np.min(y)
        bottom = np.max(y)

        # 余白を計算
        left_margin = left
        right_margin = bin_image.shape[1] - right - 1
        top_margin = top
        bottom_margin = bin_image.shape[0] - bottom - 1

        return [left_margin, right_margin, top_margin, bottom_margin]
