from typing import Optional

import math
import pathlib

import cv2
import numpy as np

import torch
from torch.utils.data import Dataset

import albumentations as A

import lib.transforms as transforms_module
from lib.transforms.sony_dice_transform import ShiftDA
from .dataset_phase import DatasetPhase


def load_x_train(x_train_path: str) -> np.ndarray:
    """X_trainの画像を読み込む

    画素値を確認するとnp.arange(0, 256, 6)が歯抜けになった213階調の画像であることがわかる
    一番暗いピクセルが1であったり扱いづらいので、0~255のスケールに直す
    """
    x_train = np.load(x_train_path)
    pixel_table_0_255_to_0_213 = {sv: dv for dv, sv in enumerate(np.setdiff1d(np.arange(256), np.arange(0, 256, 6)))}
    lebel = len(pixel_table_0_255_to_0_213)

    x_train_0_213 = np.vectorize(pixel_table_0_255_to_0_213.get)(x_train)
    x_train_norm = x_train_0_213 / (lebel - 1)
    x_train_0_255 = (x_train_norm * 255).astype(np.uint8)
    return x_train_0_255


def read_image(image_path: str) -> np.ndarray:
    image = cv2.imread(image_path)  # , cv2.IMREAD_GRAYSCALE)
    return image


class SonyDiceDataset(Dataset):
    def __init__(self,
                 phase: DatasetPhase,
                 num_classes: int,
                 image_size: int,
                 transform: dict,
                 image_npy_path: str,
                 target_npy_path: Optional[str] = None,
                 center_npy_path: Optional[str] = None,
                 center_target_npy_path: Optional[str] = None,
                 stride: int = 4,
                 ) -> None:
        self._phase = phase
        self._num_classes = num_classes
        self._image_size = image_size
        self._feature_size = image_size // stride
        self._stride = stride

        if phase != DatasetPhase.TEST:
            self.images = load_x_train(image_npy_path)
            self.targets = np.load(target_npy_path) - 1
            self.centers = np.load(center_npy_path)
            self.center_targets = np.load(center_target_npy_path) - 1
        else:
            self.images = np.load(image_npy_path)
            self.targets = None
            self.centers = None
            self.center_targets = None

        image_size = int(math.sqrt(self.images.shape[1]))
        self.images = self.images.reshape(-1, image_size, image_size)
        self._transform = self._build_transform(**transform, phase=phase)
        self.shift_da = ShiftDA()

    @ property
    def phase(self) -> DatasetPhase:
        return self._phase

    def _build_transform(self, name: str, **kwargs: dict) -> A.Compose:
        transform_class = getattr(transforms_module, name)
        return transform_class(**kwargs)

    def __getitem__(self, idx: int) -> dict:
        output = dict()

        image = self._get_image(self.images, idx)

        if self._phase != DatasetPhase.TEST:
            target = self.targets[idx]

            heatmap = np.zeros((self._num_classes, self._feature_size, self._feature_size), dtype=np.float32)
            centers = self.centers[idx]
            center_targets = self.center_targets[idx]
            centers = centers[center_targets < 254]
            center_targets = center_targets[center_targets < 254]
            num_dices_target = len(centers) - 1

            # TODO: シフトDAをAlbumentations側に追加
            if self._phase == DatasetPhase.TRAIN:
                image, centers = self.shift_da.apply(image, centers)

            num_targets = len(center_targets)
            while True:
                transformed = self._transform(image=image, keypoints=centers, class_labels=center_targets)
                transformed_image = transformed['image']
                transformed_image = transformed_image.repeat(3, 1, 1)
                transformed_centers = transformed['keypoints']
                transformed_center_targets = transformed['class_labels']
                if len(transformed_centers) == num_targets:
                    break

            for center, center_target in zip(transformed_centers, transformed_center_targets):
                center_x = int(center[0] // self._stride)
                center_y = int(center[1] // self._stride)
                heatmap[center_target][center_y][center_x] = 1.0

            output['image'] = transformed_image.to(torch.float32)
            output['target'] = torch.Tensor([target]).to(torch.int64)
            heatmap = heatmap.reshape(-1)
            output['center_target'] = torch.Tensor(heatmap).to(torch.float32)
            output['num_dices_target'] = torch.tensor(num_dices_target).to(torch.int64)

            if self._phase == DatasetPhase.TRAIN:
                dump_dir = pathlib.Path('temp')
                if dump_dir.exists():
                    self._dump(dump_dir, idx, output, center_targets, transformed_image,
                               transformed_centers, transformed_center_targets)
        else:
            transformed_image = self._transform(image=image)['image']
            transformed_image = transformed_image.repeat(3, 1, 1)
            output['image'] = transformed_image.to(torch.float32)

        return output

    def __len__(self):
        return self.images.shape[0]

    def _get_image(self, images: np.ndarray, idx: int):
        return images[idx]

    def _dump(self,
              dump_dir: pathlib.Path,
              idx: int,
              output: dict,
              center_targets: np.ndarray,
              transformed_image: torch.Tensor,
              transformed_centers: np.ndarray,
              transformed_center_targets: np.ndarray
              ) -> None:
        image_path = dump_dir / \
            f'{idx:08d}_{output["target"][0]}_{"-".join(map(str,center_targets.tolist()))}_{output["num_dices_target"]}.png'
        if not image_path.exists():
            save_image = np.transpose(transformed_image.detach().cpu().numpy(), (1, 2, 0)).copy()
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255),]
            for save_center, save_center_target in zip(transformed_centers, transformed_center_targets):
                save_image = cv2.circle(save_image, (int(round(save_center[0], 0)), int(
                    round(save_center[1], 0))), 1, colors[save_center_target], thickness=-1)

            cv2.imwrite(image_path.as_posix(), save_image)
