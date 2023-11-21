import pathlib

import cv2
import numpy as np

import torch
from torch.utils.data import Dataset

import albumentations as A

import lib.transforms as transforms_module
from .dataset_phase import DatasetPhase


def load_x_train(x_train_path: str) -> np.ndarray:
    """X_trainの画像を読み込む

    画素値を確認するとnp.arange(0, 256, 6)が歯抜けになった213階調の画像であることがわかる
    一番暗いピクセルが1であったり扱いづらいので、0~255のスケールに直す
    """
    x_train = read_image(x_train_path)
    pixel_table_0_255_to_0_213 = {sv: dv for dv, sv in enumerate(np.setdiff1d(np.arange(256), np.arange(0, 256, 6)))}
    lebel = len(pixel_table_0_255_to_0_213)

    x_train_0_213 = np.vectorize(pixel_table_0_255_to_0_213.get)(x_train)
    x_train_norm = x_train_0_213 / (lebel - 1)
    x_train_0_255 = (x_train_norm * 255).astype(np.uint8)
    return x_train_0_255


def read_image(image_path: str) -> np.ndarray:
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


class SonyDiceDataset(Dataset):
    def __init__(self,
                 phase: DatasetPhase,
                 image_dir: str,
                 num_classes: int,
                 image_size: int,
                 transform: dict,
                 ) -> None:
        self._phase = phase
        self._num_classes = num_classes
        self._image_size = image_size

        self.images = []
        self.targets = []
        self.image_paths = []
        if phase != DatasetPhase.TEST:
            for image_path in sorted(pathlib.Path(image_dir).glob('**/*.png')):
                image_path_str = image_path.as_posix()
                self.images.append(load_x_train(image_path_str))
                self.targets.append(int(image_path.parent.name))
                self.image_paths.append(image_path_str)
        else:
            for image_path in sorted(pathlib.Path(image_dir).glob('**/*.png')):
                image_path_str = image_path.as_posix()
                self.images.append(read_image(image_path_str))
                self.image_paths.append(image_path_str)

        self._transform = self._build_transform(**transform, phase=phase)

    @ property
    def phase(self) -> DatasetPhase:
        return self._phase

    def _build_transform(self, name: str, **kwargs: dict) -> A.Compose:
        transform_class = getattr(transforms_module, name)
        return transform_class(**kwargs)

    def __getitem__(self, idx: int) -> dict:
        output = dict()

        image = self.images[idx]
        image = self._transform(image=image)['image']
        output['image'] = image.to(torch.float32)
        output['image_path'] = self.image_paths[idx]

        if self._phase != DatasetPhase.TEST:
            target = self.targets[idx]
            output['target'] = torch.tensor(target).to(torch.int64)

            if self._phase == DatasetPhase.TRAIN:
                dump_dir = pathlib.Path('temp')
                if dump_dir.exists():
                    self._dump(dump_dir, idx, image, target)

        return output

    def __len__(self):
        return len(self.images)

    def _dump(self,
              dump_dir: pathlib.Path,
              idx: int,
              image: torch.Tensor,
              target: int,
              ) -> None:
        image_path = dump_dir / f'{idx:08d}_{target}.png'
        if not image_path.exists():
            save_image = np.transpose(image.detach().cpu().numpy(), (1, 2, 0)).copy()
            cv2.imwrite(image_path.as_posix(), save_image)
