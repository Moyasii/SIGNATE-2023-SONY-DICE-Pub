import copy
import pathlib
import argparse

import yaml

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import lightning.pytorch as pl

from lib.data_module import ExampleDataModule
from lib.module import ExampleModule
from lib.utils.config import get_config


def decode(heatmap: np.ndarray, k: int = 3) -> np.ndarray:
    _, num_classes, feat_h, feat_w = heatmap.shape
    heatmap = copy.deepcopy(heatmap)

    # top_k
    heatmap = heatmap.reshape(heatmap.shape[0], -1)
    indices = np.argsort(-heatmap, axis=1)[..., :k]
    scores = np.take_along_axis(heatmap, indices, axis=1)

    # decode
    class_indices, h_indices, w_indices = np.unravel_index(indices, (num_classes, feat_h, feat_w))
    pred_h = h_indices / feat_h
    pred_w = w_indices / feat_w
    predictions = np.stack((class_indices, pred_h, pred_w, scores), axis=-1)

    return predictions


def build_trainer(trainer_cfg, dst_root):
    trainer_cfg_work = copy.deepcopy(trainer_cfg)
    dst_root = pathlib.Path(dst_root).resolve()
    if 'swa' in trainer_cfg_work:
        trainer_cfg_work.pop('swa')
    return pl.Trainer(**trainer_cfg_work, default_root_dir=dst_root)


def main(args: argparse.Namespace):
    # config
    config = get_config(args.config, args.options)
    print(yaml.dump(config))

    # data
    data_module = ExampleDataModule(**config['data'])

    # model
    config['module']['model']['timm_model']['pretrained'] = False
    module = ExampleModule.load_from_checkpoint(args.checkpoint, **config['module'])
    module.eval()

    # trainer
    trainer = build_trainer(config['trainer'], args.dst_root)
    log_dir = pathlib.Path(trainer.logger.log_dir)
    print(f'log_dir: {log_dir.as_posix()}')

    # predict
    outputs = trainer.predict(module, data_module)

    # decode
    heatmaps = torch.cat([o[0] for o in outputs])
    if heatmaps.dtype == torch.bfloat16:
        heatmaps = heatmaps.to(torch.float32)
    heatmaps = heatmaps.detach().cpu().numpy()
    predictions = decode(heatmaps)
    np.save((log_dir / 'predictions').as_posix(), predictions)

    # log
    log_text_path = pathlib.Path(log_dir) / 'log.txt'
    with open(log_text_path, 'w') as f:
        f.write(f'config: {args.config}\n')
        f.write(f'checkpoint: {args.checkpoint}\n')
        f.write(f'dst_root: {args.dst_root}\n')
        f.write(f'options: {", ".join(args.options)}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('checkpoint', type=str)
    parser.add_argument('--target', type=str, default='prediction')
    parser.add_argument('--dst_root', type=str, default='./outputs/eval/')
    parser.add_argument('--options', type=str, nargs="*", default=list())
    args = parser.parse_args()

    print(f'config: {args.config}')
    print(f'checkpoint: {args.checkpoint}')
    print(f'target: {args.target}')
    print(f'dst_root: {args.dst_root}')
    print(f'options: {args.options}')
    main(args)
