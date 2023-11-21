import copy
import pathlib
import argparse

import yaml

import numpy as np
import pandas as pd

import torch
import lightning.pytorch as pl

from lib.data_module import ExampleDataModule
from lib.module import ExampleModule
from lib.utils.config import get_config


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
    predictions = torch.cat([o['prediction'] for o in outputs])
    if predictions.dtype == torch.bfloat16:
        predictions = predictions.to(torch.float32)
    predictions = predictions.detach().cpu().numpy()
    np.save((log_dir / 'predictions').as_posix(), predictions)

    image_paths = np.concatenate([o['image_path'] for o in outputs]).tolist()
    image_paths = [pathlib.Path(p) for p in image_paths]

    pred_dice_result_per_image = dict()
    for p, image_path in zip(predictions, image_paths):
        pred_dice_result = np.argmax(p) + 1
        image_no = int(image_path.name.split('_')[0])
        if image_no not in pred_dice_result_per_image:
            pred_dice_result_per_image[image_no] = 0
        pred_dice_result_per_image[image_no] += pred_dice_result

    df = pd.DataFrame(list(pred_dice_result_per_image.items()), columns=['index', 'pred'])
    df = df.sort_values(['index'])
    if not all(df['index'] == range(len(df['index']))):
        raise ValueError('推論されていないテスト画像がある')
    df.to_csv(log_dir / f'{pathlib.Path(args.config).stem}_preds.csv', index=False)

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
    parser.add_argument('--score_th', type=float, default=0.25)
    parser.add_argument('--target', type=str, default='prediction')
    parser.add_argument('--dst_root', type=str, default='./outputs/eval_testset/')
    parser.add_argument('--options', type=str, nargs="*", default=list())
    args = parser.parse_args()

    print(f'config: {args.config}')
    print(f'checkpoint: {args.checkpoint}')
    print(f'score_th: {args.score_th}')
    print(f'target: {args.target}')
    print(f'dst_root: {args.dst_root}')
    print(f'options: {args.options}')
    main(args)
