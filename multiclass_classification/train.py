import os
import copy
import pathlib
import argparse

import yaml

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, StochasticWeightAveraging
from lightning.pytorch.loggers import CSVLogger

from lib.data_module import ExampleDataModule
from lib.module import ExampleModule
from lib.utils.seed import fix_seed
from lib.utils.config import get_config
from lib.utils.tqdm_progress import LightProgressBar


def build_trainer(trainer_cfg, dst_root, experiment_name):
    trainer_cfg_work = copy.deepcopy(trainer_cfg)
    dst_root = pathlib.Path(dst_root).resolve()

    callbacks = [
        ModelCheckpoint(monitor='val/loss',
                        mode='min',
                        filename=f'{experiment_name}_best-loss_' +
                        'ep-{epoch}_loss-{val/loss:.2f}_auc-{val/acc-0.3:.2f}',
                        auto_insert_metric_name=False),
        ModelCheckpoint(filename=f'{experiment_name}_latest',
                        auto_insert_metric_name=False,
                        # every_n_epochs=4,
                        # save_top_k=-1,),
                        ),
        LightProgressBar(),
        LearningRateMonitor(logging_interval='step'),
    ]
    if 'swa' in trainer_cfg_work:
        swa_config = trainer_cfg_work.pop('swa')
        callbacks.append(StochasticWeightAveraging(**swa_config))

    logger = [
        CSVLogger(dst_root),
    ]

    return pl.Trainer(**trainer_cfg_work, callbacks=callbacks, logger=logger)


def main(args: argparse.Namespace):
    # os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "1"

    # config
    config = get_config(args.config, args.options)
    print(yaml.dump(config))

    fix_seed(config['seed'])

    # experiment name
    experiment_name = args.experiment
    print(f'experiment_name: {experiment_name}')

    # data
    data_module = ExampleDataModule(**config['data'])

    # model
    module = ExampleModule(**config['module'])

    trainer = build_trainer(config['trainer'], args.dst_root, experiment_name)
    log_dir = pathlib.Path(trainer.logger.log_dir)
    print(f'log_dir: {log_dir.as_posix()}')

    # fit
    trainer.fit(module, data_module)

    # save config
    config_path = log_dir / f'{experiment_name}.yaml'
    with open(config_path.as_posix(), 'w') as f:
        yaml.safe_dump(config, f, sort_keys=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('experiment', type=str)
    parser.add_argument('--dst_root', type=str, default='./outputs/train/')
    parser.add_argument('--options', type=str, nargs="*", default=list())
    args = parser.parse_args()

    print(f'config: {args.config}')
    print(f'experiment: {args.experiment}')
    print(f'dst_root: {args.dst_root}')
    print(f'options: {args.options}')
    main(args)
