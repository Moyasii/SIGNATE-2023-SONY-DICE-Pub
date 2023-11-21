from typing import List

import torch


# https://www.kaggle.com/datasets/hengck23/for-tpu-efficientb4-debug
class NullCollate:
    def __init__(self, stack_columns: List[str]) -> None:
        self._stack_columns = stack_columns

    def __call__(self, batch: dict) -> dict:
        d = {}
        key = batch[0].keys()
        for k in key:
            v = [b[k] for b in batch]
            if k in self._stack_columns:
                v = torch.stack(v, 0)
            d[k] = v
        # if 'target' in d.keys():
        #    d['target'] = d['target'].reshape(-1)
        return d
