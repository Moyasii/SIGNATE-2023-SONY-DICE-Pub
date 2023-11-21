import random

import numpy as np
import torch
import lightning.pytorch as pl

def fix_seed(seed: int = 2022) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    # 遅くなるけど大丈夫な場合のみ
    # (そもそもUpsampleとか使っている時点で再現性ないのでそこまでシビアになる必要とは)
    torch.backends.cudnn.deterministic = True
    pl.seed_everything(seed)

