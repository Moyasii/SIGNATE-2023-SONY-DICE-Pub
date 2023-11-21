import numpy as np
from torch.utils.data import Sampler


# https://www.kaggle.com/datasets/hengck23/for-tpu-efficientb4-debug
class BalanceSampler(Sampler):
    def __init__(self, dataset, ratio=8):
        self.r = ratio - 1
        self.dataset = dataset
        self.pos_index = np.where(self.dataset._targets > 0)[0]
        self.neg_index = np.where(self.dataset._targets == 0)[0]

        self.length = self.r * int(np.floor(len(self.neg_index) / self.r))

    def __iter__(self):
        pos_index = self.pos_index.copy()
        neg_index = self.neg_index.copy()
        np.random.shuffle(pos_index)
        np.random.shuffle(neg_index)

        neg_index = neg_index[:self.length].reshape(-1, self.r)
        pos_index = np.random.choice(pos_index, self.length // self.r).reshape(-1, 1)

        index = np.concatenate([pos_index, neg_index], -1).reshape(-1)
        return iter(index)

    def __len__(self):
        return self.length
