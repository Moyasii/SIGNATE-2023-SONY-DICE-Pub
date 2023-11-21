import torch
import torch.nn as nn
from torchvision.ops import sigmoid_focal_loss


def similiarity(x1, x2):
    # refer: https://www.kaggle.com/code/hengck23/mvccl-model-for-admani-dataset

    p12 = (x1 * x2).sum(-1)
    p1 = torch.sqrt((x1 * x1).sum(-1))
    p2 = torch.sqrt((x2 * x2).sum(-1))
    s = p12 / (p1 * p2 + 1e-6)
    return s


def criterion_global_consistency(x1, x1_projection, x2, x2_projection, alpha=-0.5):
    # refer: https://www.kaggle.com/code/hengck23/mvccl-model-for-admani-dataset

    loss = alpha * (similiarity(x1, x1_projection) + similiarity(x2, x2_projection))
    return loss


class BFWithLogitsLoss(nn.Module):
    """Binary Focal Loss"""

    def __init__(self, alpha: float = 0.25, gamma: float = 2, reduction: str = 'mean') -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return sigmoid_focal_loss(input, target, self.alpha, self.gamma, self.reduction)


class ConsistencyLoss(nn.Module):

    def __init__(self, alpha: float = -0.5, reduction: str = 'mean'):
        super().__init__()
        self._alpha = alpha
        self._reduction = reduction

    def forward(self,
                x1: torch.Tensor,
                x1_projection: torch.Tensor,
                x2: torch.Tensor,
                x2_projection: torch.Tensor) -> torch.Tensor:

        loss = criterion_global_consistency(x1, x1_projection, x2, x2_projection, self._alpha)
        if self._reduction == 'mean':
            return torch.mean(loss)
        elif self._reduction == 'sum':
            return torch.sum(loss)
        else:
            return loss
