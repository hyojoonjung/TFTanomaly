import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantileLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.register_buffer('q', torch.tensor(config.quantiles))

    def forward(self, predictions, targets):
        diff = predictions - targets
        ql = (1-self.q)*F.relu(diff) + self.q*F.relu(-diff)
        losses = ql.view(-1, ql.shape[-1]).mean(0)
        return losses

class QuantileLoss_Test(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.register_buffer('q', torch.tensor(config.quantiles))

    def forward(self, predictions, targets):
        diff = predictions - targets
        ql = (1-self.q)*F.relu(diff) + self.q*F.relu(-diff)
        losses = torch.sum(torch.mean(ql, dim=1),dim=1)
        # losses = ql.view(-1, ql.shape[-1])
        return losses