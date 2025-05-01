import torch
import torch.nn as nn
import numpy as np

class BetaMaresLoss(nn.Module):
    """
    Beta-MARes Loss: Learnable α, β for skip and L2 loss.
    """
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))

    def forward(self, outputs, targets, skip_connections=None):
        ce_loss = self.ce(outputs, targets)

        if skip_connections:
            skip = skip_connections[np.random.randint(0, len(skip_connections))]
            skip_loss = torch.mean(skip)
        else:
            skip_loss = 0.0

        l2_loss = torch.mean(outputs ** 2)

        return ce_loss + self.alpha * skip_loss + self.beta * l2_loss
