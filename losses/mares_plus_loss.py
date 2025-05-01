import torch
import torch.nn as nn
import torch.nn.functional as F

class MAResPlusLoss(nn.Module):
    """
    MARes+ Loss: GAP-based skip supervision + L2 regularization.
    """
    def __init__(self, skip_weight=0.03, l2_weight=0.01):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.skip_weight = skip_weight
        self.l2_weight = l2_weight

    def forward(self, outputs, targets, skip_connections=None):
        ce_loss = self.ce(outputs, targets)

        if skip_connections:
            skip_losses = []
            for feat in skip_connections:
                gap = F.adaptive_avg_pool2d(feat, 1).squeeze()
                if gap.ndim > 1:
                    gap = gap.view(gap.size(0), -1)
                skip_losses.append(torch.mean(gap ** 2))
            skip_loss = torch.mean(torch.stack(skip_losses))
        else:
            skip_loss = 0.0

        l2_loss = torch.mean(outputs ** 2)

        return ce_loss + self.skip_weight * skip_loss + self.l2_weight * l2_loss
