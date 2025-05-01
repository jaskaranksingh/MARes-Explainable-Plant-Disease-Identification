import torch
import torch.nn as nn

class MaresLoss(nn.Module):
    """
    MARes Loss: CE + skip feature mean + L2 penalty.
    Clippers to stabilize skip and L2 contributions.
    """
    def __init__(self, skip_weight=0.01, l2_weight=0.005,
                 skip_clip=(0.0, 10.0), l2_clip=(0.0, 1.0)):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.skip_weight = skip_weight
        self.l2_weight = l2_weight
        self.skip_clip = skip_clip
        self.l2_clip = l2_clip

    def forward(self, outputs, targets, skip_connections=None):
        ce_loss = self.ce(outputs, targets)

        # Skip connection supervision
        if skip_connections:
            skip_means = [torch.mean(fm) for fm in skip_connections]
            skip_loss = torch.mean(torch.stack(skip_means))
        else:
            skip_loss = torch.tensor(0.0, device=outputs.device)

        # L2 regularization on logits
        l2_loss = torch.mean(outputs ** 2)

        # Apply clipping
        skip_loss = torch.clamp(skip_loss, *self.skip_clip)
        l2_loss = torch.clamp(l2_loss, *self.l2_clip)

        total = ce_loss + self.skip_weight * skip_loss + self.l2_weight * l2_loss
        return total
