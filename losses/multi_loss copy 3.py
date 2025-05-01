# losses/multi_loss.py

import torch
import torch.nn as nn
import numpy as np

class MultiLoss(nn.Module):
    """
    Custom Multi-Component Loss:
    Combines cross-entropy loss with skip connection penalties and output regularization.
    """

    def __init__(self):
        super(MultiLoss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.alpha = nn.Parameter(torch.tensor(0.5))  # Learnable weighting
        self.beta = nn.Parameter(torch.tensor(0.5))

    def forward(self, outputs, targets, skip_connections):
        """
        Args:
            outputs (Tensor): Final classification logits.
            targets (Tensor): Ground-truth labels.
            skip_connections (list): List of feature maps (out1, out2, out3, out4).
        """
        ce_loss = self.cross_entropy(outputs, targets)

        # Randomly pick a skip connection
        skip_output = skip_connections[np.random.randint(0, len(skip_connections))]
        skip_loss = torch.mean(skip_output)

        # L2 regularization on outputs
        l2_loss = torch.mean(outputs ** 2)

        total_loss = ce_loss + self.alpha * skip_loss + self.beta * l2_loss
        return total_loss
