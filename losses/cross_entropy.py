import torch.nn as nn

class CELoss(nn.Module):
    """
    Standard Cross Entropy Loss for classification tasks.
    """
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, outputs, targets, skip_connections=None):
        return self.ce(outputs, targets)
