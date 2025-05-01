import torch.nn as nn

class ResNetWithSkips(nn.Module):
    """
    ResNet wrapper to extract intermediate features (skip connections).
    """
    def __init__(self, base_model, num_classes=2):
        super().__init__()
        self.base = base_model

        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool

        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4

        self.avgpool = base_model.avgpool
        self.fc = nn.Linear(base_model.fc.in_features, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)

        pooled = self.avgpool(out4)
        pooled = pooled.view(pooled.size(0), -1)
        logits = self.fc(pooled)

        return logits, [out1, out2, out3, out4]
