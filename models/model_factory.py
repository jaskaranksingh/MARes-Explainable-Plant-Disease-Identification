from models.base_models import get_resnet
from models.channel_attention import ChannelAttention
from models.skip_connections import ResNetWithSkips

import torch.nn as nn

def build_model(model_name='resnet18', num_classes=2, use_attention=False):
    """
    Builds a ResNet model with optional channel attention and skip outputs.
    """
    base = get_resnet(model_name=model_name, pretrained=True)

    if use_attention:
        in_c3 = base.layer3[-1].conv1.in_channels
        in_c4 = base.layer4[-1].conv1.in_channels

        base.layer3.add_module("ca", ChannelAttention(in_c3))
        base.layer4.add_module("ca", ChannelAttention(in_c4))

    model = ResNetWithSkips(base, num_classes=num_classes)
    return model
