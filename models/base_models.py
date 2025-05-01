import torchvision.models as models

def get_resnet(model_name='resnet18', pretrained=True):
    """
    Returns a torchvision ResNet model (without classifier head).
    """
    if model_name == 'resnet18':
        model = models.resnet18(pretrained=pretrained)
    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
    elif model_name == 'resnet152':
        model = models.resnet152(pretrained=pretrained)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    return model
