import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

def get_resnet_model(num_classes):
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)

    # Freeze backbone
    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
