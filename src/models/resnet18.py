import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


def build_resnet18(num_outputs=1):

    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_outputs)

    return model