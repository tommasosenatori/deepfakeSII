from torchvision.models import (
    resnet18,
    ResNet18_Weights
)

import torch.nn as nn


def build_resnet18():

    model = resnet18(
        weights=ResNet18_Weights.DEFAULT
    )

    in_features = model.fc.in_features

    model.fc = nn.Linear(
        in_features,
        1
    )

    return model