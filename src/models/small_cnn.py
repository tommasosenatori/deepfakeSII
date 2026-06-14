import torch.nn as nn


class SmallCNN(nn.Module):

    def __init__(self, num_outputs=1):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Linear(64, num_outputs)

    def forward(self, x):

        x = self.features(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def build_small_cnn(num_outputs=1):
    return SmallCNN(num_outputs=num_outputs)