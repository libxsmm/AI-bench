import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_classes=1000):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2
        )

    def forward(self, x):
        x = self.conv1(x)
        return x
