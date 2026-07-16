import torch
import torch.nn as nn


class Model(nn.Module):
    """
    A model that computes the Mean Squared Error loss for regression tasks.

    Parameters:
        None
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, predictions, targets):
        return torch.mean((predictions - targets) ** 2)
