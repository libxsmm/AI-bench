import torch
import torch.nn as nn


class Model(nn.Module):
    """
    A model that computes Kullback-Leibler Divergence for comparing two distributions.

    Parameters:
        None
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, predictions, targets):
        return torch.nn.functional.kl_div(
            torch.log(predictions), targets, reduction="batchmean"
        )
