import helion_mlir_backend  # noqa: F401
import torch
import torch.nn as nn

from helion_mlir_cpu_utils import identity_epilogue
from helion_mlir_cpu_utils import linear


class Model(nn.Module):
    """KernelBench-compatible MLP.

    By default each weight is packed by the first forward and reused, modeling
    deployment with constant weights; ``HELION_MLIR_CACHE_PREPACKED_WEIGHTS=0``
    packs them on every forward.
    """

    def __init__(self, input_size, layer_sizes, output_size, *args, **kwargs):
        super(Model, self).__init__()

        layers = []
        current_input_size = input_size
        for layer_size in layer_sizes:
            layers.append(nn.Linear(current_input_size, layer_size))
            current_input_size = layer_size
        layers.append(nn.Linear(current_input_size, output_size))
        self.layers = nn.ModuleList(layers)
        self._caches = [None] * len(self.layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        last = len(self.layers) - 1
        for i, layer in enumerate(self.layers):
            epilogue = identity_epilogue if i == last else torch.relu
            x, self._caches[i] = linear(x, layer, epilogue, cache=self._caches[i])
        return x
