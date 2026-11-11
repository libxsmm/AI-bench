"""Helion MLIR CPU helpers' kernel construction."""

import pytest

pytest.importorskip("helion_mlir_backend")

from helion_mlir_cpu_utils import per_config


def test_per_config_returns_one_kernel_per_config() -> None:
    @per_config(backend="mlir")
    def body(x):
        return x

    first = body((32, 1, 64))
    assert body((32, 1, 64)) is first
    assert body((32, 1, 128)) is not first
    assert first.configs[0].config["block_sizes"] == [32, 1, 64]
    assert body.body.__name__ == "body"
