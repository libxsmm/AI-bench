"""Kernels of one body that differ by tile config."""

from __future__ import annotations

import functools

import helion


def per_config(**kernel_args):
    """A kernel per tile config: ``@per_config(backend="mlir")`` on a kernel body
    makes its name take a hashable ``block_sizes`` and return that config's
    kernel. The raw body stays reachable as ``.body``."""

    def wrap(body):
        @functools.cache
        def for_config(block_sizes):
            return helion.kernel(
                body, config=helion.Config(block_sizes=list(block_sizes)), **kernel_args
            )

        for_config.body = body
        return for_config

    return wrap
