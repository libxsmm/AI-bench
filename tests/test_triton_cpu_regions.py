import importlib
from pathlib import Path

import pytest
import torch

triton = pytest.importorskip("triton")
cpu_backend = triton.backends.backends.get("cpu")
if cpu_backend is None:
    pytest.skip("Triton CPU backend is unavailable", allow_module_level=True)

region = importlib.import_module("triton_cpu_utils.region_matmul")
sfc = importlib.import_module("triton_cpu_utils.sfc_matmul")


@pytest.mark.parametrize("grid", [None, (64, 1)])
def test_region_tile_ownership(grid):
    tiles = region._make_region_tensor(512, 128, 64, 4, grid=grid)
    assert tiles.shape == (512 * 128, 2)
    assert tiles.unique(dim=0).size(0) == 512 * 128
    regions_m, regions_n = (16, 4) if grid is None else grid
    rows, columns = 512 // regions_m, 128 // regions_n
    for program, chunk in enumerate(tiles.chunk(64)):
        row_begin = program // regions_n * rows
        col_begin = program % regions_n * columns
        assert chunk[:, 0].unique().tolist() == list(range(row_begin, row_begin + rows))
        assert chunk[:, 1].unique().tolist() == list(
            range(col_begin, col_begin + columns)
        )
        assert chunk[:4].tolist() == [
            [row_begin + 4 * (program % regions_n) + row, col_begin] for row in range(4)
        ]


def test_autotuner_uses_map_for_candidate_and_cached_winner(monkeypatch):
    monkeypatch.setattr(torch, "get_num_threads", lambda: 2)
    monkeypatch.setattr(region, "_l2_cache_bytes", lambda: 2**19)
    tuner = sfc._autotuned_kernel.__wrapped__(2, region._tile_map)
    monkeypatch.setattr(tuner, "cache_results", False)
    monkeypatch.setattr(sfc, "_autotuned_kernel", lambda *args: tuner)
    launches = []

    def run(*args, **kwargs):
        factor, per_thread = kwargs["BLOCKING_FACTOR_K"], kwargs["progs_per_thread"]
        programs = kwargs["grid"](kwargs)[0]
        assert kwargs["num_cpu_threads"] == 2
        assert programs == min(16, 2 * per_thread)
        group_m = {1: 1, 2: 4}[factor]
        expected = region._make_region_tensor(4, 4, programs, group_m)
        if expected is None:
            expected = sfc._make_sfc_tensor(4, 4)
        torch.testing.assert_close(kwargs["sfc_map_ptr"], expected, rtol=0, atol=0)
        launches.append((factor, per_thread))

    def bench(kernel_call, quantiles):
        kernel_call()
        return [1.0 if launches[-1] == (2, 2) else 2.0] * len(quantiles)

    monkeypatch.setattr(sfc._sfc_matmul_kernel, "run", run)
    monkeypatch.setattr(tuner, "_do_bench", bench)
    inputs = torch.empty((128, 2048), dtype=torch.bfloat16)
    weights = torch.empty((2048, 128), dtype=torch.bfloat16)
    region.region_matmul(inputs, weights)
    assert {(1, 1), (2, 2), (2, 8)} <= set(launches[:-1])
    assert launches[-1] == (2, 2)

    launches.clear()
    region.region_matmul(inputs, weights)
    assert launches == [(2, 2)]


@pytest.mark.skipif(
    not torch.cpu.get_capabilities().get("amx_bf16"), reason="requires AMX BF16"
)
@pytest.mark.parametrize(
    ("problem", "init_args", "factor"),
    [(39, ((128,),), 1), (40, (-1.5,), 2), (41, (), 2)],
)
def test_bf16_models_match_sfc_at_same_k_split(monkeypatch, problem, init_args, factor):
    monkeypatch.setattr(triton.runtime.driver, "_active", cpu_backend.driver())
    monkeypatch.setattr(torch, "get_num_threads", lambda: 2)
    models = (
        Path(__file__).resolve().parents[1] / "backends/triton/cpu/KernelBench/level2"
    )
    path = next(models.glob(f"{problem}_*.py"))
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.Model(128, 128, *init_args).eval()
    inputs = torch.randn((128, 128), dtype=torch.bfloat16)
    launch_options = {"blocking_factor_k": factor, "progs_per_thread": 1}

    def run_regions(a, b, *, region_grid, **kwargs):
        assert region_grid == (64, 1)
        return region.region_matmul(
            a, b, region_grid=region_grid, **launch_options, **kwargs
        )

    def run_sfc(a, b, *, region_grid, **kwargs):
        return sfc.sfc_matmul(a, b, **launch_options, **kwargs)

    with torch.inference_mode():
        if problem == 39:
            model.scale.copy_(torch.linspace(-1, 1, 128))
            model.scale[::3] = 0
        monkeypatch.setattr(module, "region_matmul", run_regions)
        actual = model(inputs).clone()
        monkeypatch.setattr(module, "region_matmul", run_sfc)
        torch.testing.assert_close(actual, model(inputs), rtol=0, atol=0)
