import logging
import os
from pathlib import Path

import pytest
import torch

from ai_bench.harness import core as ai_hc
from ai_bench.harness import runner as ai_hr
from ai_bench.utils.logger import setup_logger


@pytest.fixture
def temp_csv_file():
    os.makedirs("tests/output", exist_ok=True)
    csv_path = "tests/output/test_logging.csv"
    if os.path.exists(csv_path):
        os.remove(csv_path)
    yield csv_path
    if os.path.exists(csv_path):
        os.remove(csv_path)


def _create_spec_file(specs_dir, name, content):
    specs_dir = Path(specs_dir)
    specs_dir.mkdir(parents=True, exist_ok=True)
    (specs_dir / name).write_text(content)


def _create_kernel_file(kernels_dir, name, content):
    kernels_dir = Path(kernels_dir)
    kernels_dir.mkdir(parents=True, exist_ok=True)
    (kernels_dir / name).write_text(content)


def test_csv_logging_and_env_capture(monkeypatch, temp_csv_file):
    root_dir = Path("tests/output/project_root")
    spec_dir = root_dir / "problems" / "specs" / "KernelBench" / "level1"
    kernel_dir = root_dir / "third_party" / "KernelBench" / "KernelBench" / "level1"

    spec_content = """
inputs:
  X:
    shape: [N]
    dtype: float32
bench-cpu:
  - params: [X]
    dims:
      N: 4
    flop: 8*N
    mem_bytes: (N + N) * 4 # f32
"""
    kernel_content = """
import torch
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x * 2
"""

    _create_spec_file(spec_dir, "double.yaml", spec_content)
    _create_kernel_file(kernel_dir, "double.py", kernel_content)

    # Patch project_root to use our test root (as a Path!)
    monkeypatch.setattr("ai_bench.utils.finder.project_root", lambda: root_dir)

    monkeypatch.setenv("AIBENCH_CARD", "D770")
    monkeypatch.setenv("AIBENCH_SYSTEM", "TestSystem")

    runner = ai_hr.KernelBenchRunner(
        spec_type=ai_hc.SpecKey.V_BENCH_CPU,
        device=torch.device("cpu"),
        backend=ai_hc.Backend.PYTORCH,
        flops_unit=ai_hr.FlopsUnit.TFLOPS,
        mem_bw_unit=ai_hr.MemBwUnit.GBS,
        csv_path=temp_csv_file,
        note="Unit test note",
    )
    runner.run_kernels()

    assert os.path.exists(temp_csv_file), f"CSV file was not created: {temp_csv_file}"

    import csv

    with open(temp_csv_file, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) > 0, "No rows were logged to the CSV file"
    row = rows[0]
    assert row.get("flops_unit") == "TFLOPS"
    assert row.get("mem_bw_unit") == "GB/s"
    assert row.get("note") == "Unit test note"
    assert row.get("AIBENCH_CARD") == "D770"
    assert row.get("AIBENCH_SYSTEM") == "TestSystem"


def test_env_var_log_levels(monkeypatch, caplog):
    info_txt = "info msg"
    debug_txt = "debug msg"

    def print_log(logger):
        caplog.clear()
        logger.info(info_txt)
        logger.debug(debug_txt)
        return caplog.text

    monkeypatch.delenv("AIBENCH_LOG", raising=False)
    logger = setup_logger(level=logging.INFO)
    out = print_log(logger)
    assert info_txt in out
    assert debug_txt not in out

    monkeypatch.setenv("AIBENCH_LOG", "INFO")
    logger = setup_logger()
    out = print_log(logger)
    assert info_txt in out
    assert debug_txt not in out

    monkeypatch.setenv("AIBENCH_LOG", "DEBUG")
    logger = setup_logger()
    out = print_log(logger)
    assert info_txt in out
    assert debug_txt in out
