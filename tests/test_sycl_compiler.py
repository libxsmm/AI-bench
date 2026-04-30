"""Tests for SYCL compiler error propagation."""

from pathlib import Path
import subprocess
from unittest import mock

from ai_bench.sycl.compiler import SYCLCompiler


class TestSYCLCompilerInit:
    def test_default_init(self):
        compiler = SYCLCompiler()
        assert compiler.last_compile_error == ""

    def test_last_compile_error_initialized(self):
        compiler = SYCLCompiler(compiler="/usr/bin/false")
        assert compiler.last_compile_error == ""


class TestCompileErrorPropagation:
    def test_compiler_not_found_stores_error(self):
        compiler = SYCLCompiler(compiler="nonexistent_compiler_xyz")
        result = compiler.compile(Path("/tmp/fake.cpp"))

        assert result is None
        assert "SYCL compiler not found" in compiler.last_compile_error
        assert "nonexistent_compiler_xyz" in compiler.last_compile_error
        assert "AIBENCH_SYCL_COMPILER" in compiler.last_compile_error

    def test_timeout_stores_error(self):
        compiler = SYCLCompiler()
        source = Path("/tmp/fake_kernel.cpp")

        with mock.patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="icpx", timeout=300),
        ):
            result = compiler.compile(source)

        assert result is None
        assert "timed out" in compiler.last_compile_error
        assert "fake_kernel.cpp" in compiler.last_compile_error

    def test_nonzero_returncode_stores_stderr(self):
        compiler = SYCLCompiler()
        source = Path("/tmp/bad_kernel.cpp")
        mock_result = subprocess.CompletedProcess(
            args=["icpx"],
            returncode=1,
            stdout="",
            stderr="error: use of undeclared identifier 'foo'",
        )

        with mock.patch("subprocess.run", return_value=mock_result):
            result = compiler.compile(source)

        assert result is None
        assert (
            compiler.last_compile_error == "error: use of undeclared identifier 'foo'"
        )

    def test_successful_compile_clears_error(self):
        compiler = SYCLCompiler()
        compiler.last_compile_error = "previous error"
        source = Path("/tmp/good_kernel.cpp")
        mock_result = subprocess.CompletedProcess(
            args=["icpx"], returncode=0, stdout="", stderr=""
        )

        with mock.patch("subprocess.run", return_value=mock_result):
            result = compiler.compile(source)

        assert result is not None
        assert compiler.last_compile_error == ""

    def test_error_reset_between_compiles(self):
        compiler = SYCLCompiler()
        source = Path("/tmp/fake.cpp")

        mock_fail = subprocess.CompletedProcess(
            args=["icpx"], returncode=1, stdout="", stderr="some error"
        )
        with mock.patch("subprocess.run", return_value=mock_fail):
            compiler.compile(source)
        assert compiler.last_compile_error == "some error"

        mock_ok = subprocess.CompletedProcess(
            args=["icpx"], returncode=0, stdout="", stderr=""
        )
        with mock.patch("subprocess.run", return_value=mock_ok):
            compiler.compile(source)
        assert compiler.last_compile_error == ""


class TestRunErrorPropagation:
    def test_nonzero_exit_includes_output(self):
        compiler = SYCLCompiler()
        binary = Path("/tmp/bad_binary")
        mock_result = subprocess.CompletedProcess(
            args=[str(binary)],
            returncode=1,
            stdout="partial output\n",
            stderr="segfault\n",
        )

        with mock.patch("subprocess.run", return_value=mock_result):
            result = compiler.run(binary, m=64, n=64, k=64)

        assert not result.success
        assert "Exit code 1" in result.error
        assert "partial output" in result.error
        assert "segfault" in result.error


class TestCompileAndRunErrorPropagation:
    def test_compile_failure_propagates_error(self):
        compiler = SYCLCompiler(compiler="nonexistent_compiler_xyz")
        source = Path("/tmp/fake.cpp")

        result = compiler.compile_and_run(source, m=64, n=64, k=64)

        assert not result.success
        assert "SYCL compiler not found" in result.error
        assert "nonexistent_compiler_xyz" in result.error

    def test_compile_failure_timeout_propagates(self):
        compiler = SYCLCompiler()
        source = Path("/tmp/slow_kernel.cpp")

        with mock.patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="icpx", timeout=300),
        ):
            result = compiler.compile_and_run(source, m=64, n=64, k=64)

        assert not result.success
        assert "timed out" in result.error

    def test_compile_failure_stderr_propagates(self):
        compiler = SYCLCompiler()
        source = Path("/tmp/bad.cpp")
        mock_result = subprocess.CompletedProcess(
            args=["icpx"],
            returncode=1,
            stdout="",
            stderr="fatal error: sycl.hpp not found",
        )

        with mock.patch("subprocess.run", return_value=mock_result):
            result = compiler.compile_and_run(source, m=64, n=64, k=64)

        assert not result.success
        assert "sycl.hpp not found" in result.error

    def test_fallback_message_when_no_error_stored(self):
        compiler = SYCLCompiler()
        with mock.patch.object(compiler, "compile", return_value=None):
            compiler.last_compile_error = ""
            result = compiler.compile_and_run(Path("/tmp/x.cpp"), m=1, n=1, k=1)

        assert not result.success
        assert result.error == "Compilation failed"
