"""Tests for ai_bench.utils.finder module."""

import os
from pathlib import Path

import pytest

from ai_bench.utils import finder


class TestConfiguration:
    """Tests for path configuration."""

    def setup_method(self):
        """Reset configuration before each test."""
        finder.reset_configuration()

    def teardown_method(self):
        """Reset configuration after each test."""
        finder.reset_configuration()

    def test_configure_specs_dir(self, tmp_path):
        """Test configuring specs directory."""
        specs_dir = tmp_path / "specs"
        specs_dir.mkdir()

        finder.configure(specs_dir=specs_dir)

        assert finder.specs() == specs_dir

    def test_configure_kernels_dir(self, tmp_path):
        """Test configuring kernels directory."""
        kernels_dir = tmp_path / "kernels"
        kernels_dir.mkdir()

        finder.configure(kernels_dir=kernels_dir)

        assert finder.kernel_bench_dir() == kernels_dir

    def test_configure_triton_kernels_dir(self, tmp_path):
        """Test configuring Triton kernels directory."""
        triton_dir = tmp_path / "triton"
        triton_dir.mkdir()

        finder.configure(triton_kernels_dir=triton_dir)

        assert finder.triton_kernels_dir() == triton_dir

    def test_configure_with_string_path(self, tmp_path):
        """Test configuring with string path instead of Path object."""
        specs_dir = tmp_path / "specs"
        specs_dir.mkdir()

        finder.configure(specs_dir=str(specs_dir))

        assert finder.specs() == specs_dir

    def test_configure_nonexistent_path_raises(self, tmp_path):
        """Test that configuring nonexistent path raises error."""
        nonexistent = tmp_path / "does_not_exist"

        finder.configure(specs_dir=nonexistent)

        with pytest.raises(finder.ConfigurationError, match="does not exist"):
            finder.specs()

    def test_reset_configuration(self, tmp_path):
        """Test resetting configuration."""
        specs_dir = tmp_path / "specs"
        specs_dir.mkdir()

        finder.configure(specs_dir=specs_dir)
        assert finder.specs() == specs_dir

        finder.reset_configuration()

        # After reset, should try default path (which may or may not exist)
        # Just verify the configured path is no longer used
        try:
            result = finder.specs()
            assert (
                result != specs_dir
                or result == finder.project_root() / "problems" / "specs"
            )
        except finder.ConfigurationError:
            pass  # Expected if default path doesn't exist

    def test_configure_multiple_paths(self, tmp_path):
        """Test configuring multiple paths at once."""
        specs_dir = tmp_path / "specs"
        kernels_dir = tmp_path / "kernels"
        specs_dir.mkdir()
        kernels_dir.mkdir()

        finder.configure(specs_dir=specs_dir, kernels_dir=kernels_dir)

        assert finder.specs() == specs_dir
        assert finder.kernel_bench_dir() == kernels_dir


class TestEnvironmentVariables:
    """Tests for environment variable configuration."""

    def setup_method(self):
        """Reset configuration and save env vars."""
        finder.reset_configuration()
        self._saved_env = {}
        for key in [
            "AIBENCH_SPECS_DIR",
            "AIBENCH_KERNELS_DIR",
            "AIBENCH_TRITON_KERNELS_DIR",
        ]:
            self._saved_env[key] = os.environ.get(key)

    def teardown_method(self):
        """Reset configuration and restore env vars."""
        finder.reset_configuration()
        for key, value in self._saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_specs_from_env_var(self, tmp_path):
        """Test specs path from environment variable."""
        specs_dir = tmp_path / "specs"
        specs_dir.mkdir()

        os.environ["AIBENCH_SPECS_DIR"] = str(specs_dir)

        assert finder.specs() == specs_dir

    def test_kernels_from_env_var(self, tmp_path):
        """Test kernels path from environment variable."""
        kernels_dir = tmp_path / "kernels"
        kernels_dir.mkdir()

        os.environ["AIBENCH_KERNELS_DIR"] = str(kernels_dir)

        assert finder.kernel_bench_dir() == kernels_dir

    def test_triton_kernels_from_env_var(self, tmp_path):
        """Test Triton kernels path from environment variable."""
        triton_dir = tmp_path / "triton"
        triton_dir.mkdir()

        os.environ["AIBENCH_TRITON_KERNELS_DIR"] = str(triton_dir)

        assert finder.triton_kernels_dir() == triton_dir

    def test_env_var_nonexistent_path_raises(self, tmp_path):
        """Test that env var with nonexistent path raises error."""
        os.environ["AIBENCH_SPECS_DIR"] = str(tmp_path / "does_not_exist")

        with pytest.raises(finder.ConfigurationError, match="does not exist"):
            finder.specs()

    def test_configure_takes_priority_over_env(self, tmp_path):
        """Test that explicit configuration takes priority over env var."""
        env_dir = tmp_path / "env_specs"
        config_dir = tmp_path / "config_specs"
        env_dir.mkdir()
        config_dir.mkdir()

        os.environ["AIBENCH_SPECS_DIR"] = str(env_dir)
        finder.configure(specs_dir=config_dir)

        assert finder.specs() == config_dir


class TestDotEnvLoading:
    """Tests for .env file loading."""

    def setup_method(self):
        """Reset configuration and save env vars."""
        finder.reset_configuration()
        self._saved_env = {}
        for key in ["AIBENCH_SPECS_DIR", "AIBENCH_KERNELS_DIR", "AIBENCH_CARD"]:
            self._saved_env[key] = os.environ.get(key)
            os.environ.pop(key, None)

    def teardown_method(self):
        """Reset configuration and restore env vars."""
        finder.reset_configuration()
        for key, value in self._saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_load_env_explicit_path(self, tmp_path):
        """Test loading .env from explicit path."""
        specs_dir = tmp_path / "specs"
        specs_dir.mkdir()

        env_file = tmp_path / ".env"
        env_file.write_text(f"AIBENCH_SPECS_DIR={specs_dir}\n")

        result = finder.load_env(env_file)

        assert result is True
        assert finder.is_env_loaded() is True
        assert os.environ.get("AIBENCH_SPECS_DIR") == str(specs_dir)

    def test_load_env_nonexistent_returns_false(self, tmp_path):
        """Test loading nonexistent .env returns False."""
        result = finder.load_env(tmp_path / "nonexistent.env")

        assert result is False
        assert finder.is_env_loaded() is False

    def test_load_env_multiple_variables(self, tmp_path):
        """Test loading multiple variables from .env."""
        specs_dir = tmp_path / "specs"
        kernels_dir = tmp_path / "kernels"
        specs_dir.mkdir()
        kernels_dir.mkdir()

        env_content = f"""
AIBENCH_SPECS_DIR={specs_dir}
AIBENCH_KERNELS_DIR={kernels_dir}
AIBENCH_CARD=BMG
"""
        env_file = tmp_path / ".env"
        env_file.write_text(env_content)

        finder.load_env(env_file)

        assert os.environ.get("AIBENCH_SPECS_DIR") == str(specs_dir)
        assert os.environ.get("AIBENCH_KERNELS_DIR") == str(kernels_dir)
        assert os.environ.get("AIBENCH_CARD") == "BMG"

    def test_load_env_no_override_by_default(self, tmp_path):
        """Test that existing env vars are not overridden by default."""
        os.environ["AIBENCH_CARD"] = "existing"

        env_file = tmp_path / ".env"
        env_file.write_text("AIBENCH_CARD=from_file\n")

        finder.load_env(env_file, override=False)

        assert os.environ.get("AIBENCH_CARD") == "existing"

    def test_load_env_with_override(self, tmp_path):
        """Test that override=True overrides existing env vars."""
        os.environ["AIBENCH_CARD"] = "existing"

        env_file = tmp_path / ".env"
        env_file.write_text("AIBENCH_CARD=from_file\n")

        finder.load_env(env_file, override=True)

        assert os.environ.get("AIBENCH_CARD") == "from_file"

    def test_is_env_loaded_false_initially(self):
        """Test is_env_loaded returns False initially."""
        assert finder.is_env_loaded() is False

    def test_reset_clears_env_loaded(self, tmp_path):
        """Test that reset_configuration clears env_loaded flag."""
        env_file = tmp_path / ".env"
        env_file.write_text("AIBENCH_CARD=test\n")

        finder.load_env(env_file)
        assert finder.is_env_loaded() is True

        finder.reset_configuration()
        assert finder.is_env_loaded() is False


class TestDefaultPaths:
    """Tests for default path resolution."""

    def setup_method(self):
        """Reset configuration."""
        finder.reset_configuration()

    def teardown_method(self):
        """Reset configuration."""
        finder.reset_configuration()

    def test_project_root_returns_path(self):
        """Test that project_root returns a Path."""
        root = finder.project_root()

        assert isinstance(root, Path)

    def test_project_root_contains_ai_bench(self):
        """Test that project root contains ai_bench package."""
        root = finder.project_root()

        # The project root should contain the ai_bench directory
        assert (root / "ai_bench").exists() or root.name == "ai_bench"


class TestConfigurationError:
    """Tests for ConfigurationError."""

    def test_configuration_error_message(self):
        """Test ConfigurationError has informative message."""
        error = finder.ConfigurationError("Test error message")

        assert "Test error message" in str(error)

    def test_configuration_error_is_exception(self):
        """Test ConfigurationError is an Exception."""
        error = finder.ConfigurationError("test")

        assert isinstance(error, Exception)
