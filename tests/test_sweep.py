"""Tests for ai_bench.utils.sweep module."""

from ai_bench.utils import sweep


class TestGenerateVariants:
    """Tests for generate_variants function."""

    def test_simple_sweep(self):
        """Test simple parameter sweep."""
        config = {
            "test_kernel": {
                "batch": [32, 64],
                "dim": [1024, 2048],
            }
        }

        variants = sweep.generate_variants("test_kernel", config)

        assert len(variants) == 4  # 2 x 2
        dims = [v["dims"] for v in variants]
        assert {"BATCH": 32, "DIM": 1024} in dims
        assert {"BATCH": 32, "DIM": 2048} in dims
        assert {"BATCH": 64, "DIM": 1024} in dims
        assert {"BATCH": 64, "DIM": 2048} in dims

    def test_merge_with_base_variant(self):
        """Test that sweep merges with base variant dims."""
        config = {
            "test_kernel": {
                "batch": [32, 64],
            }
        }
        base_variant = {
            "params": ["X"],
            "dtype": "float16",
            "dims": {"BATCH": 16, "DIM": 4096, "HIDDEN": 1024},
        }

        variants = sweep.generate_variants(
            "test_kernel", config, base_variant=base_variant
        )

        assert len(variants) == 2
        for v in variants:
            # Should preserve params and dtype
            assert v["params"] == ["X"]
            assert v["dtype"] == "float16"
            # Should preserve unswept dims
            assert v["dims"]["DIM"] == 4096
            assert v["dims"]["HIDDEN"] == 1024
        # Batch should be swept
        assert variants[0]["dims"]["BATCH"] == 32
        assert variants[1]["dims"]["BATCH"] == 64

    def test_linked_parameters(self):
        """Test linked parameters (e.g., width = height)."""
        config = {
            "conv_kernel": {
                "batch_size": [16, 32],
                "height": [64, 128],
                "_linked": {
                    "width": "height",
                },
            }
        }

        variants = sweep.generate_variants("conv_kernel", config)

        assert len(variants) == 4  # 2 x 2
        for v in variants:
            assert v["dims"]["WIDTH"] == v["dims"]["HEIGHT"]

    def test_fixed_parameters(self):
        """Test fixed parameters."""
        config = {
            "matmul_kernel": {
                "m": [1024, 2048],
                "_fixed": {
                    "k": 4096,
                    "n": 3072,
                },
            }
        }

        variants = sweep.generate_variants("matmul_kernel", config)

        assert len(variants) == 2
        for v in variants:
            assert v["dims"]["K"] == 4096
            assert v["dims"]["N"] == 3072

    def test_missing_kernel(self):
        """Test that missing kernel returns empty list."""
        config = {"other_kernel": {"dim": [1024]}}

        variants = sweep.generate_variants("nonexistent", config)

        assert variants == []

    def test_single_value_list(self):
        """Test single value in sweep."""
        config = {
            "kernel": {
                "batch_size": [64],
                "dim": [1024, 2048, 4096],
            }
        }

        variants = sweep.generate_variants("kernel", config)

        assert len(variants) == 3


class TestCountVariants:
    """Tests for count_variants function."""

    def test_count_simple(self):
        """Test counting variants."""
        config = {
            "kernel1": {"a": [1, 2], "b": [3, 4, 5]},  # 2 x 3 = 6
            "kernel2": {"x": [1, 2, 3, 4]},  # 4
        }

        counts = sweep.count_variants(config)

        assert counts["kernel1"] == 6
        assert counts["kernel2"] == 4

    def test_count_ignores_special_keys(self):
        """Test that _linked and _fixed don't affect count."""
        config = {
            "kernel": {
                "a": [1, 2],
                "_linked": {"b": "a"},
                "_fixed": {"c": 100},
            }
        }

        counts = sweep.count_variants(config)

        assert counts["kernel"] == 2


class TestMergeSweepIntoSpec:
    """Tests for merge_sweep_into_spec function."""

    def test_merge_basic(self):
        """Test merging sweep into spec."""
        spec = {
            "inputs": {"X": {"shape": ["BATCH", "DIM"], "dtype": "float16"}},
            "bench-gpu": [
                {
                    "params": ["X"],
                    "dtype": "float16",
                    "dims": {"BATCH": 128, "DIM": 4096},
                }
            ],
        }
        sweep_config = {
            "test_kernel": {
                "batch": [64, 128],
                "dim": [1024, 2048],
            }
        }

        result = sweep.merge_sweep_into_spec(spec, "test_kernel", sweep_config)

        assert len(result["bench-gpu"]) == 4

    def test_merge_preserves_other_fields(self):
        """Test that merge preserves non-dims fields."""
        spec = {
            "inputs": {"X": {"shape": ["BATCH"], "dtype": "float32"}},
            "ci": [{"params": ["X"], "dims": {"BATCH": 4}}],
            "bench-gpu": [
                {
                    "params": ["X"],
                    "dtype": "float16",
                    "dims": {"BATCH": 128},
                    "flop": "BATCH * 1000",
                }
            ],
        }
        sweep_config = {
            "kernel": {
                "batch": [64, 128],
            }
        }

        result = sweep.merge_sweep_into_spec(spec, "kernel", sweep_config)

        # CI should be unchanged
        assert result["ci"] == spec["ci"]
        # bench-gpu should have variants
        assert len(result["bench-gpu"]) == 2


class TestGenerateAllVariants:
    """Tests for generate_all_variants function."""

    def test_generate_all(self):
        """Test generating variants for all kernels."""
        config = {
            "kernel1": {"a": [1, 2]},
            "kernel2": {"b": [3, 4, 5]},
        }

        all_variants = sweep.generate_all_variants(config)

        assert "kernel1" in all_variants
        assert "kernel2" in all_variants
        assert len(all_variants["kernel1"]) == 2
        assert len(all_variants["kernel2"]) == 3

    def test_ignores_underscore_keys(self):
        """Test that top-level underscore keys are ignored."""
        config = {
            "_metadata": {"version": "1.0"},
            "kernel1": {"a": [1, 2]},
        }

        all_variants = sweep.generate_all_variants(config)

        assert "_metadata" not in all_variants
        assert "kernel1" in all_variants
