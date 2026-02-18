"""Tests for per-problem tolerance configuration."""

import pytest
import torch

from ai_bench.harness import core as ai_hc


class TestToleranceVKey:
    """Tests for VKey tolerance entries."""

    def test_vkey_has_rtol(self):
        """Test VKey enum includes rtol."""
        assert ai_hc.VKey.RTOL == "rtol"

    def test_vkey_has_atol(self):
        """Test VKey enum includes atol."""
        assert ai_hc.VKey.ATOL == "atol"

    def test_vkey_iteration_includes_tolerances(self):
        """Test that tolerance keys are included in VKey iteration."""
        all_keys = list(ai_hc.VKey)
        assert ai_hc.VKey.RTOL in all_keys
        assert ai_hc.VKey.ATOL in all_keys


class TestGetRtol:
    """Tests for get_rtol function."""

    def test_rtol_from_variant(self):
        """Test rtol extraction when specified in variant."""
        variant = {
            ai_hc.VKey.DIMS: {"M": 64},
            ai_hc.VKey.RTOL: 1e-3,
        }
        assert ai_hc.get_rtol(variant) == 1e-3

    def test_rtol_default_when_missing(self):
        """Test rtol falls back to default when not in variant."""
        variant = {ai_hc.VKey.DIMS: {"M": 64}}
        assert ai_hc.get_rtol(variant) == 1e-2

    def test_rtol_zero(self):
        """Test rtol can be set to zero (exact match)."""
        variant = {ai_hc.VKey.RTOL: 0.0}
        assert ai_hc.get_rtol(variant) == 0.0

    def test_rtol_large_value(self):
        """Test rtol with large tolerance (e.g., for low-precision dtypes)."""
        variant = {ai_hc.VKey.RTOL: 0.5}
        assert ai_hc.get_rtol(variant) == 0.5


class TestGetAtol:
    """Tests for get_atol function."""

    def test_atol_from_variant(self):
        """Test atol extraction when specified in variant."""
        variant = {
            ai_hc.VKey.DIMS: {"M": 64},
            ai_hc.VKey.ATOL: 1e-6,
        }
        assert ai_hc.get_atol(variant) == 1e-6

    def test_atol_default_when_missing(self):
        """Test atol falls back to default when not in variant."""
        variant = {ai_hc.VKey.DIMS: {"M": 64}}
        assert ai_hc.get_atol(variant) == 1e-5

    def test_atol_zero(self):
        """Test atol can be set to zero."""
        variant = {ai_hc.VKey.ATOL: 0.0}
        assert ai_hc.get_atol(variant) == 0.0

    def test_atol_large_value(self):
        """Test atol with large tolerance."""
        variant = {ai_hc.VKey.ATOL: 1e-1}
        assert ai_hc.get_atol(variant) == 1e-1


class TestTolerancesPerDtype:
    """Tests for per-dtype tolerance differentiation."""

    def test_different_tolerances_per_dtype_variant(self):
        """Test different tolerances for float32 vs float16 variants."""
        variant_fp32 = {
            ai_hc.VKey.PARAMS: ["A", "B"],
            ai_hc.VKey.DIMS: {"M": 4096, "N": 4096, "K": 4096},
            ai_hc.VKey.TYPE: "float32",
            ai_hc.VKey.FLOP: "2*M*N*K",
            ai_hc.VKey.RTOL: 1e-5,
            ai_hc.VKey.ATOL: 1e-8,
        }
        variant_fp16 = {
            ai_hc.VKey.PARAMS: ["A", "B"],
            ai_hc.VKey.DIMS: {"M": 4096, "N": 4096, "K": 4096},
            ai_hc.VKey.TYPE: "float16",
            ai_hc.VKey.FLOP: "2*M*N*K",
            ai_hc.VKey.RTOL: 1e-1,
            ai_hc.VKey.ATOL: 1e-2,
        }

        # float32 should be tighter
        assert ai_hc.get_rtol(variant_fp32) == 1e-5
        assert ai_hc.get_atol(variant_fp32) == 1e-8

        # float16 should be more relaxed
        assert ai_hc.get_rtol(variant_fp16) == 1e-1
        assert ai_hc.get_atol(variant_fp16) == 1e-2

    def test_mixed_tolerance_spec(self):
        """Test variant with rtol but no atol uses atol default."""
        variant = {
            ai_hc.VKey.DIMS: {"N": 64},
            ai_hc.VKey.RTOL: 5e-3,
            # atol intentionally missing
        }
        assert ai_hc.get_rtol(variant) == 5e-3
        assert ai_hc.get_atol(variant) == 1e-5  # default

    def test_mixed_tolerance_spec_reverse(self):
        """Test variant with atol but no rtol uses rtol default."""
        variant = {
            ai_hc.VKey.DIMS: {"N": 64},
            # rtol intentionally missing
            ai_hc.VKey.ATOL: 1e-7,
        }
        assert ai_hc.get_rtol(variant) == 1e-2  # default
        assert ai_hc.get_atol(variant) == 1e-7


class TestBackwardCompatibility:
    """Tests ensuring backward compatibility with existing specs."""

    def test_no_tolerance_fields_uses_defaults(self):
        """Test that specs without rtol/atol still work with defaults."""
        variant = {
            ai_hc.VKey.PARAMS: ["A", "B"],
            ai_hc.VKey.DIMS: {"M": 32, "N": 32, "K": 32},
            ai_hc.VKey.TYPE: "float32",
            ai_hc.VKey.FLOP: "2*M*N*K",
            ai_hc.VKey.MEM_BYTES: "(M*K + K*N + M*N) * 4",
        }

        assert ai_hc.get_rtol(variant) == 1e-2
        assert ai_hc.get_atol(variant) == 1e-5

    def test_existing_functions_unaffected(self):
        """Test that existing VKey functions still work alongside tolerance."""
        variant = {
            ai_hc.VKey.PARAMS: ["X"],
            ai_hc.VKey.DIMS: {"N": 128},
            ai_hc.VKey.TYPE: "float32",
            ai_hc.VKey.FLOP: "2*N",
            ai_hc.VKey.MEM_BYTES: "N*4",
            ai_hc.VKey.RTOL: 1e-3,
            ai_hc.VKey.ATOL: 1e-6,
        }

        # Existing functions should work unchanged
        assert ai_hc.get_flop(variant) == 256
        assert ai_hc.get_mem_bytes(variant) == 512
        assert ai_hc.get_variant_torch_dtype(variant) == torch.float32

        # New tolerance functions should also work
        assert ai_hc.get_rtol(variant) == 1e-3
        assert ai_hc.get_atol(variant) == 1e-6

    def test_empty_variant_defaults(self):
        """Test that minimal variant returns defaults."""
        variant = {}
        assert ai_hc.get_rtol(variant) == 1e-2
        assert ai_hc.get_atol(variant) == 1e-5


class TestYamlRoundTrip:
    """Tests for YAML serialization/deserialization of tolerances."""

    def test_tolerance_survives_yaml_roundtrip(self, tmp_path):
        """Test tolerances survive YAML write and read."""
        import yaml

        spec = {
            "inputs": {
                "X": {"shape": ["N"], "dtype": "float32"},
            },
            "bench-gpu": [
                {
                    "params": ["X"],
                    "dims": {"N": 1024},
                    "dtype": "float32",
                    "flop": "2*N",
                    "rtol": 1e-3,
                    "atol": 1e-6,
                },
                {
                    "params": ["X"],
                    "dims": {"N": 1024},
                    "dtype": "float16",
                    "flop": "2*N",
                    "rtol": 5e-2,
                    "atol": 1e-3,
                },
            ],
        }

        spec_path = tmp_path / "test_spec.yaml"
        with open(spec_path, "w") as f:
            yaml.dump(spec, f, default_flow_style=False)

        with open(spec_path) as f:
            loaded = yaml.safe_load(f)

        variants = loaded["bench-gpu"]

        # float32 variant
        assert ai_hc.get_rtol(variants[0]) == 1e-3
        assert ai_hc.get_atol(variants[0]) == 1e-6

        # float16 variant
        assert ai_hc.get_rtol(variants[1]) == 5e-2
        assert ai_hc.get_atol(variants[1]) == 1e-3

    def test_missing_tolerance_in_yaml(self, tmp_path):
        """Test YAML spec without tolerance fields returns defaults."""
        import yaml

        spec = {
            "inputs": {
                "A": {"shape": ["N", "N"], "dtype": "float32"},
            },
            "bench-gpu": [
                {
                    "params": ["A"],
                    "dims": {"N": 512},
                    "dtype": "float32",
                    "flop": "2*N*N*N",
                },
            ],
        }

        spec_path = tmp_path / "no_tol_spec.yaml"
        with open(spec_path, "w") as f:
            yaml.dump(spec, f, default_flow_style=False)

        with open(spec_path) as f:
            loaded = yaml.safe_load(f)

        variant = loaded["bench-gpu"][0]
        assert ai_hc.get_rtol(variant) == 1e-2
        assert ai_hc.get_atol(variant) == 1e-5

    def test_partial_tolerance_in_yaml(self, tmp_path):
        """Test YAML spec with only rtol specified."""
        import yaml

        spec = {
            "inputs": {
                "X": {"shape": ["N"], "dtype": "float32"},
            },
            "ci": [
                {
                    "params": ["X"],
                    "dims": {"N": 8},
                    "rtol": 1e-4,
                },
            ],
        }

        spec_path = tmp_path / "partial_tol_spec.yaml"
        with open(spec_path, "w") as f:
            yaml.dump(spec, f, default_flow_style=False)

        with open(spec_path) as f:
            loaded = yaml.safe_load(f)

        variant = loaded["ci"][0]
        assert ai_hc.get_rtol(variant) == 1e-4
        assert ai_hc.get_atol(variant) == 1e-5  # default


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
