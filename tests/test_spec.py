"""Tests for dtype mismatch warning between input and variant specs."""

import warnings

import pytest
import torch

from ai_bench.harness import core as ai_hc


class TestDtypeMismatchWarning:
    """Tests for dtype mismatch warning in get_inputs."""

    def test_warns_on_dtype_mismatch(self):
        """Test that warning is raised when input dtype differs from variant dtype."""
        variant = {
            ai_hc.VKey.PARAMS: ["X"],
            ai_hc.VKey.DIMS: {"BATCH": 32, "IN_FEAT": 128},
            ai_hc.VKey.TYPE: "bfloat16",
        }
        inputs = {
            "X": {
                ai_hc.InKey.SHAPE: ["BATCH", "IN_FEAT"],
                ai_hc.InKey.TYPE: "float16",
            },
        }

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            tensors = ai_hc.get_inputs(variant, inputs, device=torch.device("cpu"))

            assert len(w) == 1
            assert issubclass(w[0].category, UserWarning)
            assert "dtype" in str(w[0].message).lower()
            assert "float16" in str(w[0].message)
            assert "bfloat16" in str(w[0].message)

        # Tensor should still be created with input dtype.
        assert tensors[0].dtype == torch.float16

    def test_no_warning_when_dtypes_match(self):
        """Test that no warning is raised when input and variant dtypes match."""
        variant = {
            ai_hc.VKey.PARAMS: ["X"],
            ai_hc.VKey.DIMS: {"N": 64},
            ai_hc.VKey.TYPE: "float32",
        }
        inputs = {
            "X": {
                ai_hc.InKey.SHAPE: ["N"],
                ai_hc.InKey.TYPE: "float32",
            },
        }

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ai_hc.get_inputs(variant, inputs, device=torch.device("cpu"))

            assert len(w) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
