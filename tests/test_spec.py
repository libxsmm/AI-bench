"""Tests for dtype mismatch warning between input and variant specs."""

import warnings

import pytest
import torch

from ai_bench.harness import core as ai_hc


class TestDtypeMismatchWarning:
    """Tests for dtype mismatch warning in get_inputs."""

    def test_input_int_ranges(self):
        """Test integer inputs with various value ranges."""
        variant = {
            ai_hc.VKey.PARAMS: ["X", "Y", "Z"],
            ai_hc.VKey.DIMS: {"BATCH": 2, "IN_FEAT": 16},
        }
        inputs = {
            "X": {
                ai_hc.InKey.SHAPE: ["BATCH"],
                ai_hc.InKey.TYPE: "int64",
                ai_hc.InKey.RANGE: [1, 5],
            },
            "Y": {
                ai_hc.InKey.SHAPE: ["IN_FEAT"],
                ai_hc.InKey.TYPE: "int32",
                ai_hc.InKey.RANGE: ["BATCH", 7],
            },
            "Z": {
                ai_hc.InKey.SHAPE: ["BATCH"],
                ai_hc.InKey.TYPE: "int16",
                ai_hc.InKey.RANGE: ["BATCH", "IN_FEAT"],
            },
            "INVALID_RANGE": {
                ai_hc.InKey.SHAPE: ["BATCH"],
                ai_hc.InKey.TYPE: "int64",
                ai_hc.InKey.RANGE: [3, 6, 9],
            },
        }

        assert ai_hc.input_range(variant, inputs["X"]) == [1, 5]
        assert ai_hc.input_range(variant, inputs["Y"]) == [2, 7]
        assert ai_hc.input_range(variant, inputs["Z"]) == [2, 16]

        with pytest.raises(Exception):
            ai_hc.input_range(variant, inputs["INVALID_RANGE"])

        inputs = ai_hc.get_inputs(variant, inputs, device=torch.device("cpu"))
        assert len(inputs) == 3
        assert inputs[0].dtype == torch.int64
        assert inputs[1].dtype == torch.int32
        assert inputs[2].dtype == torch.int16

    def test_input_data_types(self):
        """Test inputs with various data types."""
        float_param = "T_FLOAT"
        int_param = "T_INT"
        bool_param = "T_BOOL"

        variant = {
            ai_hc.VKey.PARAMS: [float_param, int_param, bool_param],
            ai_hc.VKey.DIMS: {"BATCH": 2, "IN_FEAT": 8},
        }
        inputs = {
            float_param: {
                ai_hc.InKey.SHAPE: ["BATCH", "IN_FEAT"],
                ai_hc.InKey.TYPE: "bfloat16",
            },
            int_param: {
                ai_hc.InKey.SHAPE: ["BATCH"],
                ai_hc.InKey.TYPE: "int64",
                ai_hc.InKey.RANGE: [0, "IN_FEAT"],
            },
            bool_param: {
                ai_hc.InKey.SHAPE: ["IN_FEAT"],
                ai_hc.InKey.TYPE: "bool",
            },
        }

        input_float = inputs[float_param]
        assert ai_hc.input_is_float(input_float)
        assert not ai_hc.input_is_int(input_float)
        assert not ai_hc.input_is_bool(input_float)

        input_int = inputs[int_param]
        assert not ai_hc.input_is_float(input_int)
        assert ai_hc.input_is_int(input_int)
        assert not ai_hc.input_is_bool(input_int)

        input_bool = inputs[bool_param]
        assert not ai_hc.input_is_float(input_bool)
        assert not ai_hc.input_is_int(input_bool)
        assert ai_hc.input_is_bool(input_bool)

        int_range = ai_hc.input_range(variant, input_int)
        assert int_range == [0, 8]

        inputs = ai_hc.get_inputs(variant, inputs, device=torch.device("cpu"))
        assert len(inputs) == 3
        assert inputs[0].dtype == torch.bfloat16
        assert inputs[1].dtype == torch.int64
        assert inputs[2].dtype == torch.bool

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
