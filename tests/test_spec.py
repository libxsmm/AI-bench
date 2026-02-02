"""Tests for dtype mismatch warning between input and variant specs."""

import pytest
import torch

from ai_bench.harness import core as ai_hc


class TestInputDataTypes:
    def test_input_data_types(self):
        """Test inputs with various data types."""
        float_param = "T_FLOAT"
        int_param = "T_INT"
        bool_param = "T_BOOL"
        inherit_param = "T_INHERIT"

        variant = {
            ai_hc.VKey.PARAMS: [float_param, int_param, bool_param, inherit_param],
            ai_hc.VKey.TYPE: "float32",
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
            inherit_param: {
                ai_hc.InKey.SHAPE: ["IN_FEAT"],
                ai_hc.InKey.TYPE: ai_hc.InInputKey.INHERIT,
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

        input_inherit = inputs[inherit_param]
        assert not ai_hc.input_is_float(input_inherit)
        assert not ai_hc.input_is_int(input_inherit)
        assert not ai_hc.input_is_bool(input_inherit)
        with pytest.raises(Exception) as e:
            ai_hc.input_torch_dtype(input_inherit)
        assert "Input uses 'inherit' dtype but variant has no 'dtype' field" in str(e)
        inherit_dtype = ai_hc.input_torch_dtype(input_inherit, variant)
        assert inherit_dtype == torch.float32

        int_range = ai_hc.input_range(variant, input_int)
        assert int_range == [0, 8]

        inputs = ai_hc.get_inputs(variant, inputs, device=torch.device("cpu"))
        assert len(inputs) == 4
        assert inputs[0].dtype == torch.bfloat16
        assert inputs[1].dtype == torch.int64
        assert inputs[2].dtype == torch.bool
        assert inputs[3].dtype == torch.float32


class TestIntegerInputCreation:
    """Tests for integer input creation."""

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

    def test_integer_inheritance(self):
        """Test integer type inherited from the variant."""
        variant = {
            ai_hc.VKey.PARAMS: ["X"],
            ai_hc.VKey.TYPE: "int16",
            ai_hc.VKey.DIMS: {"BATCH": 2, "IN_FEAT": 8},
        }
        inputs = {
            "X": {
                ai_hc.InKey.SHAPE: ["BATCH"],
                ai_hc.InKey.TYPE: ai_hc.InInputKey.INHERIT,
                ai_hc.InKey.RANGE: [1, 5],
            },
        }

        inputs = ai_hc.get_inputs(variant, inputs, device=torch.device("cpu"))
        assert len(inputs) == 1
        assert inputs[0].dtype == torch.int16


class TestDtypeMismatchWarning:
    """Tests for dtype mismatch warning in get_inputs."""

    def test_warns_on_dtype_mismatch(self, caplog):
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

        with caplog.at_level("DEBUG", logger="ai_bench"):
            tensors = ai_hc.get_inputs(variant, inputs, device=torch.device("cpu"))
            assert "dtype" in caplog.text
            assert "float16" in caplog.text
            assert "bfloat16" in caplog.text

        # Tensor should still be created with input dtype.
        assert tensors[0].dtype == torch.float16

    def test_no_warning_when_dtypes_match(self, caplog):
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

        with caplog.at_level("DEBUG", logger="ai_bench"):
            ai_hc.get_inputs(variant, inputs, device=torch.device("cpu"))
            assert "dtype" not in caplog.text


class TestInputInitializations:
    """Tests input initialization transformations."""

    def test_input_inits(self):
        """Test inputs with various initializers."""
        variant = {
            ai_hc.VKey.PARAMS: [
                ai_hc.InInitKey.SCALE,
                ai_hc.InInitKey.SOFTMAX,
                ai_hc.InInitKey.ABS,
                ai_hc.InInitKey.NORMALIZE,
                ai_hc.InInitKey.SYMMETRIC,
                ai_hc.InInitKey.TRI_UPPER,
                ai_hc.InInitKey.TRI_LOWER,
                ai_hc.InInitKey.TRANSPOSE,
                ai_hc.InInitKey.UNIFORM,
                ai_hc.InInitKey.RADEMACHER,
                "multiple_inits",
            ],
            ai_hc.VKey.DIMS: {"BATCH": 2, "IN_FEAT": 4},
        }
        inputs = {
            ai_hc.InInitKey.SCALE: {
                ai_hc.InKey.SHAPE: ["IN_FEAT"],
                ai_hc.InKey.TYPE: "float16",
                ai_hc.InKey.INITS: [ai_hc.InInitKey.SCALE],
            },
            ai_hc.InInitKey.SOFTMAX: {
                ai_hc.InKey.SHAPE: ["BATCH", "IN_FEAT"],
                ai_hc.InKey.TYPE: "float16",
                ai_hc.InKey.INITS: [ai_hc.InInitKey.SOFTMAX],
            },
            ai_hc.InInitKey.ABS: {
                ai_hc.InKey.SHAPE: ["IN_FEAT"],
                ai_hc.InKey.TYPE: "int16",
                ai_hc.InKey.RANGE: [-5, 5],
                ai_hc.InKey.INITS: [ai_hc.InInitKey.ABS],
            },
            ai_hc.InInitKey.NORMALIZE: {
                ai_hc.InKey.SHAPE: ["BATCH", "IN_FEAT"],
                ai_hc.InKey.TYPE: "float16",
                ai_hc.InKey.INITS: [ai_hc.InInitKey.NORMALIZE],
            },
            ai_hc.InInitKey.SYMMETRIC: {
                ai_hc.InKey.SHAPE: ["IN_FEAT", "IN_FEAT"],
                ai_hc.InKey.TYPE: "float16",
                ai_hc.InKey.INITS: [ai_hc.InInitKey.SYMMETRIC],
            },
            ai_hc.InInitKey.TRI_UPPER: {
                ai_hc.InKey.SHAPE: ["BATCH", "IN_FEAT"],
                ai_hc.InKey.TYPE: "float16",
                ai_hc.InKey.INITS: [ai_hc.InInitKey.TRI_UPPER],
            },
            ai_hc.InInitKey.TRI_LOWER: {
                ai_hc.InKey.SHAPE: ["BATCH", "IN_FEAT"],
                ai_hc.InKey.TYPE: "float32",
                ai_hc.InKey.INITS: [ai_hc.InInitKey.TRI_LOWER],
            },
            ai_hc.InInitKey.TRANSPOSE: {
                ai_hc.InKey.SHAPE: ["BATCH", "IN_FEAT"],
                ai_hc.InKey.TYPE: "int32",
                ai_hc.InKey.RANGE: [-3, 3],
                ai_hc.InKey.INITS: [ai_hc.InInitKey.TRANSPOSE],
            },
            ai_hc.InInitKey.UNIFORM: {
                ai_hc.InKey.SHAPE: ["BATCH", "IN_FEAT"],
                ai_hc.InKey.TYPE: "float32",
                ai_hc.InKey.INITS: [ai_hc.InInitKey.UNIFORM],
            },
            ai_hc.InInitKey.RADEMACHER: {
                ai_hc.InKey.SHAPE: ["BATCH", "IN_FEAT"],
                ai_hc.InKey.TYPE: "int8",
                ai_hc.InKey.RANGE: [-10, 10],
                ai_hc.InKey.INITS: [ai_hc.InInitKey.RADEMACHER],
            },
            "multiple_inits": {
                ai_hc.InKey.SHAPE: ["BATCH", "IN_FEAT"],
                ai_hc.InKey.TYPE: "float16",
                ai_hc.InKey.INITS: [
                    ai_hc.InInitKey.SCALE,
                    ai_hc.InInitKey.SOFTMAX,
                    ai_hc.InInitKey.SCALE,
                    ai_hc.InInitKey.ABS,
                ],
            },
        }

        invalid_variant = {
            ai_hc.VKey.PARAMS: ["INVALID"],
            ai_hc.VKey.DIMS: {"BATCH": 2, "IN_FEAT": 4},
        }
        invalid_inputs = {
            "INVALID": {
                ai_hc.InKey.SHAPE: ["BATCH", "IN_FEAT"],
                ai_hc.InKey.TYPE: "float16",
                ai_hc.InKey.INITS: [
                    ai_hc.InInitKey.SCALE,
                    "invalid_init",
                    ai_hc.InInitKey.SCALE,
                ],
            },
        }

        with pytest.raises(Exception) as e:
            ai_hc.get_inputs(
                invalid_variant, invalid_inputs, device=torch.device("cpu")
            )
        assert "invalid_init" in str(e)

        inputs = ai_hc.get_inputs(variant, inputs, device=torch.device("cpu"))
        assert len(inputs) == 11
        assert inputs[0].dtype == torch.float16
        assert inputs[2].dtype == torch.int16
        assert inputs[3].dtype == torch.float16
        assert inputs[7].dtype == torch.int32
        assert inputs[7].shape == (4, 2)
        assert inputs[8].dtype == torch.float32
        assert all(x >= -1.0 or x <= 1.0 for x in inputs[8].flatten().tolist())
        assert inputs[9].dtype == torch.int8
        assert inputs[9].shape == (2, 4)
        assert all(x == -1 or x == 1 for x in inputs[9].flatten().tolist())
        assert inputs[10].dtype == torch.float16


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
