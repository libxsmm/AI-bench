import torch
from torch import nn

from ai_bench import utils as ai_utils


class TestMemoryCounter:
    """Tests for memory counter."""

    def test_torch_function_wrapper_model(self):
        """Test memory count for module wrapping a function."""

        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()

            def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
                return torch.matmul(a, b)

        M = 4
        N = 8
        K = 16

        mod = Model()
        a = torch.empty(M, K, dtype=torch.float32)
        b = torch.empty(K, N, dtype=torch.float32)

        bytes_ref = (
            M * K
            + K * N  # Inputs
            + M * N  # Result
        ) * a.element_size()
        mem_bytes = ai_utils.count_torch_memory_bytes(mod, [a, b])

        assert bytes_ref == mem_bytes
        assert isinstance(mem_bytes, int)

    def test_torch_model_with_internal_params(self):
        """Test memory count for module with internal parameters."""

        BATCH = 4
        IN_FEAT = 8
        OUT_FEAT = 16

        mod = nn.Linear(IN_FEAT, OUT_FEAT, bias=True)
        x = torch.empty(BATCH, IN_FEAT, dtype=torch.float32)

        bytes_ref = (
            BATCH * IN_FEAT  # Input
            + IN_FEAT * OUT_FEAT  # Weight
            + OUT_FEAT  # Bias
            + BATCH * OUT_FEAT  # Result
        ) * x.element_size()
        mem_bytes = ai_utils.count_torch_memory_bytes(mod, [x])

        assert bytes_ref == mem_bytes

    def test_torch_functional_wrapper_model(self):
        """Test memory count for modele wrapping a torch stateless network."""

        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()

            def forward(
                self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
            ) -> torch.Tensor:
                return torch.nn.functional.scaled_dot_product_attention(q, k, v)

        Z = 2
        H = 4
        N_CTX = 8
        D_HEAD = 16

        mod = Model()
        q = torch.rand(Z, H, N_CTX, D_HEAD, dtype=torch.bfloat16)
        k = torch.rand(Z, H, N_CTX, D_HEAD, dtype=torch.bfloat16)
        v = torch.rand(Z, H, N_CTX, D_HEAD, dtype=torch.bfloat16)

        bytes_ref = (2 * Z * H * (N_CTX * D_HEAD + N_CTX * D_HEAD)) * q.element_size()
        mem_bytes = ai_utils.count_torch_memory_bytes(mod, [q, k, v])

        assert bytes_ref == mem_bytes

    def test_torch_model_multiple_outputs(self):
        """Test memory count for module with mutliple results."""

        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()

            def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                val = torch.add(x, x)
                return val, val

        M = 4
        N = 8

        mod = Model()
        x = torch.empty(M, N, dtype=torch.float32)

        bytes_ref = (3 * M * N) * x.element_size()
        mem_bytes = ai_utils.count_torch_memory_bytes(mod, [x])

        assert bytes_ref == mem_bytes
        assert isinstance(mem_bytes, int)

    def test_torch_model_multiple_layers(self):
        """Test memory count for module with mutliple layers."""

        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()

                self.network = nn.Sequential(
                    nn.Linear(4, 8, bias=False), nn.ReLU(), nn.Linear(8, 2, bias=True)
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.network(x)

        M = 16

        mod = Model()
        x = torch.empty(M, 4, dtype=torch.float32)

        bytes_ref = (
            (M * 4 + 4 * 8 + M * 8)  # Linear without bias
            + (M * 8 + M * 8)  # ReLU
            + (M * 8 + 8 * 2 + 2 + M * 2)  # Linear with bias
        ) * x.element_size()
        mem_bytes = ai_utils.count_torch_memory_bytes(mod, [x])

        assert bytes_ref == mem_bytes

    def test_torch_model_with_params_and_buffers(self):
        """Test memory count for module with parameters and buffers."""

        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.matmul = nn.Linear(4, 8, bias=False)
                self.add_value = nn.Parameter(torch.randn(8))
                self.sub_value = nn.Buffer(torch.randn(8))

            def forward(self, x):
                x = self.matmul(x)
                x = x + self.add_value
                x = x - self.sub_value
                return x

        M = 16

        mod = Model()
        x = torch.empty(M, 4, dtype=torch.float32)

        bytes_ref = (
            (M * 4 + 4 * 8 + M * 8)  # Linear without bias
            + (8)  # Parameter
            + (8)  # Buffer
        ) * x.element_size()
        mem_bytes = ai_utils.count_torch_memory_bytes(mod, [x])

        assert bytes_ref == mem_bytes
