import torch

from ai_bench import utils as ai_utils


class TestFlopCounter:
    """Tests for flop counter."""

    def test_torch_func_flop(self):
        """Test flop estimate for plain torch function."""

        def torch_fn(a, b):
            return torch.matmul(a, b)

        M = 4
        N = 8
        K = 32
        flop_ref = 2 * M * N * K

        a = torch.empty(M, K, dtype=torch.float32)
        b = torch.empty(K, N, dtype=torch.float32)
        flop_est = ai_utils.count_torch_flop(torch_fn, [a, b])

        assert flop_ref == flop_est
        assert isinstance(flop_est, int)

    def test_torch_model_flop(self):
        """Test flop estimate for a torch model."""

        class Model(torch.nn.Module):
            def __init__(self, in_size, out_size):
                super().__init__()

                layers = []
                layers.append(torch.nn.Linear(in_size, out_size, bias=True))
                layers.append(torch.nn.ReLU())

                self.network = torch.nn.Sequential(*layers)

            def forward(self, x):
                return self.network(x)

        BATCH = 4
        IN_SIZE = 8
        OUT_SIZE = 16

        # Torch flop counter does not include element-wise operations
        # like bias addition and ReLU.
        # For example, see:
        # https://discuss.pytorch.org/t/should-the-bias-in-a-linear-layer-be-considered-when-estimating-flop/224158
        flop_ref = 2 * BATCH * IN_SIZE * OUT_SIZE

        x = torch.empty(BATCH, IN_SIZE, dtype=torch.float32)
        model = Model(IN_SIZE, OUT_SIZE)
        flop_est = ai_utils.count_torch_flop(model.forward, [x])

        assert flop_ref == flop_est
