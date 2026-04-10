import torch
import torch.nn as nn

import ai_bench.mlir


@torch.compile(
    dynamic=False, backend=ai_bench.mlir.cpu_backend(ai_bench.mlir.cpu_pipeline)
)
class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.0):
        """
        Initialize the LSTM model.

        :param input_size: The number of expected features in the input `x`
        :param hidden_size: The number of features in the hidden state `h`
        :param num_layers: Number of recurrent layers
        :param output_size: The number of output features
        :param dropout: If non-zero, introduces a Dropout layer on the outputs of each LSTM layer except the last layer, with dropout probability equal to `dropout`
        """
        super(Model, self).__init__()
        # Initialize hidden state with random values
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=False,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, h0, c0):
        """
        Forward pass through the LSTM model.

        :param x: The input tensor, shape (batch_size, sequence_length, input_size)
        :return: The output tensor, shape (batch_size, sequence_length, output_size)
        """

        # Forward propagate LSTM
        out, state = self.lstm(
            x, (h0, c0)
        )  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])  # out: tensor of shape (batch_size, output_size)

        return state[0]
