import torch
import torch.nn as nn

class LSTMEmotionModel(nn.Module):
    """
    LSTM model for sequence-based emotion recognition.

    Args:
        input_dim: Dimensionality of input embeddings per timestep
        hidden_dim: Number of hidden units in LSTM
        output_dim: Number of output labels per timestep
        num_layers: Number of LSTM layers
        bidirectional: Whether to use bidirectional LSTM
        dropout: Dropout probability between LSTM layers
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, 
                 bidirectional=False, dropout=0.0):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # If bidirectional, hidden_dim doubles
        fc_input_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(fc_input_dim, output_dim)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            out: Tensor of shape (batch_size, seq_len, output_dim)
        """
        # lstm_out: (batch, seq_len, num_directions*hidden_dim)
        lstm_out, _ = self.lstm(x)

        # Apply linear layer to each timestep
        out = self.fc(lstm_out)  # (batch, seq_len, output_dim)
        return out
