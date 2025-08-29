import torch
import torch.nn as nn
import torch.nn.functional as F

class AudioEncoder(nn.Module):
    """
    Audio encoder for COVAREP features with robust preprocessing
    """
    def __init__(self, input_dim=74, hidden_dim=256, num_layers=3, dropout=0.3):
        super(AudioEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Input projection layer to handle feature scaling
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # LSTM layers
        self.lstm = nn.LSTM(
            hidden_dim,  # Use projected dimension
            hidden_dim, 
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # Output normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x, mask=None):
        """
        Forward pass for audio encoder
        
        Args:
            x: Audio features [batch_size, seq_len, input_dim]
            mask: Optional padding mask [batch_size, seq_len]
            
        Returns:
            output: Encoded features [batch_size, seq_len, hidden_dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # Handle numerical issues in audio features
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Project input features
        x = self.input_projection(x)  # [batch_size, seq_len, hidden_dim]
        
        # LSTM processing
        if mask is not None:
            # Pack padded sequence for efficiency
            lengths = mask.sum(dim=1).cpu()
            x_packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths, batch_first=True, enforce_sorted=False
            )
            output_packed, (hidden, cell) = self.lstm(x_packed)
            output, _ = nn.utils.rnn.pad_packed_sequence(
                output_packed, batch_first=True
            )
        else:
            output, (hidden, cell) = self.lstm(x)
        
        # Apply layer normalization
        output = self.layer_norm(output)
        
        return output

class SimpleAudioEncoder(nn.Module):
    """
    Simplified audio encoder for initial testing
    """
    def __init__(self, input_dim=74, hidden_dim=256, num_layers=2, dropout=0.3):
        super(SimpleAudioEncoder, self).__init__()
        
        # Simple LSTM without input projection
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Handle numerical issues
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # LSTM processing
        output, (hidden, cell) = self.lstm(x)
        
        # Normalization and dropout
        output = self.layer_norm(output)
        output = self.dropout(output)
        
        return output

# Test the audio encoder
if __name__ == "__main__":
    # Test with sample audio features
    batch_size = 4
    seq_len = 300
    input_dim = 74
    hidden_dim = 256
    
    # Create sample data
    sample_audio = torch.randn(batch_size, seq_len, input_dim)
    
    # Add some problematic values to test robustness
    sample_audio[0, 5:10, :] = float('inf')
    sample_audio[1, 15:20, :] = float('-inf')
    sample_audio[2, 25:30, :] = float('nan')
    
    print("Testing Audio Encoders:")
    print(f"Input shape: {sample_audio.shape}")
    print(f"Input contains inf: {torch.isinf(sample_audio).any()}")
    print(f"Input contains nan: {torch.isnan(sample_audio).any()}")
    
    # Test Simple Audio Encoder
    simple_encoder = SimpleAudioEncoder(input_dim=input_dim, hidden_dim=hidden_dim)
    simple_output = simple_encoder(sample_audio)
    print(f"\nSimple Encoder Output: {simple_output.shape}")
    print(f"Output range: [{simple_output.min():.3f}, {simple_output.max():.3f}]")
    print(f"Output contains inf: {torch.isinf(simple_output).any()}")
    print(f"Output contains nan: {torch.isnan(simple_output).any()}")
    
    # Test Advanced Audio Encoder
    advanced_encoder = AudioEncoder(input_dim=input_dim, hidden_dim=hidden_dim)
    advanced_output = advanced_encoder(sample_audio)
    print(f"\nAdvanced Encoder Output: {advanced_output.shape}")
    print(f"Output range: [{advanced_output.min():.3f}, {advanced_output.max():.3f}]")
    print(f"Output contains inf: {torch.isinf(advanced_output).any()}")
    print(f"Output contains nan: {torch.isnan(advanced_output).any()}")
    
    print("\n✓ Audio encoders working correctly!")
    
    # Count parameters
    simple_params = sum(p.numel() for p in simple_encoder.parameters())
    advanced_params = sum(p.numel() for p in advanced_encoder.parameters())
    print(f"\nParameter counts:")
    print(f"  Simple encoder: {simple_params:,}")
    print(f"  Advanced encoder: {advanced_params:,}")