import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Then keep the original imports:
from src.models.lstm_model import LSTMEmotionModel
from src.models.audio_encoder import AudioEncoder

class AttentionFusionModel(nn.Module):
    """
    Multimodal emotion recognition model with attention-based fusion
    """
    def __init__(self, 
                 text_input_dim=300, 
                 audio_input_dim=74,
                 hidden_dim=256, 
                 output_dim=7,
                 num_layers=4,
                 dropout=0.3,
                 num_attention_heads=8,
                 pretrained_text_model_path=None):
        super(AttentionFusionModel, self).__init__()
        
        self.hidden_dim = hidden_dim
        
        # Text encoder - can load pretrained weights
        self.text_encoder = LSTMEmotionModel(
            input_dim=text_input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,  # Will ignore final layer
            num_layers=num_layers,
            bidirectional=False,
            dropout=dropout
        )
        
        # Load pretrained text model weights if provided
        if pretrained_text_model_path:
            print(f"Loading pretrained text model from {pretrained_text_model_path}")
            state_dict = torch.load(pretrained_text_model_path, map_location='cpu', weights_only=True)
            self.text_encoder.load_state_dict(state_dict, strict=False)
            # Freeze text encoder initially (optional)
            # for param in self.text_encoder.parameters():
            #     param.requires_grad = False
        
        # Audio encoder
        self.audio_encoder = AudioEncoder(
            input_dim=audio_input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # Cross-modal attention mechanism
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Self-attention for fused features
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Modality-specific projection heads
        self.text_projection = nn.Linear(hidden_dim, hidden_dim)
        self.audio_projection = nn.Linear(hidden_dim, hidden_dim)
        
        # Fusion layers
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Final prediction layers
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
    def forward(self, text_input, audio_input, return_attention=False):
        """
        Forward pass for multimodal emotion recognition
        
        Args:
            text_input: [batch_size, text_seq_len, text_dim]
            audio_input: [batch_size, audio_seq_len, audio_dim]
            return_attention: Whether to return attention weights
            
        Returns:
            predictions: [batch_size, seq_len, output_dim]
            attention_weights: (optional) attention visualization
        """
        batch_size = text_input.size(0)
        
        # Encode modalities
        text_features = self.get_text_features(text_input)  # [batch, text_seq, hidden]
        audio_features = self.audio_encoder(audio_input)    # [batch, audio_seq, hidden]
        
        # Project features for attention
        text_proj = self.text_projection(text_features)
        audio_proj = self.audio_projection(audio_features)
        
        # Cross-modal attention: text attends to audio
        text_attended, text_audio_attn = self.cross_attention(
            query=text_proj,
            key=audio_proj,
            value=audio_proj
        )
        
        # Cross-modal attention: audio attends to text  
        audio_attended, audio_text_attn = self.cross_attention(
            query=audio_proj,
            key=text_proj,
            value=text_proj
        )
        
        # Align sequence lengths by pooling audio to match text length
        text_seq_len = text_features.size(1)
        if audio_attended.size(1) != text_seq_len:
            # Average pool audio to match text sequence length
            audio_attended = F.adaptive_avg_pool1d(
                audio_attended.transpose(1, 2), 
                text_seq_len
            ).transpose(1, 2)
        
        # Concatenate attended features
        fused_features = torch.cat([text_attended, audio_attended], dim=-1)
        
        # Apply fusion layer
        fused_features = self.fusion_layer(fused_features)
        
        # Self-attention on fused features
        fused_attended, self_attn = self.self_attention(
            fused_features, fused_features, fused_features
        )
        
        # Final prediction
        predictions = self.classifier(fused_attended)
        
        if return_attention:
            attention_weights = {
                'text_audio_attention': text_audio_attn,
                'audio_text_attention': audio_text_attn,
                'self_attention': self_attn
            }
            return predictions, attention_weights
        
        return predictions
    
    def get_text_features(self, text_input):
        """Extract features from text encoder without final classification"""
        # Forward through LSTM only, skip the final classification layer
        lstm_out, _ = self.text_encoder.lstm(text_input)
        return lstm_out
    
    def freeze_text_encoder(self):
        """Freeze text encoder parameters"""
        for param in self.text_encoder.parameters():
            param.requires_grad = False
            
    def unfreeze_text_encoder(self):
        """Unfreeze text encoder parameters"""
        for param in self.text_encoder.parameters():
            param.requires_grad = True

# Test the attention fusion model
if __name__ == "__main__":
    # Model configuration
    text_dim = 300
    audio_dim = 74
    hidden_dim = 256
    output_dim = 7
    batch_size = 4
    text_seq_len = 98
    audio_seq_len = 300
    
    # Create sample inputs
    text_input = torch.randn(batch_size, text_seq_len, text_dim)
    audio_input = torch.randn(batch_size, audio_seq_len, audio_dim)
    
    print("Testing Attention Fusion Model:")
    print(f"Text input shape: {text_input.shape}")
    print(f"Audio input shape: {audio_input.shape}")
    
    # Create model
    model = AttentionFusionModel(
        text_input_dim=text_dim,
        audio_input_dim=audio_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_attention_heads=8,
        pretrained_text_model_path=None  # Set to your model path when available
    )
    
    # Forward pass
    predictions = model(text_input, audio_input)
    print(f"\nPredictions shape: {predictions.shape}")
    print(f"Predictions range: [{predictions.min():.3f}, {predictions.max():.3f}]")
    
    # Test with attention weights
    predictions_with_attn, attention_weights = model(text_input, audio_input, return_attention=True)
    print(f"\nAttention weights available:")
    for name, attn in attention_weights.items():
        print(f"  {name}: {attn.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    
    print("\n✓ Attention Fusion Model working correctly!")