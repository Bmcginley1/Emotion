import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torch.nn as nn
from mmsdk.mmdatasdk import mmdataset
import numpy as np
import matplotlib.pyplot as plt

# Data paths
DATA_ROOT = "data/raw/mosei"
CSD_FILES = {
    "glove_vectors": os.path.join(DATA_ROOT, "CMU_MOSEI_TimestampedWordVectors.csd"),
    "COVAREP": os.path.join(DATA_ROOT, "CMU_MOSEI_COVAREP.csd"),
    "All Labels": os.path.join(DATA_ROOT, "CMU_MOSEI_Labels.csd")
}

print("Loading MOSEI dataset with text and audio features...")
dataset_obj = mmdataset(CSD_FILES)

# Extract sequences
text_seq = dataset_obj["glove_vectors"]
audio_seq = dataset_obj["COVAREP"] 
labels_seq = dataset_obj["All Labels"]

print(f"Available utterances in text: {len(text_seq.data)}")
print(f"Available utterances in audio: {len(audio_seq.data)}")
print(f"Available utterances in labels: {len(labels_seq.data)}")

# Find common utterances across all modalities
common_utt_ids = set(text_seq.data.keys()).intersection(
    set(audio_seq.data.keys())).intersection(
    set(labels_seq.data.keys()))

print(f"Common utterances across all modalities: {len(common_utt_ids)}")

# Explore COVAREP features
sample_utt = list(common_utt_ids)[0]
sample_audio = audio_seq.data[sample_utt]["features"][:].astype('float32')
sample_text = text_seq.data[sample_utt]["features"][:].astype('float32')
sample_labels = labels_seq.data[sample_utt]["features"][:].astype('float32')

print(f"\n=== FEATURE ANALYSIS FOR SAMPLE UTTERANCE: {sample_utt} ===")
print(f"Audio features shape: {sample_audio.shape}")
print(f"Text features shape: {sample_text.shape}")
print(f"Labels shape: {sample_labels.shape}")

print(f"\nAudio feature statistics:")
# Clean audio data for analysis
clean_audio = sample_audio.copy()
clean_audio = np.nan_to_num(clean_audio, nan=0.0, posinf=1.0, neginf=-1.0)
finite_mask = np.isfinite(sample_audio)

print(f"  Min: {clean_audio.min():.4f}")
print(f"  Max: {clean_audio.max():.4f}")
print(f"  Mean: {clean_audio.mean():.4f}")
print(f"  Std: {clean_audio.std():.4f}")
print(f"  Has NaN: {np.isnan(sample_audio).any()}")
print(f"  Has Inf: {np.isinf(sample_audio).any()}")
print(f"  Finite values: {finite_mask.sum()}/{sample_audio.size} ({finite_mask.mean()*100:.1f}%)")

# Check sequence lengths across modalities
text_len = sample_text.shape[0]
audio_len = sample_audio.shape[0]
labels_len = sample_labels.shape[0]

print(f"\nSequence lengths:")
print(f"  Text: {text_len}")
print(f"  Audio: {audio_len}")
print(f"  Labels: {labels_len}")

# Analyze multiple utterances to understand length distribution
print(f"\n=== LENGTH DISTRIBUTION ANALYSIS ===")
text_lengths = []
audio_lengths = []
audio_dims = []

for utt_id in list(common_utt_ids)[:100]:  # Sample first 100
    text_feat = text_seq.data[utt_id]["features"][:].astype('float32')
    audio_feat = audio_seq.data[utt_id]["features"][:].astype('float32')
    
    text_lengths.append(text_feat.shape[0])
    audio_lengths.append(audio_feat.shape[0])
    audio_dims.append(audio_feat.shape[1])

print(f"Text lengths - Min: {min(text_lengths)}, Max: {max(text_lengths)}, Mean: {np.mean(text_lengths):.1f}")
print(f"Audio lengths - Min: {min(audio_lengths)}, Max: {max(audio_lengths)}, Mean: {np.mean(audio_lengths):.1f}")
print(f"Audio feature dimensions: {set(audio_dims)}")

# Plot feature distributions
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Audio feature value distribution (use cleaned data)
clean_audio_flat = clean_audio.flatten()
axes[0].hist(clean_audio_flat, bins=50, alpha=0.7, range=(-10, 10))  # Focus on reasonable range
axes[0].set_title('COVAREP Feature Value Distribution (Cleaned)')
axes[0].set_xlabel('Feature Value')
axes[0].set_ylabel('Frequency')

# Sequence length comparison
axes[1].scatter(text_lengths[:50], audio_lengths[:50], alpha=0.6)
axes[1].set_xlabel('Text Sequence Length')
axes[1].set_ylabel('Audio Sequence Length')
axes[1].set_title('Text vs Audio Sequence Lengths')
axes[1].set_yscale('log')  # Use log scale due to large differences

plt.tight_layout()
plt.savefig('audio_feature_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\n=== RECOMMENDATIONS FOR AUDIO ENCODER ===")
print(f"1. Audio feature dimension: {sample_audio.shape[1]}")
print(f"2. Need to handle sequence length alignment between text and audio")
print(f"3. Consider normalization due to wide value range")
print(f"4. Check for and handle NaN/Inf values in preprocessing")

# Basic audio encoder architecture suggestion
print(f"\n=== SUGGESTED AUDIO ENCODER ARCHITECTURE ===")
audio_dim = sample_audio.shape[1]
print(f"""
class AudioEncoder(nn.Module):
    def __init__(self, input_dim={audio_dim}, hidden_dim=256, num_layers=3, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers, 
            batch_first=True, dropout=dropout, bidirectional=False
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x):
        # Handle NaN/Inf values
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        output, (hidden, cell) = self.lstm(x)
        output = self.layer_norm(output)
        return output
""")