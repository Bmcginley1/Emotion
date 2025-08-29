import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
import matplotlib.pyplot as plt
from src.models.attention_fusion import AttentionFusionModel
from src.data.audio.multimodal_dataloader import get_multimodal_dataloader

# -----------------------------
# Hyperparameters
# -----------------------------
batch_size = 8  # Reduced due to larger model
text_input_dim = 300
audio_input_dim = 74
hidden_dim = 256
output_dim = 7
num_layers = 4
num_attention_heads = 8
num_epochs = 50
learning_rate = 5e-4
val_split = 0.15
dropout = 0.3
weight_decay = 5e-5
patience = 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path to your pretrained text model
PRETRAINED_TEXT_MODEL = "src/trained_models/best_model_audio.pt"

# -----------------------------
# Load Multimodal Data
# -----------------------------
print("Loading multimodal dataset...")
full_dataloader, audio_scaler = get_multimodal_dataloader(
    batch_size=1, 
    shuffle=True, 
    max_audio_length=300
)

# Extract dataset for splitting
dataset = full_dataloader.dataset
val_size = int(len(dataset) * val_split)
train_size = len(dataset) - val_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create data loaders
train_loader = torch.utils.data.DataLoader(
    train_dataset, 
    batch_size=batch_size, 
    shuffle=True
)
val_loader = torch.utils.data.DataLoader(
    val_dataset, 
    batch_size=batch_size, 
    shuffle=False
)

print(f"Dataset split: {train_size} train, {val_size} validation")

# -----------------------------
# Initialize Model
# -----------------------------
model = AttentionFusionModel(
    text_input_dim=text_input_dim,
    audio_input_dim=audio_input_dim,
    hidden_dim=hidden_dim,
    output_dim=output_dim,
    num_layers=num_layers,
    dropout=dropout,
    num_attention_heads=num_attention_heads,
    pretrained_text_model_path=PRETRAINED_TEXT_MODEL
)

model.to(device)

# Custom loss function (same as your text model)
class MixedRangeMSELoss(nn.Module):
    def __init__(self):
        super(MixedRangeMSELoss, self).__init__()
    
    def forward(self, input, target):
        sentiment_loss = ((input[:, :, 0] - target[:, :, 0]) ** 2).mean()
        emotion_loss = ((input[:, :, 1:] - target[:, :, 1:]) ** 2).mean()
        
        total_loss = (sentiment_loss + emotion_loss) / 2
        return total_loss

criterion = MixedRangeMSELoss()
optimizer = optim.Adam(
    model.parameters(),
    lr=learning_rate,
    weight_decay=weight_decay
)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 'min', patience=5, factor=0.5
)

print(f"Model loaded on {device}")
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

# -----------------------------
# Training Loop
# -----------------------------
train_losses = []
val_losses = []
best_val_loss = float('inf')
epochs_no_improve = 0

print("\nStarting multimodal training...")

for epoch in range(num_epochs):
    # Training
    model.train()
    epoch_loss = 0.0
    num_batches = 0
    
    for batch_idx, batch in enumerate(train_loader):
        text_input = batch['text'].to(device)
        audio_input = batch['audio'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        predictions = model(text_input, audio_input)
        loss = criterion(predictions, labels)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        epoch_loss += loss.item()
        num_batches += 1
        
        # Progress update
        if (batch_idx + 1) % 50 == 0:
            print(f"  Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}")
    
    avg_train_loss = epoch_loss / num_batches
    train_losses.append(avg_train_loss)
    
    # Validation
    model.eval()
    val_loss = 0.0
    num_val_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            text_input = batch['text'].to(device)
            audio_input = batch['audio'].to(device)
            labels = batch['labels'].to(device)
            
            predictions = model(text_input, audio_input)
            loss = criterion(predictions, labels)
            
            val_loss += loss.item()
            num_val_batches += 1
    
    avg_val_loss = val_loss / num_val_batches
    val_losses.append(avg_val_loss)
    scheduler.step(avg_val_loss)
    
    print(f"Epoch [{epoch+1}/{num_epochs}], "
          f"Train Loss: {avg_train_loss:.4f}, "
          f"Val Loss: {avg_val_loss:.4f}")
    
    # Early stopping and model saving
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), "src/trained_models/best_multimodal_model.pt")
        print(f"  → New best model saved! Val Loss: {avg_val_loss:.4f}")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

# -----------------------------
# Plot Results
# -----------------------------
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label="Train Loss", alpha=0.8)
plt.plot(val_losses, label="Validation Loss", alpha=0.8)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Multimodal Model Training Progress")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("multimodal_training_curve.png", dpi=300, bbox_inches='tight')
plt.show()

# -----------------------------
# Final Results
# -----------------------------
print(f"\n=== TRAINING COMPLETE ===")
print(f"Best validation loss: {best_val_loss:.4f}")
print(f"Model saved as: src/trained_models/best_multimodal_model.pt")
print(f"\nHyperparameters used:")
print(f"  Batch size: {batch_size}")
print(f"  Hidden dim: {hidden_dim}")
print(f"  Learning rate: {learning_rate}")
print(f"  Attention heads: {num_attention_heads}")
print(f"  Dropout: {dropout}")
print(f"  Weight decay: {weight_decay}")

# Compare with text-only baseline
print(f"\nComparison with text-only model:")
print(f"  Text-only best val loss: ~0.044")  # From your previous results
print(f"  Multimodal best val loss: {best_val_loss:.4f}")
if best_val_loss < 0.044:
    improvement = ((0.044 - best_val_loss) / 0.044) * 100
    print(f"  Improvement: {improvement:.1f}% better than text-only")
else:
    print(f"  Multimodal model needs further tuning")