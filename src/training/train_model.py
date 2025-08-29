import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
from src.models.lstm_model import LSTMEmotionModel
from src.data.text.mosei_exploration import get_mosei_dataloader
import matplotlib.pyplot as plt

# -----------------------------
# Hyperparameters
# -----------------------------
batch_size = 16  # Reduced for better gradients
hidden_dim = 256
num_epochs = 50
num_layers = 4
learning_rate = 5e-4
val_split = 0.15
dropout = 0.3
weight_decay = 5e-5
patience = 5
tolerance = 1e-6
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Load DataLoader
# -----------------------------
full_dataloader = get_mosei_dataloader(batch_size=1, shuffle=True)  # batch_size=1 for splitting

# Extract dataset object from DataLoader
dataset = full_dataloader.dataset
val_size = int(len(dataset) * val_split)
train_size = len(dataset) - val_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# -----------------------------
# Initialize model
# -----------------------------
example_batch, _ = next(iter(train_loader))
input_dim = example_batch.shape[2]
output_dim = _.shape[2]

model = LSTMEmotionModel(
    input_dim=input_dim,
    hidden_dim=hidden_dim,
    output_dim=output_dim,
    num_layers=num_layers,
    bidirectional=False,
    dropout=dropout
)
# Custom loss to handle different ranges: sentiment [-3,3] vs emotions [0,3]
class MixedRangeMSELoss(nn.Module):
    def __init__(self):
        super(MixedRangeMSELoss, self).__init__()
    
    def forward(self, input, target):
        sentiment_loss = ((input[:, :, 0] - target[:, :, 0]) ** 2).mean()
        emotion_loss = ((input[:, :, 1:] - target[:, :, 1:]) ** 2).mean()
        
        # Average instead of sum to normalize by number of components
        total_loss = (sentiment_loss + emotion_loss) / 2
        return total_loss

criterion = MixedRangeMSELoss()
optimizer = optim.Adam(
    model.parameters(),
    lr=learning_rate,
    weight_decay=weight_decay
)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)  # Increased patience
model.to(device)

# -----------------------------
# Training Loop with Validation
# -----------------------------
train_losses = []
val_losses = []
best_val_loss = float('inf')
epochs_no_improve = 0
for epoch in range(num_epochs):
    # --- Training ---
    model.train()
    epoch_loss = 0.0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        
        # Add gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()

        epoch_loss += loss.item() * batch_X.size(0)
    avg_train_loss = epoch_loss / len(train_loader.dataset)
    train_losses.append(avg_train_loss)

    # --- Validation ---
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            val_loss += loss.item() * batch_X.size(0)
    avg_val_loss = val_loss / len(val_loader.dataset)
    val_losses.append(avg_val_loss)
    scheduler.step(avg_val_loss)

    print(f"Epoch [{epoch+1}/{num_epochs}], "
          f"Train Loss: {avg_train_loss:.4f}, "
          f"Val Loss: {avg_val_loss:.4f}")

    if avg_val_loss < best_val_loss - tolerance:
        best_val_loss = avg_val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), "src/trained_models/best_model_audio.pt")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

plt.figure(figsize=(8,5))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.show()

print("Batch Size:", batch_size)
print("Hidden Dimension:", hidden_dim)
print("Number of Epochs:", num_epochs)
print("Number of Layers:", num_layers)
print("Learning Rate:", learning_rate)
print("Validation Split:", val_split)
print("Dropout Rate:", dropout)
print("Weight Decay:", weight_decay)
print("Patience:", patience)