import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from src.models.lstm_model import LSTMEmotionModel
from src.data.mosei_exploration import get_mosei_dataloader
import numpy as np

# -----------------------------
# Config
# -----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
MODEL_PATH = "src/trained_models/best_model_audio.pt"

# Use same loss function as training
class MixedRangeMSELoss(nn.Module):
    def __init__(self):
        super(MixedRangeMSELoss, self).__init__()
    
    def forward(self, input, target):
        sentiment_loss = ((input[:, :, 0] - target[:, :, 0]) ** 2).mean()
        emotion_loss = ((input[:, :, 1:] - target[:, :, 1:]) ** 2).mean()
        
        total_loss = (sentiment_loss + emotion_loss) / 2
        return total_loss

# -----------------------------
# Load data and create test split
# -----------------------------
full_dataloader = get_mosei_dataloader(batch_size=1, shuffle=False)
dataset = full_dataloader.dataset

# Same split as training: 85% train, 15% val, but now we want a separate test set
test_size = int(len(dataset) * 0.2)
train_val_size = len(dataset) - test_size
train_val_dataset, test_dataset = random_split(dataset, [train_val_size, test_size])

test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Dataset sizes:")
print(f"  Total: {len(dataset)}")
print(f"  Train+Val: {train_val_size}")
print(f"  Test: {test_size}")

# -----------------------------
# Load model with correct architecture
# -----------------------------
example_batch, example_target = next(iter(test_loader))
input_dim = example_batch.shape[2]
output_dim = example_target.shape[2]

model = LSTMEmotionModel(
    input_dim=input_dim,
    hidden_dim=256,  # Updated to match your trained model
    output_dim=output_dim,
    num_layers=4,    # Updated to match your trained model
    bidirectional=False,
    dropout=0.3      # Updated to match your trained model
)

# Load trained weights
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
model.to(DEVICE)
model.eval()

print(f"Model loaded: {input_dim} -> {256}h x {4}L -> {output_dim}")

# -----------------------------
# Evaluate on test set
# -----------------------------
criterion = MixedRangeMSELoss()
total_loss = 0.0
total_samples = 0

# For regression metrics
all_predictions = []
all_targets = []

print("\nEvaluating on test set...")

with torch.no_grad():
    for i, (batch_X, batch_y) in enumerate(test_loader):
        batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
        
        # Forward pass
        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)
        
        # Accumulate loss
        total_loss += loss.item() * batch_X.size(0)
        total_samples += batch_X.size(0)
        
        # Store predictions and targets for analysis
        all_predictions.append(predictions.cpu())
        all_targets.append(batch_y.cpu())
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(test_loader)} batches")

# Calculate average test loss
avg_test_loss = total_loss / total_samples

# Concatenate all predictions and targets
all_predictions = torch.cat(all_predictions, dim=0)  # [samples, timesteps, features]
all_targets = torch.cat(all_targets, dim=0)

print(f"\n=== TEST RESULTS ===")
print(f"Test Loss: {avg_test_loss:.4f}")
print(f"Test Samples: {total_samples}")

# Calculate Mean Absolute Error (MAE) for interpretability
mae = torch.mean(torch.abs(all_predictions - all_targets))
print(f"Mean Absolute Error: {mae:.4f}")

# Calculate percentage accuracy with tolerance thresholds
print(f"\n=== PERCENTAGE ACCURACY ===")
tolerances = [0.01, 0.05, 0.1, 0.2]
dimension_names = ['Sentiment', 'Happy', 'Sad', 'Anger', 'Surprise', 'Disgust', 'Fear']

for tolerance in tolerances:
    # Calculate accuracy for each dimension
    dim_accuracies = []
    for dim in range(output_dim):
        pred_dim = all_predictions[:, :, dim]
        target_dim = all_targets[:, :, dim]
        
        # Count predictions within tolerance
        within_tolerance = torch.abs(pred_dim - target_dim) <= tolerance
        accuracy = torch.mean(within_tolerance.float()) * 100
        dim_accuracies.append(accuracy.item())
    
    # Overall accuracy (all dimensions must be within tolerance)
    all_within_tolerance = torch.abs(all_predictions - all_targets) <= tolerance
    all_dims_correct = torch.all(all_within_tolerance, dim=2)  # All 7 dimensions correct
    overall_accuracy = torch.mean(all_dims_correct.float()) * 100
    
    print(f"\nTolerance ±{tolerance}:")
    print(f"  Overall (all dimensions): {overall_accuracy:.1f}%")
    for dim, acc in enumerate(dim_accuracies):
        print(f"  {dimension_names[dim]}: {acc:.1f}%")

# Per-dimension analysis (sentiment + emotions)
print(f"\nPer-dimension MAE:")
for dim in range(output_dim):
    dim_mae = torch.mean(torch.abs(all_predictions[:, :, dim] - all_targets[:, :, dim]))
    print(f"  {dimension_names[dim]}: {dim_mae:.4f}")

# Show prediction ranges
print(f"\nPrediction ranges:")
print(f"  Predictions: [{all_predictions.min():.3f}, {all_predictions.max():.3f}]")
print(f"  Targets: [{all_targets.min():.3f}, {all_targets.max():.3f}]")

print(f"\n=== SUMMARY ===")
if avg_test_loss < 0.05:
    print("✓ Model performance looks good (test loss < 0.05)")
elif avg_test_loss < 0.10:
    print("○ Model performance is reasonable (test loss < 0.10)")
else:
    print("⚠ Model performance might need improvement (test loss >= 0.10)")

print(f"\nFor reference:")
print(f"- Lower MAE is better (0 = perfect predictions)")
print(f"- Test loss should be close to your validation loss (~0.044)")
print(f"- If test loss >> validation loss, model may be overfitting")