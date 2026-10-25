
from pathlib import Path
import csv
import random

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
from PIL import Image

from autoencoder import ConvAutoencoder

# Configuration
SEED = 42
EPOCHS = 30
BATCH_SIZE = 4
LEARNING_RATE = 0.001

TRAIN_DIR = Path(
    "datasets/processed/metal_nut/train/good"
)
OUTPUT = Path("outputs/autoencoder/metal_nut")
OUTPUT.mkdir(parents=True, exist_ok=True)

# Reproducibility
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)

# Image preprocessing
transform = v2.Compose([
    v2.Resize((256, 256)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
])


class HealthyDataset(Dataset):
    def __init__(self, folder):
        self.images = sorted(folder.glob("*.png"))
        assert len(self.images) == 176, (
            f"Expected 176 images, found {len(self.images)}"
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        with Image.open(self.images[index]) as image:
            return transform(image.convert("RGB"))


dataset = HealthyDataset(TRAIN_DIR)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=(device.type == "cuda"),
    generator=torch.Generator().manual_seed(SEED),
)

# Initialize model
model = ConvAutoencoder().to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)

history = []

print(f"Training images: {len(dataset)}")
print(f"Epochs: {EPOCHS}")
print("Starting autoencoder training...\n")

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    total_images = 0

    for images in loader:
        images = images.to(
            device, non_blocking=True
        )

        optimizer.zero_grad()

        reconstructed = model(images)

        loss = criterion(
            reconstructed, images
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        total_images += images.size(0)

    epoch_loss = total_loss / total_images

    history.append({
        "epoch": epoch,
        "train_mse": epoch_loss,
    })

    print(
        f"Epoch {epoch:02d}/{EPOCHS} "
        f"| Training MSE: {epoch_loss:.6f}",
        flush=True,
    )

# Save checkpoint
checkpoint_path = OUTPUT / "model.pt"

torch.save({
    "model_state_dict": model.state_dict(),
    "epochs": EPOCHS,
    "seed": SEED,
    "image_size": 256,
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "train_images": len(dataset),
    "final_train_mse": history[-1]["train_mse"],
}, checkpoint_path)

# Save training history
history_path = OUTPUT / "training_history.csv"

with history_path.open(
    "w", newline="", encoding="utf-8"
) as f:
    writer = csv.DictWriter(
        f, fieldnames=["epoch", "train_mse"]
    )
    writer.writeheader()
    writer.writerows(history)

print("\n====== TRAINING COMPLETE ======")
print(f"Initial MSE: {history[0]['train_mse']:.6f}")
print(f"Final MSE:   {history[-1]['train_mse']:.6f}")
print(f"Model saved: {checkpoint_path}")
print(f"History:     {history_path}")
print("===============================")
