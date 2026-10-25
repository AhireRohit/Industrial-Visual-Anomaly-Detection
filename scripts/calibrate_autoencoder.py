
from pathlib import Path
import csv
import json

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import v2

from autoencoder import ConvAutoencoder

# Configuration
OUTPUT = Path("outputs/autoencoder/metal_nut")
CHECKPOINT = OUTPUT / "model.pt"
VAL_DIR = Path("datasets/processed/metal_nut/val/good")

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

assert CHECKPOINT.is_file(), "Autoencoder checkpoint missing"

images = sorted(VAL_DIR.glob("*.png"))
assert len(images) == 44, f"Expected 44, found {len(images)}"

# Load trained autoencoder
model = ConvAutoencoder().to(device)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=device,
    weights_only=True,
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

transform = v2.Compose([
    v2.Resize((256, 256)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
])

rows = []

print(f"Using device: {device}")
print("Calibrating on 44 healthy validation images...\n")

with torch.inference_mode():
    for i, path in enumerate(images, start=1):
        with Image.open(path) as image:
            tensor = transform(
                image.convert("RGB")
            ).unsqueeze(0).to(device)

        reconstruction = model(tensor)

        # Image-level anomaly score:
        # Mean squared reconstruction error
        mse = torch.mean(
            (tensor - reconstruction) ** 2
        ).item()

        rows.append({
            "image": path.name,
            "anomaly_score": float(mse),
        })

        print(
            f"[{i:02d}/44] {path.name} "
            f"| MSE: {mse:.6f}"
        )

scores = np.array(
    [row["anomaly_score"] for row in rows],
    dtype=np.float64,
)

assert np.isfinite(scores).all()

# 95th percentile of healthy validation scores
threshold = float(
    np.quantile(scores, 0.95, method="higher")
)

validation_fpr = float(
    np.mean(scores > threshold)
)

# Save validation scores
csv_path = OUTPUT / "validation_scores.csv"

with csv_path.open(
    "w", newline="", encoding="utf-8"
) as f:
    writer = csv.DictWriter(
        f, fieldnames=["image", "anomaly_score"]
    )
    writer.writeheader()
    writer.writerows(rows)

# Save calibration
config = {
    "category": "metal_nut",
    "model": "convolutional_autoencoder",
    "checkpoint": str(CHECKPOINT),
    "scoring_method": "image_mean_squared_error",
    "threshold_method": "healthy_validation_95th_percentile",
    "validation_count": len(scores),
    "image_threshold": threshold,
    "observed_validation_fpr": validation_fpr,
}

json_path = OUTPUT / "threshold.json"

json_path.write_text(
    json.dumps(config, indent=2),
    encoding="utf-8",
)

print("\n====== AUTOENCODER CALIBRATION ======")
print(f"Validation images: {len(scores)}")
print(f"Minimum MSE:       {scores.min():.6f}")
print(f"Median MSE:        {np.median(scores):.6f}")
print(f"Maximum MSE:       {scores.max():.6f}")
print(f"Image threshold:   {threshold:.6f}")
print(f"Validation FPR:    {validation_fpr:.2%}")
print("=====================================")

print("\nSaved:")
print(csv_path)
print(json_path)
