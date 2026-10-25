
from pathlib import Path
import csv
import json
import math

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import v2

from autoencoder import ConvAutoencoder

ROOT = Path("outputs/autoencoder/metal_nut")
OUTPUT = ROOT / "topk_1pct"
OUTPUT.mkdir(parents=True, exist_ok=True)

VAL_DIR = Path("datasets/processed/metal_nut/val/good")
CHECKPOINT = ROOT / "model.pt"

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

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

images = sorted(VAL_DIR.glob("*.png"))
assert len(images) == 44

rows = []

with torch.inference_mode():
    for path in images:
        with Image.open(path) as image:
            x = transform(
                image.convert("RGB")
            ).unsqueeze(0).to(device)

        reconstructed = model(x)

        # Squared error averaged across RGB channels
        error_map = (
            (x - reconstructed) ** 2
        ).mean(dim=1)

        # Select highest-error 1% of pixels
        flat = error_map.flatten()
        k = math.ceil(flat.numel() * 0.01)

        top_errors = torch.topk(
            flat, k=k
        ).values

        score = top_errors.mean().item()

        rows.append({
            "image": path.name,
            "anomaly_score": score,
        })

scores = np.array([
    row["anomaly_score"] for row in rows
])

assert np.isfinite(scores).all()

threshold = float(
    np.quantile(scores, 0.95, method="higher")
)

fpr = float(np.mean(scores > threshold))

with (OUTPUT / "validation_scores.csv").open(
    "w", newline="", encoding="utf-8"
) as f:
    writer = csv.DictWriter(
        f, fieldnames=["image", "anomaly_score"]
    )
    writer.writeheader()
    writer.writerows(rows)

config = {
    "category": "metal_nut",
    "model": "convolutional_autoencoder",
    "checkpoint": str(CHECKPOINT),
    "scoring_method": "top_1_percent_pixel_mse",
    "top_fraction": 0.01,
    "validation_count": len(images),
    "image_threshold": threshold,
    "observed_validation_fpr": fpr,
    "status": "exploratory",
}

(OUTPUT / "threshold.json").write_text(
    json.dumps(config, indent=2),
    encoding="utf-8",
)

print("\n====== TOP-1% CALIBRATION ======")
print(f"Images:         {len(images)}")
print(f"Minimum score:  {scores.min():.6f}")
print(f"Median score:   {np.median(scores):.6f}")
print(f"Maximum score:  {scores.max():.6f}")
print(f"Threshold:      {threshold:.6f}")
print(f"Validation FPR: {fpr:.2%}")
print("================================")
print("Saved to:", OUTPUT)
