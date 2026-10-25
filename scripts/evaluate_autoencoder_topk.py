
from pathlib import Path
import csv
import json
import math
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import v2
from sklearn.metrics import roc_auc_score

from autoencoder import ConvAutoencoder

# Paths
TEST_ROOT = Path("datasets/MVTecAD/metal_nut/test")
OUTPUT = Path("outputs/autoencoder/metal_nut/topk_1pct")
CHECKPOINT = OUTPUT.parent / "model.pt"

# Load calibrated threshold
config = json.loads(
    (OUTPUT / "threshold.json").read_text(
        encoding="utf-8"
    )
)
threshold = float(config["image_threshold"])

# Device
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {device}")
print(f"Fixed anomaly threshold: {threshold:.6f}")

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

# Evaluate all healthy and defective test images
with torch.inference_mode():
    for folder in sorted(TEST_ROOT.iterdir()):
        if not folder.is_dir():
            continue

        images = sorted(folder.glob("*.png"))
        print(f"\nEvaluating {folder.name}: {len(images)} images")

        for path in images:
            with Image.open(path) as image:
                tensor = transform(
                    image.convert("RGB")
                ).unsqueeze(0).to(device)

            reconstruction = model(tensor)

            # Per-pixel reconstruction error, averaged over RGB
            error_map = (
                (tensor - reconstruction) ** 2
            ).mean(dim=1)

            # Select the 1% highest-error pixels
            flat = error_map.flatten()
            k = math.ceil(flat.numel() * 0.01)

            mse = torch.topk(
                flat, k=k
            ).values.mean().item()

            actual = int(folder.name != "good")
            predicted = int(mse > threshold)

            rows.append({
                "image": f"{folder.name}/{path.name}",
                "defect_type": folder.name,
                "actual": actual,
                "predicted": predicted,
                "anomaly_score": float(mse),
                "correct": actual == predicted,
            })

rows.sort(key=lambda x: x["image"])

assert len(rows) == 115, f"Expected 115, got {len(rows)}"
assert sum(r["actual"] for r in rows) == 93
assert sum(r["actual"] == 0 for r in rows) == 22
assert len({r["image"] for r in rows}) == 115

# Confusion matrix
tp = sum(
    r["actual"] == 1 and r["predicted"] == 1
    for r in rows
)
tn = sum(
    r["actual"] == 0 and r["predicted"] == 0
    for r in rows
)
fp = sum(
    r["actual"] == 0 and r["predicted"] == 1
    for r in rows
)
fn = sum(
    r["actual"] == 1 and r["predicted"] == 0
    for r in rows
)

# Performance metrics
accuracy = (tp + tn) / len(rows)
recall = tp / (tp + fn) if tp + fn else 0.0
precision = tp / (tp + fp) if tp + fp else 0.0
fpr = fp / (fp + tn) if fp + tn else 0.0

auroc = roc_auc_score(
    [r["actual"] for r in rows],
    [r["anomaly_score"] for r in rows],
)

metrics = {
    "model": "convolutional_autoencoder_top1pct",
    "category": "metal_nut",
    "threshold": threshold,
    "test_images": len(rows),
    "true_positives": tp,
    "true_negatives": tn,
    "false_positives": fp,
    "false_negatives": fn,
    "accuracy": accuracy,
    "recall": recall,
    "precision": precision,
    "false_positive_rate": fpr,
    "image_auroc": float(auroc),
}

# Save predictions
csv_path = OUTPUT / "test_predictions.csv"

with csv_path.open(
    "w", newline="", encoding="utf-8"
) as f:
    writer = csv.DictWriter(
        f, fieldnames=rows[0].keys()
    )
    writer.writeheader()
    writer.writerows(rows)

# Save metrics
json_path = OUTPUT / "test_metrics.json"

json_path.write_text(
    json.dumps(metrics, indent=2),
    encoding="utf-8",
)

print("\n========== AUTOENCODER TEST RESULTS ==========")
print(f"Total images:        {len(rows)}")
print(f"True positives:      {tp}")
print(f"True negatives:      {tn}")
print(f"False positives:     {fp}")
print(f"False negatives:     {fn}")
print(f"Accuracy:            {accuracy:.2%}")
print(f"Defect recall:       {recall:.2%}")
print(f"Precision:           {precision:.2%}")
print(f"False-positive rate: {fpr:.2%}")
print(f"Image AUROC:         {auroc:.4f}")
print("==============================================")

print("\nDefect recall by category:")

for defect_type in sorted(
    {r["defect_type"] for r in rows} - {"good"}
):
    subset = [
        r for r in rows
        if r["defect_type"] == defect_type
    ]

    detected = sum(r["predicted"] for r in subset)
    total = len(subset)

    print(
        f"{defect_type}: {detected}/{total} "
        f"({detected / total:.2%})"
    )

print("\nSaved:")
print(csv_path)
print(json_path)
