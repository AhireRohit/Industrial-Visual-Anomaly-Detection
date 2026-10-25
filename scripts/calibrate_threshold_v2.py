
from pathlib import Path
import csv
import json

import numpy as np
import torch

from anomalib.engine import Engine
from anomalib.models import Patchcore

checkpoint = Path("outputs/patchcore/metal_nut_v2/model.ckpt")
val_dir = Path("datasets/processed/metal_nut/val/good")
output_dir = Path("outputs/patchcore/metal_nut_v2")

assert checkpoint.is_file(), "Checkpoint not found"
assert len(list(val_dir.glob("*.png"))) == 44

model = Patchcore(
    backbone="resnet18",
    layers=["layer2", "layer3"],
    coreset_sampling_ratio=0.1,
)

engine = Engine(
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=1,
    logger=False,
    default_root_dir=str(output_dir),
)

print("Scoring 44 healthy validation images...")

predictions = engine.predict(
    model=model,
    ckpt_path=str(checkpoint),
    data_path=str(val_dir),
    return_predictions=True,
)

assert predictions is not None

rows = []

for batch in predictions:
    scores = torch.as_tensor(
        batch.pred_score
    ).detach().cpu().reshape(-1).tolist()

    paths = batch.image_path
    if isinstance(paths, (str, Path)):
        paths = [paths]

    assert len(paths) == len(scores)

    for path, score in zip(paths, scores):
        rows.append((Path(path).name, float(score)))

rows.sort()
assert len(rows) == 44
assert len({name for name, _ in rows}) == 44

scores = np.array([score for _, score in rows])
assert np.isfinite(scores).all()

# Initial image-level threshold, 95th percentile
threshold = float(
    np.quantile(scores, 0.95, method="higher")
)

# Save per-image scores
csv_path = output_dir / "validation_scores.csv"

with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["image", "anomaly_score"])
    writer.writerows(rows)

# Save calibration configuration
config = {
    "category": "metal_nut",
    "checkpoint": str(checkpoint),
    "method": "healthy_validation_95th_percentile",
    "validation_count": len(scores),
    "image_threshold": threshold,
    "observed_validation_fpr": float(
        np.mean(scores > threshold)
    ),
}

json_path = output_dir / "threshold.json"
json_path.write_text(
    json.dumps(config, indent=2),
    encoding="utf-8",
)

print("\n--- Calibration Results ---")
print(f"Images:        {len(scores)}")
print(f"Minimum score: {scores.min():.4f}")
print(f"Median score:  {np.median(scores):.4f}")
print(f"Maximum score: {scores.max():.4f}")
print(f"Threshold:     {threshold:.4f}")
print(
    f"Validation FPR: "
    f"{config['observed_validation_fpr']:.2%}"
)
print(f"\nSaved: {csv_path}")
print(f"Saved: {json_path}")
