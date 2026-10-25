
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from anomalib.engine import Engine
from anomalib.models import Patchcore

parser = argparse.ArgumentParser()
parser.add_argument(
    "--category",
    required=True,
    choices=["cable", "tile"],
)
args = parser.parse_args()

category = args.category

output_dir = Path("outputs/patchcore") / category
checkpoint = output_dir / "model.ckpt"
val_dir = Path("datasets/processed") / category / "val/good"

assert checkpoint.is_file(), f"Missing checkpoint: {checkpoint}"

expected_count = {"cable": 45, "tile": 46}[category]
assert len(list(val_dir.glob("*.png"))) == expected_count

model = Patchcore(
    backbone="resnet18",
    layers=["layer2", "layer3"],
    coreset_sampling_ratio=0.1,
)

engine = Engine(
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=1,
    logger=False,
    default_root_dir=str(output_dir / "calibration"),
)

print(f"Calibrating {category.upper()}...")
print(f"Healthy validation images: {expected_count}")

predictions = engine.predict(
    model=model,
    ckpt_path=str(checkpoint),
    data_path=str(val_dir),
    return_predictions=True,
)

assert predictions is not None

rows = []

for batch in predictions:
    scores = (
        torch.as_tensor(batch.pred_score)
        .detach()
        .cpu()
        .reshape(-1)
        .tolist()
    )

    paths = batch.image_path
    if isinstance(paths, (str, Path)):
        paths = [paths]

    assert len(paths) == len(scores)

    for path, score in zip(paths, scores):
        rows.append((Path(path).name, float(score)))

rows.sort()

assert len(rows) == expected_count
assert len({name for name, _ in rows}) == expected_count

scores = np.array([score for _, score in rows])
assert np.isfinite(scores).all()

threshold = float(
    np.quantile(scores, 0.95, method="higher")
)

config = {
    "category": category,
    "checkpoint": str(checkpoint),
    "method": "healthy_validation_95th_percentile",
    "validation_count": len(scores),
    "image_threshold": threshold,
    "observed_validation_fpr": float(
        np.mean(scores > threshold)
    ),
}

output_dir.mkdir(parents=True, exist_ok=True)

with (output_dir / "validation_scores.csv").open(
    "w", newline="", encoding="utf-8"
) as file:
    writer = csv.writer(file)
    writer.writerow(["image", "anomaly_score"])
    writer.writerows(rows)

(output_dir / "threshold.json").write_text(
    json.dumps(config, indent=2),
    encoding="utf-8",
)

print("\n====== CALIBRATION RESULTS ======")
print(f"Category:       {category}")
print(f"Images:         {len(scores)}")
print(f"Minimum:        {scores.min():.4f}")
print(f"Median:         {np.median(scores):.4f}")
print(f"Maximum:        {scores.max():.4f}")
print(f"Threshold:      {threshold:.4f}")
print(f"Validation FPR: {config['observed_validation_fpr']:.2%}")
print("=================================")

print(f"\nSaved: {output_dir / 'threshold.json'}")
