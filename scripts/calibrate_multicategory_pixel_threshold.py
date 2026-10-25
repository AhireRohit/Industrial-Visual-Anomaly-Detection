
import argparse
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

expected_count = {"cable": 45, "tile": 46}[category]

assert checkpoint.is_file(), f"Missing checkpoint: {checkpoint}"
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
    default_root_dir=str(output_dir / "pixel_calibration"),
)

print(f"Calibrating {category.upper()} pixel threshold...")

predictions = engine.predict(
    model=model,
    ckpt_path=str(checkpoint),
    data_path=str(val_dir),
    return_predictions=True,
)

assert predictions is not None

all_pixel_scores = []
image_count = 0

for batch in predictions:
    maps = torch.as_tensor(batch.anomaly_map).detach().cpu()

    if maps.ndim == 2:
        maps = maps.unsqueeze(0)
    elif maps.ndim == 4 and maps.shape[1] == 1:
        maps = maps[:, 0]

    if maps.ndim != 3:
        raise ValueError(
            f"Unexpected anomaly-map shape: {tuple(maps.shape)}"
        )

    assert torch.isfinite(maps).all()

    for anomaly_map in maps:
        all_pixel_scores.append(
            anomaly_map.numpy().reshape(-1)
        )
        image_count += 1

assert image_count == expected_count

scores = np.concatenate(all_pixel_scores)

threshold = float(np.quantile(scores, 0.99))
validation_fpr = float(np.mean(scores > threshold))

result = {
    "category": category,
    "method": "healthy_validation_pixel_percentile",
    "percentile": 99,
    "validation_images": image_count,
    "pixel_count": int(scores.size),
    "pixel_threshold": threshold,
    "observed_validation_pixel_fpr": validation_fpr,
}

output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "pixel_threshold.json"

output_path.write_text(
    json.dumps(result, indent=2),
    encoding="utf-8",
)

print("\n====== PIXEL CALIBRATION RESULTS ======")
print(f"Category:       {category}")
print(f"Images:         {image_count}")
print(f"Total pixels:   {scores.size:,}")
print(f"Minimum:        {scores.min():.4f}")
print(f"Median:         {np.median(scores):.4f}")
print(f"Maximum:        {scores.max():.4f}")
print(f"Pixel threshold: {threshold:.4f}")
print(f"Validation pixel exceedance: {validation_fpr:.2%}")
print("=======================================")
print(f"\nSaved: {output_path}")
