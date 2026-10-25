
from pathlib import Path
import json

import numpy as np
import torch

from anomalib.engine import Engine
from anomalib.models import Patchcore

OUTPUT = Path("outputs/patchcore/metal_nut_v2")
CHECKPOINT = OUTPUT / "model.ckpt"
VAL_DIR = Path("datasets/processed/metal_nut/val/good")

assert CHECKPOINT.is_file()
assert len(list(VAL_DIR.glob("*.png"))) == 44

model = Patchcore(
    backbone="resnet18",
    layers=["layer2", "layer3"],
    coreset_sampling_ratio=0.1,
)

engine = Engine(
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=1,
    logger=False,
    default_root_dir=str(OUTPUT / "pixel_calibration"),
)

print("Calibrating pixel threshold on healthy images...")

predictions = engine.predict(
    model=model,
    ckpt_path=str(CHECKPOINT),
    data_path=str(VAL_DIR),
    return_predictions=True,
)

assert predictions is not None

all_pixel_scores = []
image_count = 0

for batch in predictions:
    maps = torch.as_tensor(
        batch.anomaly_map
    ).detach().cpu()

    if maps.ndim == 2:
        maps = maps.unsqueeze(0)
    elif maps.ndim == 4 and maps.shape[1] == 1:
        maps = maps[:, 0]
    elif maps.ndim != 3:
        raise ValueError(
            f"Unexpected map shape: {tuple(maps.shape)}"
        )

    assert torch.isfinite(maps).all()

    for anomaly_map in maps:
        all_pixel_scores.append(
            anomaly_map.numpy().reshape(-1)
        )
        image_count += 1

assert image_count == 44

scores = np.concatenate(all_pixel_scores)

# Allow approximately 1% of healthy validation pixels
# to exceed the localisation threshold.
threshold = float(np.quantile(scores, 0.99))

validation_fpr = float(np.mean(scores > threshold))

result = {
    "category": "metal_nut",
    "method": "healthy_validation_pixel_percentile",
    "percentile": 99,
    "validation_images": image_count,
    "pixel_count": int(scores.size),
    "pixel_threshold": threshold,
    "observed_validation_pixel_fpr": validation_fpr,
}

path = OUTPUT / "pixel_threshold.json"

path.write_text(
    json.dumps(result, indent=2),
    encoding="utf-8",
)

print("\n====== PIXEL THRESHOLD CALIBRATION ======")
print(f"Validation images: {image_count}")
print(f"Total pixels:      {scores.size:,}")
print(f"Minimum score:     {scores.min():.4f}")
print(f"Median score:      {np.median(scores):.4f}")
print(f"Maximum score:     {scores.max():.4f}")
print(f"Pixel threshold:   {threshold:.4f}")
print(f"Validation pixel FPR: {validation_fpr:.2%}")
print("=========================================")
print(f"\nSaved: {path}")
