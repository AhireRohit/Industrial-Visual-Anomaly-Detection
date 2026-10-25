
from pathlib import Path
import csv
import json

import numpy as np
import torch
from PIL import Image

from anomalib.engine import Engine
from anomalib.models import Patchcore

ROOT = Path("datasets/MVTecAD/metal_nut")
OUTPUT = Path("outputs/patchcore/metal_nut")
CHECKPOINT = OUTPUT / "model.ckpt"
MASK_DIR = OUTPUT / "predicted_masks"

# Load frozen pixel threshold
config = json.loads(
    (OUTPUT / "pixel_threshold.json").read_text(
        encoding="utf-8"
    )
)
threshold = float(config["pixel_threshold"])

assert CHECKPOINT.is_file()
MASK_DIR.mkdir(parents=True, exist_ok=True)

print(f"Pixel threshold: {threshold:.4f}")

# Load trained model
model = Patchcore(
    backbone="resnet18",
    layers=["layer2", "layer3"],
    coreset_sampling_ratio=0.1,
)

engine = Engine(
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=1,
    logger=False,
    default_root_dir=str(OUTPUT / "mask_eval"),
)

totals = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
rows = []

for folder in sorted((ROOT / "test").iterdir()):
    if not folder.is_dir():
        continue

    print(f"Processing: {folder.name}", flush=True)

    predictions = engine.predict(
        model=model,
        ckpt_path=str(CHECKPOINT),
        data_path=str(folder),
        return_predictions=True,
    )

    assert predictions is not None

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
                f"Unexpected anomaly map shape: {maps.shape}"
            )

        paths = batch.image_path
        if isinstance(paths, (str, Path)):
            paths = [paths]

        assert len(paths) == len(maps)

        for path, anomaly_map in zip(paths, maps):
            path = Path(path)
            scores = anomaly_map.numpy()

            assert np.isfinite(scores).all()

            h, w = scores.shape

            # Binary prediction from frozen threshold
            predicted = scores > threshold

            # Load actual defect mask
            if folder.name == "good":
                actual = np.zeros(
                    (h, w), dtype=bool
                )
            else:
                gt_path = (
                    ROOT / "ground_truth"
                    / folder.name
                    / f"{path.stem}_mask.png"
                )

                assert gt_path.is_file(), gt_path

                gt_image = Image.open(gt_path).convert("L")
                gt_image = gt_image.resize(
                    (w, h),
                    Image.Resampling.NEAREST,
                )
                actual = np.asarray(gt_image) > 0

            # Pixel confusion matrix
            tp = int(np.sum(predicted & actual))
            tn = int(np.sum(~predicted & ~actual))
            fp = int(np.sum(predicted & ~actual))
            fn = int(np.sum(~predicted & actual))

            for key, value in zip(
                ("tp", "tn", "fp", "fn"),
                (tp, tn, fp, fn),
            ):
                totals[key] += value

            # Individual-image overlap
            union = tp + fp + fn

            iou = tp / union if union else None
            dice = (
                2 * tp / (2 * tp + fp + fn)
                if (2 * tp + fp + fn)
                else None
            )

            rows.append({
                "image": f"{folder.name}/{path.name}",
                "defect_type": folder.name,
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "iou": iou,
                "dice": dice,
            })

            # Save predicted binary mask
            destination = MASK_DIR / folder.name
            destination.mkdir(
                parents=True, exist_ok=True
            )

            mask_image = Image.fromarray(
                (predicted.astype(np.uint8) * 255),
                mode="L",
            )
            mask_image.save(
                destination / path.name
            )

assert len(rows) == 115
assert sum(r["defect_type"] != "good" for r in rows) == 93

# Global pixel metrics (micro-averaged)
tp, tn, fp, fn = (
    totals[k] for k in ("tp", "tn", "fp", "fn")
)

precision = tp / (tp + fp) if tp + fp else 0.0
recall = tp / (tp + fn) if tp + fn else 0.0
iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
dice = (
    2 * tp / (2 * tp + fp + fn)
    if 2 * tp + fp + fn else 0.0
)
fpr = fp / (fp + tn) if fp + tn else 0.0

metrics = {
    "category": "metal_nut",
    "test_images": len(rows),
    "pixel_threshold": threshold,
    "tp": tp,
    "tn": tn,
    "fp": fp,
    "fn": fn,
    "pixel_precision": precision,
    "pixel_recall": recall,
    "pixel_fpr": fpr,
    "mask_iou": iou,
    "dice": dice,
    "aggregation": "micro_all_test_pixels",
}

# Save metrics
(OUTPUT / "mask_metrics.json").write_text(
    json.dumps(metrics, indent=2),
    encoding="utf-8",
)

with (OUTPUT / "mask_per_image.csv").open(
    "w", newline="", encoding="utf-8"
) as f:
    writer = csv.DictWriter(
        f, fieldnames=rows[0].keys()
    )
    writer.writeheader()
    writer.writerows(rows)

print("\n========== MASK EVALUATION ==========")
print(f"Images evaluated: {len(rows)}")
print(f"Pixel precision:  {precision:.2%}")
print(f"Pixel recall:     {recall:.2%}")
print(f"Pixel FPR:        {fpr:.2%}")
print(f"Mask IoU:         {iou:.4f}")
print(f"Dice score:       {dice:.4f}")
print("=====================================")

print("\nWorst 5 defective images by IoU:")
defects = [
    r for r in rows if r["defect_type"] != "good"
]
for r in sorted(defects, key=lambda x: x["iou"])[:5]:
    print(f"{r['image']}: IoU={r['iou']:.4f}")

print("\nSaved:")
print(OUTPUT / "mask_metrics.json")
print(OUTPUT / "mask_per_image.csv")
print(MASK_DIR)
