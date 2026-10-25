
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score

from anomalib.engine import Engine
from anomalib.models import Patchcore


parser = argparse.ArgumentParser()
parser.add_argument(
    "--category",
    choices=["cable", "tile"],
    required=True,
)
args = parser.parse_args()

category = args.category
root = Path("datasets/MVTecAD") / category
output = Path("outputs/patchcore") / category
checkpoint = output / "model.ckpt"
mask_dir = output / "predicted_masks"

expected_counts = {"cable": (150, 92), "tile": (117, 84)}
expected_total, expected_defects = expected_counts[category]

config = json.loads(
    (output / "pixel_threshold.json").read_text(
        encoding="utf-8"
    )
)
threshold = float(config["pixel_threshold"])

assert checkpoint.is_file(), checkpoint
mask_dir.mkdir(parents=True, exist_ok=True)

print(f"\nEvaluating {category.upper()} localisation")
print(f"Frozen pixel threshold: {threshold:.4f}")

model = Patchcore(
    backbone="resnet18",
    layers=["layer2", "layer3"],
    coreset_sampling_ratio=0.1,
)

engine = Engine(
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=1,
    logger=False,
    default_root_dir=str(output / "localization_eval"),
)

totals = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
all_scores = []
all_labels = []
rows = []

for folder in sorted((root / "test").iterdir()):
    if not folder.is_dir():
        continue

    print(f"Processing: {folder.name}", flush=True)

    predictions = engine.predict(
        model=model,
        ckpt_path=str(checkpoint),
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

        if maps.ndim != 3:
            raise ValueError(
                f"Unexpected map shape: {tuple(maps.shape)}"
            )

        paths = batch.image_path
        if isinstance(paths, (str, Path)):
            paths = [paths]

        assert len(paths) == len(maps)

        for path, anomaly_map in zip(paths, maps):
            path = Path(path)
            scores = anomaly_map.numpy()

            assert np.isfinite(scores).all()

            height, width = scores.shape
            predicted = scores > threshold

            if folder.name == "good":
                actual = np.zeros(
                    (height, width), dtype=bool
                )
            else:
                gt_path = (
                    root / "ground_truth"
                    / folder.name
                    / f"{path.stem}_mask.png"
                )
                assert gt_path.is_file(), gt_path

                with Image.open(gt_path) as image:
                    gt = image.convert("L").resize(
                        (width, height),
                        Image.Resampling.NEAREST,
                    )
                    actual = np.asarray(gt) > 0

            tp = int(np.sum(predicted & actual))
            tn = int(np.sum(~predicted & ~actual))
            fp = int(np.sum(predicted & ~actual))
            fn = int(np.sum(~predicted & actual))

            for key, value in zip(
                ("tp", "tn", "fp", "fn"),
                (tp, tn, fp, fn),
            ):
                totals[key] += value

            union = tp + fp + fn
            denominator = 2 * tp + fp + fn

            rows.append({
                "image": f"{folder.name}/{path.name}",
                "defect_type": folder.name,
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "iou": tp / union if union else None,
                "dice": (
                    2 * tp / denominator
                    if denominator else None
                ),
            })

            all_scores.append(
                scores.astype(np.float32).reshape(-1)
            )
            all_labels.append(
                actual.astype(np.uint8).reshape(-1)
            )

            destination = mask_dir / folder.name
            destination.mkdir(
                parents=True, exist_ok=True
            )

            Image.fromarray(
                predicted.astype(np.uint8) * 255
            ).save(destination / path.name)

assert len(rows) == expected_total
assert sum(
    r["defect_type"] != "good" for r in rows
) == expected_defects
assert len({r["image"] for r in rows}) == expected_total

tp, tn, fp, fn = (
    totals[key] for key in ("tp", "tn", "fp", "fn")
)

precision = tp / (tp + fp) if tp + fp else 0.0
recall = tp / (tp + fn) if tp + fn else 0.0
fpr = fp / (fp + tn) if fp + tn else 0.0
iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
dice = (
    2 * tp / (2 * tp + fp + fn)
    if 2 * tp + fp + fn else 0.0
)

pixel_auroc = roc_auc_score(
    np.concatenate(all_labels),
    np.concatenate(all_scores),
)

metrics = {
    "category": category,
    "test_images": len(rows),
    "pixel_threshold": threshold,
    "pixel_auroc": float(pixel_auroc),
    "pixel_precision": float(precision),
    "pixel_recall": float(recall),
    "pixel_fpr": float(fpr),
    "mask_iou": float(iou),
    "dice": float(dice),
    "aggregation": "micro_all_test_pixels",
}

(output / "localization_metrics.json").write_text(
    json.dumps(metrics, indent=2),
    encoding="utf-8",
)

with (output / "mask_per_image.csv").open(
    "w", newline="", encoding="utf-8"
) as file:
    writer = csv.DictWriter(
        file, fieldnames=rows[0].keys()
    )
    writer.writeheader()
    writer.writerows(rows)

print("\n====== LOCALISATION RESULTS ======")
print(f"Category:        {category}")
print(f"Images evaluated: {len(rows)}")
print(f"Pixel AUROC:      {pixel_auroc:.4f}")
print(f"Pixel precision:  {precision:.2%}")
print(f"Pixel recall:     {recall:.2%}")
print(f"Pixel FPR:        {fpr:.2%}")
print(f"Mask IoU:         {iou:.4f}")
print(f"Dice score:       {dice:.4f}")
print("==================================")

print("\nWorst 5 defective images by IoU:")
defects = [
    r for r in rows if r["defect_type"] != "good"
]
for row in sorted(defects, key=lambda x: x["iou"])[:5]:
    print(f'{row["image"]}: IoU={row["iou"]:.4f}')

print(f"\nSaved results to: {output}")
