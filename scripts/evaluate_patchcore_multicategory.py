
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, f1_score

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

test_root = Path("datasets/MVTecAD") / category / "test"
output_dir = Path("outputs/patchcore") / category
checkpoint = output_dir / "model.ckpt"

assert test_root.is_dir()
assert checkpoint.is_file()

config = json.loads(
    (output_dir / "threshold.json").read_text(
        encoding="utf-8"
    )
)
threshold = float(config["image_threshold"])

expected_counts = {
    "cable": (150, 92, 58),
    "tile": (117, 84, 33),
}
expected_total, expected_defects, expected_good = (
    expected_counts[category]
)

print(f"\nEvaluating {category.upper()}")
print(f"Frozen image threshold: {threshold:.4f}")

model = Patchcore(
    backbone="resnet18",
    layers=["layer2", "layer3"],
    coreset_sampling_ratio=0.1,
)

engine = Engine(
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=1,
    logger=False,
    default_root_dir=str(output_dir / "evaluation"),
)

rows = []

for folder in sorted(test_root.iterdir()):
    if not folder.is_dir():
        continue

    print(f"Testing: {folder.name}", flush=True)

    predictions = engine.predict(
        model=model,
        ckpt_path=str(checkpoint),
        data_path=str(folder),
        return_predictions=True,
    )

    assert predictions is not None

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
            assert np.isfinite(score)

            actual = 0 if folder.name == "good" else 1
            predicted = int(score > threshold)

            rows.append({
                "image": f"{folder.name}/{Path(path).name}",
                "defect_type": folder.name,
                "actual": actual,
                "predicted": predicted,
                "anomaly_score": float(score),
                "correct": int(actual == predicted),
            })

rows.sort(key=lambda row: row["image"])

assert len(rows) == expected_total
assert sum(r["actual"] for r in rows) == expected_defects
assert sum(1 - r["actual"] for r in rows) == expected_good
assert len({r["image"] for r in rows}) == expected_total

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

accuracy = (tp + tn) / len(rows)
precision = tp / (tp + fp) if tp + fp else 0.0
recall = tp / (tp + fn) if tp + fn else 0.0
fpr = fp / (fp + tn) if fp + tn else 0.0

auroc = roc_auc_score(
    [r["actual"] for r in rows],
    [r["anomaly_score"] for r in rows],
)

f1 = f1_score(
    [r["actual"] for r in rows],
    [r["predicted"] for r in rows],
)

metrics = {
    "category": category,
    "image_threshold": threshold,
    "test_images": len(rows),
    "true_positives": tp,
    "true_negatives": tn,
    "false_positives": fp,
    "false_negatives": fn,
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "false_positive_rate": float(fpr),
    "image_auroc": float(auroc),
}

output_dir.mkdir(parents=True, exist_ok=True)

with (output_dir / "test_predictions.csv").open(
    "w", newline="", encoding="utf-8"
) as file:
    writer = csv.DictWriter(
        file,
        fieldnames=rows[0].keys(),
    )
    writer.writeheader()
    writer.writerows(rows)

(output_dir / "test_metrics.json").write_text(
    json.dumps(metrics, indent=2),
    encoding="utf-8",
)

print("\n========== TEST RESULTS ==========")
print(f"Category:        {category}")
print(f"Total images:    {len(rows)}")
print(f"True positives:  {tp}")
print(f"True negatives:  {tn}")
print(f"False positives: {fp}")
print(f"False negatives: {fn}")
print(f"Accuracy:        {accuracy:.2%}")
print(f"Precision:       {precision:.2%}")
print(f"Recall:          {recall:.2%}")
print(f"F1 score:        {f1:.4f}")
print(f"False-pos rate:  {fpr:.2%}")
print(f"Image AUROC:     {auroc:.4f}")
print("==================================")

print("\nRecall by defect type:")
for defect_type in sorted({
    r["defect_type"] for r in rows
    if r["actual"] == 1
}):
    subset = [
        r for r in rows
        if r["defect_type"] == defect_type
    ]
    detected = sum(r["predicted"] for r in subset)

    print(
        f"  {defect_type}: "
        f"{detected}/{len(subset)} detected "
        f"({detected / len(subset):.2%})"
    )

print(f"\nResults saved to: {output_dir}")
