
from pathlib import Path
import csv
import json

import torch
from sklearn.metrics import roc_auc_score

from anomalib.engine import Engine
from anomalib.models import Patchcore

# Paths
root = Path("datasets/MVTecAD/metal_nut/test")
output = Path("outputs/patchcore/metal_nut_v2")
checkpoint = output / "model.ckpt"

# Load the frozen validation threshold
config = json.loads(
    (output / "threshold.json").read_text(encoding="utf-8")
)
threshold = float(config["image_threshold"])

print(f"Using fixed threshold: {threshold:.4f}")

# Load model
model = Patchcore(
    backbone="resnet18",
    layers=["layer2", "layer3"],
    coreset_sampling_ratio=0.1,
)

engine = Engine(
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=1,
    logger=False,
    default_root_dir=str(output / "evaluation"),
)

# Evaluate every defect category and healthy images
rows = []

for folder in sorted(root.iterdir()):
    if not folder.is_dir():
        continue

    print(f"\nEvaluating: {folder.name}", flush=True)

    predictions = engine.predict(
        model=model,
        ckpt_path=str(checkpoint),
        data_path=str(folder),
        return_predictions=True,
    )

    assert predictions is not None

    for batch in predictions:
        scores = torch.as_tensor(
            batch.pred_score
        ).detach().cpu().reshape(-1).tolist()

        paths = batch.image_path
        if isinstance(paths, (str, Path)):
            paths = [paths]

        assert len(paths) == len(scores)

        for path, score in zip(paths, scores):
            path = Path(path)
            assert path.parent.name == folder.name

            actual = 0 if folder.name == "good" else 1
            predicted = int(score > threshold)

            rows.append({
                "image": f"{folder.name}/{path.name}",
                "defect_type": folder.name,
                "actual": actual,
                "predicted": predicted,
                "anomaly_score": float(score),
                "correct": actual == predicted,
            })

rows.sort(key=lambda x: x["image"])

assert len(rows) == 115, f"Expected 115, got {len(rows)}"
assert sum(r["actual"] for r in rows) == 93
assert sum(not r["actual"] for r in rows) == 22

# Confusion matrix values
tp = sum(r["actual"] == 1 and r["predicted"] == 1 for r in rows)
tn = sum(r["actual"] == 0 and r["predicted"] == 0 for r in rows)
fp = sum(r["actual"] == 0 and r["predicted"] == 1 for r in rows)
fn = sum(r["actual"] == 1 and r["predicted"] == 0 for r in rows)

accuracy = (tp + tn) / len(rows)
recall = tp / (tp + fn)
precision = tp / (tp + fp) if tp + fp else 0.0
fpr = fp / (fp + tn)

auroc = roc_auc_score(
    [r["actual"] for r in rows],
    [r["anomaly_score"] for r in rows],
)

metrics = {
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

# Save results
output.mkdir(parents=True, exist_ok=True)

with (output / "test_predictions.csv").open(
    "w", newline="", encoding="utf-8"
) as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

(output / "test_metrics.json").write_text(
    json.dumps(metrics, indent=2),
    encoding="utf-8",
)

print("\n========== TEST RESULTS ==========")
print(f"Total images:   {len(rows)}")
print(f"True positives: {tp}")
print(f"True negatives: {tn}")
print(f"False positives:{fp}")
print(f"False negatives:{fn}")
print(f"Accuracy:       {accuracy:.2%}")
print(f"Defect recall:  {recall:.2%}")
print(f"Precision:      {precision:.2%}")
print(f"False-positive rate: {fpr:.2%}")
print(f"Image AUROC:    {auroc:.4f}")
print("==================================")

print("\nSaved test_predictions.csv and test_metrics.json")
