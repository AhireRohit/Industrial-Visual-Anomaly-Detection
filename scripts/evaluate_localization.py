from pathlib import Path
import json

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score

from anomalib.engine import Engine
from anomalib.models import Patchcore

ROOT = Path("datasets/MVTecAD/metal_nut")
TEST_ROOT = ROOT / "test"
GT_ROOT = ROOT / "ground_truth"
OUTPUT = Path("outputs/patchcore/metal_nut")
CHECKPOINT = OUTPUT / "model.ckpt"

assert CHECKPOINT.is_file(), "Checkpoint not found"

model = Patchcore(
    backbone="resnet18",
    layers=["layer2", "layer3"],
    coreset_sampling_ratio=0.1,
)

engine = Engine(
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=1,
    logger=False,
    default_root_dir=str(OUTPUT / "pixel_eval"),
)

pixel_scores = []
pixel_labels = []
image_counter = 0

for folder in sorted(TEST_ROOT.iterdir()):
    if not folder.is_dir():
        continue

    print(f"Evaluating pixel localisation for: {folder.name}")

    predictions = engine.predict(
        model=model,
        ckpt_path=str(CHECKPOINT),
        data_path=str(folder),
        return_predictions=True,
    )

    assert predictions is not None

    for batch in predictions:
        anomaly_maps = torch.as_tensor(batch.anomaly_map).detach().cpu()

        if anomaly_maps.ndim == 2:
            anomaly_maps = anomaly_maps.unsqueeze(0)
        elif anomaly_maps.ndim == 4 and anomaly_maps.shape[1] == 1:
            anomaly_maps = anomaly_maps[:, 0]
        elif anomaly_maps.ndim != 3:
            raise ValueError(
                f"Unexpected anomaly_map shape: {tuple(anomaly_maps.shape)}"
            )

        paths = batch.image_path
        if isinstance(paths, (str, Path)):
            paths = [paths]

        assert len(paths) == len(anomaly_maps)

        for path, anomaly_map in zip(paths, anomaly_maps):
            path = Path(path)
            anomaly_map = anomaly_map.numpy().astype(np.float32)
            h, w = anomaly_map.shape

            if folder.name == "good":
                mask = np.zeros((h, w), dtype=np.uint8)
            else:
                mask_path = GT_ROOT / folder.name / f"{path.stem}_mask.png"
                assert mask_path.is_file(), f"Missing mask: {mask_path}"

                mask = Image.open(mask_path).convert("L")
                mask = mask.resize((w, h), Image.Resampling.NEAREST)
                mask = (np.array(mask) > 0).astype(np.uint8)

            pixel_scores.append(anomaly_map.reshape(-1))
            pixel_labels.append(mask.reshape(-1))
            image_counter += 1

all_scores = np.concatenate(pixel_scores)
all_labels = np.concatenate(pixel_labels)

pixel_auroc = roc_auc_score(all_labels, all_scores)

metrics = {
    "category": "metal_nut",
    "test_images": image_counter,
    "total_pixels": int(all_labels.size),
    "anomalous_pixels": int(all_labels.sum()),
    "normal_pixels": int((all_labels == 0).sum()),
    "pixel_auroc": float(pixel_auroc),
}

output_path = OUTPUT / "pixel_metrics.json"
output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

print("\n========== PIXEL-LEVEL RESULTS ==========")
print(f"Images evaluated: {image_counter}")
print(f"Total pixels:     {all_labels.size}")
print(f"Anomalous pixels: {all_labels.sum()}")
print(f"Pixel AUROC:      {pixel_auroc:.4f}")
print("=========================================")
print(f"\nSaved: {output_path}")