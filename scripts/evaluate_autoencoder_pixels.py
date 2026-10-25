
from pathlib import Path
import json

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import v2
from sklearn.metrics import roc_auc_score

from autoencoder import ConvAutoencoder

ROOT = Path("datasets/MVTecAD/metal_nut")
OUTPUT = Path("outputs/autoencoder/metal_nut")
CHECKPOINT = OUTPUT / "model.pt"

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

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

pixel_scores = []
pixel_labels = []
image_count = 0

print("Using device:", device)

with torch.inference_mode():
    for folder in sorted((ROOT / "test").iterdir()):
        if not folder.is_dir():
            continue

        images = sorted(folder.glob("*.png"))
        print(f"Evaluating {folder.name}: {len(images)} images")

        for path in images:
            with Image.open(path) as image:
                x = transform(
                    image.convert("RGB")
                ).unsqueeze(0).to(device)

            reconstruction = model(x)

            # Squared reconstruction error per pixel
            error_map = (
                (x - reconstruction) ** 2
            ).mean(dim=1)[0]

            scores = error_map.cpu().numpy()
            h, w = scores.shape

            # Ground-truth defect mask
            if folder.name == "good":
                mask = np.zeros((h, w), dtype=np.uint8)
            else:
                mask_path = (
                    ROOT / "ground_truth"
                    / folder.name
                    / f"{path.stem}_mask.png"
                )
                assert mask_path.is_file(), mask_path

                with Image.open(mask_path) as gt:
                    gt = gt.convert("L").resize(
                        (w, h),
                        Image.Resampling.NEAREST,
                    )
                    mask = (np.array(gt) > 0).astype(np.uint8)

            pixel_scores.append(scores.reshape(-1))
            pixel_labels.append(mask.reshape(-1))
            image_count += 1

assert image_count == 115

all_scores = np.concatenate(pixel_scores)
all_labels = np.concatenate(pixel_labels)

assert np.isfinite(all_scores).all()
assert int(all_labels.sum()) == 883024

pixel_auroc = roc_auc_score(
    all_labels,
    all_scores,
)

metrics = {
    "model": "convolutional_autoencoder",
    "category": "metal_nut",
    "scoring_method": "per_pixel_squared_reconstruction_error",
    "test_images": image_count,
    "total_pixels": int(all_labels.size),
    "anomalous_pixels": int(all_labels.sum()),
    "pixel_auroc": float(pixel_auroc),
    "evaluation_status": "exploratory",
}

output_path = OUTPUT / "pixel_metrics.json"

output_path.write_text(
    json.dumps(metrics, indent=2),
    encoding="utf-8",
)

print("\n====== AUTOENCODER PIXEL RESULTS ======")
print(f"Images evaluated: {image_count}")
print(f"Total pixels:     {all_labels.size:,}")
print(f"Anomalous pixels: {all_labels.sum():,}")
print(f"Pixel AUROC:      {pixel_auroc:.4f}")
print("======================================")
print("Saved:", output_path)
