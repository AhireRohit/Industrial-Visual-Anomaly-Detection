
from pathlib import Path

import torch
from PIL import Image

from anomalib.engine import Engine
from anomalib.models import Patchcore
from anomalib.visualization import visualize_anomaly_map

# Saved model and validation image
checkpoint = Path("outputs/patchcore/metal_nut/model.ckpt")
val_dir = Path("datasets/processed/metal_nut/val/good")
output_dir = Path("outputs/patchcore/metal_nut/previews")

assert checkpoint.exists(), "Checkpoint missing"
images = sorted(val_dir.glob("*.png"))
assert len(images) == 44, "Validation images missing"

image_path = images[0]
output_dir.mkdir(parents=True, exist_ok=True)

# Recreate model configuration
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

# Load checkpoint and run inference
predictions = engine.predict(
    model=model,
    ckpt_path=str(checkpoint),
    data_path=str(image_path),
    return_predictions=True,
)

assert predictions, "No predictions returned"

prediction = predictions[0]
score = torch.as_tensor(prediction.pred_score).flatten()[0].item()
anomaly_map = prediction.anomaly_map.squeeze().detach().cpu()

# Create heatmap (normalized for visualization only)
heatmap = visualize_anomaly_map(
    anomaly_map,
    normalize=True,
    colormap=True,
)

original = Image.open(image_path).convert("RGB")
original = original.resize(heatmap.size)

overlay = Image.blend(original, heatmap, alpha=0.45)

# Combine images side by side
width, height = original.size
result = Image.new("RGB", (width * 3, height))

result.paste(original, (0, 0))
result.paste(heatmap, (width, 0))
result.paste(overlay, (width * 2, 0))

output_path = output_dir / "first_heatmap.png"
result.save(output_path)

print("Image:", image_path)
print(f"Anomaly score: {score:.6f}")
print("Anomaly map shape:", tuple(anomaly_map.shape))
print("Saved:", output_path)
