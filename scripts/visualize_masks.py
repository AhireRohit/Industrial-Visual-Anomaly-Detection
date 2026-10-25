
from pathlib import Path
import csv
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path("datasets/MVTecAD/metal_nut")
OUTPUT = Path("outputs/patchcore/metal_nut")

with (OUTPUT / "mask_per_image.csv").open(
    newline="", encoding="utf-8"
) as f:
    rows = list(csv.DictReader(f))

# Five defective images with the lowest IoU
worst = sorted(
    (r for r in rows if r["defect_type"] != "good"),
    key=lambda r: float(r["iou"])
)[:5]

SIZE = 256
PAD = 12
ROW_HEIGHT = 300
WIDTH = SIZE * 4 + PAD * 5

gallery = Image.new(
    "RGB",
    (WIDTH, 55 + len(worst) * ROW_HEIGHT),
    "white"
)
draw = ImageDraw.Draw(gallery)

titles = [
    "Original",
    "Ground truth",
    "Predicted mask",
    "Error comparison"
]

for j, title in enumerate(titles):
    draw.text(
        (PAD + j * (SIZE + PAD), 15),
        title,
        fill="black"
    )

for i, row in enumerate(worst):
    relative = Path(row["image"])
    defect_type = relative.parent.name
    filename = relative.name

    # Original image
    original = Image.open(
        ROOT / "test" / relative
    ).convert("RGB").resize((SIZE, SIZE))

    # Ground-truth mask
    gt_path = (
        ROOT / "ground_truth" /
        defect_type /
        f"{relative.stem}_mask.png"
    )

    gt = Image.open(gt_path).convert("L")
    gt = gt.resize(
        (SIZE, SIZE),
        Image.Resampling.NEAREST
    )
    gt_mask = np.array(gt) > 0

    # Predicted mask from Step 17
    pred_path = (
        OUTPUT / "predicted_masks" /
        defect_type / filename
    )

    pred = Image.open(pred_path).convert("L")
    pred = pred.resize(
        (SIZE, SIZE),
        Image.Resampling.NEAREST
    )
    pred_mask = np.array(pred) > 0

    # Color-coded comparison overlay
    comparison = np.array(original).copy()

    regions = [
        (pred_mask & gt_mask, [0, 220, 60]),
        (pred_mask & ~gt_mask, [255, 40, 40]),
        (~pred_mask & gt_mask, [255, 220, 0]),
    ]

    for mask, color in regions:
        comparison[mask] = (
            0.35 * comparison[mask] +
            0.65 * np.array(color)
        ).astype(np.uint8)

    comparison_image = Image.fromarray(comparison)

    images = [
        original,
        Image.fromarray(
            (gt_mask * 255).astype(np.uint8)
        ).convert("RGB"),
        Image.fromarray(
            (pred_mask * 255).astype(np.uint8)
        ).convert("RGB"),
        comparison_image,
    ]

    y = 55 + i * ROW_HEIGHT

    draw.text(
        (PAD, y),
        f"{row['image']} | IoU: {float(row['iou']):.3f}",
        fill="black"
    )

    for j, img in enumerate(images):
        x = PAD + j * (SIZE + PAD)
        gallery.paste(img, (x, y + 24))

save_path = OUTPUT / "mask_comparison.png"
gallery.save(save_path)

print(f"Saved: {save_path}")
print("Visualized:", len(worst), "images")
