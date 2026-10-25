
from pathlib import Path
import csv

from PIL import Image, ImageDraw

ROOT = Path("datasets/MVTecAD/metal_nut")
RESULTS = Path("outputs/patchcore/metal_nut")
THRESHOLD = 12.249866485595703

with (RESULTS / "test_predictions.csv").open(
    newline="", encoding="utf-8"
) as f:
    rows = list(csv.DictReader(f))

errors = [
    row for row in rows
    if row["actual"] != row["predicted"]
]

print(f"Total misclassified images: {len(errors)}")

SIZE = 300
ROW_HEIGHT = 355

gallery = Image.new(
    "RGB",
    (SIZE * 2 + 40, len(errors) * ROW_HEIGHT),
    "white",
)
draw = ImageDraw.Draw(gallery)

for i, row in enumerate(errors):
    relative = Path(row["image"])
    image_path = ROOT / "test" / relative

    original = Image.open(image_path).convert("RGB")
    original = original.resize((SIZE, SIZE))

    annotated = original.copy()

    if row["actual"] == "1":
        mask_path = (
            ROOT / "ground_truth"
            / relative.parent
            / f"{relative.stem}_mask.png"
        )

        mask = Image.open(mask_path).convert("L")
        mask = mask.resize(
            (SIZE, SIZE), Image.Resampling.NEAREST
        )

        red = Image.new("RGBA", (SIZE, SIZE), (255, 0, 0, 0))
        red.putalpha(
            mask.point(lambda value: 140 if value > 0 else 0)
        )

        annotated = Image.alpha_composite(
            original.convert("RGBA"), red
        ).convert("RGB")

    y = i * ROW_HEIGHT

    error_type = (
        "MISSED DEFECT"
        if row["actual"] == "1"
        else "FALSE ALARM"
    )

    label = (
        f"{error_type} | {relative} | "
        f"Score: {float(row['anomaly_score']):.3f}"
    )

    draw.text((10, y + 5), label, fill="black")

    gallery.paste(original, (10, y + 30))
    gallery.paste(annotated, (SIZE + 30, y + 30))

    draw.text(
        (10, y + 332), "Original", fill="black"
    )
    draw.text(
        (SIZE + 30, y + 332),
        "Ground truth (red)" if row["actual"] == "1"
        else "Healthy - no defect",
        fill="black",
    )

    print(label)

output = RESULTS / "failure_gallery.png"
gallery.save(output)

print(f"\nSaved: {output}")
