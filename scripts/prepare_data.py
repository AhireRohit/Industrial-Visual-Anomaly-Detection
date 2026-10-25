
from pathlib import Path
import json
import shutil

manifest = json.loads(
    Path("configs/metal_nut_split.json").read_text(encoding="utf-8")
)

output = Path("datasets/processed/metal_nut")

for split_name in ("train", "val"):
    images = manifest[split_name]
    destination = output / split_name / "good"
    destination.mkdir(parents=True, exist_ok=True)

    for image_path in images:
        source = Path(image_path)
        assert source.is_file(), f"Missing image: {source}"

        shutil.copy2(source, destination / source.name)

    actual = sorted(destination.glob("*.png"))
    assert len(actual) == len(images), (
        f"{split_name}: expected {len(images)}, found {len(actual)}"
    )

    print(f"{split_name}: {len(actual)} images ready")

print("Dataset preparation complete!")
