
from pathlib import Path
import json
import shutil

CATEGORIES = ["cable", "tile"]

for category in CATEGORIES:
    manifest_path = Path("configs") / f"{category}_split.json"

    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )

    output_root = Path("datasets/processed") / category

    print(f"\nPreparing {category.upper()}")

    train_paths = set(manifest["train"])
    val_paths = set(manifest["val"])

    assert not train_paths & val_paths, (
        f"{category}: Training/validation overlap detected"
    )

    for split_name in ("train", "val"):
        images = manifest[split_name]
        destination = output_root / split_name / "good"
        destination.mkdir(parents=True, exist_ok=True)

        for image_path in images:
            source = Path(image_path)

            assert source.is_file(), (
                f"Missing image: {source}"
            )

            shutil.copy2(
                source,
                destination / source.name
            )

        expected = {Path(p).name for p in images}
        actual = {p.name for p in destination.glob("*.png")}

        assert actual == expected, (
            f"{category}/{split_name}: "
            "Unexpected or missing images"
        )

        print(f"{split_name}: {len(actual)} images ready")

print("\nAll datasets prepared successfully!")
