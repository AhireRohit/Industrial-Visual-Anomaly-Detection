
from pathlib import Path
import json
import random

DATASET_ROOT = Path("datasets/MVTecAD")
CONFIG_DIR = Path("configs")
CATEGORIES = {
    "cable": 224,
    "tile": 230,
}
SEED = 42
TRAIN_RATIO = 0.80

for category, expected_count in CATEGORIES.items():
    good_dir = DATASET_ROOT / category / "train" / "good"
    images = sorted(good_dir.glob("*.png"))

    assert len(images) == expected_count, (
        f"{category}: Expected {expected_count} images, "
        f"found {len(images)}"
    )

    rng = random.Random(SEED)
    rng.shuffle(images)

    split_index = int(len(images) * TRAIN_RATIO)
    train = images[:split_index]
    val = images[split_index:]

    split = {
        "category": category,
        "seed": SEED,
        "train": [p.as_posix() for p in train],
        "val": [p.as_posix() for p in val],
    }

    assert not set(split["train"]) & set(split["val"])

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    output = CONFIG_DIR / f"{category}_split.json"
    output.write_text(
        json.dumps(split, indent=2),
        encoding="utf-8"
    )

    print(f"\n{category.upper()}")
    print(f"Training images: {len(train)}")
    print(f"Validation images: {len(val)}")
    print(f"Saved: {output}")

print("\nAll splits created successfully!")
