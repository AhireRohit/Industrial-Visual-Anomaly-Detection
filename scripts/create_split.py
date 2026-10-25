
from pathlib import Path
import random
import json

ROOT = Path("datasets/MVTecAD/metal_nut")
OUTPUT = Path("configs/metal_nut_split.json")
SEED = 42

images = sorted((ROOT / "train" / "good").glob("*.png"))

assert len(images) == 220, f"Expected 220 images, found {len(images)}"

rng = random.Random(SEED)
rng.shuffle(images)

train = images[:176]
val = images[176:]

split = {
    "category": "metal_nut",
    "seed": SEED,
    "train": [p.as_posix() for p in train],
    "val": [p.as_posix() for p in val],
}

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(split, indent=2), encoding="utf-8")

assert not set(split["train"]) & set(split["val"])

print(f"Training images: {len(train)}")
print(f"Validation images: {len(val)}")
print(f"Split saved to: {OUTPUT}")
