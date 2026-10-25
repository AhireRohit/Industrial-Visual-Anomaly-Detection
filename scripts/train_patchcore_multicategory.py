
import argparse
import json
from pathlib import Path

import torch
from torchvision.transforms import v2

from anomalib.data import Folder
from anomalib.data.utils import TestSplitMode, ValSplitMode
from anomalib.engine import Engine
from anomalib.models import Patchcore


parser = argparse.ArgumentParser()
parser.add_argument(
    "--category",
    choices=["cable", "tile"],
    required=True,
)
args = parser.parse_args()

category = args.category
torch.manual_seed(42)

device = "gpu" if torch.cuda.is_available() else "cpu"

print(f"Category: {category}")
print(f"Device: {device}", flush=True)

manifest = json.loads(
    (Path("configs") / f"{category}_split.json")
    .read_text(encoding="utf-8")
)

expected_train = len(manifest["train"])
expected_val = len(manifest["val"])

data_root = Path("datasets/processed") / category
output_dir = Path("outputs/patchcore") / category

transform = v2.Compose([
    v2.Resize((224, 224)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
])

data = Folder(
    name=category,
    root=str(data_root),
    normal_dir="train/good",
    normal_test_dir="val/good",
    train_batch_size=1,
    eval_batch_size=1,
    num_workers=0,
    normal_split_ratio=0,
    test_split_mode=TestSplitMode.FROM_DIR,
    val_split_mode=ValSplitMode.SAME_AS_TEST,
    train_augmentations=transform,
    val_augmentations=transform,
    test_augmentations=transform,
)

data.setup()

assert len(data.train_dataloader().dataset) == expected_train
assert len(data.val_dataloader().dataset) == expected_val

print(f"Training images: {expected_train}")
print(f"Validation images: {expected_val}")

model = Patchcore(
    backbone="resnet18",
    layers=["layer2", "layer3"],
    coreset_sampling_ratio=0.1,
)

engine = Engine(
    accelerator=device,
    devices=1,
    max_epochs=1,
    limit_val_batches=0,
    enable_checkpointing=True,
    logger=False,
    default_root_dir=str(output_dir),
)

print("\nStarting PatchCore fitting...", flush=True)

engine.fit(model=model, datamodule=data)

output_dir.mkdir(parents=True, exist_ok=True)
checkpoint = output_dir / "model.ckpt"

engine.trainer.save_checkpoint(str(checkpoint))

print("\nTraining completed successfully!")
print(f"Checkpoint: {checkpoint}")
print(
    f"Model size: "
    f"{checkpoint.stat().st_size / (1024 ** 2):.2f} MB"
)
