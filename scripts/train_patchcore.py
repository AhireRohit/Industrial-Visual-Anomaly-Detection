
from pathlib import Path

import torch
from torchvision.transforms import v2

from anomalib.data import Folder
from anomalib.data.utils import TestSplitMode, ValSplitMode
from anomalib.engine import Engine
from anomalib.models import Patchcore

# Reproducibility
torch.manual_seed(42)

# Hardware
device = "gpu" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}", flush=True)

# Image preprocessing
transform = v2.Compose([
    v2.Resize((224, 224)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
])

# Our fixed dataset split
data = Folder(
    name="metal_nut",
    root="datasets/processed/metal_nut",
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

assert len(data.train_dataloader().dataset) == 176
assert len(data.val_dataloader().dataset) == 44

# Initialize PatchCore
model = Patchcore(
    backbone="resnet18",
    layers=["layer2", "layer3"],
    coreset_sampling_ratio=0.1,
)

# Configure training
engine = Engine(
    accelerator=device,
    devices=1,
    max_epochs=1,
    limit_val_batches=0,
    enable_checkpointing=True,
    logger=False,
    default_root_dir="outputs/patchcore/metal_nut",
)

print("Starting PatchCore fitting...", flush=True)

# Fit using healthy training images only.
# Validation is intentionally disabled during fitting.
engine.fit(model=model, datamodule=data)

# Save a Lightning checkpoint including the memory bank.
checkpoint_path = Path("outputs/patchcore/metal_nut/model.ckpt")
checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

engine.trainer.save_checkpoint(str(checkpoint_path))

print("\nPatchCore fitting completed!")
print(f"Checkpoint saved: {checkpoint_path}")
print(f"Checkpoint size: {checkpoint_path.stat().st_size / (1024**2):.2f} MB")
