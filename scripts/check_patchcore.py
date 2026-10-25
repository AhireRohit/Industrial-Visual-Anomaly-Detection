
import torch
from torchvision.transforms import v2

from anomalib.data import Folder
from anomalib.data.utils import TestSplitMode, ValSplitMode
from anomalib.models import Patchcore

print("GPU available:", torch.cuda.is_available())

transform = v2.Compose([
    v2.Resize((224, 224)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
])

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

train_count = len(data.train_dataloader().dataset)
val_count = len(data.val_dataloader().dataset)

print("Training images:", train_count)
print("Validation images:", val_count)

assert train_count == 176
assert val_count == 44

model = Patchcore(
    backbone="resnet18",
    layers=["layer2", "layer3"],
    coreset_sampling_ratio=0.1,
)

print("PatchCore initialized successfully!")
