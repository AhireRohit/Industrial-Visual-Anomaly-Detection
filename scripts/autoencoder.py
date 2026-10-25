
from pathlib import Path

import torch
from torch import nn
from PIL import Image
from torchvision.transforms import v2


class ConvAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1),
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


if __name__ == "__main__":
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    images = sorted(
        Path("datasets/processed/metal_nut/train/good")
        .glob("*.png")
    )
    assert images, "No training images found"

    transform = v2.Compose([
        v2.Resize((256, 256)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ])

    image = Image.open(images[0]).convert("RGB")
    x = transform(image).unsqueeze(0).to(device)

    model = ConvAutoencoder().to(device)
    model.eval()

    with torch.no_grad():
        reconstructed = model(x)

    assert reconstructed.shape == x.shape
    assert torch.isfinite(reconstructed).all()

    parameters = sum(
        p.numel() for p in model.parameters()
    )

    print("Device:", device)
    print("Input shape:", tuple(x.shape))
    print("Output shape:", tuple(reconstructed.shape))
    print("Parameters:", f"{parameters:,}")
    print("Autoencoder smoke test PASSED")
    print("Model is untrained.")
