
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision.transforms import v2

from autoencoder import ConvAutoencoder

OUTPUT = Path("outputs/autoencoder/metal_nut")
CHECKPOINT = OUTPUT / "model.pt"

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Load trained model
model = ConvAutoencoder().to(device)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=device,
    weights_only=True,
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

transform = v2.Compose([
    v2.Resize((256, 256)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
])

# One healthy validation image
healthy = sorted(
    Path("datasets/processed/metal_nut/val/good")
    .glob("*.png")
)[0]

# One defective test image (qualitative demonstration only)
defective = sorted(
    Path("datasets/MVTecAD/metal_nut/test/bent")
    .glob("*.png")
)[0]

samples = [
    ("HEALTHY VALIDATION IMAGE", healthy),
    ("DEFECTIVE TEST IMAGE", defective),
]

SIZE = 256
PAD = 12
ROW_HEIGHT = 300

gallery = Image.new(
    "RGB",
    (SIZE * 4 + PAD * 5, ROW_HEIGHT * 2),
    "white",
)
draw = ImageDraw.Draw(gallery)

for i, (label, path) in enumerate(samples):
    image = Image.open(path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.inference_mode():
        reconstruction = model(tensor)

    original = tensor[0].cpu().permute(1, 2, 0).numpy()
    rebuilt = reconstruction[0].cpu().permute(1, 2, 0).numpy()

    # Absolute pixel reconstruction error
    error = np.mean(
        np.abs(original - rebuilt), axis=2
    )

    mse = float(np.mean((original - rebuilt) ** 2))

    # Fixed visualization scale across both images
    intensity = np.clip(error / 0.20, 0, 1)

    heatmap = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    heatmap[:, :, 0] = (intensity * 255).astype(np.uint8)

    # Red overlay proportional to reconstruction error
    red = np.zeros_like(original)
    red[:, :, 0] = 1.0

    alpha = (intensity * 0.65)[..., None]
    overlay = original * (1 - alpha) + red * alpha

    def to_image(array):
        return Image.fromarray(
            (np.clip(array, 0, 1) * 255).astype(np.uint8)
        )

    panels = [
        to_image(original),
        to_image(rebuilt),
        Image.fromarray(heatmap),
        to_image(overlay),
    ]

    y = i * ROW_HEIGHT

    draw.text(
        (PAD, y + 5),
        f"{label} | MSE: {mse:.6f} | {path.name}",
        fill="black",
    )

    names = [
        "Original",
        "Reconstructed",
        "Absolute error",
        "Error overlay",
    ]

    for j, (panel, name) in enumerate(zip(panels, names)):
        x = PAD + j * (SIZE + PAD)

        gallery.paste(panel, (x, y + 28))
        draw.text(
            (x, y + 286),
            name,
            fill="black",
        )

    print(f"{label}: MSE = {mse:.6f}")

save_path = OUTPUT / "reconstruction_comparison.png"
gallery.save(save_path)

print("Saved:", save_path)
