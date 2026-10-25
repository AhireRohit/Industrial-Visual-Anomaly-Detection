"""Generate reproducible benchmark visuals from saved PatchCore evaluation files.

Run from any working directory:
    python scripts/generate_benchmark_charts.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
MODEL_ROOT = ROOT / "outputs" / "patchcore"
FIGURE_DIR = ROOT / "docs" / "figures"
CATEGORIES = {
    "metal_nut": "Metal Nut",
    "cable": "Cable",
    "tile": "Tile",
}


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing benchmark results: {path}\n"
            "Run the category's image and localisation evaluations first."
        )
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def collect_results() -> list[dict]:
    rows = []
    for key, label in CATEGORIES.items():
        folder = MODEL_ROOT / key
        image = load_json(folder / "test_metrics.json")
        if key == "metal_nut":
            pixel = load_json(folder / "pixel_metrics.json")
            masks = load_json(folder / "mask_metrics.json")
        else:
            combined = load_json(folder / "localization_metrics.json")
            pixel = combined
            masks = combined

        rows.append({
            "category": label,
            "test_images": int(image["test_images"]),
            "image_auroc": float(image["image_auroc"]),
            "image_accuracy": float(image["accuracy"]),
            "image_precision": float(image["precision"]),
            "image_recall": float(image["recall"]),
            "image_false_positive_rate": float(image["false_positive_rate"]),
            "pixel_auroc": float(pixel["pixel_auroc"]),
            "pixel_precision": float(masks["pixel_precision"]),
            "pixel_recall": float(masks["pixel_recall"]),
            "mask_iou": float(masks["mask_iou"]),
            "dice": float(masks["dice"]),
        })
    return rows


def plot_comparison(rows: list[dict], columns: tuple[str, str],
                    captions: tuple[str, str], title: str,
                    output_name: str) -> Path:
    names = [row["category"] for row in rows]
    positions = np.arange(len(names))
    width = 0.34
    fig, ax = plt.subplots(figsize=(9, 5))

    for index, (column, caption) in enumerate(zip(columns, captions)):
        bars = ax.bar(
            positions + (index - 0.5) * width,
            [row[column] for row in rows],
            width=width,
            label=caption,
        )
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)

    ax.set_xticks(positions, names)
    ax.set_ylabel("Score (higher is better)")
    ax.set_ylim(0, 1.08)
    ax.set_title(title)
    ax.legend(loc="upper center",
              bbox_to_anchor=(0.5, -0.12),
              ncol=2,
              frameon=False,
    )
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    fig.tight_layout()

    destination = FIGURE_DIR / output_name
    fig.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return destination


def main() -> None:
    rows = collect_results()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = FIGURE_DIR / "benchmark_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    image_chart = plot_comparison(
        rows,
        ("image_auroc", "pixel_auroc"),
        ("Image AUROC", "Pixel AUROC"),
        "Anomaly detection performance — MVTec AD",
        "detection_auroc.png",
    )
    mask_chart = plot_comparison(
        rows,
        ("mask_iou", "dice"),
        ("Mask IoU", "Dice"),
        "Binary defect localisation — MVTec AD",
        "localization_overlap.png",
    )

    print("Benchmark artifacts created:")
    for path in (csv_path, image_chart, mask_chart):
        print(f"  {path.relative_to(ROOT)}")
    print("\nSummary:")
    for row in rows:
        print(
            f"{row['category']:10s}  image AUROC={row['image_auroc']:.4f}  "
            f"pixel AUROC={row['pixel_auroc']:.4f}  "
            f"IoU={row['mask_iou']:.4f}  Dice={row['dice']:.4f}"
        )


if __name__ == "__main__":
    main()
