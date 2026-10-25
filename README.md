# Industrial Visual Anomaly Detection and Defect Localisation

An industrial visual-inspection system that learns from defect-free product images to detect anomalies and localise suspicious regions.

Built using **PyTorch, Anomalib, PatchCore, OpenCV, Streamlit and SQLite**.

## Overview

Industrial inspection systems often face a shortage of labelled defective samples. This project investigates whether models trained exclusively on healthy products can identify defects such as scratches, discoloration and structural damage.

The project compares two approaches:

- **PatchCore:** Extracts pretrained visual features and compares image patches against a memory bank of healthy examples.
- **Convolutional Autoencoder:** Learns to reconstruct healthy products and uses reconstruction errors as anomaly scores.

An interactive Streamlit dashboard supports individual and batch inspection, anomaly visualisation, report export and persistent inspection history.

## Results

Experiments were conducted on the Metal Nut category of MVTec AD.

### Image-level detection

| Metric | PatchCore V1 | PatchCore V2 | Autoencoder |
|---|---:|---:|---:|
| Accuracy | 96.52% | 95.65% | 18.26% |
| Defect recall | 96.77% | 98.92% | 2.15% |
| Precision | 98.90% | 95.83% | 40.00% |
| False-positive rate | 4.55% | 18.18% | 13.64% |
| Image AUROC | 0.9897 | 0.9946 | 0.3099 |

### Pixel-level localisation

| Metric | PatchCore V1 | PatchCore V2 | Autoencoder |
|---|---:|---:|---:|
| Pixel AUROC | 0.9791 | — | 0.7444 |
| Pixel recall | 93.58% | 96.61% | — |
| Mask IoU | 63.23% | 62.85% | — |
| Dice score | 77.47% | 77.19% | — |

IoU and Dice are micro-averaged across test pixels. Unreported measurements are marked with a dash.

**Key finding:** PatchCore considerably outperformed the reconstruction-based autoencoder. However, its predicted masks sometimes extend beyond the actual defective regions, particularly for small scratches and discolorations.

These results were obtained on one benchmark category. Subsequent experiments were exploratory and should not be interpreted as independent test-set validation.

## Dataset

**MVTec AD — Metal Nut**

| Partition | Images |
|---|---:|
| Healthy training | 176 |
| Healthy validation | 44 |
| Benchmark test | 115 |
| Defective test images | 93 |
| Healthy test images | 22 |

Healthy training samples were divided using a fixed random seed of 42.

Validation images were used to calibrate anomaly thresholds, while the benchmark test images were used for evaluation.

The original dataset is available from [MVTec AD](https://www.mvtec.com/research-teaching/datasets/mvtec-ad). Its licence restricts usage to the permitted noncommercial purposes. Dataset images are not included in this repository.

## Model Architecture

### PatchCore

- Feature extractor: ResNet-18
- Feature layers: `layer2`, `layer3`
- Memory-bank coreset ratio: 0.1
- Effective input resolution: 256 × 256
- Training: Healthy images only
- Inference: Nearest-neighbour anomaly scoring

PatchCore V1 uses fixed validation-calibrated thresholds:

- Image threshold: `12.2499`
- Pixel threshold: `10.4767`

### Convolutional Autoencoder

- Four convolutional encoder layers
- Four transposed-convolution decoder layers
- Parameters: 1,380,035
- Training epochs: 30
- Optimizer: Adam
- Learning rate: 0.001
- Loss: Mean Squared Error

Two image scoring methods were evaluated: full-image reconstruction MSE and Top-1% pixel reconstruction MSE.

## Inspection Dashboard

The Streamlit application provides:

- Single-image inspection
- Batch-image inspection
- Normal/anomaly classification
- Anomaly scores and calibrated thresholds
- Heatmaps and image overlays
- Binary defect masks
- CSV and PNG/ZIP export
- End-to-end processing-time measurements
- Persistent SQLite inspection history

### Run the application

After installing dependencies, preparing the dataset, fitting the model and generating calibration files:

```bash
python -m streamlit run dashboard.py
```

The application will be available at `http://localhost:8501`.

The project was developed on Windows with Python 3.12 and an NVIDIA RTX 3050 laptop GPU.

## Project Structure

```text
Anomaly detection/
├── configs/
│   └── metal_nut_split.json
├── scripts/
│   ├── create_split.py
│   ├── prepare_data.py
│   ├── train_patchcore.py
│   ├── train_patchcore_v2.py
│   ├── autoencoder.py
│   ├── train_autoencoder.py
│   ├── evaluate_patchcore.py
│   ├── evaluate_masks.py
│   └── ...
├── app.py
├── batch_app.py
├── dashboard.py
├── inspection_history.py
├── datasets/                 # Not tracked by Git
└── outputs/                  # Not tracked by Git
```

Model checkpoints, datasets and inspection databases are generated locally and excluded from version control.

## Limitations

- Results currently cover only the Metal Nut category.
- Small, low-contrast defects remain challenging.
- Some segmentation masks overestimate the defective region.
- Calibration on a limited healthy validation set does not guarantee the same false-positive rate in deployment.
- Performance on independently photographed factory components has not been established.

## Planned Extensions

- Extend PatchCore to Cable and Tile.
- Benchmark training-data efficiency.
- Evaluate robustness to brightness, blur and compression.
- Measure warm-model inference latency and GPU memory usage.
- Add automated tests and Docker deployment.
- Produce a demonstration video and benchmark report.

## Intended Use

This project is a research and portfolio demonstration of industrial visual anomaly detection. It is not validated for production quality-control decisions.