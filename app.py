
from pathlib import Path
from tempfile import TemporaryDirectory
from datetime import datetime, timezone
import hashlib
import csv
import io
import json

import numpy as np
import streamlit as st
import torch
from PIL import Image, ImageOps

from anomalib.engine import Engine
from anomalib.models import Patchcore


# ==========================================
# CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

ROOT = BASE_DIR / "outputs" / "patchcore" / "metal_nut"
CHECKPOINT = ROOT / "model.ckpt"

IMAGE_CONFIG_PATH = ROOT / "threshold.json"
PIXEL_CONFIG_PATH = ROOT / "pixel_threshold.json"

MODEL_NAME = "PatchCore"
MODEL_VERSION = "v1"
CATEGORY = "metal_nut"

st.set_page_config(
    page_title="Industrial AI Inspection",
    page_icon="🔍",
    layout="wide",
)

# Verify model files
required_files = [
    CHECKPOINT,
    IMAGE_CONFIG_PATH,
    PIXEL_CONFIG_PATH,
]

for path in required_files:
    if not path.is_file():
        st.error(f"Required file missing: {path}")
        st.stop()

# Load thresholds
with IMAGE_CONFIG_PATH.open(encoding="utf-8") as f:
    image_config = json.load(f)

with PIXEL_CONFIG_PATH.open(encoding="utf-8") as f:
    pixel_config = json.load(f)

IMAGE_THRESHOLD = float(image_config["image_threshold"])
PIXEL_THRESHOLD = float(pixel_config["pixel_threshold"])


# ==========================================
# MODEL INFERENCE
# ==========================================

def inspect_image(image):
    """Run PatchCore inference on one image."""

    with TemporaryDirectory() as temporary:
        image_path = Path(temporary) / "input.png"
        image.save(image_path)

        model = Patchcore(
            backbone="resnet18",
            layers=["layer2", "layer3"],
            coreset_sampling_ratio=0.1,
        )

        engine = Engine(
            accelerator=(
                "gpu" if torch.cuda.is_available() else "cpu"
            ),
            devices=1,
            logger=False,
            default_root_dir=str(ROOT / "dashboard_runs"),
        )

        predictions = engine.predict(
            model=model,
            ckpt_path=str(CHECKPOINT),
            data_path=str(image_path),
            return_predictions=True,
        )

        if not predictions:
            raise RuntimeError("No prediction returned.")

        prediction = predictions[0]

        score = float(
            torch.as_tensor(
                prediction.pred_score
            ).reshape(-1)[0].item()
        )

        anomaly_map = (
            torch.as_tensor(prediction.anomaly_map)
            .squeeze()
            .detach()
            .cpu()
            .numpy()
        )

        if anomaly_map.ndim != 2:
            raise ValueError(
                f"Unexpected anomaly map shape: "
                f"{anomaly_map.shape}"
            )

        if not np.isfinite(anomaly_map).all():
            raise ValueError(
                "Anomaly map contains invalid values."
            )

        return score, anomaly_map


# ==========================================
# VISUALIZATION
# ==========================================

def create_visualizations(image, anomaly_map):
    """Generate heatmap, overlay and binary mask."""

    height, width = anomaly_map.shape

    original = image.resize(
        (width, height),
        Image.Resampling.BILINEAR,
    )

    # Heatmap normalization for visualization only
    low = np.percentile(anomaly_map, 2)
    high = np.percentile(anomaly_map, 98)

    normalized = np.clip(
        (anomaly_map - low) / (high - low + 1e-8),
        0,
        1,
    )

    intensity = (
        normalized * 255
    ).astype(np.uint8)

    # Blue -> Green -> Red heatmap
    green = np.clip(
        255 - 2 * np.abs(
            intensity.astype(np.int16) - 127
        ),
        0,
        255,
    ).astype(np.uint8)

    heatmap_rgb = np.stack(
        [
            intensity,
            green,
            255 - intensity,
        ],
        axis=-1,
    )

    heatmap = Image.fromarray(
        heatmap_rgb.astype(np.uint8)
    )

    overlay = Image.blend(
        original,
        heatmap,
        alpha=0.40,
    )

    # Real calibrated pixel threshold
    binary_mask = (
        anomaly_map > PIXEL_THRESHOLD
    ).astype(np.uint8) * 255

    mask_image = Image.fromarray(binary_mask)

    return original, heatmap, overlay, mask_image


# ==========================================
# CSV EXPORT
# ==========================================

def create_csv_report(result, filename):
    """Create a downloadable inspection report."""

    report = {
        "timestamp_utc": result["timestamp"],
        "image": filename,
        "category": CATEGORY,
        "model": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "decision": result["decision"],
        "anomaly_score": result["score"],
        "image_threshold": IMAGE_THRESHOLD,
        "pixel_threshold": PIXEL_THRESHOLD,
    }

    buffer = io.StringIO()

    writer = csv.DictWriter(
        buffer,
        fieldnames=list(report.keys()),
    )

    writer.writeheader()
    writer.writerow(report)

    return buffer.getvalue()


# ==========================================
# DASHBOARD HEADER
# ==========================================

st.title("Industrial Visual Anomaly Detection")

st.caption(
    "PatchCore | Metal Nut | Model V1"
)

st.write(
    "Upload a metal nut photograph to detect unusual "
    "visual patterns and localize suspicious regions."
)

# ==========================================
# IMAGE UPLOAD
# ==========================================

uploaded = st.file_uploader(
    "Choose an image",
    type=["png", "jpg", "jpeg", "bmp"],
    accept_multiple_files=False,
)

if uploaded is not None:

    uploaded_bytes = uploaded.getvalue()

    image_hash = hashlib.sha256(
        uploaded_bytes
    ).hexdigest()

    image = ImageOps.exif_transpose(
        Image.open(io.BytesIO(uploaded_bytes))
    ).convert("RGB")

    st.image(
        image,
        caption="Uploaded image",
        width=350,
    )

    # ======================================
    # INSPECTION BUTTON
    # ======================================

    if st.button(
        "Inspect Product",
        type="primary",
    ):

        with st.spinner("Running PatchCore inspection..."):

            try:
                score, anomaly_map = inspect_image(image)

                decision = (
                    "ANOMALY"
                    if score > IMAGE_THRESHOLD
                    else "NORMAL"
                )

                # Preserve result across Streamlit reruns
                st.session_state["inspection_result"] = {
                    "image_hash": image_hash,
                    "filename": uploaded.name,
                    "score": score,
                    "anomaly_map": anomaly_map,
                    "decision": decision,
                    "timestamp": datetime.now(
                        timezone.utc
                    ).isoformat(),
                }

            except Exception as error:
                st.error(
                    f"Inspection failed: {error}"
                )
                st.stop()

    # ======================================
    # DISPLAY SAVED INSPECTION RESULT
    # ======================================

    result = st.session_state.get(
        "inspection_result"
    )

    # Do not display an old result for a new upload
    if (
        result is not None
        and result["image_hash"] == image_hash
    ):

        score = result["score"]
        anomaly_map = result["anomaly_map"]
        decision = result["decision"]

        is_anomaly = decision == "ANOMALY"

        st.divider()

        st.subheader("Inspection Result")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Decision",
            decision,
        )

        col2.metric(
            "Anomaly Score",
            f"{score:.4f}",
        )

        col3.metric(
            "Alert Threshold",
            f"{IMAGE_THRESHOLD:.4f}",
        )

        if is_anomaly:
            st.error(
                "Suspicious visual patterns detected."
            )
        else:
            st.success(
                "Image classified as normal."
            )

        # ==================================
        # DEFECT LOCALISATION
        # ==================================

        st.subheader("Defect Localisation")

        original, heatmap, overlay, mask_image = (
            create_visualizations(
                image,
                anomaly_map,
            )
        )

        panels = st.columns(4)

        panels[0].image(
            original,
            caption="Original Image",
            use_container_width=True,
        )

        panels[1].image(
            heatmap,
            caption="Anomaly Heatmap",
            use_container_width=True,
        )

        panels[2].image(
            overlay,
            caption="Heatmap Overlay",
            use_container_width=True,
        )

        panels[3].image(
            mask_image,
            caption="Predicted Binary Mask",
            use_container_width=True,
        )

        st.caption(
            "Heatmap colors are normalized for display. "
            "The binary mask uses the calibrated pixel "
            "threshold. Anomaly scores are not probabilities."
        )

        # ==================================
        # INSPECTION DETAILS
        # ==================================

        with st.expander("Inspection Details"):

            st.write(f"**File:** {uploaded.name}")
            st.write(f"**Category:** {CATEGORY}")
            st.write(f"**Model:** {MODEL_NAME}")
            st.write(f"**Version:** {MODEL_VERSION}")
            st.write(f"**Device:** {'CUDA' if torch.cuda.is_available() else 'CPU'}")

            st.write(
                f"**Anomaly score:** {score:.6f}"
            )

            st.write(
                f"**Image threshold:** {IMAGE_THRESHOLD:.6f}"
            )

            st.write(
                f"**Pixel threshold:** {PIXEL_THRESHOLD:.6f}"
            )

            st.write(
                f"**Inspection time (UTC):** "
                f"{result['timestamp']}"
            )

        # ==================================
        # EXPORT RESULTS
        # ==================================

        st.divider()
        st.subheader("Export Inspection Results")

        csv_data = create_csv_report(
            result,
            uploaded.name,
        )

        mask_buffer = io.BytesIO()

        mask_image.save(
            mask_buffer,
            format="PNG",
        )

        col_a, col_b = st.columns(2)

        with col_a:
            st.download_button(
                label="Download Inspection CSV",
                data=csv_data,
                file_name="inspection_result.csv",
                mime="text/csv",
                on_click="ignore",
                use_container_width=True,
            )

        with col_b:
            st.download_button(
                label="Download Defect Mask PNG",
                data=mask_buffer.getvalue(),
                file_name="predicted_mask.png",
                mime="image/png",
                on_click="ignore",
                use_container_width=True,
            )

        st.caption(
            "Inspection results were generated using "
            "PatchCore V1 with fixed validation-calibrated "
            "thresholds. Results are for research and "
            "demonstration, not production quality assurance."
        )
