"""Industrial Visual Anomaly Detection dashboard (PatchCore V1 / Metal Nut).

Run from the project directory with:
    python -m streamlit run dashboard.py
"""

from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from zipfile import ZIP_DEFLATED, ZipFile
import csv
import hashlib
import io
import json
import re

import numpy as np
from PIL import Image, ImageOps
import streamlit as st
import torch

from anomalib.engine import Engine
from anomalib.models import Patchcore

from inspection_history import (
    initialize_database,
    save_inspection_run,
    get_recent_inspections,
)


# ---------------------------- Configuration ----------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "outputs" / "patchcore" / "metal_nut"
CHECKPOINT = MODEL_DIR / "model.ckpt"
IMAGE_THRESHOLD_FILE = MODEL_DIR / "threshold.json"
PIXEL_THRESHOLD_FILE = MODEL_DIR / "pixel_threshold.json"
MODEL_VERSION = "PatchCore V1"
CATEGORY = "metal_nut"

st.set_page_config(
    page_title="Industrial AI Inspection",
    page_icon="🏭",
    layout="wide",
)

for required in (CHECKPOINT, IMAGE_THRESHOLD_FILE, PIXEL_THRESHOLD_FILE):
    if not required.is_file():
        st.error(f"Missing required model file: {required}")
        st.stop()

with IMAGE_THRESHOLD_FILE.open(encoding="utf-8") as file:
    IMAGE_THRESHOLD = float(json.load(file)["image_threshold"])
with PIXEL_THRESHOLD_FILE.open(encoding="utf-8") as file:
    PIXEL_THRESHOLD = float(json.load(file)["pixel_threshold"])


# -------------------------- Image / I/O helpers ------------------------
def read_image(image_bytes: bytes) -> Image.Image:
    """Normalize orientation and ensure 3-channel RGB input."""
    with Image.open(io.BytesIO(image_bytes)) as source:
        return ImageOps.exif_transpose(source).convert("RGB")


def upload_signature(files) -> str:
    """Invalidate old results whenever files, names, or order change."""
    digest = hashlib.sha256()
    for uploaded in files:
        data = uploaded.getvalue()
        digest.update(uploaded.name.encode("utf-8"))
        digest.update(len(data).to_bytes(8, "little"))
        digest.update(data)
    return digest.hexdigest()


def safe_stem(name: str) -> str:
    """Make a safe ZIP member name, including for duplicate filenames."""
    stem = Path(name.replace("\\", "/")).stem
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", stem).strip("_")[:80] or "image"


def to_batches(prediction_batches):
    """Flatten Anomalib's batched inference results."""
    for batch in prediction_batches:
        scores = torch.as_tensor(batch.pred_score).detach().cpu().reshape(-1)
        maps = torch.as_tensor(batch.anomaly_map).detach().cpu()

        if maps.ndim == 2:
            maps = maps.unsqueeze(0)
        elif maps.ndim == 4 and maps.shape[1] == 1:
            maps = maps[:, 0]
        if maps.ndim != 3:
            raise ValueError(f"Unexpected anomaly-map shape: {tuple(maps.shape)}")

        paths = batch.image_path
        if isinstance(paths, (str, Path)):
            paths = [paths]
        if not (len(paths) == len(scores) == len(maps)):
            raise RuntimeError("Prediction paths/scores/maps have different lengths")

        for path, score, anomaly_map in zip(paths, scores, maps):
            array = anomaly_map.numpy()
            value = float(score.item())
            if not np.isfinite(array).all() or not np.isfinite(value):
                raise ValueError("Non-finite anomaly prediction")
            yield Path(path).stem, value, array


# ------------------------- Model / inference ---------------------------
def run_inspection(files):
    """Run one model over all uploads; time the entire call, not GPU-only inference."""
    if not files:
        raise ValueError("Upload at least one image")

    started = perf_counter()
    results_by_index = {}

    with TemporaryDirectory() as temporary:
        folder = Path(temporary)
        for index, uploaded in enumerate(files):
            image = read_image(uploaded.getvalue())
            image.save(folder / f"{index:04d}.png")

        model = Patchcore(
            backbone="resnet18",
            layers=["layer2", "layer3"],
            coreset_sampling_ratio=0.1,
        )
        engine = Engine(
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            logger=False,
            default_root_dir=str(MODEL_DIR / "dashboard_runs"),
        )
        batches = engine.predict(
            model=model,
            ckpt_path=str(CHECKPOINT),
            data_path=str(folder),
            return_predictions=True,
        )
        if not batches:
            raise RuntimeError("PatchCore returned no predictions")

        for identifier, score, anomaly_map in to_batches(batches):
            index = int(identifier)
            if index in results_by_index or not 0 <= index < len(files):
                raise RuntimeError("Unexpected or repeated prediction index")
            results_by_index[index] = {
                "score": score,
                "anomaly_map": anomaly_map,
                "decision": "ANOMALY" if score > IMAGE_THRESHOLD else "NORMAL",
            }

    if len(results_by_index) != len(files):
        raise RuntimeError("The model did not return a result for every uploaded image")

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total_seconds = perf_counter() - started
    timestamp = datetime.now(timezone.utc).isoformat()

    entries = []
    for index, uploaded in enumerate(files):
        result = results_by_index[index]
        entries.append({
            "index": index,
            "filename": uploaded.name,
            "mask_file": f"masks/{index:04d}_{safe_stem(uploaded.name)}.png",
            "score": result["score"],
            "anomaly_map": result["anomaly_map"],
            "decision": result["decision"],
            "timestamp": timestamp,
        })

    return entries, total_seconds


# ------------------------------ Visualization --------------------------
def make_visuals(image: Image.Image, anomaly_map: np.ndarray):
    height, width = anomaly_map.shape
    original = image.resize((width, height), Image.Resampling.BILINEAR)

    # ONLY for the colorful display, never for thresholding.
    low, high = np.percentile(anomaly_map, (2, 98))
    normalized = np.clip((anomaly_map - low) / (high - low + 1e-8), 0, 1)
    intensity = (normalized * 255).astype(np.uint8)
    green = np.clip(
        255 - 2 * np.abs(intensity.astype(np.int16) - 127), 0, 255
    ).astype(np.uint8)
    heatmap = Image.fromarray(np.stack(
        (intensity, green, 255 - intensity), axis=-1
    ))
    overlay = Image.blend(original, heatmap, alpha=0.40)

    # Use the RAW anomaly map and the calibrated pixel threshold.
    mask = Image.fromarray(
        ((anomaly_map > PIXEL_THRESHOLD).astype(np.uint8) * 255)
    )
    return original, heatmap, overlay, mask


def show_inspection(entry: dict, image_bytes: bytes):
    a, b, c = st.columns(3)
    a.metric("Decision", entry["decision"])
    b.metric("Anomaly Score", f'{entry["score"]:.4f}')
    c.metric("Image Threshold", f"{IMAGE_THRESHOLD:.4f}")

    if entry["decision"] == "ANOMALY":
        st.error("Suspicious visual patterns detected — human review recommended.")
    else:
        st.success("Classified as normal by this model.")

    visuals = make_visuals(read_image(image_bytes), entry["anomaly_map"])
    columns = st.columns(4)
    for column, visual, caption in zip(
        columns,
        visuals,
        ("Original", "Anomaly Heatmap", "Heatmap Overlay", "Binary Mask"),
    ):
        column.image(visual, caption=caption, use_container_width=True)

    st.caption(
        "Heatmap colors are normalized per image for display. "
        "Binary masks use the raw scores and the fixed pixel threshold. "
        "Anomaly scores are not probabilities."
    )
    with st.expander("Inspection details"):
        st.write(f'**Source:** {entry["filename"]}')
        st.write(f"**Model:** {MODEL_VERSION}")
        st.write(f"**Category:** {CATEGORY}")
        st.write(f"**Pixel threshold:** {PIXEL_THRESHOLD:.6f}")
        st.write(f'**Inspected at (UTC):** {entry["timestamp"]}')


# --------------------------- Report generation -------------------------
def create_exports(entries, total_seconds: float):
    csv_file = io.StringIO()
    columns = [
        "timestamp_utc", "image", "category", "model_version", "decision",
        "anomaly_score", "image_threshold", "pixel_threshold", "mask_file",
        "batch_end_to_end_seconds", "average_seconds_per_image",
    ]
    writer = csv.DictWriter(csv_file, fieldnames=columns)
    writer.writeheader()

    zip_file = io.BytesIO()
    with ZipFile(zip_file, mode="w", compression=ZIP_DEFLATED) as archive:
        for entry in entries:
            writer.writerow({
                "timestamp_utc": entry["timestamp"],
                "image": entry["filename"],
                "category": CATEGORY,
                "model_version": MODEL_VERSION,
                "decision": entry["decision"],
                "anomaly_score": entry["score"],
                "image_threshold": IMAGE_THRESHOLD,
                "pixel_threshold": PIXEL_THRESHOLD,
                "mask_file": entry["mask_file"],
                "batch_end_to_end_seconds": round(total_seconds, 6),
                "average_seconds_per_image": round(total_seconds / len(entries), 6),
            })
            array = (entry["anomaly_map"] > PIXEL_THRESHOLD).astype(np.uint8) * 255
            mask_buffer = io.BytesIO()
            Image.fromarray(array).save(mask_buffer, format="PNG")
            archive.writestr(entry["mask_file"], mask_buffer.getvalue())

    return csv_file.getvalue(), zip_file.getvalue()


# ---------------------------------- UI ---------------------------------
st.title("Industrial Visual Anomaly Detection")
st.caption("Industrial inspection | PatchCore V1 | Metal Nut")

# SQLite stores inspection metadata between app restarts.
try:
    initialize_database()
except Exception as error:
    st.error(f"Could not initialize inspection history: {error}")
    st.stop()

single_tab, batch_tab, history_tab = st.tabs([
    "Single Inspection", "Batch Inspection", "Inspection History",
])

with single_tab:
    st.subheader("Inspect a Product")
    uploaded = st.file_uploader(
        "Upload one metal nut image",
        type=["png", "jpg", "jpeg", "bmp"],
        key="single_file",
    )
    if uploaded is not None:
        uploaded_bytes = uploaded.getvalue()
        try:
            st.image(read_image(uploaded_bytes), width=300)
        except Exception as error:
            st.error(f"Unable to open image: {error}")
            st.stop()

        signature = upload_signature([uploaded])
        if st.button("Inspect Product", type="primary", key="inspect_single"):
            try:
                with st.spinner("Inspecting image..."):
                    entries, elapsed = run_inspection([uploaded])
                # Save session results first, even if SQLite is unavailable.
                st.session_state["single_result"] = {
                    "signature": signature,
                    "entries": entries,
                    "elapsed": elapsed,
                    "saved_run_id": None,
                }
                # Only runs after the user clicks Inspect, not on Streamlit reruns.
                try:
                    run_id = save_inspection_run(
                        entries=entries,
                        files=[uploaded],
                        elapsed_seconds=elapsed,
                        mode="single",
                        image_threshold=IMAGE_THRESHOLD,
                        pixel_threshold=PIXEL_THRESHOLD,
                        model_version=MODEL_VERSION,
                    )
                    st.session_state["single_result"]["saved_run_id"] = run_id
                except Exception as db_error:
                    st.warning(f"Inspection worked, but history was not saved: {db_error}")
            except Exception as error:
                st.error(f"Inspection failed: {error}")

        saved = st.session_state.get("single_result")
        if saved and saved["signature"] == signature:
            st.divider()
            st.subheader("Inspection Result")
            show_inspection(saved["entries"][0], uploaded_bytes)
            st.metric("End-to-End Inspection Time", f'{saved["elapsed"]:.3f} s')
            st.caption("Includes image preparation, model loading, and inference.")

            csv_data, zip_data = create_exports(saved["entries"], saved["elapsed"])
            left, right = st.columns(2)
            left.download_button(
                "Download Inspection CSV", csv_data,
                file_name="inspection.csv", mime="text/csv", on_click="ignore",
                use_container_width=True,
            )
            right.download_button(
                "Download Mask ZIP", zip_data,
                file_name="inspection_mask.zip", mime="application/zip",
                on_click="ignore", use_container_width=True,
            )

with batch_tab:
    st.subheader("Inspect Multiple Products")
    files = st.file_uploader(
        "Upload multiple metal nut images",
        type=["png", "jpg", "jpeg", "bmp"],
        accept_multiple_files=True,
        key="batch_files",
    )
    if files:
        st.write(f"Selected images: **{len(files)}**")
        signature = upload_signature(files)
        if st.button("Inspect All", type="primary", key="inspect_batch"):
            try:
                with st.spinner("Inspecting batch..."):
                    entries, elapsed = run_inspection(files)
                # Save session results first, even if SQLite is unavailable.
                st.session_state["batch_result"] = {
                    "signature": signature,
                    "entries": entries,
                    "elapsed": elapsed,
                    "saved_run_id": None,
                }
                try:
                    run_id = save_inspection_run(
                        entries=entries,
                        files=files,
                        elapsed_seconds=elapsed,
                        mode="batch",
                        image_threshold=IMAGE_THRESHOLD,
                        pixel_threshold=PIXEL_THRESHOLD,
                        model_version=MODEL_VERSION,
                    )
                    st.session_state["batch_result"]["saved_run_id"] = run_id
                except Exception as db_error:
                    st.warning(f"Inspection worked, but history was not saved: {db_error}")
            except Exception as error:
                st.error(f"Batch inspection failed: {error}")

        saved = st.session_state.get("batch_result")
        if saved and saved["signature"] == signature:
            entries = saved["entries"]
            elapsed = saved["elapsed"]
            flagged = sum(entry["decision"] == "ANOMALY" for entry in entries)
            st.divider()
            st.subheader("Batch Summary")

            # FIXED: five values returned, five variables unpacked.
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Total Images", len(entries))
            c2.metric("Anomalies Flagged", flagged)
            c3.metric("Normal Images", len(entries) - flagged)
            c4.metric("Batch Processing Time", f"{elapsed:.2f} s")
            c5.metric("Average / Image", f"{elapsed / len(entries):.3f} s")
            st.caption(
                "Times are end-to-end (including one model load); "
                "average time is total batch duration divided by the image count, "
                "not warm-model per-image latency."
            )

            st.dataframe(
                [{
                    "Image": e["filename"],
                    "Decision": e["decision"],
                    "Anomaly Score": round(e["score"], 4),
                    "Mask File": e["mask_file"],
                } for e in entries],
                use_container_width=True,
                hide_index=True,
            )

            st.subheader("Inspect an Individual Result")
            selected = st.selectbox(
                "Choose an image from the batch",
                options=range(len(entries)),
                format_func=lambda i: (
                    f'{i + 1}. {entries[i]["filename"]} — {entries[i]["decision"]}'
                ),
                key="selected_batch_item",
            )
            show_inspection(entries[selected], files[selected].getvalue())

            st.divider()
            st.subheader("Export Batch Results")
            csv_data, zip_data = create_exports(entries, elapsed)
            left, right = st.columns(2)
            left.download_button(
                "Download Batch CSV", csv_data,
                file_name="batch_results.csv", mime="text/csv", on_click="ignore",
                use_container_width=True,
            )
            right.download_button(
                "Download Masks ZIP", zip_data,
                file_name="batch_masks.zip", mime="application/zip",
                on_click="ignore", use_container_width=True,
            )

# ---------------------------- History tab ------------------------------
with history_tab:
    st.subheader("Inspection History")
    st.caption(
        "Previous inspection metadata stored locally in SQLite. "
        "Original image bytes, heatmaps and masks are not stored in history."
    )

    max_rows = st.selectbox(
        "Recent inspection records to load",
        options=[100, 250, 500, 1000],
        index=0,
        key="history_limit",
    )
    decision_filter = st.selectbox(
        "Filter by decision",
        options=["All", "ANOMALY", "NORMAL"],
        index=0,
        key="history_decision_filter",
    )

    try:
        history_rows = get_recent_inspections(limit=max_rows)
    except Exception as error:
        st.error(f"Unable to load history: {error}")
        history_rows = []

    if decision_filter != "All":
        history_rows = [
            row for row in history_rows if row["decision"] == decision_filter
        ]

    if not history_rows:
        st.info("No saved inspections match this filter. Run a new inspection first.")
    else:
        flagged = sum(row["decision"] == "ANOMALY" for row in history_rows)
        m1, m2, m3 = st.columns(3)
        m1.metric("Records displayed", len(history_rows))
        m2.metric("Anomalies flagged", flagged)
        m3.metric("Normal results", len(history_rows) - flagged)

        # Preserve the SQLite result, including timestamps, thresholds and timing.
        st.dataframe(
            history_rows,
            use_container_width=True,
            hide_index=True,
        )

        csv_buffer = io.StringIO()
        fieldnames = list(history_rows[0].keys())
        writer = csv.DictWriter(csv_buffer, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history_rows)
        st.download_button(
            "Download Filtered History CSV",
            data=csv_buffer.getvalue(),
            file_name="inspection_history.csv",
            mime="text/csv",
            on_click="ignore",
            use_container_width=True,
        )
        st.caption(
            "History shows the most recent records up to the selected limit. "
            "The decision filter is applied to those loaded records."
        )

st.divider()
st.caption(
    "Research demonstration using MVTec AD. Predictions need human review; "
    "this system has not been validated for production quality control."
)
