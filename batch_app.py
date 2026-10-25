
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile, ZIP_DEFLATED
import csv
import hashlib
import io
import json

import numpy as np
import streamlit as st
import torch
from PIL import Image, ImageOps

from anomalib.engine import Engine
from anomalib.models import Patchcore


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "outputs/patchcore/metal_nut"

CHECKPOINT = MODEL_DIR / "model.ckpt"

IMAGE_THRESHOLD = json.loads(
    (MODEL_DIR / "threshold.json").read_text()
)["image_threshold"]

PIXEL_THRESHOLD = json.loads(
    (MODEL_DIR / "pixel_threshold.json").read_text()
)["pixel_threshold"]


st.set_page_config(
    page_title="Batch Industrial Inspection",
    page_icon="🏭",
    layout="wide",
)

st.title("Batch Industrial Inspection")
st.caption("PatchCore V1 | Metal Nut")

uploaded = st.file_uploader(
    "Upload multiple metal nut images",
    type=["png", "jpg", "jpeg", "bmp"],
    accept_multiple_files=True,
)

if uploaded:
    st.write(f"Selected images: **{len(uploaded)}**")

    signature = hashlib.sha256()

    for file in uploaded:
        signature.update(file.name.encode())
        signature.update(file.getvalue())

    upload_signature = signature.hexdigest()

    if st.button("Inspect All Images", type="primary"):

        with st.spinner("Running batch inspection..."):

            try:
                with TemporaryDirectory() as tmp:
                    tmp_dir = Path(tmp)
                    filename_map = {}

                    # Prepare unique filenames
                    for index, file in enumerate(uploaded):
                        image = ImageOps.exif_transpose(
                            Image.open(
                                io.BytesIO(file.getvalue())
                            )
                        ).convert("RGB")

                        temp_name = f"{index:04d}.png"
                        image.save(tmp_dir / temp_name)

                        filename_map[temp_name] = file.name

                    # Load PatchCore once
                    model = Patchcore(
                        backbone="resnet18",
                        layers=["layer2", "layer3"],
                        coreset_sampling_ratio=0.1,
                    )

                    engine = Engine(
                        accelerator=(
                            "gpu"
                            if torch.cuda.is_available()
                            else "cpu"
                        ),
                        devices=1,
                        logger=False,
                        default_root_dir=str(
                            MODEL_DIR / "batch_runs"
                        ),
                    )

                    predictions = engine.predict(
                        model=model,
                        ckpt_path=str(CHECKPOINT),
                        data_path=str(tmp_dir),
                        return_predictions=True,
                    )

                    assert predictions is not None

                    rows = []
                    masks = {}

                    for batch in predictions:
                        scores = torch.as_tensor(
                            batch.pred_score
                        ).detach().cpu().reshape(-1)

                        maps = torch.as_tensor(
                            batch.anomaly_map
                        ).detach().cpu()

                        if maps.ndim == 2:
                            maps = maps.unsqueeze(0)
                        elif (
                            maps.ndim == 4
                            and maps.shape[1] == 1
                        ):
                            maps = maps[:, 0]

                        assert maps.ndim == 3

                        paths = batch.image_path

                        if isinstance(paths, (str, Path)):
                            paths = [paths]

                        assert len(paths) == len(scores)
                        assert len(paths) == len(maps)

                        for path, score, anomaly_map in zip(
                            paths, scores, maps
                        ):
                            temp_name = Path(path).name
                            filename = filename_map[temp_name]

                            score = float(score.item())
                            anomaly_map = anomaly_map.numpy()

                            if not np.isfinite(
                                anomaly_map
                            ).all():
                                raise ValueError(
                                    "Invalid anomaly map"
                                )

                            decision = (
                                "ANOMALY"
                                if score > IMAGE_THRESHOLD
                                else "NORMAL"
                            )

                            binary = (
                                anomaly_map > PIXEL_THRESHOLD
                            ).astype(np.uint8) * 255

                            mask = Image.fromarray(binary)

                            mask_buffer = io.BytesIO()
                            mask.save(
                                mask_buffer,
                                format="PNG",
                            )

                            masks[temp_name] = (
                                mask_buffer.getvalue()
                            )

                            rows.append({
                                "image": filename,
                                "decision": decision,
                                "anomaly_score": score,
                                "image_threshold": IMAGE_THRESHOLD,
                                "pixel_threshold": PIXEL_THRESHOLD,
                                "model": "PatchCore",
                                "model_version": "v1",
                            })

                    assert len(rows) == len(uploaded)
                    assert len(masks) == len(uploaded)

                    # Generate CSV
                    csv_buffer = io.StringIO()
                    writer = csv.DictWriter(
                        csv_buffer,
                        fieldnames=list(rows[0].keys()),
                    )
                    writer.writeheader()
                    writer.writerows(rows)

                    # Generate ZIP of binary masks
                    zip_buffer = io.BytesIO()

                    with ZipFile(
                        zip_buffer,
                        "w",
                        ZIP_DEFLATED,
                    ) as archive:
                        for name, data in sorted(masks.items()):
                            archive.writestr(
                                f"masks/{name}",
                                data,
                            )

                    st.session_state["batch_results"] = {
                        "signature": upload_signature,
                        "rows": rows,
                        "csv": csv_buffer.getvalue(),
                        "zip": zip_buffer.getvalue(),
                    }

            except Exception as error:
                st.error(f"Batch inspection failed: {error}")
                st.stop()

    result = st.session_state.get("batch_results")

    if (
        result is not None
        and result["signature"] == upload_signature
    ):
        rows = result["rows"]

        anomaly_count = sum(
            row["decision"] == "ANOMALY"
            for row in rows
        )

        normal_count = len(rows) - anomaly_count

        st.divider()
        st.subheader("Batch Results")

        c1, c2, c3 = st.columns(3)

        c1.metric("Total inspected", len(rows))
        c2.metric("Anomalies flagged", anomaly_count)
        c3.metric("Normal", normal_count)

        st.dataframe(
            rows,
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Export Results")

        left, right = st.columns(2)

        with left:
            st.download_button(
                "Download Batch CSV",
                data=result["csv"],
                file_name="batch_results.csv",
                mime="text/csv",
                on_click="ignore",
                use_container_width=True,
            )

        with right:
            st.download_button(
                "Download All Masks ZIP",
                data=result["zip"],
                file_name="batch_masks.zip",
                mime="application/zip",
                on_click="ignore",
                use_container_width=True,
            )

        st.caption(
            "All results use PatchCore V1 and fixed "
            "validation-calibrated thresholds. "
            "Flagged anomalies require human review."
        )
