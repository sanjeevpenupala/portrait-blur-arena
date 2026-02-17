import tempfile
import time

import cv2
import numpy as np
import streamlit as st
from huggingface_hub import hf_hub_download
from PIL import Image
from ultralytics import YOLO
from ultralytics.models.sam import SAM2Predictor, SAM3SemanticPredictor


def merge_masks(masks: np.ndarray, h: int, w: int) -> np.ndarray:
    """Merge multiple masks into a single binary mask at (h, w) resolution."""
    merged = np.zeros((h, w), dtype=np.uint8)
    for mask in masks:
        if mask.shape != (h, w):
            mask = cv2.resize(
                mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
            )
        merged = np.maximum(merged, mask.astype(np.uint8))
    return merged


def render_segmentation_overlay(original: np.ndarray, merged: np.ndarray) -> np.ndarray:
    """Draw green overlay and contours on the image from a merged binary mask."""
    overlay = original.copy()

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(merged, cv2.MORPH_CLOSE, kernel)

    bool_mask = closed.astype(bool)
    color = (0, 255, 0)
    overlay[bool_mask] = (overlay[bool_mask] * 0.5 + np.array(color) * 0.5).astype(
        np.uint8
    )

    dilated = cv2.dilate(closed, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, color, 2)

    return overlay


def blur_background(
    original: np.ndarray, merged: np.ndarray, blur_strength: int = 51
) -> np.ndarray:
    """Blur the background using the raw mask with feathered edges."""
    h, w = original.shape[:2]

    # Resize mask to full resolution with LINEAR interpolation for soft edges
    if merged.shape != (h, w):
        merged = cv2.resize(merged, (w, h), interpolation=cv2.INTER_LINEAR)

    # Convert to float alpha mask [0, 1]
    alpha = merged.astype(np.float32)
    if alpha.max() > 1:
        alpha = alpha / 255.0

    # Feather the edges with Gaussian blur on the mask itself
    alpha = cv2.GaussianBlur(alpha, (15, 15), sigmaX=5)

    # Blur the full image
    blurred = cv2.GaussianBlur(original, (blur_strength, blur_strength), sigmaX=0)

    # Blend: person stays sharp, background gets blurred
    alpha_3ch = alpha[:, :, np.newaxis]
    result = (original * alpha_3ch + blurred * (1 - alpha_3ch)).astype(np.uint8)

    return result


@st.cache_resource
def load_model():
    """Download SAM3 weights and initialize the predictor."""
    model_path = hf_hub_download(repo_id="1038lab/sam3", filename="sam3.pt")
    overrides = dict(
        conf=0.25,
        task="segment",
        mode="predict",
        model=model_path,
    )
    return SAM3SemanticPredictor(overrides=overrides)


def run_segmentation(predictor, image: Image.Image):
    """Run SAM3 segmentation. Returns (overlay, merged_mask, elapsed) or (None, None, elapsed)."""
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        image.save(f, format="JPEG")
        temp_path = f.name

    original = np.array(image)
    h, w = original.shape[:2]

    start = time.perf_counter()
    predictor.set_image(temp_path)
    results = predictor(text=["person"])
    elapsed = time.perf_counter() - start

    if results and results[0].masks is not None:
        masks = results[0].masks.data.cpu().numpy()
        merged = merge_masks(masks, h, w)
        overlay = render_segmentation_overlay(original, merged)
        return Image.fromarray(overlay), merged, elapsed
    return None, None, elapsed


@st.cache_resource
def load_sam2_models():
    """Load YOLO (person detector) and SAM2 predictor."""
    yolo = YOLO("yolo11n.pt")
    sam2 = SAM2Predictor(overrides=dict(model="sam2.1_l.pt"))
    return yolo, sam2


def run_sam2_segmentation(yolo, sam2, image: Image.Image):
    """YOLO person detection → SAM2 segmentation. Returns (overlay, blurred, elapsed) or (None, None, elapsed)."""
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        image.save(f, format="JPEG")
        temp_path = f.name

    original = np.array(image)
    h, w = original.shape[:2]

    start = time.perf_counter()
    detections = yolo(temp_path, classes=[0], verbose=False)
    boxes = detections[0].boxes

    if boxes is None or len(boxes) == 0:
        elapsed = time.perf_counter() - start
        return None, None, elapsed

    sam2.set_image(temp_path)
    results = sam2(bboxes=boxes.xyxy)
    elapsed = time.perf_counter() - start

    if results and results[0].masks is not None:
        masks = results[0].masks.data.cpu().numpy()
        merged = merge_masks(masks, h, w)
        overlay = render_segmentation_overlay(original, merged)
        return Image.fromarray(overlay), merged, elapsed
    return None, None, elapsed


st.set_page_config(page_title="Portrait Background Blur", layout="wide")
st.title("Portrait Background Blur — SAM 2.1 vs SAM3")

try:
    with st.spinner("Loading SAM3 model (downloads ~3.2 GB on first run)..."):
        sam3_predictor = load_model()
    with st.spinner("Loading YOLO + SAM 2.1 Large (downloads on first run)..."):
        yolo, sam2_predictor = load_sam2_models()
except Exception as e:
    st.error(f"Failed to load models: {e}")
    st.stop()

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    original = np.array(image)

    st.image(image, caption="Original", width=400)

    with st.spinner("Running segmentation..."):
        sam2_overlay, sam2_mask, sam2_time = run_sam2_segmentation(
            yolo, sam2_predictor, image
        )
        sam3_overlay, sam3_mask, sam3_time = run_segmentation(sam3_predictor, image)

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("SAM 2.1")
        st.caption(f"Inference: {sam2_time:.2f}s")
        if sam2_overlay is not None:
            st.image(sam2_overlay, caption="Mask Overlay", use_container_width=True)
        else:
            st.warning("No humans detected.")

    with right:
        st.subheader("SAM3")
        st.caption(f"Inference: {sam3_time:.2f}s")
        if sam3_overlay is not None:
            st.image(sam3_overlay, caption="Mask Overlay", use_container_width=True)
        else:
            st.warning("No humans detected.")

    if sam2_mask is not None or sam3_mask is not None:
        st.divider()
        blur_strength = st.slider(
            "Blur Strength", min_value=5, max_value=101, value=51, step=2
        )
        st.write("")

        blur_left, blur_right = st.columns(2)

        with blur_left:
            if sam2_mask is not None:
                sam2_blur = blur_background(original, sam2_mask, blur_strength)
                st.image(sam2_blur, caption="SAM 2.1 — Background Blur", use_container_width=True)

        with blur_right:
            if sam3_mask is not None:
                sam3_blur = blur_background(original, sam3_mask, blur_strength)
                st.image(sam3_blur, caption="SAM3 — Background Blur", use_container_width=True)
