import tempfile

import cv2
import numpy as np
import streamlit as st
from huggingface_hub import hf_hub_download
from PIL import Image
from ultralytics.models.sam import SAM3SemanticPredictor


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


def run_segmentation(predictor, image: Image.Image) -> Image.Image | None:
    """Run SAM3 segmentation and return image with mask overlay."""
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        image.save(f, format="JPEG")
        temp_path = f.name

    predictor.set_image(temp_path)
    results = predictor(text=["person"])

    original = np.array(image)
    overlay = original.copy()

    if results and results[0].masks is not None:
        masks = results[0].masks.data.cpu().numpy()

        for mask in masks:
            if mask.shape != original.shape[:2]:
                mask = cv2.resize(
                    mask.astype(np.uint8),
                    (original.shape[1], original.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            else:
                mask = mask.astype(bool)

            color = (0, 255, 0)
            overlay[mask] = (
                overlay[mask] * 0.5 + np.array(color) * 0.5
            ).astype(np.uint8)

            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(overlay, contours, -1, color, 2)

        return Image.fromarray(overlay)
    return None


st.set_page_config(page_title="Human Segmentation", layout="wide")
st.title("Human Segmentation with SAM3")

try:
    with st.spinner("Loading SAM3 model..."):
        predictor = load_model()
except Exception as e:
    st.error(f"Failed to load SAM3 model: {e}")
    st.stop()

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original")
        st.image(image, use_container_width=True)
    with col2:
        st.subheader("Segmentation")
        with st.spinner("Segmenting humans..."):
            result_image = run_segmentation(predictor, image)
        if result_image is not None:
            st.image(result_image, use_container_width=True)
        else:
            st.warning("No humans detected in the image.")
