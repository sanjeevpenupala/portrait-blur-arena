# SAM3 Human Segmentation App — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a Streamlit app that accepts an uploaded image and displays the original alongside a segmentation overlay of all detected humans using SAM3.

**Architecture:** Single-file Streamlit app (`app.py`). Model loaded once via `@st.cache_resource`. Segmentation via `SAM3SemanticPredictor` with text prompt `["person"]`. Mask overlay rendered with OpenCV.

**Tech Stack:** Python 3.13, Streamlit, ultralytics (SAM3), OpenCV, Pillow, huggingface_hub, PyTorch

---

### Task 1: Install Dependencies

**Step 1: Add all required packages**

```bash
uv add streamlit ultralytics opencv-python-headless Pillow huggingface-hub torch torchvision
```

Note: `torch` and `torchvision` are large. This will take a minute.

**Step 2: Verify installation**

```bash
uv run python -c "import streamlit; import ultralytics; import cv2; import PIL; import huggingface_hub; print('All imports OK')"
```

Expected: `All imports OK`

**Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "Add project dependencies"
```

---

### Task 2: Model Loading with Auto-Download

**Files:**
- Create: `app.py` (replace existing `main.py` scaffold)

**Step 1: Write model loading function**

Write `app.py` with the cached model loader. This function downloads `sam3.pt` from HuggingFace on first run and initializes `SAM3SemanticPredictor`.

```python
import streamlit as st
from huggingface_hub import hf_hub_download
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
```

Note on `half=True`: Only use this if running on CUDA GPU. On CPU/MPS it will either error or be slower. We'll detect device and set accordingly. For now, omit `half` and let ultralytics handle dtype.

**Step 2: Add a minimal Streamlit page to verify model loads**

```python
st.set_page_config(page_title="Human Segmentation", layout="wide")
st.title("Human Segmentation with SAM3")

with st.spinner("Loading SAM3 model..."):
    predictor = load_model()

st.success("Model loaded.")
```

**Step 3: Run the app to verify model downloads and loads**

```bash
uv run streamlit run app.py
```

Expected: Browser opens, spinner shows while downloading weights (~2GB), then "Model loaded." appears. Kill with Ctrl+C after verifying.

**Step 4: Commit**

```bash
git add app.py
git commit -m "Add SAM3 model loading with auto-download"
```

---

### Task 3: File Upload UI

**Files:**
- Modify: `app.py`

**Step 1: Add file uploader and image display**

After the model loading section, add:

```python
uploaded_file = st.file_uploader(
    "Upload an image", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    from PIL import Image

    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original")
        st.image(image, use_container_width=True)
    with col2:
        st.subheader("Segmentation")
        st.info("Segmentation will appear here.")
```

**Step 2: Run and verify**

```bash
uv run streamlit run app.py
```

Expected: Upload an image, see it displayed in the left column with placeholder text in the right column.

**Step 3: Commit**

```bash
git add app.py
git commit -m "Add image upload and two-column layout"
```

---

### Task 4: Segmentation Pipeline

**Files:**
- Modify: `app.py`

**Step 1: Add segmentation and overlay logic**

Replace the placeholder in the right column with the actual segmentation pipeline:

```python
import tempfile
import numpy as np
import cv2
from PIL import Image


def run_segmentation(predictor, image: Image.Image) -> Image.Image:
    """Run SAM3 segmentation and return image with mask overlay."""
    # Save to temp file — ultralytics expects a file path
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        image.save(f, format="JPEG")
        temp_path = f.name

    predictor.set_image(temp_path)
    results = predictor(text=["person"])

    # Convert original to numpy (BGR for OpenCV)
    original = np.array(image)
    overlay = original.copy()

    if results and results[0].masks is not None:
        masks = results[0].masks.data.cpu().numpy()

        for mask in masks:
            # Resize mask to image dimensions if needed
            if mask.shape != original.shape[:2]:
                mask = cv2.resize(
                    mask.astype(np.uint8),
                    (original.shape[1], original.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            else:
                mask = mask.astype(bool)

            # Semi-transparent green fill
            color = (0, 255, 0)
            overlay[mask] = (
                overlay[mask] * 0.5 + np.array(color) * 0.5
            ).astype(np.uint8)

            # Draw contour outline
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(overlay, contours, -1, color, 2)

        return Image.fromarray(overlay)
    else:
        return None
```

**Step 2: Wire it into the UI**

Replace the `st.info` placeholder in the right column:

```python
    with col2:
        st.subheader("Segmentation")
        with st.spinner("Segmenting humans..."):
            result_image = run_segmentation(predictor, image)
        if result_image is not None:
            st.image(result_image, use_container_width=True)
        else:
            st.warning("No humans detected in the image.")
```

**Step 3: Run and verify with a test image**

```bash
uv run streamlit run app.py
```

Upload a photo with one or more people. Expected: left shows original, right shows same image with green semi-transparent overlay and contour outlines on detected humans.

**Step 4: Commit**

```bash
git add app.py
git commit -m "Add SAM3 segmentation pipeline with mask overlay"
```

---

### Task 5: Clean Up and Final Polish

**Files:**
- Modify: `app.py`
- Delete: `main.py` (replaced by `app.py`)

**Step 1: Add error handling for model load failure**

Wrap model loading in try/except:

```python
try:
    with st.spinner("Loading SAM3 model..."):
        predictor = load_model()
except Exception as e:
    st.error(f"Failed to load SAM3 model: {e}")
    st.stop()
```

**Step 2: Delete unused `main.py`**

```bash
rm main.py
```

**Step 3: Run final verification**

```bash
uv run streamlit run app.py
```

Test with:
- Image with one person
- Image with multiple people
- Image with no people (should show warning)

**Step 4: Commit**

```bash
git add -A
git commit -m "Add error handling and remove unused main.py"
```
