# Human Segmentation App Design

## Goal

Streamlit app that lets users upload an image and see all humans segmented using SAM3 via ultralytics.

## Architecture

Single-file Streamlit app (`app.py`). No backend, no database.

## Model

- **SAM3SemanticPredictor** from `ultralytics.models.sam`
- Text prompt: `["person"]`
- Config: `conf=0.25, task="segment", mode="predict", model="sam3.pt", half=True`
- Weights auto-downloaded from `huggingface_hub` (`1038lab/sam3`) on first run
- Model cached with `@st.cache_resource` to avoid reloading per interaction

## UI Layout

1. Title and brief description
2. `st.file_uploader` accepting jpg, jpeg, png
3. Two columns (`st.columns(2)`):
   - Left: original uploaded image
   - Right: image with segmentation overlay

## Segmentation Pipeline

1. Read uploaded file into PIL Image
2. Save to temp file (ultralytics expects file path)
3. `predictor.set_image(temp_path)`
4. `results = predictor(text=["person"])`
5. Extract masks from results
6. Overlay masks on original image: semi-transparent colored fill + contour outline (OpenCV)
7. Display both images side-by-side

## Dependencies (via `uv add`)

- `ultralytics` — SAM3 interface
- `streamlit` — UI
- `opencv-python-headless` — mask overlay and contour drawing
- `Pillow` — image handling
- `huggingface-hub` — auto-download model weights
- `torch`, `torchvision` — model backend

## Model Weight Download

Use `huggingface_hub.hf_hub_download` to fetch `sam3.safetensors` from `1038lab/sam3` into a local cache. Convert or reference as needed for ultralytics.

## Error Handling

- Show `st.warning` if no humans detected
- Show `st.error` if model fails to load
- Show spinner during inference
