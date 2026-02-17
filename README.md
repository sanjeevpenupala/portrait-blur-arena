# Portrait Background Blur

A Streamlit app that segments humans and blurs the background, comparing two segmentation models side-by-side: [SAM3](https://huggingface.co/1038lab/sam3) (text-prompted) and [SAM 2.1](https://docs.ultralytics.com/models/sam-2/) (YOLO-detected bounding boxes), both via [Ultralytics](https://docs.ultralytics.com/).

Upload a photo and the app displays:
- **Mask overlays** for both models with green highlight and contour outlines
- **Background blur** results with an adjustable blur strength slider
- **Inference timing** for each model

## Prerequisites

- **Python 3.13+**
- **[uv](https://docs.astral.sh/uv/)** — Python package manager:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
- **Disk space** — Model weights are downloaded automatically on first run:
  - SAM3 (~3.2 GB) — cached by HuggingFace Hub in `~/.cache/huggingface/`
  - SAM 2.1 Large + YOLO11n — cached by Ultralytics

## Getting Started

1. **Clone the repository**

   ```bash
   git clone https://github.com/sanjeevpenupala/sam-portrait-background-blur.git
   cd sam-portrait-background-blur
   ```

2. **Install dependencies**

   ```bash
   uv sync
   ```

3. **Run the app**

   ```bash
   uv run streamlit run app.py
   ```

   Opens in your browser at `http://localhost:8501`.

4. **Upload an image** and compare the results. Use the blur strength slider to adjust the background blur intensity.

## How It Works

### Segmentation

- **SAM3** — Uses text-prompted semantic segmentation (`["person"]`). Downloads `sam3.pt` from HuggingFace.
- **SAM 2.1** — Uses YOLO11n to detect people (bounding boxes), then feeds those boxes to SAM 2.1 Large for segmentation. Downloads `yolo11n.pt` and `sam2.1_l.pt` via Ultralytics.

### Mask Overlay

All detected masks are merged into a single binary mask, cleaned with morphological closing, and rendered as a semi-transparent green overlay with external contour outlines.

### Background Blur

The raw binary mask is converted to a soft alpha channel via Gaussian blur (feathered edges), then used to blend the sharp foreground with a Gaussian-blurred background. The blur kernel size is adjustable via the UI slider.

## Performance Notes

- **GPU (CUDA)** — PyTorch uses it automatically. Inference takes a few seconds.
- **CPU only** — Works but slower (10-30+ seconds per model).
- **Apple Silicon (MPS)** — PyTorch MPS backend may be used automatically.

## Project Structure

```
portrait-background-blur/
├── app.py              # Streamlit app (models, segmentation, blur, UI)
├── pyproject.toml      # Project metadata and dependencies
├── uv.lock             # Locked dependency versions
└── .python-version     # Python version (3.13)
```

## Dependencies

| Package | Purpose |
|---------|---------|
| `streamlit` | Web UI framework |
| `ultralytics` | SAM3, SAM 2.1, and YOLO model interfaces |
| `opencv-python-headless` | Mask processing, contour drawing, blur |
| `Pillow` | Image I/O |
| `huggingface-hub` | Auto-download SAM3 weights |
| `torch` / `torchvision` | Model backend |

## Troubleshooting

**"Failed to load models"**
- Check your internet connection (models download on first run).
- For SAM3, clear partial cache: `rm -rf ~/.cache/huggingface/hub/models--1038lab--sam3/`

**"No humans detected"**
- SAM3 uses a confidence threshold of 0.25. SAM 2.1 depends on YOLO detection.
- Try a different image with more clearly visible people.

**Slow inference**
- Expected on CPU. Use a CUDA GPU for faster results.
- Large images take longer.
