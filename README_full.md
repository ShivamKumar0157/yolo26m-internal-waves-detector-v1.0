# 🌊 YOLO26m Internal Waves Detector

A deep-learning pipeline that detects **Internal Waves (IW)** in SAR (Synthetic Aperture Radar) satellite imagery using a custom-trained YOLO26m object detection model, deployed as a live, interactive web app.

**🔗 Live demo:** [yolo26m-app-waves-detector-v10-klatih5z5rnibcjn9ucvwp.streamlit.app](https://yolo26m-app-waves-detector-v10-klatih5z5rnibcjn9ucvwp.streamlit.app/)

---

## 🧭 Why Detect Internal Waves?

Internal waves are gravity waves that propagate *within* the ocean, along density boundaries (pycnoclines) between layers of different temperature and salinity, rather than on the sea surface. Although they're subsurface phenomena, they leave a visible signature on the ocean surface that SAR satellites can capture as characteristic bright/dark streak patterns.

Detecting and mapping internal waves matters for several reasons:

- **Ocean mixing & climate modeling** — Internal waves are a major mechanism for transporting heat, nutrients, and energy between ocean layers, influencing regional and global climate models.
- **Submarine & underwater navigation** — Strong internal waves create sharp density gradients that can affect sonar propagation and submarine buoyancy, making their detection valuable for naval and underwater operations.
- **Offshore engineering** — Oil rigs, underwater pipelines, and cables can experience significant structural stress from internal wave-induced currents; early detection supports safer offshore infrastructure planning.
- **Marine biology & fisheries** — Internal waves drive nutrient upwelling, which affects plankton distribution and, in turn, fish aggregation — useful for fisheries research.
- **Remote sensing automation** — Manually scanning SAR imagery for internal wave signatures is slow and inconsistent; an automated detector enables large-scale, repeatable monitoring across ocean basins.

This project automates that detection step using a custom-trained object detector, turning raw SAR imagery into labeled, quantified detections with confidence scores.

---

## 🛰️ Data Preprocessing Pipeline

Raw SAR data cannot be fed directly into a YOLO model — it needs to go through a preprocessing pipeline to convert calibrated radar data into clean, trainable images:

| Step | Purpose |
|---|---|
| **1. Thermal Noise Removal** | Eliminates background interference inherent to the satellite's radar receiver, improving the signal-to-noise ratio across image swaths. |
| **2. Radiometric Calibration** | Converts raw digital numbers into normalized physical radar cross-section values (backscatter coefficient, σ₀). |
| **3. Speckle Filtering** | Applies a spatial filter to suppress the granular "salt-and-pepper" noise typical of SAR imagery, while preserving the sharp surface signatures of internal waves. |
| **4. Spatial Downsampling (10x)** | Reduces image scale by a factor of 10, cutting data volume and improving memory/computation efficiency without losing the macro-structure of the waves. |
| **5. Export to .TIF** | Processed data arrays are exported as GeoTIFF files to preserve spatial metadata and geographic coordinates. |
| **6. Format Conversion (.TIF → .PNG)** | Converts the high-depth geospatial data into standard PNG format, enabling bounding-box annotation and direct ingestion into the YOLO training pipeline. |

**Additional refinement:** the dataset was further refined by cropping SAR images to isolate open-ocean regions and specific IW packets, minimizing land interference that would otherwise confuse the detector.

### Methodology Flow
```
Raw SAR image → Crop to open-ocean/IW region → Data Augmentation → YOLO (.pt)
                                                        ↑                ↓
                                                        └── Hyperparameter tuning ←── Results
```

---

## 🏋️ Model Training

The model is a custom-trained **YOLO26m** (medium variant), trained on the preprocessed and annotated SAR dataset using the Ultralytics framework.

```python
!pip install ultralytics
import torch
from ultralytics import YOLO

model = YOLO('yolo26m.pt')

results26m = model.train(
    data='/content/drive/MyDrive/data.yaml',   # Path to dataset config
    epochs=100,
    imgsz=640,
    batch=32,
    lr0=0.0005,
    optimizer='SGD',
    mosaic=0.0,
    mixup=0.0,
    erasing=0.0,
    amp=False,
    hsv_h=0.0,
    hsv_s=0.0,
    hsv_v=0.0,
    flipud=0.0,
    fliplr=0.5,
    device=0,
    name='3_yolo26m_custom'
)
```

### Key training parameters explained

| Parameter | Value | Why |
|---|---|---|
| `imgsz` | 640 | Standard YOLO input resolution — balances detection accuracy and training/inference speed. Since SAR wave signatures are large-scale patterns (not tiny objects), 640px preserves enough detail without needing a larger, slower input size. |
| `batch` | 32 | Chosen to fit within available GPU VRAM while keeping gradient estimates stable; lower if you hit out-of-memory errors, higher if you have more GPU headroom. |
| `epochs` | 100 | Enough passes over the dataset for the model to converge on a single-class detection task without excessive overfitting risk. |
| `lr0` | 0.0005 | A conservative initial learning rate, suited to fine-tuning on a smaller, specialized dataset rather than training from scratch. |
| `optimizer` | SGD | More stable convergence for small/medium datasets compared to Adam-family optimizers in this setup. |
| `mosaic`, `mixup`, `erasing` | 0.0 (disabled) | These augmentations combine/occlude multiple images — turned off here because internal wave patterns are large, continuous structures that mosaic/erasing would fragment or distort, hurting learning. |
| `hsv_h`, `hsv_s`, `hsv_v` | 0.0 (disabled) | SAR imagery is grayscale-derived intensity data, not natural RGB color — hue/saturation/value jitter (designed for photos) adds no useful variation and can be actively harmful. |
| `flipud` | 0.0 | Vertical flips disabled — wave propagation direction relative to the sensor is a meaningful, orientation-dependent feature. |
| `fliplr` | 0.5 | Horizontal flips enabled at 50% — a valid augmentation since left-right mirroring doesn't break the physical meaning of the wave pattern. |
| `amp` | False | Automatic Mixed Precision disabled for full-precision training stability. |
| `device` | 0 | Trains on GPU (index 0) for practical training speed. |

Full training code and experimentation is available in [`train.ipynb`](./train.ipynb).

---

## 🖥️ Web App (Deployed)

The trained `best.pt` weights power a **Streamlit** web app that lets anyone upload a SAR image and get instant internal-wave detections with bounding boxes and confidence scores.

**Try it now:** [yolo26m-app-waves-detector-v10-klatih5z5rnibcjn9ucvwp.streamlit.app](https://yolo26m-app-waves-detector-v10-klatih5z5rnibcjn9ucvwp.streamlit.app/)

To test it quickly without your own data:
1. Download [`sample image.jpg`](./sample%20image.jpg) from this repo.
2. Open the live app link above.
3. Upload the downloaded image and view the detection result.

### App features
- Upload any SAR image (JPG/PNG/BMP/WEBP)
- Adjustable **confidence threshold** and **IoU (NMS) threshold** sliders
- Annotated output image with bounding boxes and per-detection confidence
- Runs entirely on free-tier CPU hosting (no GPU required for inference)

---

## ⚙️ Requirements

### Software
- Python 3.10+ (tested up to 3.14 on Streamlit Cloud)
- [Ultralytics](https://github.com/ultralytics/ultralytics) (YOLO framework)
- PyTorch & Torchvision
- Streamlit
- OpenCV (headless)
- Pillow, NumPy

All pinned in [`requirements.txt`](./requirements.txt).

### Hardware
| | Training | Inference (deployed app) |
|---|---|---|
| **GPU** | Recommended (e.g. Colab T4/A100) — used `device=0` for training | Not required — deployed app runs on free CPU hosting |
| **RAM** | 12GB+ recommended for batch size 32 at 640px | ~1–2GB sufficient for single-image inference |
| **Storage** | Depends on dataset size; model weights (`best.pt`) are ~43MB | Same — model is loaded once and cached |

---

## 📁 Repository Structure

```
yolo26m-internal-waves-detector-v1.0/
├── best.pt              # Trained YOLO26m model weights
├── data.yaml            # Dataset config used for training
├── train.ipynb          # Full training notebook (Colab)
├── streamlit_app.py     # Deployed web app source code
├── requirements.txt     # Python dependencies
├── sample image.jpg     # Sample SAR image to test the app
└── README.md            # This file
```

---

## 📌 Notes

- This model is trained for a **single-class detection task** (internal waves / "iw").
- SAR imagery preprocessing (noise removal, calibration, speckle filtering) was performed upstream of this repo; only the final PNG-converted, annotated dataset is used for training here.
- The confidence/IoU thresholds in the app are adjustable at inference time — no retraining needed to tune detection sensitivity.
