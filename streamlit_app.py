import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO
import torch

# -----------------------------------------------------------------
# PAGE CONFIG (must be the first Streamlit command)
# -----------------------------------------------------------------
st.set_page_config(
    page_title="YOLO26 Object Detector",
    page_icon="🎯",
    layout="wide",
)

MODEL_PATH = "best.pt"  # <-- put your trained .pt file next to this script with this name


# -----------------------------------------------------------------
# LOAD MODEL (cached so it only loads once per session, not on every rerun)
# -----------------------------------------------------------------
@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(MODEL_PATH)
    model.to(device)
    return model, device


model, device = load_model()
CLASS_NAMES = model.names


# -----------------------------------------------------------------
# INFERENCE
# -----------------------------------------------------------------
def detect(image: Image.Image, conf_threshold: float, iou_threshold: float):
    results = model.predict(
        source=image,
        conf=conf_threshold,
        iou=iou_threshold,
        device=device,
        verbose=False,
    )
    result = results[0]

    # result.plot() returns BGR (OpenCV convention) — flip to RGB for correct display colors
    annotated_bgr = result.plot()
    annotated_rgb = annotated_bgr[..., ::-1]

    boxes = result.boxes
    detections = []
    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = CLASS_NAMES.get(cls_id, str(cls_id))
            detections.append((label, conf))

    return annotated_rgb, detections


# -----------------------------------------------------------------
# CUSTOM STYLING
# -----------------------------------------------------------------
st.markdown(
    """
    <style>
    .main .block-container { max-width: 1100px; }
    h1 { text-align: center; }
    .subtitle { text-align: center; color: #666; margin-top: -10px; margin-bottom: 30px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------
# UI
# -----------------------------------------------------------------
st.markdown("<h1>🎯 Custom YOLO26 Object Detector</h1>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtitle'>Upload an image and the model will detect and localize the trained object.</div>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("⚙️ Settings")
    conf_threshold = st.slider("Confidence threshold", 0.05, 1.0, 0.25, 0.05)
    iou_threshold = st.slider("IoU threshold (NMS)", 0.05, 1.0, 0.45, 0.05)
    st.caption(f"Running on: **{device.upper()}**")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Input Image")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp", "webp"])
    if uploaded_file is not None:
        input_image = Image.open(uploaded_file).convert("RGB")
        st.image(input_image, use_container_width=True)

with col2:
    st.subheader("Detection Result")
    if uploaded_file is not None:
        with st.spinner("Running detection..."):
            annotated_rgb, detections = detect(input_image, conf_threshold, iou_threshold)
        st.image(annotated_rgb, use_container_width=True)

        if detections:
            st.success(f"Detected {len(detections)} object(s)")
            for label, conf in detections:
                st.write(f"• **{label}** — {conf:.1%} confidence")
        else:
            st.warning("No objects detected. Try lowering the confidence threshold.")
    else:
        st.info("Upload an image on the left to see results here.")
