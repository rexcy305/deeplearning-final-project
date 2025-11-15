import numpy as np
import streamlit as st
from ultralytics import YOLO
import os

# Global cache untuk default model saja
_cached_default_model = None


def load_model(custom_model_file=None, model_path=None):
    """
    PRIORITAS:
    1. Jika user upload custom model -> gunakan itu (selalu reload).
    2. Jika tidak, gunakan model_path jika diberikan.
    3. Jika tidak, gunakan default (cached).
    """

    global _cached_default_model

    # ---------------------------
    # 1. CUSTOM MODEL (selalu di-load ulang)
    # ---------------------------
    if custom_model_file is not None:
        temp_model_path = os.path.join("models", custom_model_file.name)
        os.makedirs("models", exist_ok=True)

        # save uploaded file
        with open(temp_model_path, "wb") as f:
            f.write(custom_model_file.read())

        st.sidebar.success(f"Custom model loaded: {custom_model_file.name}")
        return YOLO(temp_model_path) 


    # ---------------------------
    # 2. MODEL DARI PATH
    # ---------------------------
    if model_path is not None:
        return YOLO(model_path)


    # ---------------------------
    # 3. DEFAULT MODEL (pakai cache)
    # ---------------------------
    if _cached_default_model is None:
        default_path = "yolov8s.pt"   # bisa ubah jadi yolov11n.pt
        _cached_default_model = YOLO(default_path)

    return _cached_default_model

def detect_image_numpy(img_bgr, conf=0.25, imgsz=640, model=None):
    """Perform YOLO inference on np image."""

    # model dikirim dari luar, tidak ambil cached lagi
    results = model.predict(img_bgr, imgsz=imgsz, conf=conf, verbose=False)
    r = results[0]

    boxes = []
    names = model.model.names if hasattr(model.model, 'names') else {}

    for box in r.boxes:
        x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
        conf_score = float(box.conf[0])
        cls_id = int(box.cls[0])
        cls_name = names.get(cls_id, str(cls_id))
        boxes.append((x1, y1, x2, y2, conf_score, cls_name, cls_id))

    return boxes