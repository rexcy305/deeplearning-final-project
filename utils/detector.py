# utils/detector.py
import numpy as np
import streamlit as st
from ultralytics import YOLO
import os

_model = None

def load_model(model_path=None, custom_model_file=None):
    """
    Load YOLO model.
    - Jika user upload model .pt, maka itu yang dipakai.
    - Jika tidak, gunakan model default (yolov11n.pt).
    """
    global _model
    if _model is not None:
        return _model

    try:
        if custom_model_file is not None:
            # Simpan model upload ke file temporer
            temp_model_path = os.path.join("models", custom_model_file.name)
            os.makedirs("models", exist_ok=True)
            with open(temp_model_path, "wb") as f:
                f.write(custom_model_file.read())
            st.sidebar.success(f"Custom model '{custom_model_file.name}' loaded.")
            _model = YOLO(temp_model_path)
        elif model_path:
            _model = YOLO(model_path)
        else:
            # Gunakan default jika tidak ada upload
            default_path = "yolov8s.pt"
            _model = YOLO(default_path)
        return _model

    except Exception as e:
        st.error(f"Failed to load model: {e}")
        raise RuntimeError(f"Failed to load model: {e}")

def detect_image_numpy(img_bgr, conf=0.25, imgsz=640):
    """
    img_bgr: OpenCV BGR numpy array
    returns: list of boxes [(x1,y1,x2,y2,conf,cls_name,cls_id)] and annotated image (copy)
    """
    model = load_model()
    # Ultralytics accepts BGR numpy arrays
    results = model.predict(img_bgr, imgsz=imgsz, conf=conf, verbose=False)
    r = results[0]
    boxes = []
    annotated = img_bgr.copy()
    # names mapping
    names = model.model.names if hasattr(model, 'model') and hasattr(model.model, 'names') else {}
    for box in r.boxes:
        x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
        conf_score = float(box.conf[0])
        cls_id = int(box.cls[0])
        cls_name = names.get(cls_id, str(cls_id))
        boxes.append((x1, y1, x2, y2, conf_score, cls_name, cls_id))
    return boxes, annotated