# utils/detector.py
import numpy as np
from ultralytics import YOLO

_model = None

def load_model(model_path=None):
    global _model
    if _model is not None:
        return _model
    try:
        if model_path:
            _model = YOLO(model_path)
        else:
            # try load default yolov11 (if present) else fallback to yolov8n (if ultralytics has it)
            try:
                _model = YOLO("yolov11.pt")
            except Exception:
                _model = YOLO("yolov8n.pt")
        return _model
    except Exception as e:
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