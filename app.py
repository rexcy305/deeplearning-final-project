import streamlit as st
from PIL import Image
import cv2, tempfile, os, time, numpy as np
from utils.enhancements import (
    hist_eq_color,
    clahe_color,
    contrast_stretch,
    compute_brightness_contrast_metrics,
)
from utils.detector import detect_image_numpy, load_model
from utils.counting import count_all, count_in_region
from utils.reporting import save_metrics_rows

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="YOLOv11 Web App - Enhancement + Counting + Region",
    layout="wide",
)
st.title("🔍 YOLOv11 Web App — Image Enhancement + Counting + Region Box")

# -----------------------------
# LOAD MODEL
# -----------------------------
with st.spinner("Loading YOLO model..."):
    try:
        model = load_model()
        st.success("✅ Model loaded successfully.")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

# -----------------------------
# SIDEBAR SETTINGS
# -----------------------------
st.sidebar.header("⚙️ Settings")
upload_type = st.sidebar.radio("Upload type", ["Image", "Video"])
enh_choice = st.sidebar.selectbox(
    "Enhancement", ["Ori", "Histogram Equalization (HE)", "CLAHE", "Contrast Stretching"]
)
conf_thres = st.sidebar.slider("Confidence threshold", 0.1, 0.9, 0.25, 0.05)
imgsz = st.sidebar.selectbox("Inference size (px)", [320, 480, 640, 960], index=2)

st.sidebar.markdown("---")
st.sidebar.subheader("📍 Region Settings")
use_region = st.sidebar.checkbox("Show / use region for region-counting", value=True)
region_spec = None
if use_region:
    st.sidebar.markdown("Enter region coordinates (x1, y1, x2, y2)")
    rx1 = st.sidebar.number_input("x1", min_value=0, value=50)
    ry1 = st.sidebar.number_input("y1", min_value=0, value=50)
    rx2 = st.sidebar.number_input("x2", min_value=1, value=400)
    ry2 = st.sidebar.number_input("y2", min_value=1, value=300)
    region_spec = (rx1, ry1, rx2, ry2)

st.sidebar.markdown("---")
enable_counting = st.sidebar.checkbox("Enable Counting", value=True)
st.sidebar.markdown("---")
uploaded_file = st.sidebar.file_uploader(
    "📁 Upload image or video", type=["jpg", "jpeg", "png", "mp4", "avi", "mov", "mkv"]
)

# -----------------------------
# SESSION STATE FOR REPORT
# -----------------------------
if "report_rows" not in st.session_state:
    st.session_state.report_rows = []
report_rows = st.session_state.report_rows

# -----------------------------
# FUNCTIONS
# -----------------------------
def apply_enh(img_bgr, choice):
    if choice == "Ori":
        return img_bgr
    if choice == "Histogram Equalization (HE)":
        return hist_eq_color(img_bgr)
    if choice == "CLAHE":
        return clahe_color(img_bgr)
    if choice == "Contrast Stretching":
        return contrast_stretch(img_bgr)
    return img_bgr


# -----------------------------
# MAIN PROCESS (IMAGE)
# -----------------------------
if uploaded_file is None:
    st.info("Upload an image or video to start.")
    st.stop()

if upload_type == "Image":
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        st.error("Cannot read image.")
        st.stop()

    # Metrics before
    before_metrics = compute_brightness_contrast_metrics(img)
    enhanced = apply_enh(img.copy(), enh_choice)
    after_metrics = compute_brightness_contrast_metrics(enhanced)

    # --- Visual comparison (Before vs After)
    colA, colB = st.columns(2)
    colA.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Original Image", use_container_width=True)
    colB.image(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB),
               caption=f"Enhanced Image ({enh_choice})",
               use_container_width=True)

    # --- Detection
    to_detect = enhanced if enh_choice != "Ori" else img
    boxes, _ = detect_image_numpy(to_detect, conf=conf_thres, imgsz=imgsz)

    # --- Compute average confidence
    avg_conf = np.mean([b[4] for b in boxes]) if len(boxes) > 0 else 0.0

    # --- Draw results
    display_img = to_detect.copy()
    for b in boxes:
        x1, y1, x2, y2, conf, cls_name, cls_id = b
        cv2.rectangle(display_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(display_img, f"{cls_name} {conf:.2f}",
                    (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # --- Region Box with label
    region_counts = {}
    if use_region and region_spec is not None:
        rx1, ry1, rx2, ry2 = region_spec
        cv2.rectangle(display_img, (int(rx1), int(ry1)), (int(rx2), int(ry2)), (0, 0, 255), 2)
        cv2.putText(display_img, "Region Area", (int(rx1) + 5, int(ry1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        region_counts = count_in_region(boxes, region_spec)
        if region_counts:
            total_region = sum(region_counts.values())
            cv2.putText(display_img, f"Objs in Region: {total_region}",
                        (int(rx1) + 5, int(ry1) + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # --- Counting
    total_counts = count_all(boxes) if enable_counting else {}
    total_detections = sum(total_counts.values()) if total_counts else 0

    # --- Display result
    col1, col2 = st.columns([1.5, 1])
    col1.image(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB),
               caption="Detection Result with Region Box",
               use_container_width=True)

    with col2:
        st.subheader("📊 Detection Summary")
        st.write(f"**Enhancement:** {enh_choice}")
        st.write(f"**Total Detections:** {total_detections}")
        st.write(f"**Average Confidence (Accuracy):** {avg_conf:.2f}")
        st.write("**Objects per Class:**")
        st.json(total_counts)
        if use_region:
            st.write("**Objects inside Region:**")
            st.json(region_counts)

        st.markdown("**Brightness / Contrast (Before → After)**")
        st.json({"Before": before_metrics, "After": after_metrics})

        # Save data
        if st.button("💾 Add to Report"):
            row = {
                "file": uploaded_file.name,
                "mode": enh_choice,
                "brightness_before": before_metrics["brightness_mean"],
                "contrast_before": before_metrics["contrast_std"],
                "brightness_after": after_metrics["brightness_mean"],
                "contrast_after": after_metrics["contrast_std"],
                "total_count": total_detections,
                "avg_confidence": avg_conf,
                "counts": total_counts,
                "counts_in_region": region_counts,
            }
            st.session_state.report_rows.append(row)
            st.success("Added to report memory!")

        if st.button("📤 Export Report to Excel"):
            if not st.session_state.report_rows:
                st.warning("No rows in memory. Click 'Add to Report' first.")
            else:
                out = save_metrics_rows(st.session_state.report_rows, out_path="outputs/reports.xlsx")
                st.success(f"Saved report to {out}")

# -----------------------------
# VIDEO (same detection logic)
# -----------------------------
elif upload_type == "Video":
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1])
    tfile.write(uploaded_file.read())
    tfile.flush()
    video_path = tfile.name

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Cannot open video.")
        st.stop()

    stframe = st.empty()
    progress_bar = st.progress(0)
    frame_idx = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        enhanced = apply_enh(frame.copy(), enh_choice)
        to_detect = enhanced if enh_choice != "Ori" else frame
        boxes, _ = detect_image_numpy(to_detect, conf=conf_thres, imgsz=imgsz)
        for b in boxes:
            x1, y1, x2, y2, conf, cls_name, cls_id = b
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"{cls_name} {conf:.2f}",
                        (int(x1), int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if use_region and region_spec is not None:
            rx1, ry1, rx2, ry2 = region_spec
            cv2.rectangle(frame, (int(rx1), int(ry1)), (int(rx2), int(ry2)), (0, 0, 255), 2)
            cv2.putText(frame, "Region Area", (int(rx1) + 5, int(ry1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
        progress_bar.progress(min(frame_idx / total_frames, 1.0))

    cap.release()
    st.success("Video processing finished — screenshot this for report.")