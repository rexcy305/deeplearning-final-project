import cv2
import numpy as np

def hist_eq_color(img):
    if len(img.shape) == 2:
        return cv2.equalizeHist(img)
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y_eq = cv2.equalizeHist(y)
    ycrcb_eq = cv2.merge([y_eq, cr, cb])
    return cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2BGR)

def clahe_color(img, clip=2.0, grid=(8,8)):
    if len(img.shape) == 2:
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid)
        return clahe.apply(img)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid)
    l2 = clahe.apply(l)
    lab2 = cv2.merge((l2, a, b))
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

def contrast_stretch(img):
    out = img.copy()
    for c in range(out.shape[2]) if out.ndim == 3 else range(1):
        channel = out[:,:,c] if out.ndim == 3 else out
        p2, p98 = np.percentile(channel, (2, 98))
        channel_stretched = np.clip((channel - p2) * 255.0 / (p98 - p2 + 1e-8), 0, 255).astype('uint8')
        if out.ndim == 3:
            out[:,:,c] = channel_stretched
        else:
            out = channel_stretched
    return out

def compute_brightness_contrast_metrics(img):
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    mean = float(gray.mean())
    std = float(gray.std())
    return {"brightness_mean": mean, "contrast_std": std}