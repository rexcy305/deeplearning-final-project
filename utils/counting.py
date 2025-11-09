# utils/counting.py
def centroid(box):
    x1,y1,x2,y2 = box[0],box[1],box[2],box[3]
    return ((x1+x2)/2.0, (y1+y2)/2.0)

def count_all(boxes):
    """Count all detections per class in the whole frame."""
    counts = {}
    for b in boxes:
        cls = b[5]  # class name
        counts[cls] = counts.get(cls, 0) + 1
    return counts

def count_in_region(boxes, region):
    """
    region: (x1,y1,x2,y2)
    boxes: list as returned by detector
    returns dict counts per class inside region
    """
    x1r, y1r, x2r, y2r = region
    counts = {}
    for b in boxes:
        cx, cy = centroid(b)
        if x1r <= cx <= x2r and y1r <= cy <= y2r:
            cls = b[5]
            counts[cls] = counts.get(cls, 0) + 1
    return counts