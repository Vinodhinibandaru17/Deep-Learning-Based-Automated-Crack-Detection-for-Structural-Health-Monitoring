"""Post-processing + crack measurements (no torch needed).

Everything the website displays about a predicted mask comes from here:
cleaning, area %, skeleton length, width, bounding boxes, severity, overlay,
and a classical (non-learned) crack detector used as a fallback baseline.
"""
import cv2
import numpy as np
from skimage.morphology import skeletonize


def clean_mask(mask, min_area=30):
    """Drop connected components smaller than min_area pixels."""
    mask = (np.asarray(mask) > 0).astype(np.uint8)
    if min_area <= 0 or mask.sum() == 0:
        return mask.astype(bool)
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros_like(mask)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] >= min_area:
            keep[lab == i] = 1
    return keep.astype(bool)


def boxes_from_mask(mask, merge_frac=0.015):
    """Mask-derived detections: merge nearby fragments, then one box per group."""
    m = (np.asarray(mask) > 0).astype(np.uint8)
    if m.sum() == 0:
        return []
    h, w = m.shape
    k = max(3, int(merge_frac * max(h, w)) | 1)
    merged = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(merged, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, bw, bh = (int(st[i, c]) for c in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))
        out.append({"x": x, "y": y, "w": bw, "h": bh, "crack_px": int(m[lab == i].sum())})
    return sorted(out, key=lambda b: -b["crack_px"])


def skeleton_of(mask):
    return skeletonize(np.asarray(mask) > 0)


def crack_width_px(mask):
    """Mean crack width in pixels = area / skeleton length (0 if no crack)."""
    m = np.asarray(mask) > 0
    if not m.any():
        return 0.0
    length = max(int(skeleton_of(m).sum()), 1)
    return float(m.sum() / length)


def severity_label(area_pct, max_width_mm=None):
    """Heuristic traffic-light label. Indicative only, not an engineering rating."""
    levels = ["None", "Low", "Moderate", "High"]
    a = 0 if area_pct < 0.05 else 1 if area_pct < 0.5 else 2 if area_pct < 2.0 else 3
    w = 0
    if max_width_mm is not None:
        w = 1 if max_width_mm < 0.3 else 2 if max_width_mm < 1.0 else 3
    return levels[max(a, w)]


def crack_stats(mask, mm_per_px=None):
    m = np.asarray(mask) > 0
    h, w = m.shape
    area = int(m.sum())
    base = {"area_px": area, "area_pct": 100.0 * area / (h * w), "length_px": 0, "mean_width_px": 0.0,
            "max_width_px": 0.0, "n_regions": 0}
    if area:
        skel = skeleton_of(m)
        dist = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 3)
        widths = 2.0 * dist[skel] if skel.any() else np.array([0.0])
        n, _ = cv2.connectedComponents(m.astype(np.uint8), connectivity=8)
        base.update(length_px=int(skel.sum()), mean_width_px=float(area / max(skel.sum(), 1)),
                    max_width_px=float(widths.max()), n_regions=int(n - 1))
    if mm_per_px:
        base["length_mm"] = base["length_px"] * mm_per_px
        base["mean_width_mm"] = base["mean_width_px"] * mm_per_px
        base["max_width_mm"] = base["max_width_px"] * mm_per_px
    base["severity"] = severity_label(base["area_pct"], base.get("max_width_mm") if area else None)
    return base


def overlay(img_rgb, mask, boxes=None, alpha=0.55, color=(255, 40, 40), show_skeleton=False):
    out = img_rgb.astype(np.float32).copy()
    m = np.asarray(mask) > 0
    out[m] = (1 - alpha) * out[m] + alpha * np.array(color, np.float32)
    out = out.astype(np.uint8)
    if show_skeleton and m.any():
        out[skeleton_of(m)] = (255, 255, 0)
    thick = max(2, int(round(max(out.shape[:2]) / 250)))
    for i, b in enumerate(boxes or [], 1):
        cv2.rectangle(out, (b["x"], b["y"]), (b["x"] + b["w"], b["y"] + b["h"]), (0, 220, 60), thick)
        cv2.putText(out, str(i), (b["x"] + 3, b["y"] + 6 + 8 * thick), cv2.FONT_HERSHEY_SIMPLEX,
                    0.35 * thick + 0.3, (0, 220, 60), max(1, thick - 1), cv2.LINE_AA)
    return out


def classical_crack_mask(img_rgb):
    """Non-learned baseline: CLAHE -> black-hat (dark thin structures) -> adaptive threshold."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    k = max(9, (min(h, w) // 25) | 1)
    bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))).astype(np.float32)
    thr = max(15.0, float(bh.mean() + 2.5 * bh.std()))
    m = (bh > thr).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return clean_mask(m, max(25, int(0.0003 * h * w)))
