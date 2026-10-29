"""NumPy/OpenCV augmentation + robustness perturbations (no torch needed)."""
import random
import cv2
import numpy as np
from .config import MEAN, STD


def normalize_np(img_rgb):
    x = img_rgb.astype(np.float32) / 255.0
    return (x - np.array(MEAN, np.float32)) / np.array(STD, np.float32)


def add_shadow(img, rng=random, strength=None):
    """Darken a random soft-edged polygon (simulates cast shadows)."""
    h, w = img.shape[:2]
    pts = np.array([[rng.randint(0, w - 1), rng.randint(0, h - 1)] for _ in range(rng.randint(3, 6))], np.int32)
    m = np.zeros((h, w), np.float32)
    cv2.fillConvexPoly(m, cv2.convexHull(pts), 1.0)
    m = cv2.GaussianBlur(m, (0, 0), sigmaX=max(h, w) / 30)
    f = strength if strength is not None else rng.uniform(0.35, 0.7)  # remaining brightness
    out = img.astype(np.float32) * (1 - m[..., None] * (1 - f))
    return np.clip(out, 0, 255).astype(np.uint8)


def random_augment(img, mask=None, rng=random, zoom=True):
    """Flip / rot90 / zoom-crop / brightness-contrast / shadow / blur / noise."""
    if rng.random() < 0.5:
        img = img[:, ::-1]; mask = mask[:, ::-1] if mask is not None else None
    if rng.random() < 0.5:
        img = img[::-1]; mask = mask[::-1] if mask is not None else None
    k = rng.randint(0, 3)
    if k:
        img = np.rot90(img, k); mask = np.rot90(mask, k) if mask is not None else None
    img = np.ascontiguousarray(img); mask = np.ascontiguousarray(mask) if mask is not None else None
    h, w = img.shape[:2]
    if zoom and rng.random() < 0.4:
        s = rng.uniform(0.7, 1.0)
        ch, cw = int(h * s), int(w * s)
        y0, x0 = rng.randint(0, h - ch), rng.randint(0, w - cw)
        img = cv2.resize(img[y0:y0 + ch, x0:x0 + cw], (w, h), interpolation=cv2.INTER_LINEAR)
        if mask is not None:
            mask = cv2.resize(mask[y0:y0 + ch, x0:x0 + cw], (w, h), interpolation=cv2.INTER_NEAREST)
    a, b = rng.uniform(0.75, 1.25), rng.uniform(-25, 25)
    img = np.clip(img.astype(np.float32) * a + b, 0, 255).astype(np.uint8)
    if rng.random() < 0.3:
        img = add_shadow(img, rng)
    if rng.random() < 0.2:
        img = cv2.GaussianBlur(img, (3, 3), 0)
    if rng.random() < 0.2:
        img = np.clip(img + np.random.normal(0, 6, img.shape), 0, 255).astype(np.uint8)
    return img, mask


# ---- robustness perturbations (severity 1..3) -------------------------------
PERTURBATIONS = ["dark", "bright", "low_contrast", "shadow", "blur", "noise", "jpeg"]


def perturb_image(img, kind, severity=1):
    s = int(np.clip(severity, 1, 3)) - 1
    if kind == "dark":
        return np.clip(img.astype(np.float32) * [0.7, 0.5, 0.3][s], 0, 255).astype(np.uint8)
    if kind == "bright":
        return np.clip(img.astype(np.float32) * [1.3, 1.6, 2.0][s] + [10, 20, 30][s], 0, 255).astype(np.uint8)
    if kind == "low_contrast":
        f = [0.7, 0.5, 0.3][s]
        return np.clip((img.astype(np.float32) - 128) * f + 128, 0, 255).astype(np.uint8)
    if kind == "shadow":
        return add_shadow(img, random.Random(7), strength=[0.6, 0.45, 0.3][s])
    if kind == "blur":
        k = [3, 7, 11][s]
        return cv2.GaussianBlur(img, (k, k), 0)
    if kind == "noise":
        rs = np.random.RandomState(0)
        return np.clip(img + rs.normal(0, [8, 16, 28][s], img.shape), 0, 255).astype(np.uint8)
    if kind == "jpeg":
        ok, enc = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, [40, 20, 8][s]])
        return cv2.cvtColor(cv2.imdecode(enc, 1), cv2.COLOR_BGR2RGB)
    raise ValueError(kind)
