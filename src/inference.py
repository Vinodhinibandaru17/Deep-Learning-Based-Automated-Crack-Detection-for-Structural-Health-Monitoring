"""Inference helpers used by the website and evaluate.py. torch is imported lazily so the
site still opens (with the classical baseline) if torch is not installed."""
import cv2
import numpy as np
from .augment import normalize_np


def load_seg_model(path, device="cpu"):
    import torch
    from .models import build_seg_model
    ck = torch.load(str(path), map_location=device, weights_only=False)
    model = build_seg_model(ck["arch"], ck.get("base", 32))
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), ck


def predict_seg_prob(model, img_rgb, size=256, device="cpu"):
    """Crack probability map at the ORIGINAL image resolution (float32, 0..1)."""
    import torch
    h, w = img_rgb.shape[:2]
    size = max(64, size // 16 * 16)
    x = cv2.resize(img_rgb, (size, size), interpolation=cv2.INTER_AREA)
    t = torch.from_numpy(normalize_np(x)).permute(2, 0, 1)[None].float().to(device)
    with torch.no_grad():
        p = torch.sigmoid(model(t))[0, 0].cpu().numpy()
    return cv2.resize(p, (w, h), interpolation=cv2.INTER_LINEAR)


def load_cls_model(path, device="cpu"):
    import torch
    from .models import build_classifier
    ck = torch.load(str(path), map_location=device, weights_only=False)
    model = build_classifier(ck["arch"], pretrained=False)
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), ck


def predict_cls_prob(model, img_rgb, size=224, device="cpu"):
    import torch
    x = cv2.resize(img_rgb, (size, size), interpolation=cv2.INTER_AREA)
    t = torch.from_numpy(normalize_np(x)).permute(2, 0, 1)[None].float().to(device)
    with torch.no_grad():
        return float(torch.sigmoid(model(t).squeeze()).item())


def load_yolo(path):
    from ultralytics import YOLO
    return YOLO(str(path))


def predict_yolo(model, img_rgb, conf=0.25):
    r = model.predict(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), conf=conf, verbose=False)[0]
    out = []
    for b in r.boxes:
        x1, y1, x2, y2 = (int(v) for v in b.xyxy[0].tolist())
        out.append({"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1, "conf": float(b.conf[0])})
    return out
