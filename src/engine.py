"""Shared evaluation helpers for training and robustness scripts."""
import time
import numpy as np
import torch
from .metrics import mean_dicts, seg_metrics


@torch.no_grad()
def collect_probs(model, loader, device):
    """Returns probs[N,H,W], gts[N,H,W] (numpy) and seconds-per-image."""
    model.eval()
    P, G, t0, n = [], [], time.time(), 0
    for x, y in loader:
        p = torch.sigmoid(model(x.to(device)))
        if device.type == "cuda":
            torch.cuda.synchronize()
        P.append(p[:, 0].cpu().numpy()); G.append(y[:, 0].numpy()); n += len(x)
    return np.concatenate(P), np.concatenate(G), (time.time() - t0) / max(n, 1)


def per_image_metrics(probs, gts, thr=0.5):
    return [seg_metrics(p >= thr, g > 0.5) for p, g in zip(probs, gts)]


def best_threshold(probs, gts, grid=np.arange(0.2, 0.81, 0.05)):
    scores = [(mean_dicts(per_image_metrics(probs, gts, t))["dice"], float(t)) for t in grid]
    return max(scores)[1]
