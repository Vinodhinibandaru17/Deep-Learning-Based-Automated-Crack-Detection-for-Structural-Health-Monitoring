"""Objective 6 - robustness analysis.

For every trained segmentation model it measures Dice/IoU
  * on clean test images
  * under controlled perturbations (dark, bright, low contrast, shadow, blur, noise, jpeg) x 3 severities
  * stratified by crack thickness (thin / medium / thick third of the test set)
  * optionally on an UNSEEN dataset (--cross_csv), e.g. train on DeepCrack, test on CRACK500

python -m src.evaluate --models unet attention_unet
python -m src.evaluate --models unet attention_unet --cross_csv data/crack500_split.csv
"""
import argparse, json
import numpy as np
import torch
from torch.utils.data import DataLoader
from .analysis import crack_width_px
from .augment import PERTURBATIONS
from .config import DATA_DIR, RESULTS_DIR, WEIGHTS_DIR
from .datasets import SegDataset
from .engine import collect_probs, per_image_metrics
from .inference import load_seg_model
from .metrics import mean_dicts


def _loader(csv, split, size, perturb=None, bs=16):
    return DataLoader(SegDataset(csv, split, size, perturb=perturb), bs, num_workers=2)


def _mean(P, G, thr, idx=None):
    rows = per_image_metrics(P, G, thr)
    if idx is not None:
        rows = [rows[i] for i in idx]
    m = mean_dicts(rows)
    return {"dice": m.get("dice", float("nan")), "iou": m.get("iou", float("nan")), "n": len(rows)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["unet", "attention_unet"])
    ap.add_argument("--csv", default=str(DATA_DIR / "seg_split.csv"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--cross_csv", default=None, help="split CSV of a different dataset (unseen surfaces)")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for name in a.models:
        model, ck = load_seg_model(WEIGHTS_DIR / f"seg_{name}.pt", dev)
        size, thr = ck.get("img_size", 256), ck.get("threshold", 0.5)
        P, G, _ = collect_probs(model, _loader(a.csv, a.split, size), dev)
        out = {"model": name, "threshold": thr, "clean": _mean(P, G, thr), "perturbed": {}, "by_width": {}}

        widths = np.array([crack_width_px(g > 0.5) for g in G])          # px at training resolution
        has = np.where(widths > 0)[0]
        if len(has) >= 3:
            lo, hi = np.percentile(widths[has], [33.3, 66.6])
            groups = {"thin": has[widths[has] <= lo], "medium": has[(widths[has] > lo) & (widths[has] <= hi)],
                      "thick": has[widths[has] > hi]}
            out["width_edges_px"] = [float(lo), float(hi)]
            out["by_width"] = {k: _mean(P, G, thr, v) for k, v in groups.items() if len(v)}

        for kind in PERTURBATIONS:
            out["perturbed"][kind] = {}
            for sev in (1, 2, 3):
                Pp, Gp, _ = collect_probs(model, _loader(a.csv, a.split, size, (kind, sev)), dev)
                out["perturbed"][kind][str(sev)] = _mean(Pp, Gp, thr)
            print(f"{name:16s} {kind:13s} dice " + " ".join(f"{out['perturbed'][kind][s]['dice']:.3f}" for s in "123"))

        if a.cross_csv:
            Pc, Gc, _ = collect_probs(model, _loader(a.cross_csv, "all", size), dev)
            out["cross_dataset"] = {"csv": a.cross_csv, **_mean(Pc, Gc, thr)}
        (RESULTS_DIR / f"robustness_{name}.json").write_text(json.dumps(out, indent=2))
        print(f"[{name}] clean dice {out['clean']['dice']:.3f}  by width:",
              {k: round(v['dice'], 3) for k, v in out["by_width"].items()})


if __name__ == "__main__":
    main()
