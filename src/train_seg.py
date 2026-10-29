"""Train a segmentation model and write weights + metrics.

python -m src.train_seg --model unet           --loss bce
python -m src.train_seg --model attention_unet --loss bce_dice
"""
import argparse, json, random, time
import numpy as np
import torch
from torch.utils.data import DataLoader
from .config import DATA_DIR, IMG_SIZE, RESULTS_DIR, SEG_MODELS, WEIGHTS_DIR
from .datasets import SegDataset
from .engine import best_threshold, collect_probs, per_image_metrics
from .losses import get_loss
from .metrics import mean_dicts
from .models import build_seg_model, count_params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(DATA_DIR / "seg_split.csv"))
    ap.add_argument("--model", default="unet", choices=list(SEG_MODELS))
    ap.add_argument("--loss", default=None, choices=["bce", "bce_dice", "focal_dice"])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--img_size", type=int, default=IMG_SIZE)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--tag", default="", help="suffix for output names (ablations)")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    a.loss = a.loss or ("bce" if a.model == "unet" else "bce_dice")
    name = a.model + (f"_{a.tag}" if a.tag else "")

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr = DataLoader(SegDataset(a.csv, "train", a.img_size, augment=True), a.bs, shuffle=True,
                    num_workers=a.workers, drop_last=True, pin_memory=dev.type == "cuda")
    va = DataLoader(SegDataset(a.csv, "val", a.img_size), a.bs, num_workers=a.workers)
    te = DataLoader(SegDataset(a.csv, "test", a.img_size), a.bs, num_workers=a.workers)
    print(f"device={dev} train={len(tr.dataset)} val={len(va.dataset)} test={len(te.dataset)} loss={a.loss}")

    model = build_seg_model(a.model, a.base).to(dev)
    crit, opt = get_loss(a.loss), torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    amp = dev.type == "cuda"
    scaler = torch.amp.GradScaler(dev.type, enabled=amp)
    hist = {"train_loss": [], "val_dice": [], "val_iou": []}
    best, ckpt_path = -1.0, WEIGHTS_DIR / f"seg_{name}.pt"

    for ep in range(1, a.epochs + 1):
        model.train(); tot, t0 = 0.0, time.time()
        for x, y in tr:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev.type, enabled=amp):
                loss = crit(model(x), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tot += loss.item() * len(x)
        sched.step()
        P, G, _ = collect_probs(model, va, dev)
        m = mean_dicts(per_image_metrics(P, G, 0.5))
        hist["train_loss"].append(tot / max(len(tr.dataset), 1)); hist["val_dice"].append(m["dice"]); hist["val_iou"].append(m["iou"])
        print(f"ep {ep:03d}/{a.epochs} loss {hist['train_loss'][-1]:.4f} val dice {m['dice']:.4f} iou {m['iou']:.4f} ({time.time()-t0:.0f}s)")
        if m["dice"] > best:
            best = m["dice"]
            torch.save({"model": model.state_dict(), "arch": a.model, "base": a.base, "img_size": a.img_size,
                        "loss": a.loss, "threshold": 0.5}, ckpt_path)

    ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
    model.load_state_dict(ck["model"])
    Pv, Gv, _ = collect_probs(model, va, dev)
    thr = best_threshold(Pv, Gv)
    ck["threshold"] = float(thr); torch.save(ck, ckpt_path)
    Pt, Gt, sec = collect_probs(model, te, dev)
    test = mean_dicts(per_image_metrics(Pt, Gt, thr))
    out = {"task": "segmentation", "name": name, "arch": a.model, "label": SEG_MODELS[a.model], "loss": a.loss,
           "params": count_params(model), "img_size": a.img_size, "epochs": a.epochs, "threshold": float(thr),
           "ms_per_image": 1000 * sec, "device": dev.type, "n_test": len(te.dataset), "test": test, "history": hist}
    (RESULTS_DIR / f"seg_{name}_metrics.json").write_text(json.dumps(out, indent=2))
    print("TEST", {k: round(v, 4) for k, v in test.items()}, f"thr={thr:.2f}")


if __name__ == "__main__":
    main()
