"""Train a crack / no-crack classifier.

python -m src.train_cls --model resnet18
python -m src.train_cls --model efficientnet_b0
"""
import argparse, json, random, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .config import CLS_MODELS, CLS_SIZE, DATA_DIR, RESULTS_DIR, WEIGHTS_DIR
from .datasets import ClsDataset
from .metrics import cls_metrics
from .models import build_classifier, count_params


@torch.no_grad()
def predict(model, loader, dev):
    model.eval(); P, Y, t0 = [], [], time.time()
    for x, y in loader:
        P.append(torch.sigmoid(model(x.to(dev)).squeeze(1)).cpu().numpy()); Y.append(y.numpy())
    n = sum(len(p) for p in P)
    return np.concatenate(Y), np.concatenate(P), (time.time() - t0) / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(DATA_DIR / "cls_split.csv"))
    ap.add_argument("--model", default="resnet18", choices=list(CLS_MODELS))
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--size", type=int, default=CLS_SIZE)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dtr = ClsDataset(a.csv, "train", a.size, augment=True)
    tr = DataLoader(dtr, a.bs, shuffle=True, num_workers=a.workers, drop_last=len(dtr) > a.bs)
    va = DataLoader(ClsDataset(a.csv, "val", a.size), a.bs, num_workers=a.workers)
    te = DataLoader(ClsDataset(a.csv, "test", a.size), a.bs, num_workers=a.workers)
    lab = dtr.labels(); pos_w = float((lab == 0).sum() / max((lab == 1).sum(), 1))   # class imbalance
    print(f"device={dev} train={len(dtr)} pos_weight={pos_w:.2f}")

    model = build_classifier(a.model).to(dev)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_w, device=dev))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    best, path = -1.0, WEIGHTS_DIR / f"cls_{a.model}.pt"
    hist = {"train_loss": [], "val_f1": [], "val_acc": []}
    for ep in range(1, a.epochs + 1):
        model.train(); tot = 0.0
        for x, y in tr:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad(); loss = crit(model(x).squeeze(1), y); loss.backward(); opt.step(); tot += loss.item() * len(x)
        yv, pv, _ = predict(model, va, dev); m = cls_metrics(yv, pv)
        hist["train_loss"].append(tot / len(dtr)); hist["val_f1"].append(m["f1"]); hist["val_acc"].append(m["accuracy"])
        print(f"ep {ep:02d}/{a.epochs} loss {hist['train_loss'][-1]:.4f} val acc {m['accuracy']:.4f} f1 {m['f1']:.4f}")
        if m["f1"] > best:
            best = m["f1"]; torch.save({"model": model.state_dict(), "arch": a.model, "size": a.size}, path)

    model.load_state_dict(torch.load(path, map_location=dev, weights_only=False)["model"])
    yt, pt, sec = predict(model, te, dev); test = cls_metrics(yt, pt)
    out = {"task": "classification", "name": a.model, "label": CLS_MODELS[a.model], "params": count_params(model),
           "size": a.size, "epochs": a.epochs, "ms_per_image": 1000 * sec, "device": dev.type, "n_test": len(yt),
           "test": test, "history": hist}
    (RESULTS_DIR / f"cls_{a.model}_metrics.json").write_text(json.dumps(out, indent=2))
    print("TEST", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in test.items()})


if __name__ == "__main__":
    main()
