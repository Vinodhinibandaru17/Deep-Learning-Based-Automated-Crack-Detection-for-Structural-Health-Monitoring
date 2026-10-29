"""Dataset preparation (no torch): pair images with masks, build splits, make
classification patches, summarise. Works with DeepCrack, CRACK500, CFD, khanhha's
merged set, SDNET2018, Ozgenel/"Surface Crack" folders, ...

CLI examples
    python -m src.data_prep seg --root data/DeepCrack
    python -m src.data_prep cls-from-seg --seg_csv data/seg_split.csv
    python -m src.data_prep cls --root data/SDNET2018 --max_per_class 4000
"""
import argparse
import json
import random
import re
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from .config import DATA_DIR, IMG_EXT, RESULTS_DIR

_SUFFIX = re.compile(r"(_mask|_masks|_lab|_label|_labels|_gt|_groundtruth|-mask|-lab|-gt|_seg)$", re.I)
_MASK_TOKENS = {"lab", "labs", "mask", "masks", "gt", "gts", "label", "labels", "annotation", "annotations",
                "groundtruth", "ground_truth", "seg", "segmentation"}


def _key(p):
    return _SUFFIX.sub("", p.stem).lower()


def _is_mask_path(p, root):
    if _SUFFIX.search(p.stem):
        return True
    for part in p.relative_to(root).parts[:-1]:
        if set(re.split(r"[^a-z0-9]+", part.lower())) & _MASK_TOKENS:
            return True
    return False


def read_mask(path):
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(path)
    return (m > (0 if m.max() <= 1 else 127)).astype(np.uint8)


def auto_find_pairs(root):
    """Walk `root`; masks = files in label/mask-like folders or with a mask suffix."""
    root = Path(root)
    imgs, masks = {}, {}
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() not in IMG_EXT or not p.is_file():
            continue
        (masks if _is_mask_path(p, root) else imgs)[_key(p)] = p
    keys = sorted(set(imgs) & set(masks))
    print(f"found {len(imgs)} images, {len(masks)} masks -> {len(keys)} pairs")
    return [(imgs[k], masks[k]) for k in keys]


def make_seg_split(pairs, out_csv=None, ratios=(0.7, 0.15, 0.15), seed=42):
    pairs = list(pairs)
    random.Random(seed).shuffle(pairs)
    n = len(pairs); a = int(ratios[0] * n); b = a + int(ratios[1] * n)
    rows = [{"image_path": str(i), "mask_path": str(m), "split": "train" if k < a else "val" if k < b else "test"}
            for k, (i, m) in enumerate(pairs)]
    df = pd.DataFrame(rows)
    out_csv = Path(out_csv or DATA_DIR / "seg_split.csv")
    df.to_csv(out_csv, index=False)
    print(df.split.value_counts().to_dict(), "->", out_csv)
    summarize(df, "segmentation" if out_csv.name == "seg_split.csv" else f"segmentation ({out_csv.stem})", out_csv)
    return df


def label_from_path(p, root):
    pos = {"positive", "crack", "cracked", "cracks", "pos", "yes"}
    neg = {"negative", "no_crack", "nocrack", "uncracked", "noncrack", "non_crack", "neg", "no", "intact"}
    for part in reversed(p.relative_to(root).parts[:-1]):
        low = part.lower()
        if low in pos or re.fullmatch(r"c[dpw]", low):
            return 1
        if low in neg or re.fullmatch(r"u[dpw]", low):
            return 0
    return None


def _stratified_split(df, seed, ratios=(0.7, 0.15, 0.15)):
    parts = []
    for _, g in df.groupby("label"):
        g = g.sample(frac=1, random_state=seed).reset_index(drop=True)
        a = int(ratios[0] * len(g)); b = a + int(ratios[1] * len(g))
        g["split"] = ["train" if i < a else "val" if i < b else "test" for i in range(len(g))]
        parts.append(g)
    return pd.concat(parts).reset_index(drop=True)


def make_cls_split(root, out_csv=None, max_per_class=None, seed=42):
    """Classification folders (SDNET2018 C*/U*, Positive/Negative, crack/no_crack ...)."""
    root = Path(root)
    rows = []
    for p in root.rglob("*"):
        if p.suffix.lower() in IMG_EXT and p.is_file():
            lab = label_from_path(p, root)
            if lab is not None:
                rows.append({"image_path": str(p), "label": lab})
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No labelled images found. Folder names must look like Positive/Negative, "
                         "crack/no_crack or SDNET2018's CD/UD, CP/UP, CW/UW.")
    if max_per_class:
        df = df.groupby("label", group_keys=False).apply(lambda g: g.sample(min(len(g), max_per_class), random_state=seed))
    df = _stratified_split(df, seed)
    out_csv = Path(out_csv or DATA_DIR / "cls_split.csv")
    df.to_csv(out_csv, index=False)
    print(df.groupby(["split", "label"]).size().unstack(fill_value=0), "->", out_csv)
    summarize(df, "classification", out_csv)
    return df


def make_cls_from_seg(seg_csv, out_dir=None, patch=128, min_crack_px=60, neg_ratio=1.5, seed=42):
    """Build a crack / no-crack patch dataset from segmentation masks (inherits the
    image-level split so no patch of a test image is ever used for training)."""
    rng = random.Random(seed)
    df = pd.read_csv(seg_csv)
    out_dir = Path(out_dir or DATA_DIR / "cls_patches")
    rows = []
    for split, g in df.groupby("split"):
        pos, neg = [], []
        for r in g.itertuples():
            img = cv2.imread(r.image_path); m = read_mask(r.mask_path)
            if img is None:
                continue
            if m.shape != img.shape[:2]:
                m = cv2.resize(m, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            stem = Path(r.image_path).stem
            for y in range(0, img.shape[0] - patch + 1, patch):
                for x in range(0, img.shape[1] - patch + 1, patch):
                    c = int(m[y:y + patch, x:x + patch].sum())
                    (pos if c >= min_crack_px else neg if c == 0 else []).append((img[y:y + patch, x:x + patch], f"{stem}_{x}_{y}"))
        rng.shuffle(neg); neg = neg[: int(neg_ratio * max(len(pos), 1))]
        for lab, items in ((1, pos), (0, neg)):
            d = out_dir / split / ("crack" if lab else "no_crack"); d.mkdir(parents=True, exist_ok=True)
            for patch_img, name in items:
                f = d / f"{name}.png"; cv2.imwrite(str(f), patch_img)
                rows.append({"image_path": str(f), "label": lab, "split": split})
    out = pd.DataFrame(rows)
    out_csv = DATA_DIR / "cls_split.csv"
    out.to_csv(out_csv, index=False)
    print(out.groupby(["split", "label"]).size().unstack(fill_value=0), "->", out_csv)
    summarize(out, "classification", out_csv)
    return out


def summarize(df, task, csv_path):
    """Writes results/dataset_summary.json (shown on the website's Datasets page)."""
    p = RESULTS_DIR / "dataset_summary.json"
    s = json.loads(p.read_text()) if p.exists() else {}
    s[task] = {"csv": str(csv_path), "n": int(len(df)), "per_split": df.split.value_counts().to_dict(),
               **({"per_class": {str(k): int(v) for k, v in df.label.value_counts().items()}} if "label" in df else {})}
    p.write_text(json.dumps(s, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["seg", "cls", "cls-from-seg"])
    ap.add_argument("--root"); ap.add_argument("--out_csv", default=None); ap.add_argument("--seg_csv", default=str(DATA_DIR / "seg_split.csv"))
    ap.add_argument("--max_per_class", type=int); ap.add_argument("--patch", type=int, default=128)
    a = ap.parse_args()
    if a.task == "seg":
        make_seg_split(auto_find_pairs(a.root), a.out_csv)
    elif a.task == "cls":
        make_cls_split(a.root, max_per_class=a.max_per_class)
    else:
        make_cls_from_seg(a.seg_csv, patch=a.patch)
