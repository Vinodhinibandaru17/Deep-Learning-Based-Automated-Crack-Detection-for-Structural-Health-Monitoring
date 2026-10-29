"""Derive a YOLO detection dataset (boxes) from the segmentation masks (optional, Objective 3).

python -m src.make_yolo_dataset --seg_csv data/seg_split.csv --out data/yolo
"""
import argparse, shutil
from pathlib import Path
import cv2
import pandas as pd
from .analysis import boxes_from_mask
from .config import DATA_DIR
from .data_prep import read_mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg_csv", default=str(DATA_DIR / "seg_split.csv"))
    ap.add_argument("--out", default=str(DATA_DIR / "yolo"))
    a = ap.parse_args()
    out = Path(a.out).resolve()
    df = pd.read_csv(a.seg_csv)
    for r in df.itertuples():
        (out / "images" / r.split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / r.split).mkdir(parents=True, exist_ok=True)
        img = cv2.imread(r.image_path); h, w = img.shape[:2]
        m = read_mask(r.mask_path)
        if m.shape != (h, w):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        stem = Path(r.image_path).stem
        shutil.copy(r.image_path, out / "images" / r.split / Path(r.image_path).name)
        lines = [f"0 {(b['x'] + b['w'] / 2) / w:.6f} {(b['y'] + b['h'] / 2) / h:.6f} {b['w'] / w:.6f} {b['h'] / h:.6f}"
                 for b in boxes_from_mask(m)]
        (out / "labels" / r.split / f"{stem}.txt").write_text("\n".join(lines))
    (out / "data.yaml").write_text(f"path: {out}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: crack\n")
    print("YOLO dataset ready ->", out / "data.yaml")


if __name__ == "__main__":
    main()
