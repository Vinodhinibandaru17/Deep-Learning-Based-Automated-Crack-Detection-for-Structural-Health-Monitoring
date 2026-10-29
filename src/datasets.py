"""PyTorch datasets for segmentation and classification."""
import random
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from .augment import normalize_np, perturb_image, random_augment
from .config import CLS_SIZE, IMG_SIZE
from .data_prep import read_mask


def _read_rgb(path):
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _to_tensor(img):
    return torch.from_numpy(normalize_np(img)).permute(2, 0, 1).float()


class SegDataset(Dataset):
    """Returns (image[3,H,W], mask[1,H,W]). split='all' uses every row.
    perturb=(kind, severity) applies a robustness perturbation to the image."""

    def __init__(self, csv_path, split="train", img_size=IMG_SIZE, augment=False, perturb=None):
        df = pd.read_csv(csv_path)
        if split != "all":
            df = df[df["split"] == split]
        self.items = list(zip(df["image_path"], df["mask_path"]))
        self.size, self.augment, self.perturb = img_size, augment, perturb

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        ip, mp = self.items[i]
        img, m = _read_rgb(ip), read_mask(mp).astype(np.float32)
        s = self.size
        img = cv2.resize(img, (s, s), interpolation=cv2.INTER_AREA)
        m = (cv2.resize(m, (s, s), interpolation=cv2.INTER_AREA) > 0.25).astype(np.float32)  # keep thin cracks
        if self.augment:
            img, m = random_augment(img, m, random)
        if self.perturb:
            img = perturb_image(img, *self.perturb)
        return _to_tensor(img), torch.from_numpy(np.ascontiguousarray(m))[None]


class ClsDataset(Dataset):
    """Returns (image[3,H,W], label float)."""

    def __init__(self, csv_path, split="train", size=CLS_SIZE, augment=False):
        df = pd.read_csv(csv_path)
        if split != "all":
            df = df[df["split"] == split]
        self.items = list(zip(df["image_path"], df["label"].astype(float)))
        self.size, self.augment = size, augment

    def __len__(self):
        return len(self.items)

    def labels(self):
        return np.array([l for _, l in self.items])

    def __getitem__(self, i):
        p, lab = self.items[i]
        img = cv2.resize(_read_rgb(p), (self.size, self.size), interpolation=cv2.INTER_AREA)
        if self.augment:
            img, _ = random_augment(img, None, random, zoom=False)
        return _to_tensor(img), torch.tensor(lab, dtype=torch.float32)
