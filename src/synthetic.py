"""Synthetic crack images. ONLY for smoke-testing the pipeline / app layout.
Never report results from synthetic data as real project results."""
import argparse
import random
from pathlib import Path
import cv2
import numpy as np
from .augment import add_shadow


def _texture(size, rng):
    base = rng.randint(110, 175)
    img = np.full((size, size, 3), base, np.float32)
    for s in (4, 16, 64):
        n = np.random.RandomState(rng.randint(0, 10**6)).normal(0, 1, (s, s)).astype(np.float32)
        img += cv2.resize(n, (size, size), interpolation=cv2.INTER_CUBIC)[..., None] * (60.0 / s ** 0.5)
    img += np.random.RandomState(rng.randint(0, 10**6)).normal(0, 4, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def _draw_crack(mask, rng, size, thick_max=4):
    x, y = rng.randint(0, size - 1), rng.choice([0, size - 1]) if rng.random() < 0.5 else rng.randint(0, size - 1)
    ang = rng.uniform(0, 2 * np.pi)
    thick = rng.randint(1, thick_max)
    for i in range(rng.randint(40, 90)):
        ang += rng.uniform(-0.45, 0.45)
        step = rng.uniform(3, 7)
        nx, ny = int(x + step * np.cos(ang)), int(y + step * np.sin(ang))
        cv2.line(mask, (x, y), (nx, ny), 255, thick)
        if rng.random() < 0.06:                       # small branch
            bx, by, ba = nx, ny, ang + rng.choice([-1, 1]) * rng.uniform(0.6, 1.2)
            for _ in range(rng.randint(5, 15)):
                ba += rng.uniform(-0.4, 0.4)
                cx, cy = int(bx + 4 * np.cos(ba)), int(by + 4 * np.sin(ba))
                cv2.line(mask, (bx, by), (cx, cy), 255, 1)
                bx, by = cx, cy
        x, y = nx, ny
        if not (0 <= x < size and 0 <= y < size):
            break


def synth_pair(size=256, seed=None, crack=True):
    rng = random.Random(seed)
    img = _texture(size, rng)
    mask = np.zeros((size, size), np.uint8)
    if crack:
        for _ in range(rng.randint(1, 2)):
            _draw_crack(mask, rng, size)
        dark = cv2.GaussianBlur(mask, (0, 0), 0.8).astype(np.float32) / 255.0
        img = np.clip(img.astype(np.float32) * (1 - 0.75 * dark[..., None]), 0, 255).astype(np.uint8)
    if rng.random() < 0.4:
        img = add_shadow(img, rng)
    return cv2.GaussianBlur(img, (3, 3), 0), mask


def generate_seg_dataset(out_dir, n=120, size=256, seed=0):
    out = Path(out_dir); (out / "images").mkdir(parents=True, exist_ok=True); (out / "masks").mkdir(exist_ok=True)
    for i in range(n):
        img, m = synth_pair(size, seed + i, crack=(i % 10 != 0))
        cv2.imwrite(str(out / "images" / f"syn_{i:04d}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out / "masks" / f"syn_{i:04d}.png"), m)
    return out


def generate_samples(out_dir, n=4, size=384):
    out = Path(out_dir); out.mkdir(exist_ok=True); (out / "masks").mkdir(exist_ok=True)
    for i in range(n):
        img, m = synth_pair(size, 9000 + i, crack=(i != n - 1))
        cv2.imwrite(str(out / f"synthetic_{i + 1:02d}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out / "masks" / f"synthetic_{i + 1:02d}.png"), m)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic"); ap.add_argument("--n", type=int, default=120)
    a = ap.parse_args()
    print("written to", generate_seg_dataset(a.out, a.n))
