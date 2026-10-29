"""Shrink checkpoints to fp16 so every file is < 25 MB (GitHub web-upload limit).
Loading still works: load_state_dict casts back to fp32.   python -m src.export_weights"""
import torch
from .config import WEIGHTS_DIR

for p in sorted(WEIGHTS_DIR.glob("*.pt")):
    if p.name.startswith("yolo"):
        continue
    ck = torch.load(p, map_location="cpu", weights_only=False)
    before = p.stat().st_size / 1e6
    ck["model"] = {k: (v.half() if v.is_floating_point() else v) for k, v in ck["model"].items()}
    torch.save(ck, p)
    print(f"{p.name}: {before:.1f} MB -> {p.stat().st_size / 1e6:.1f} MB")
