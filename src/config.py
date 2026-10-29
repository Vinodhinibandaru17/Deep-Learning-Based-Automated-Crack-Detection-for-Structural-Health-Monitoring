"""Central configuration: paths, sizes and model names."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
WEIGHTS_DIR = ROOT / "weights"
RESULTS_DIR = ROOT / "results"
SAMPLES_DIR = ROOT / "sample_images"
for _d in (DATA_DIR, WEIGHTS_DIR, RESULTS_DIR, SAMPLES_DIR):
    try:
        _d.mkdir(exist_ok=True)
    except OSError:
        pass

IMG_SIZE = 256          # segmentation training / inference size (multiple of 16)
CLS_SIZE = 224          # classification input size
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

SEG_MODELS = {
    "unet": "Baseline U-Net (BCE)",
    "attention_unet": "Enhanced Attention U-Net (BCE+Dice)",
}
CLS_MODELS = {
    "resnet18": "Baseline ResNet-18",
    "efficientnet_b0": "EfficientNet-B0",
    "mobilenet_v3_small": "MobileNetV3-Small (lightweight)",
}
YOLO_WEIGHTS = WEIGHTS_DIR / "yolo_crack.pt"
