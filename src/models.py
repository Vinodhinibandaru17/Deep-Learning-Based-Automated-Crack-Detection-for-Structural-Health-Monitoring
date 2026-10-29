"""Models: baseline U-Net, Attention U-Net (enhancement) and CNN classifiers."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Sequential):
    def __init__(self, i, o):
        super().__init__(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                         nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class AttentionGate(nn.Module):
    """Additive attention gate (Oktay et al., 2018): re-weights skip features using the
    coarser decoder signal, which suppresses background texture and keeps thin cracks."""

    def __init__(self, f_g, f_l, f_int):
        super().__init__()
        self.wg = nn.Sequential(nn.Conv2d(f_g, f_int, 1), nn.BatchNorm2d(f_int))
        self.wx = nn.Sequential(nn.Conv2d(f_l, f_int, 1), nn.BatchNorm2d(f_int))
        self.psi = nn.Sequential(nn.Conv2d(f_int, 1, 1), nn.BatchNorm2d(1), nn.Sigmoid())

    def forward(self, g, x):
        return x * self.psi(F.relu(self.wg(g) + self.wx(x), inplace=True))


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, attention):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.att = AttentionGate(out_ch, out_ch, max(out_ch // 2, 8)) if attention else None
        self.conv = DoubleConv(out_ch * 2, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        if self.att is not None:
            skip = self.att(x, skip)
        return self.conv(torch.cat([skip, x], 1))


class UNet(nn.Module):
    def __init__(self, in_ch=3, n_classes=1, base=32, attention=False):
        super().__init__()
        b = base
        self.inc = DoubleConv(in_ch, b)
        self.d1, self.d2 = DoubleConv(b, 2 * b), DoubleConv(2 * b, 4 * b)
        self.d3, self.d4 = DoubleConv(4 * b, 8 * b), DoubleConv(8 * b, 16 * b)
        self.pool = nn.MaxPool2d(2)
        self.u4, self.u3 = Up(16 * b, 8 * b, attention), Up(8 * b, 4 * b, attention)
        self.u2, self.u1 = Up(4 * b, 2 * b, attention), Up(2 * b, b, attention)
        self.out = nn.Conv2d(b, n_classes, 1)

    def forward(self, x):                       # returns logits
        x1 = self.inc(x)
        x2 = self.d1(self.pool(x1)); x3 = self.d2(self.pool(x2))
        x4 = self.d3(self.pool(x3)); x5 = self.d4(self.pool(x4))
        y = self.u4(x5, x4); y = self.u3(y, x3); y = self.u2(y, x2); y = self.u1(y, x1)
        return self.out(y)


def build_seg_model(name, base=32):
    if name == "unet":
        return UNet(base=base, attention=False)
    if name == "attention_unet":
        return UNet(base=base, attention=True)
    raise ValueError(f"unknown segmentation model {name}")


def build_classifier(name, pretrained=True):
    """Single-logit binary classifier (crack / no crack)."""
    import torchvision.models as tvm

    def make(ctor, weights):
        try:
            return ctor(weights=weights if pretrained else None)
        except Exception as e:                  # no internet -> random init
            print(f"[warn] could not load pretrained weights ({e}); using random init")
            return ctor(weights=None)

    if name == "resnet18":
        m = make(tvm.resnet18, tvm.ResNet18_Weights.DEFAULT)
        m.fc = nn.Linear(m.fc.in_features, 1)
    elif name == "efficientnet_b0":
        m = make(tvm.efficientnet_b0, tvm.EfficientNet_B0_Weights.DEFAULT)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, 1)
    elif name == "mobilenet_v3_small":
        m = make(tvm.mobilenet_v3_small, tvm.MobileNet_V3_Small_Weights.DEFAULT)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, 1)
    else:
        raise ValueError(f"unknown classifier {name}")
    return m


def count_params(m):
    return int(sum(p.numel() for p in m.parameters()))
