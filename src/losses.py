"""Loss functions for the heavy foreground/background imbalance of crack masks."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(logits, target, eps=1.0):
    p = torch.sigmoid(logits.float())
    inter = (p * target).sum((1, 2, 3))
    return (1 - (2 * inter + eps) / (p.sum((1, 2, 3)) + target.sum((1, 2, 3)) + eps)).mean()


class BCEDiceLoss(nn.Module):
    def __init__(self, w_bce=0.5, w_dice=0.5):
        super().__init__(); self.wb, self.wd = w_bce, w_dice

    def forward(self, logits, target):
        logits = logits.float()
        return self.wb * F.binary_cross_entropy_with_logits(logits, target) + self.wd * dice_loss(logits, target)


class FocalDiceLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0, w_focal=0.5, w_dice=0.5):
        super().__init__(); self.a, self.g, self.wf, self.wd = alpha, gamma, w_focal, w_dice

    def forward(self, logits, target):
        logits = logits.float()
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        pt = torch.exp(-bce)
        at = self.a * target + (1 - self.a) * (1 - target)
        return self.wf * (at * (1 - pt) ** self.g * bce).mean() + self.wd * dice_loss(logits, target)


def get_loss(name):
    return {"bce": nn.BCEWithLogitsLoss, "bce_dice": BCEDiceLoss, "focal_dice": FocalDiceLoss}[name]()
