"""Metrics: segmentation (IoU, Dice, P, R, pixel acc) and classification."""
import numpy as np


def seg_metrics(pred, gt, eps=1e-7):
    p, g = np.asarray(pred).astype(bool), np.asarray(gt).astype(bool)
    tp = float((p & g).sum()); fp = float((p & ~g).sum())
    fn = float((~p & g).sum()); tn = float((~p & ~g).sum())
    return {
        "iou": (tp + eps) / (tp + fp + fn + eps),
        "dice": (2 * tp + eps) / (2 * tp + fp + fn + eps),
        "precision": (tp + eps) / (tp + fp + eps),
        "recall": (tp + eps) / (tp + fn + eps),
        "pixel_acc": (tp + tn) / max(p.size, 1),
    }


def mean_dicts(rows):
    if not rows:
        return {}
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


def auc_score(y_true, y_prob):
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    pos, neg = y_prob[y_true == 1], y_prob[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def cls_metrics(y_true, y_prob, thr=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_prob) >= thr).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum()); tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum()); fn = int(((y_pred == 0) & (y_true == 1)).sum())
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    return {
        "accuracy": (tp + tn) / max(len(y_true), 1),
        "precision": prec, "recall": rec,
        "f1": 2 * prec * rec / max(prec + rec, 1e-9),
        "auc": auc_score(y_true, y_prob),
        "confusion": [[tn, fp], [fn, tp]],
    }
