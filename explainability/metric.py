"""Explainability scores s_obj and s_bg (Eq. 1), the aligned rate A (Eq. 2), and the regimes."""

import csv
import math
from pathlib import Path

import torch

from explainability.plot import safe_name

ALIGNED_MIN = 66.7        # A >= 66.7%: aligned-explainability
UNEXPLAINABLE_MIN = 33.3  # 33.3% <= A < 66.7%: unexplainability; below: inverted-explainability


def obj_bg_means(scores, mask):
    """Mean patch score over the object tokens and over the background tokens (Eq. 1).

    scores: (B, N) score of every patch token, in raster order.
    mask:   (B, rows, cols) bool with rows * cols = N, True for object tokens.
    Returns (s_obj, s_bg), each (B,). An image with no object or no background tokens gets NaN.
    """
    obj = mask.reshape(scores.shape).float()  # (B, N)
    bg = 1.0 - obj                            # (B, N)
    s_obj = (scores * obj).sum(dim=1) / obj.sum(dim=1)
    s_bg = (scores * bg).sum(dim=1) / bg.sum(dim=1)
    return s_obj, s_bg


def summarize(s_obj, s_bg):
    """The aligned rate A (Eq. 2) over images where both means are defined, and its regime.

    s_obj, s_bg: (n_images,) per-image scores, possibly NaN.
    Returns dict(n_images, n_valid, A, regime), with A in percent.
    """
    valid = torch.isfinite(s_obj) & torch.isfinite(s_bg)  # (n_images,)
    n_valid = int(valid.sum())
    A = 100.0 * float((s_obj[valid] > s_bg[valid]).sum()) / n_valid
    if A >= ALIGNED_MIN:
        regime = "aligned"
    elif A >= UNEXPLAINABLE_MIN:
        regime = "unexplainable"
    else:
        regime = "inverted"
    return dict(n_images=len(s_obj), n_valid=n_valid, A=A, regime=regime)


def save(out_dir, model_name, layer, depth, names, labels, s_obj, s_bg):
    """Write per-image scores, append a summary row to results.csv, and return the summary.

    names:  list of n_images image names.
    labels: (n_images,) ImageNet-1k class indices.
    s_obj, s_bg: (n_images,) per-image scores, possibly NaN.
    layer is 0-indexed; files report it 1-indexed as "<layer> of <depth>", as in the paper.
    """
    summary = summarize(s_obj, s_bg)
    out_dir = Path(out_dir)

    scores_path = out_dir / "scores" / f"{safe_name(model_name)}_L{layer + 1}.csv"
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    with scores_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "label", "s_obj", "s_bg"])
        for name, label, obj, bg in zip(names, labels.tolist(), s_obj.tolist(), s_bg.tolist()):
            writer.writerow([name, label, _fmt(obj), _fmt(bg)])

    results_path = out_dir / "results.csv"
    new_file = not results_path.exists()
    with results_path.open("a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["model", "layer", "n_images", "n_valid", "A", "regime"])
        writer.writerow([model_name, f"{layer + 1} of {depth}", summary["n_images"],
                         summary["n_valid"], f"{summary['A']:.2f}", summary["regime"]])

    print(f"{model_name}, layer {layer + 1} of {depth}: A = {summary['A']:.1f}% "
          f"({summary['regime']}), {summary['n_valid']} of {summary['n_images']} images valid")
    return summary


def _fmt(x):
    return "nan" if math.isnan(x) else f"{x:.6g}"
