"""2D histogram of per-image (s_obj, s_bg), as in the explainability figures of the paper."""

from pathlib import Path

import torch
from matplotlib.figure import Figure


def histogram(out_dir, model_name, layer, s_obj, s_bg, summary):
    """Save the histogram as <out_dir>/histograms/<model>_L<layer>.png.

    s_obj, s_bg: (n_images,) per-image scores; images where either is NaN are skipped.
    layer: 0-indexed; the file name reports it 1-indexed.
    summary: the dict returned by metric.summarize.
    Points below the diagonal (s_obj > s_bg) are aligned, points above it inverted.
    """
    valid = torch.isfinite(s_obj) & torch.isfinite(s_bg)
    obj, bg = s_obj[valid].numpy(), s_bg[valid].numpy()

    fig = Figure(figsize=(4, 4))
    ax = fig.subplots()
    ax.hist2d(obj, bg, bins=50, cmap="Blues", cmin=1)
    lo, hi = min(obj.min(), bg.min()), max(obj.max(), bg.max())
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1)
    ax.set_xlabel(r"$s_\mathrm{obj}$")
    ax.set_ylabel(r"$s_\mathrm{bg}$")
    ax.set_title(f"{model_name}\nA = {summary['A']:.1f}% ({summary['regime']})", fontsize=9)
    fig.tight_layout()

    path = Path(out_dir) / "histograms" / f"{safe_name(model_name)}_L{layer + 1}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)


def safe_name(model_name):
    """Model name usable as a file name, e.g. "hf-hub:org/model" -> "hf-hub_org_model"."""
    return model_name.replace("/", "_").replace(":", "_")
