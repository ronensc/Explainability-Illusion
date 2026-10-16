# The Explainability Illusion: Must ViTs Exhibit Patch-Decoding Interpretability? <!-- omit in toc -->

[![Python 3.12](https://img.shields.io/badge/python-3.12+-blue?logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3120/)
[![torch](https://img.shields.io/badge/torch-2.9+-green?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![torchvision](https://img.shields.io/badge/torchvision-0.24+-green?logo=pytorch&logoColor=white)](https://pytorch.org/)

### [Project page](https://ronensc.github.io/Explainability-Illusion) | [Arxiv](https://arxiv.org/abs/2610.25289)  <!-- omit in toc -->

#### [Ronen Schaffer](https://medium.com/@ronen.schaffer/)<sup>\*</sup>, [Hila Manor](https://hilamanor.github.io/)<sup>\*</sup>, [Tomer Michaeli](https://tomer.net.technion.ac.il/) <!-- omit in toc -->



This repository contains the official code release for ***The Explainability Illusion: Must ViTs Exhibit Patch-Decoding Interpretability?***.

**TL;DR:** We show the widespread presumption of patch-decoding explainability is not an inherent property of Vision Transformers, but an outcome that cannot be assumed to hold a-priori.



## Table of Contents <!-- omit in toc -->

- [Change Log](#change-log)
- [Requirements](#requirements)
- [Usage](#usage)
  - [Real-World ViTs Explainability](#real-world-vits-explainability)
    - [SigLIP model](#siglip-model)
  - [Synthetic ViTs Explainability](#synthetic-vits-explainability)
- [Code layout](#code-layout)
- [Citation](#citation)

## Change Log

- **16.10.26**: Initial release with evaluation code for timm ViT classifiers, DINOv2 classifiers, and CLIP models. Weights for SigLIP and the synthetic-data experiment code are TBD.

## Requirements

Running the code requires two datasets, which we cannot redistribute:

1. **ImageNet-1k training images** ([image-net.org](https://www.image-net.org/)), in the standard layout with one folder per class:
   ```
   <imagenet-root>/n01440764/n01440764_10026.JPEG
   ```
2. **PartImageNet++ annotations** (Li et al., "PartImageNet++ Dataset: Scaling up Part-based Models for Robust Recognition", ECCV 2024; [project](https://github.com/LixiaoTHU/PartImageNetPP)), which annotate a subset of the ImageNet-1k training images.
   Download them from [Hugging Face](https://huggingface.co/datasets/lixiao20/PartImageNetPP):
   ```
   <annotations-root>/json/n01440764.json      # one file per class
   <annotations-root>/discarded_data.json
   ```

The evaluation uses the 99,677 annotated images of PartImageNet++.

```bash
pip install -r requirements.txt
```

Pretrained weights are downloaded on first use through `timm`, `torch.hub`, and OpenCLIP.

## Usage

### Real-World ViTs Explainability

To compute the explainability scores $s_\mathrm{obj}$ and $s_\mathrm{bg}$ (Eq. 1) and the aligned rate $\mathcal{A}$ (Eq. 2) of a model, run the corresponding script with the `--imagenet-root` and `--annotations-root` arguments pointing to the datasets.
This reproduces the results reported in Tab. 1 of the paper.



There is one script per model family.
Each script reads the patch tokens of the layer analyzed in the paper, scores every patch token, averages the scores over the object and background tokens of each image, and reports $\mathcal{A}$ and the resulting explainability regime.

```bash
DATA="--imagenet-root <imagenet-root> --annotations-root <annotations-root>"

# ViT classifiers from timm (Tab. S2 and S3 list the identifiers of all models in the paper)
python run_vit.py --model vit_tiny_patch16_224.augreg_in21k_ft_in1k $DATA

# DINOv2 classifiers with their official ImageNet-1k linear heads, from torch.hub
python run_dinov2.py --model dinov2_vits14_lc $DATA

# Contrastive models from OpenCLIP: an architecture and a pretrained tag or checkpoint file
python run_clip.py --model ViT-B-32 --pretrained openai $DATA
```

A full run over PartImageNet++ takes a few minutes for a small model on a single GPU.
For a quick check, `--max-images 500` evaluates a fixed random subset of 500 images, which gives a rough estimate of $\mathcal{A}$.

Other options are `--out-dir` (default `outputs`), `--batch-size`, `--num-workers`, and `--device`.

#### Outputs <!-- omit in toc -->

Each run prints $\mathcal{A}$ and writes to `--out-dir`:

- `results.csv`, with one row per run: model, analyzed layer, number of images, number of valid images, $\mathcal{A}$ (%), and regime.
- `scores/<model>_L<layer>.csv`, with $s_\mathrm{obj}$ and $s_\mathrm{bg}$ for every image.
- `histograms/<model>_L<layer>.png`, the 2D histogram of $(s_\mathrm{obj}, s_\mathrm{bg})$ over all images.

An image is valid when it has at least one object token and one background token, where a token is assigned to the object if more than half of its pixels lie inside the object mask.
$\mathcal{A}$ is computed over the valid images.

#### SigLIP model

Since the original pre-trained [SigLIP](https://github.com/merveenoyan/siglip) image encoders do not employ a class token, we train a class-token ViT-B/32 SigLIP on [LAION-400M](https://laion.ai/blog/laion-400-open-dataset/). The trained checkpoint is pending release.

### Synthetic ViTs Explainability

Code for our controlled experiments using synthetic data is pending release.

## Code layout

```
run_vit.py            timm ViT classifiers: layer before the last, final norm, classification head
run_dinov2.py         DINOv2 classifiers: last layer, final norm, official linear head
run_clip.py           OpenCLIP models: layer before the last, final norm, projection, text similarity
imagenet_classes.py   class names for the CLIP text prompts
explainability/
  data.py             PartImageNet++ images and object masks on the token grid
  metric.py           s_obj and s_bg (Eq. 1), A (Eq. 2), regimes, and CSV outputs
  plot.py             2D histograms
```

On its first batch, each script also checks that its patch-scoring function, applied to the class token of the last layer, reproduces the model's own output.
This confirms that patch tokens are decoded with the model's own final normalization and head.

## Citation

If you found our work useful for your research, please cite our paper:

```latex
@article{schaffer2026explainability,
  title={The Explainability Illusion: Must ViTs Exhibit Patch-Decoding Interpretability?},
  author={Schaffer, Ronen and Manor, Hila and Michaeli, Tomer},
  journal={arXiv preprint arXiv:2610.25289},
  year={2026}
}
```
