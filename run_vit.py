"""Explainability of a timm ViT classifier on PartImageNet++.

Each patch token of the layer before the last is passed through the model's final norm and
classification head, and its score is the logit of the image's true class (Sec. 3.1). Scores
are averaged over the object and background tokens of every image (Eq. 1) and summarized by
the aligned rate A (Eq. 2).

Example:
    python run_vit.py --model vit_tiny_patch16_224.augreg_in21k_ft_in1k \
        --imagenet-root /data/imagenet/train --annotations-root /data/PartImageNetPP
"""

import argparse

import timm
import torch
import torchvision.transforms as T
from tqdm import tqdm

from explainability import data, metric, plot


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="timm model identifier (see Tab. S2)")
    data.add_common_args(parser)
    args = parser.parse_args()

    model = timm.create_model(args.model, pretrained=True).to(args.device).eval()
    depth = len(model.blocks)
    layer = depth - 2  # the layer before the last, whose patch tokens the loss never reads
    cfg = model.pretrained_cfg
    size = cfg["input_size"][-1]

    # As for the classifiers in the paper, the whole image is resized to S x S, without cropping.
    dataset = data.PartImageNetPP(
        args.imagenet_root, args.annotations_root,
        image_transform=T.Compose([
            T.Resize((size, size), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(cfg.get("mean", timm.data.IMAGENET_DEFAULT_MEAN), cfg.get("std", timm.data.IMAGENET_DEFAULT_STD)),
        ]),
        mask_transform=T.Resize((size, size), interpolation=T.InterpolationMode.BILINEAR),
        patch_size=model.patch_embed.patch_size[0],
    )

    def score(tokens, labels):
        """True-class logit of every token.

        tokens: (B, N, D), already passed through the final norm.
        labels: (B,) class indices.
        Returns (B, N).
        """
        weight, bias = model.head.weight[labels], model.head.bias[labels]  # (B, D), (B,)
        return torch.einsum("bnd,bd->bn", tokens, weight) + bias[:, None]

    all_obj, all_bg, all_labels, all_names = [], [], [], []
    with torch.no_grad():
        for i, (images, labels, masks, names) in enumerate(tqdm(data.make_loader(dataset, args))):
            # images (B, 3, S, S), labels (B,), masks (B, S/p, S/p)
            images, labels, masks = images.to(args.device), labels.to(args.device), masks.to(args.device)
            # patches (B, N, D): patch tokens of the layer before the last.
            # prefix_last (B, 1, D): class token of the last layer.
            (patches, _), (_, prefix_last) = model.forward_intermediates(
                images, indices=[layer, depth - 1], norm=True, return_prefix_tokens=True,
                output_fmt="NLC", intermediates_only=True)

            if i == 0:
                # The same score applied to the last layer's class token must reproduce the
                # model's own logit, confirming that it uses the model's own final norm and head.
                logits = model(images).gather(1, labels[:, None])[:, 0]  # (B,)
                torch.testing.assert_close(score(prefix_last[:, :1], labels)[:, 0], logits, rtol=1e-4, atol=1e-3)

            s_obj, s_bg = metric.obj_bg_means(score(patches, labels), masks)
            all_obj.append(s_obj.cpu())
            all_bg.append(s_bg.cpu())
            all_labels.append(labels.cpu())
            all_names.extend(names)

    s_obj, s_bg = torch.cat(all_obj), torch.cat(all_bg)
    summary = metric.save(args.out_dir, args.model, layer, depth, all_names, torch.cat(all_labels), s_obj, s_bg)
    plot.histogram(args.out_dir, args.model, layer, s_obj, s_bg, summary)


if __name__ == "__main__":
    main()
