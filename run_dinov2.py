"""Explainability of a DINOv2 classifier on PartImageNet++.

DINOv2's official ImageNet-1k linear head reads the concatenation of the last layer's class token
and the mean of its patch tokens. The loss therefore reaches the last layer's patch tokens, and we
analyze that layer (Sec. 3.1). To score a patch token, we pass it through the backbone's final norm
and place it in the class-token slot of the head, keeping the mean of the patch tokens in the
other slot (App. A). A patch's score is the logit of the image's true class. Scores are averaged
over the object and background tokens of every image (Eq. 1) and summarized by the aligned rate A
(Eq. 2).

Example:
    python run_dinov2.py --model dinov2_vits14_lc \
        --imagenet-root /data/imagenet/train --annotations-root /data/PartImageNetPP
"""

import argparse

import torch
import torchvision.transforms as T
from tqdm import tqdm

from explainability import data, metric, plot

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SIZE = 224


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True,
                        help="torch.hub DINOv2 classifier, e.g. dinov2_vits14_lc or dinov2_vits14_reg_lc")
    data.add_common_args(parser)
    args = parser.parse_args()

    # layers=1: the head reads the class token and mean patch token of the last layer only.
    model = torch.hub.load("facebookresearch/dinov2", args.model, layers=1).to(args.device).eval()
    backbone, head = model.backbone, model.linear_head  # head: Linear(2 * D, 1000)
    depth = backbone.n_blocks
    layer = depth - 1  # the last layer, whose patch tokens the head reads
    dim = backbone.embed_dim

    # The shorter side is resized to S and the center S x S is cropped, for both image and mask.
    geometry = [T.Resize(SIZE, interpolation=T.InterpolationMode.BILINEAR), T.CenterCrop(SIZE)]
    dataset = data.PartImageNetPP(
        args.imagenet_root, args.annotations_root,
        image_transform=T.Compose(geometry + [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]),
        mask_transform=T.Compose(geometry),
        patch_size=backbone.patch_size,
    )

    def score(tokens, mean_patch, labels):
        """True-class logit of every token placed in the class-token slot of the head.

        tokens:     (B, N, D), already passed through the final norm.
        mean_patch: (B, D) mean of the image's normed patch tokens, for the other slot.
        labels:     (B,) class indices.
        Returns (B, N).
        """
        w_cls = head.weight[labels, :dim]  # (B, D) weights of the class-token slot
        w_avg = head.weight[labels, dim:]  # (B, D) weights of the mean-patch slot
        per_image = (mean_patch * w_avg).sum(dim=1) + head.bias[labels]  # (B,)
        return torch.einsum("bnd,bd->bn", tokens, w_cls) + per_image[:, None]

    all_obj, all_bg, all_labels, all_names = [], [], [], []
    with torch.no_grad():
        for i, (images, labels, masks, names) in enumerate(tqdm(data.make_loader(dataset, args))):
            # images (B, 3, S, S), labels (B,), masks (B, S/p, S/p)
            images, labels, masks = images.to(args.device), labels.to(args.device), masks.to(args.device)
            # patches (B, N, D): normed patch tokens of the last layer, class and register tokens removed.
            # cls (B, D): normed class token of the last layer.
            ((patches, cls),) = backbone.get_intermediate_layers(images, n=1, return_class_token=True, norm=True)
            mean_patch = patches.mean(dim=1)  # (B, D)

            if i == 0:
                # The same score applied to the class token itself must reproduce the model's own
                # logit, confirming that it uses the model's own final norm, head, and token layout.
                logits = model(images).gather(1, labels[:, None])[:, 0]  # (B,)
                torch.testing.assert_close(score(cls[:, None], mean_patch, labels)[:, 0], logits, rtol=1e-4, atol=1e-3)

            s_obj, s_bg = metric.obj_bg_means(score(patches, mean_patch, labels), masks)
            all_obj.append(s_obj.cpu())
            all_bg.append(s_bg.cpu())
            all_labels.append(labels.cpu())
            all_names.extend(names)

    s_obj, s_bg = torch.cat(all_obj), torch.cat(all_bg)
    summary = metric.save(args.out_dir, args.model, layer, depth, all_names, torch.cat(all_labels), s_obj, s_bg)
    plot.histogram(args.out_dir, args.model, layer, s_obj, s_bg, summary)


if __name__ == "__main__":
    main()
