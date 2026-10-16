"""Explainability of an OpenCLIP image encoder on PartImageNet++.

Each patch token of the layer before the last is passed through the encoder's final norm and
projection, and its score is the cosine similarity with the text embedding of
"a photo of a {class}" for the image's true class (Sec. 3.1). Scores are averaged over the object
and background tokens of every image (Eq. 1) and summarized by the aligned rate A (Eq. 2).

Examples:
    python run_clip.py --model ViT-B-32 --pretrained openai \
        --imagenet-root /data/imagenet/train --annotations-root /data/PartImageNetPP
    python run_clip.py --model ViT-B-16 --pretrained /ckpts/scaling_laws_checkpoint.pt ...
"""

import argparse
from pathlib import Path

import open_clip
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from tqdm import tqdm

from explainability import data, metric, plot
from imagenet_classes import IMAGENET_CLASSES


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="OpenCLIP architecture, e.g. ViT-B-32")
    parser.add_argument("--pretrained", required=True,
                        help="OpenCLIP pretrained tag (e.g. openai, metaclip_fullcc) or a checkpoint file")
    data.add_common_args(parser)
    args = parser.parse_args()

    # Older checkpoints, such as the scaling-law ones, need full unpickling to load.
    extra = dict(weights_only=False) if Path(args.pretrained).is_file() else {}
    model, _, preprocess = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained, **extra)
    model = model.to(args.device).eval()
    tokenizer = open_clip.get_tokenizer(args.model)
    visual = model.visual
    depth = len(visual.transformer.resblocks)
    layer = depth - 2  # the layer before the last, whose patch tokens the loss never reads
    size = visual.image_size[0] if isinstance(visual.image_size, (tuple, list)) else visual.image_size
    patch = visual.patch_size[0] if isinstance(visual.patch_size, (tuple, list)) else visual.patch_size
    normalize = preprocess.transforms[-1]
    assert isinstance(normalize, T.Normalize), f"expected Normalize as the last transform, got {normalize}"

    # CLIP's own preprocessing: the shorter side is resized to S and the center S x S is cropped.
    # The mask gets the same geometry with nearest-neighbor resizing.
    dataset = data.PartImageNetPP(
        args.imagenet_root, args.annotations_root,
        image_transform=T.Compose([
            T.Resize(size, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(size),
            T.ToTensor(),
            normalize,
        ]),
        mask_transform=T.Compose([
            T.Resize(size, interpolation=T.InterpolationMode.NEAREST),
            T.CenterCrop(size),
        ]),
        patch_size=patch,
    )

    with torch.no_grad():
        prompts = [f"a photo of a {name}" for name in IMAGENET_CLASSES]
        text = torch.cat([model.encode_text(tokenizer(prompts[i:i + 250]).to(args.device))
                          for i in range(0, len(prompts), 250)])
        text = F.normalize(text, dim=-1)  # (1000, E)

    def score(tokens, labels):
        """Cosine similarity of every token with the true class's text embedding.

        tokens: (B, N, D), already passed through the final norm (ln_post).
        labels: (B,) class indices.
        Returns (B, N).
        """
        embedded = F.normalize(tokens @ visual.proj, dim=-1)  # (B, N, E)
        return torch.einsum("bne,be->bn", embedded, text[labels])

    all_obj, all_bg, all_labels, all_names = [], [], [], []
    with torch.no_grad():
        for i, (images, labels, masks, names) in enumerate(tqdm(data.make_loader(dataset, args))):
            # images (B, 3, S, S), labels (B,), masks (B, S/p, S/p)
            images, labels, masks = images.to(args.device), labels.to(args.device), masks.to(args.device)
            out = visual.forward_intermediates(
                images, indices=[layer, depth - 1], normalize_intermediates=True, intermediates_only=True,
                output_fmt="NLC", output_extra_tokens=True)
            patches = out["image_intermediates"][0]              # (B, N, D) layer before the last
            cls_last = out["image_intermediates_prefix"][1]      # (B, 1, D) class token of the last layer

            if i == 0:
                # The same score applied to the last layer's class token must reproduce the
                # similarity of CLIP's own image embedding, confirming that it uses the encoder's
                # own final norm and projection.
                image_emb = F.normalize(model.encode_image(images), dim=-1)  # (B, E)
                similarity = (image_emb * text[labels]).sum(dim=1)            # (B,)
                torch.testing.assert_close(score(cls_last, labels)[:, 0], similarity, rtol=1e-4, atol=1e-4)

            s_obj, s_bg = metric.obj_bg_means(score(patches, labels), masks)
            all_obj.append(s_obj.cpu())
            all_bg.append(s_bg.cpu())
            all_labels.append(labels.cpu())
            all_names.extend(names)

    name = f"{args.model}_{Path(args.pretrained).stem if Path(args.pretrained).is_file() else args.pretrained}"
    s_obj, s_bg = torch.cat(all_obj), torch.cat(all_bg)
    summary = metric.save(args.out_dir, name, layer, depth, all_names, torch.cat(all_labels), s_obj, s_bg)
    plot.histogram(args.out_dir, name, layer, s_obj, s_bg, summary)


if __name__ == "__main__":
    main()
