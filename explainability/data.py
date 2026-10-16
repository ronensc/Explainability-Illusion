"""PartImageNet++ images with whole-object masks on the model's token grid."""

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_utils
from scipy import ndimage
from torch.utils.data import DataLoader, Dataset, Subset


class PartImageNetPP(Dataset):
    """Yields (image, label, mask, name) for every annotated PartImageNet++ image.

    image: the output of `image_transform`, e.g. a (3, S, S) float tensor.
    label: ImageNet-1k class index.
    mask:  (S/p, S/p) bool tensor for an S x S input and patch size p. A token is part of the
           object when more than half of its pixels lie inside the object mask.
    name:  image file name without extension, e.g. "n01440764_10026".

    The object mask is the union of all part annotations of the image, with holes filled.
    `mask_transform` receives it as a (1, H, W) float tensor and must return a (1, S, S)
    tensor with the same geometry (resize and crop) as `image_transform`.
    """

    def __init__(self, imagenet_root, annotations_root, image_transform, mask_transform, patch_size):
        imagenet_root, annotations_root = Path(imagenet_root), Path(annotations_root)
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.patch_size = patch_size

        discarded = {item["file_name"] for item in json.loads((annotations_root / "discarded_data.json").read_text())}

        # One JSON file per class, named by its WordNet ID; sorted IDs give ImageNet-1k class order.
        json_paths = sorted((annotations_root / "json").glob("*.json"))
        assert len(json_paths) == 1000, f"expected 1000 class files, found {len(json_paths)}"

        self.samples = []
        for label, json_path in enumerate(json_paths):
            data = json.loads(json_path.read_text())
            annotations = {}
            for ann in data["annotations"]:
                annotations.setdefault(ann["image_id"], []).append(ann["segmentation"])
            for image in data["images"]:
                path = imagenet_root / image["file_name"]  # e.g. .../n01440764/n01440764_10026.JPEG
                if path.name in discarded or image["id"] not in annotations:
                    continue
                self.samples.append(dict(
                    path=path,
                    label=label,
                    polygons=annotations[image["id"]],
                    height=image["height"],
                    width=image["width"],
                ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = self.image_transform(Image.open(sample["path"]).convert("RGB"))

        h, w = sample["height"], sample["width"]
        mask = np.zeros((h, w), dtype=bool)
        for polygons in sample["polygons"]:
            mask |= mask_utils.decode(mask_utils.merge(mask_utils.frPyObjects(polygons, h, w))).astype(bool)
        mask = ndimage.binary_fill_holes(mask)
        mask = self.mask_transform(torch.from_numpy(mask).float()[None])[0]  # (S, S)

        # Assign each token to the object by majority vote over its pixels.
        p = self.patch_size
        rows, cols = mask.shape[0] // p, mask.shape[1] // p
        mask = mask.reshape(rows, p, cols, p).mean(dim=(1, 3)) > 0.5  # (S/p, S/p)

        return image, sample["label"], mask, sample["path"].stem


def add_common_args(parser):
    parser.add_argument("--imagenet-root", required=True,
                        help="ImageNet-1k training images, one folder per class (e.g. .../train/n01440764/)")
    parser.add_argument("--annotations-root", required=True,
                        help="PartImageNet++ annotations, containing json/ and discarded_data.json")
    parser.add_argument("--out-dir", default="outputs")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Evaluate a fixed random subset of this many images, for a quick run")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")


def make_loader(dataset, args):
    """Batches of (images, labels, masks, names), in dataset order, keeping the last partial batch."""
    if args.max_images is not None:
        order = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(0))
        dataset = Subset(dataset, order[:args.max_images].tolist())
    return DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                      shuffle=False, pin_memory=True)
