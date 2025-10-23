import random
from typing import Optional, Tuple
from PIL import Image
import numpy as np
import torch
from monai import transforms
from torchvision.transforms import Resize, CenterCrop, Normalize
import torchvision.transforms.functional as tvf
from skimage import io


def remap_to_uint8(array: np.ndarray, percentiles: Optional[Tuple[float, float]] = None) -> np.ndarray:
    """Remap values in input so the output range is :math:`[0, 255]`.

    Percentiles can be used to specify the range of values to remap.
    This is useful to discard outliers in the input data.

    :param array: Input array.
    :param percentiles: Percentiles of the input values that will be mapped to ``0`` and ``255``.
        Passing ``None`` is equivalent to using percentiles ``(0, 100)`` (but faster).
    :returns: Array with ``0`` and ``255`` as minimum and maximum values.
    """
    array = array.astype(float)
    if percentiles is not None:
        len_percentiles = len(percentiles)
        if len_percentiles != 2:
            message = 'The value for percentiles should be a sequence of length 2,' f' but has length {len_percentiles}'
            raise ValueError(message)
        a, b = percentiles
        if a >= b:
            raise ValueError(f'Percentiles must be in ascending order, but a sequence "{percentiles}" was passed')
        if a < 0 or b > 100:
            raise ValueError(f'Percentiles must be in the range [0, 100], but a sequence "{percentiles}" was passed')
        cutoff: np.ndarray = np.percentile(array, percentiles)
        array = np.clip(array, *cutoff)
    array -= array.min()
    array /= array.max()
    array *= 255
    return array.astype(np.uint8)


def preprocess_img(img):
    img = remap_to_uint8(np.asarray(img))
    return Image.fromarray(img).convert("L")


def resize_according_to_long_side(x, size):
    flag = False
    if isinstance(x, Image.Image):
        w, h = x.size
    else:
        if x.dim() == 2:
            x.unsqueeze_(0)
            flag = True
        h, w = x.shape[1:]
    ratio = float(size / float(max(h, w)))
    new_w, new_h = round(w * ratio), round(h * ratio)
    x = tvf.resize(x, (new_h, new_w), antialias=None)
    if flag:
        x.squeeze_(0)
    # box = box * ratio
    return x


class RandomResize(object):
    def __init__(self, sizes):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes

    def __call__(self, x):
        size = random.choice(self.sizes)
        x = resize_according_to_long_side(x, size)
        return x


class Pad:
    def __init__(self, size):
        self.size = size
        
    def __call__(self, x):
        if x.dim() == 2:
            h, w = x.shape
        else:
            h, w = x.shape[1:]
        dw = self.size - w
        dh = self.size - h
        top = round(dh / 2.0 - 0.1)
        left = round(dw / 2.0 - 0.1)

        if x.dim() > 2:
            out = torch.zeros((x.shape[0], self.size, self.size))
            out[:, top:top+h, left:left+w] = x
        else:
            out = torch.ones((self.size, self.size))
            out[top:top+h, left:left+w] = 0

        return out


class ExpandChannels:
    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        assert data.shape[0] == 1
        return torch.repeat_interleave(data, 3, dim=0)


class ImgInversion:
    def __call__(self, ds):
        data =  ds.pixel_array.astype(float)
        if ds.PhotometricInterpretation == "MONOCHROME1":
            data = np.amax(data) - data
        data = data - np.min(data)
        data = data / np.max(data)
        img_grayscale = (data * 255).astype(np.uint8)
        return Image.fromarray(img_grayscale)


def input_transformations(name="ddpm", crop_size=448, split="train"):
    if name is None:
        trg_transforms = transforms.Compose(
                [
                    transforms.LoadImaged(keys=["image"], reader="pilreader"),
                    transforms.EnsureChannelFirstd(keys=["image"]),
                    transforms.Lambdad(
                        keys=["image"],
                        func=lambda x: x[0, :, :][None, ],
                    ),
                ],
                map_items=True
            )
    elif name == "ddpm":
        trg_transforms = transforms.Compose(
                [
                    transforms.LoadImaged(keys=["image"], reader="pilreader"),
                    transforms.EnsureChannelFirstd(keys=["image"]),
                    transforms.Lambdad(
                        keys=["image"],
                        func=lambda x: x[0, :, :][None, ],
                    ),
                    transforms.Rotate90d(keys=["image"], k=-1, spatial_axes=(0, 1)),  # Fix flipped image read
                    transforms.Flipd(keys=["image"], spatial_axis=1),  # Fix flipped image read
                    transforms.ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True),
                    transforms.Lambdad(keys=["image", "bbox"], func=lambda x: Resize(512, antialias=None)(x)),
                    transforms.CenterSpatialCropd(keys=["image", "bbox"], roi_size=(512, 512)),
                    transforms.ToTensord(keys=["image"]),
                ],
                map_items=True
            )
    elif name == "biovil_t":
        trg_transforms = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image"], reader="pilreader"),
                transforms.EnsureChannelFirstd(keys=["image"]),
                transforms.Lambdad(
                    keys=["image"],
                    func=lambda x: x[0, :, :][None, ],
                ),
                transforms.Rotate90d(keys=["image"], k=-1, spatial_axes=(0, 1)),  # Fix flipped image read
                transforms.Flipd(keys=["image"], spatial_axis=1),  # Fix flipped image read
                transforms.ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True),
                transforms.Lambdad(keys=["image", "bbox"], func=lambda x: Resize(512, antialias=None)(x)),
                transforms.Lambdad(keys=["image", "bbox"], func=lambda x: CenterCrop(crop_size)(x)),
                transforms.ToTensord(keys=["image"]),
                transforms.Lambdad(keys=["image"], func=lambda x: ExpandChannels()(x)),
            ],
            map_items=True
        )
    elif name == "medrpg":
        trg_transforms = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image"], reader="pilreader"),
                transforms.EnsureChannelFirstd(keys=["image"]),
                transforms.Lambdad(
                    keys=["image"],
                    func=lambda x: x[0, :, :][None, ],
                ),
                transforms.Lambdad(keys=["image", "bbox", "mask"], func=lambda x: RandomResize([crop_size])(x)),
                transforms.ToTensord(keys=["image"]),
                transforms.Lambdad(keys=["image"], func=lambda x: ExpandChannels()(x)),
                transforms.Lambdad(keys=["image"], func=lambda x: Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(x)),
                transforms.Lambdad(keys=["image", "bbox", "mask"], func=lambda x: Pad(crop_size)(x)),
            ],
            map_items=True
        )
    elif name == "vindr":
        temp_transforms = [
                transforms.LoadImaged(keys=["image"], reader="pilreader"),
                # transforms.Lambdad(keys=["image"], func=lambda x: ImgInversion()(x)),
                transforms.EnsureChannelFirstd(keys=["image"]),
                transforms.Lambdad(
                    keys=["image"],
                    func=lambda x: x[0, :, :][None, ],
                ),
                transforms.Rotate90d(keys=["image"], k=-1, spatial_axes=(0, 1)),  # Fix flipped image read
                transforms.Flipd(keys=["image"], spatial_axis=1),  # Fix flipped image read
                transforms.ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True),
                transforms.Lambdad(keys=["image", "bbox"], func=lambda x: Resize(512, antialias=None)(x)),
                transforms.CenterSpatialCropd(keys=["image", "bbox"], roi_size=(512, 512)),
            ]
        if split == "train":
            temp_transforms += [
                transforms.RandFlipd(keys=["image", "bbox"], prob=0.1, spatial_axis=1),
                transforms.RandAffined(keys=["image", "bbox"], prob=0.1, rotate_range=np.pi / 4, scale_range=(1.2, 1.2), translate_range=(200, 40), padding_mode="zeros"),
                transforms.AdjustContrastd(keys=["image"], gamma=2.0, invert_image=True, retain_stats=True),
                transforms.RandGaussianSharpend(keys=["image"])
            ]
        temp_transforms += [
            transforms.ToTensord(keys=["image"]),
            # transforms.Lambdad(keys=["image"], func=lambda x: ExpandChannels()(x)),
            # transforms.Lambdad(keys=["image"], func=lambda x: Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(x)),
            ]

        trg_transforms = transforms.Compose(
                temp_transforms,
                map_items=True
            )
    else:
        raise ValueError("Unknown transform name")
    
    return trg_transforms


def checkCoord(x, dim):
    if x < 0:
        x = 0
    if x > dim:
        x = dim
    return x