"""
Feature extraction for CT linear probing using a pretrained DenseNet121 3D encoder.

This script converts each input CT volume into a fixed-length feature embedding.
The output is saved as one HDF5 file per case, with the feature stored under
the key "y_hat".

Supported modes:
1. Image-only mode:
   - Center crop or pad the full CT volume to a fixed size.
2. ROI mode:
   - If foreground masks are provided, crop around the foreground center before
     feature extraction.

Expected input:
- CT images in NIfTI format: *.nii.gz
- Optional foreground masks with the same filenames as the CT images.

Example:
    python scripts/extract_feat_LP_densenet.py \
        --input /path/to/images \
        --output ./outputs/features \
        --checkpoint ./outputs/pretrain/best_encoder_only.pth

Example with ROI masks:
    python scripts/extract_feat_LP_densenet.py \
        --input /path/to/images \
        --masks_path /path/to/masks \
        --output ./outputs/features \
        --checkpoint ./outputs/pretrain/best_encoder_only.pth
"""

import argparse
import multiprocessing as mp
import os
import sys
import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from monai.data import Dataset, ThreadDataLoader
from monai.transforms import Compose, DeleteItemsd, MapTransform, ResizeWithPadOrCropd, ToTensord
from tqdm import tqdm

warnings.filterwarnings("ignore")

# Make the script runnable from the repository root.
# Repository structure:
#   cvpr2026-window-aware-ctfm/
#   ├── models/
#   │   └── densenet121_3d_encoder.py
#   └── scripts/
#       └── extract_feat_LP_densenet.py
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    from models.densenet121_3d_encoder import densenet121_3d_encoder
except ImportError:
    # Fallback for users who run this script in the same folder as the model file.
    from densenet121_3d_encoder import densenet121_3d_encoder


def set_multiprocessing_start_method() -> None:
    """Use spawn mode to avoid multiprocessing issues in some environments."""
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass


class MaskCenterCropd(MapTransform):
    """
    Crop image and mask around the foreground center of a mask.

    If the mask has no foreground voxels, the transform falls back to an image
    center crop. This transform supports arrays with shape:
        image: (C, D, H, W)
        mask : (C, D, H, W) or (D, H, W)
    """

    def __init__(
        self,
        keys: Sequence[str],
        mask_key: str,
        roi_size: Tuple[int, int, int] = (152, 152, 152),
        fg_labels: Sequence[int] = (1,),
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.mask_key = mask_key
        self.roi_size = tuple(roi_size)
        self.fg_labels = list(fg_labels)

    def __call__(self, data: Dict) -> Dict:
        d = dict(data)

        mask = d[self.mask_key]
        mask_3d = mask[0] if mask.ndim == 4 else mask
        fg_mask = np.isin(mask_3d, self.fg_labels)

        if fg_mask.sum() > 0:
            coords = np.argwhere(fg_mask)
            center = coords.mean(axis=0).astype(int)  # (D, H, W)
        else:
            # Fallback to image center if the foreground mask is empty.
            ref = d[self.keys[0]]
            if ref.ndim == 4:
                _, depth, height, width = ref.shape
            else:
                depth, height, width = ref.shape
            center = np.array([depth // 2, height // 2, width // 2], dtype=int)

        roi_d, roi_h, roi_w = self.roi_size
        center_d, center_h, center_w = center

        for key in self.key_iterator(d):
            arr = d[key]

            if arr.ndim == 4:
                # Input shape: (C, D, H, W)
                channels, depth, height, width = arr.shape

                start_d = max(center_d - roi_d // 2, 0)
                start_h = max(center_h - roi_h // 2, 0)
                start_w = max(center_w - roi_w // 2, 0)

                end_d = min(start_d + roi_d, depth)
                end_h = min(start_h + roi_h, height)
                end_w = min(start_w + roi_w, width)

                cropped = arr[:, start_d:end_d, start_h:end_h, start_w:end_w]

                out = np.zeros((channels, roi_d, roi_h, roi_w), dtype=arr.dtype)
                out[:, : cropped.shape[1], : cropped.shape[2], : cropped.shape[3]] = cropped
                d[key] = out

            elif arr.ndim == 3:
                # Input shape: (D, H, W)
                depth, height, width = arr.shape

                start_d = max(center_d - roi_d // 2, 0)
                start_h = max(center_h - roi_h // 2, 0)
                start_w = max(center_w - roi_w // 2, 0)

                end_d = min(start_d + roi_d, depth)
                end_h = min(start_h + roi_h, height)
                end_w = min(start_w + roi_w, width)

                cropped = arr[start_d:end_d, start_h:end_h, start_w:end_w]

                out = np.zeros((roi_d, roi_h, roi_w), dtype=arr.dtype)
                out[: cropped.shape[0], : cropped.shape[1], : cropped.shape[2]] = cropped
                d[key] = out

            else:
                raise ValueError(f"Unsupported ndim for key '{key}': {arr.ndim}")

        return d


class SimpleITKLoadImaged(MapTransform):
    """Load NIfTI images with SimpleITK and convert them to (C, D, H, W)."""

    def __init__(self, keys: Sequence[str], allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data: Dict) -> Dict:
        d = dict(data)

        for key in self.key_iterator(d):
            image_path = d[key]
            image = sitk.ReadImage(image_path)
            array = sitk.GetArrayFromImage(image).astype(np.float32)  # (D, H, W)

            # Add channel dimension: (D, H, W) -> (C, D, H, W)
            array = np.expand_dims(array, axis=0)

            # SimpleITK spacing is stored as (x, y, z). Convert it to (D, H, W).
            spacing_xyz = image.GetSpacing()
            spacing_dhw = (spacing_xyz[2], spacing_xyz[1], spacing_xyz[0])

            d[key] = array
            d[f"{key}_properties"] = {
                "spacing": spacing_dhw,
                "origin": image.GetOrigin(),
                "direction": image.GetDirection(),
            }

        return d


class CopyMaskd(MapTransform):
    """Copy mask arrays for debugging or later deletion in the transform pipeline."""

    def __init__(self, keys: Sequence[str], mask_key: Sequence[str], allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)
        self.mask_key = mask_key

    def __call__(self, data: Dict) -> Dict:
        d = dict(data)
        for key in self.mask_key:
            d[f"{key}_original"] = d[key].copy()
        return d


class CTWindowd(MapTransform):
    """
    Apply CT windowing and normalize HU values to [0, 1].

    Default setting:
        soft-tissue window, center = 50, width = 380

    This matches the feature extraction setting used for the final submitted
    DenseNet121 3D model.
    """

    def __init__(
        self,
        keys: Sequence[str],
        center: float = 50.0,
        width: float = 380.0,
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.center = float(center)
        self.width = float(width)

    def __call__(self, data: Dict) -> Dict:
        d = dict(data)

        low = self.center - self.width / 2.0
        high = self.center + self.width / 2.0

        for key in self.key_iterator(d):
            x = d[key].astype(np.float32, copy=False)
            x = np.clip(x, low, high)
            x = (x - low) / max(self.width, 1e-8)
            d[key] = x.astype(np.float32)

        return d


def torch_resample_to_spacing(
    data: np.ndarray,
    current_spacing: Tuple[float, float, float],
    new_spacing: Tuple[float, float, float],
    is_seg: bool = False,
    device: str = "cuda",
) -> np.ndarray:
    """
    Resample a 3D volume to the target voxel spacing using PyTorch.

    Args:
        data: Input array with shape (C, D, H, W).
        current_spacing: Current spacing in (D, H, W) order.
        new_spacing: Target spacing in (D, H, W) order.
        is_seg: Use nearest-neighbor interpolation for segmentation masks.
        device: Device used for interpolation.
    """

    if data.ndim != 4:
        raise ValueError(f"Expected input shape (C, D, H, W), got {data.shape}")

    old_shape = np.array(data.shape[1:], dtype=np.float32)  # (D, H, W)
    current_spacing = np.array(current_spacing, dtype=np.float32)
    new_spacing = np.array(new_spacing, dtype=np.float32)

    new_shape = np.round(old_shape * current_spacing / new_spacing).astype(int)
    new_shape = np.maximum(new_shape, 1)

    x = torch.from_numpy(data).float().unsqueeze(0).to(device)  # (1, C, D, H, W)

    if is_seg:
        mode = "nearest"
        align_corners = None
    else:
        mode = "trilinear"
        align_corners = False

    y = F.interpolate(
        x,
        size=tuple(new_shape.tolist()),
        mode=mode,
        align_corners=align_corners,
    )

    return y.squeeze(0).cpu().numpy().astype(np.float32)


class ResampleToSpacingd(MapTransform):
    """Resample image and mask keys to a common voxel spacing."""

    def __init__(
        self,
        keys: Sequence[str],
        target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        allow_missing_keys: bool = False,
        is_seg_list: Optional[Sequence[bool]] = None,
        device: str = "cuda",
    ):
        super().__init__(keys, allow_missing_keys)

        if is_seg_list is None:
            is_seg_list = [False] * len(keys)

        if len(keys) != len(is_seg_list):
            raise ValueError("keys and is_seg_list must have the same length.")

        self.target_spacing = tuple(target_spacing)
        self.is_seg_list = list(is_seg_list)
        self.device = device if torch.cuda.is_available() else "cpu"

    def __call__(self, data: Dict) -> Dict:
        d = dict(data)

        for key, is_seg in zip(self.key_iterator(d), self.is_seg_list):
            arr = d[key]
            props = d.get(f"{key}_properties", None)
            if props is None:
                raise ValueError(f"Missing metadata: {key}_properties")

            d[key] = torch_resample_to_spacing(
                data=arr,
                current_spacing=props["spacing"],
                new_spacing=self.target_spacing,
                is_seg=is_seg,
                device=self.device,
            )

        return d


def parse_spatial_size(value: str) -> Tuple[int, int, int]:
    """Parse a spatial size from a string such as '152,152,152'."""
    parts = [int(v.strip()) for v in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("spatial size must contain three integers, e.g. 152,152,152")
    return tuple(parts)


def parse_spacing(value: str) -> Tuple[float, float, float]:
    """Parse target spacing from a string such as '1.0,1.0,1.0'."""
    parts = [float(v.strip()) for v in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("spacing must contain three numbers, e.g. 1.0,1.0,1.0")
    return tuple(parts)


def build_datalist(imgs_path: str, masks_path: Optional[str] = None) -> List[Dict]:
    """
    Build the list of input cases.

    In ROI mode, a mask file with the same filename must exist under masks_path.
    """

    if not os.path.isdir(imgs_path):
        raise FileNotFoundError(f"Input image directory not found: {imgs_path}")

    imgs_files = sorted([f for f in os.listdir(imgs_path) if f.endswith(".nii.gz")])

    if masks_path is not None:
        if not os.path.isdir(masks_path):
            raise FileNotFoundError(f"Mask directory not found: {masks_path}")
        imgs_files = [f for f in imgs_files if os.path.exists(os.path.join(masks_path, f))]

    datalist = []
    for img_file in imgs_files:
        img_id = img_file.replace(".nii.gz", "")
        img_full_path = os.path.join(imgs_path, img_file)
        mask_full_path = os.path.join(masks_path, img_file) if masks_path is not None else None

        if not os.path.exists(img_full_path):
            raise FileNotFoundError(f"Image not found: {img_full_path}")

        if mask_full_path is not None:
            if not os.path.exists(mask_full_path):
                raise FileNotFoundError(f"Mask not found: {mask_full_path}")
            datalist.append({"image": img_full_path, "mask": mask_full_path, "filename": img_id})
        else:
            datalist.append({"image": img_full_path, "filename": img_id})

    return datalist


def build_transforms(args: argparse.Namespace) -> Compose:
    """Create preprocessing transforms for image-only or ROI-mask mode."""

    if args.masks_path is None:
        return Compose(
            [
                SimpleITKLoadImaged(keys=["image"]),
                CTWindowd(keys=["image"], center=args.window_center, width=args.window_width),
                ResampleToSpacingd(
                    keys=["image"],
                    target_spacing=args.target_spacing,
                    is_seg_list=[False],
                    device=args.device,
                ),
                ResizeWithPadOrCropd(keys=["image"], spatial_size=list(args.spatial_size)),
                ToTensord(keys=["image"]),
            ]
        )

    return Compose(
        [
            SimpleITKLoadImaged(keys=["image", "mask"]),
            CopyMaskd(keys=["mask"], mask_key=["mask"]),
            CTWindowd(keys=["image"], center=args.window_center, width=args.window_width),
            ResampleToSpacingd(
                keys=["image", "mask"],
                target_spacing=args.target_spacing,
                is_seg_list=[False, True],
                device=args.device,
            ),
            MaskCenterCropd(
                keys=["image", "mask"],
                mask_key="mask",
                roi_size=args.spatial_size,
                fg_labels=args.fg_labels,
            ),
            ResizeWithPadOrCropd(keys=["image", "mask"], spatial_size=list(args.spatial_size)),
            DeleteItemsd(keys=["mask_original"]),
            ToTensord(keys=["image", "mask"]),
        ]
    )


def load_encoder(checkpoint_path: str, device: torch.device, dropout: float = 0.2) -> torch.nn.Module:
    """
    Load DenseNet121 3D encoder weights.

    The checkpoint may be either:
    - an encoder state_dict directly, or
    - a larger checkpoint dictionary containing "model_state_dict".
    """

    model = densenet121_3d_encoder(in_channels=1, dropout=dropout)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    return model


def dump_debug_volume(array: np.ndarray, save_path: str) -> None:
    """Save a preprocessed 3D volume for debugging."""
    if array.shape[0] == 1:
        array = array[0]

    image = sitk.GetImageFromArray(array)
    image.SetSpacing((1.0, 1.0, 1.0))
    sitk.WriteImage(image, save_path)


def extract_features(args: argparse.Namespace) -> None:
    """Run feature extraction and save one HDF5 file per CT volume."""

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    os.makedirs(args.dest, exist_ok=True)

    if args.dump_dir is not None:
        os.makedirs(args.dump_dir, exist_ok=True)

    model = load_encoder(args.checkpoint, device=device, dropout=args.dropout)
    print(f"Loaded encoder checkpoint from: {args.checkpoint}")

    datalist = build_datalist(args.imgs_path, args.masks_path)
    print(f"Found {len(datalist)} cases")

    transforms = build_transforms(args)
    dataset = Dataset(data=datalist, transform=transforms)
    dataloader = ThreadDataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    processed_count = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Extracting features")):
            if not isinstance(batch, dict):
                raise TypeError(f"Expected batch to be dict, got {type(batch)}")

            images = batch["image"]
            masks = batch.get("mask", None)
            filenames = batch.get(
                "filename",
                [f"batch_{batch_idx}_sample_{i}" for i in range(images.shape[0])],
            )

            if not isinstance(filenames, (list, tuple)):
                filenames = [filenames]

            if len(filenames) != images.shape[0]:
                raise ValueError(
                    f"Number of filenames ({len(filenames)}) does not match batch size ({images.shape[0]})."
                )

            # Optionally save preprocessed images and masks to inspect the pipeline.
            if args.dump_dir:
                for i, filename in enumerate(filenames):
                    image_np = images[i].cpu().numpy()
                    dump_debug_volume(image_np, os.path.join(args.dump_dir, f"{filename}_image.nii.gz"))

                    if masks is not None:
                        mask_np = masks[i].cpu().numpy()
                        dump_debug_volume(mask_np, os.path.join(args.dump_dir, f"{filename}_mask.nii.gz"))

            images = images.to(device, non_blocking=True)

            # The encoder returns multi-level feature maps: [f1, f2, f3, f4].
            outputs = model(images)

            # Global average pooling is applied to each feature level.
            # The pooled features are concatenated into the final CT embedding.
            image_embeddings = torch.cat(
                [F.adaptive_avg_pool3d(output, 1) for output in outputs],
                dim=1,
            )
            image_embeddings = image_embeddings.view(image_embeddings.shape[0], -1)
            image_embeddings = image_embeddings.detach().cpu().numpy()

            for i, filename in enumerate(filenames):
                output_path = os.path.join(args.dest, f"{filename}.h5")
                with h5py.File(output_path, "w") as hf:
                    hf.create_dataset("y_hat", data=image_embeddings[i])
                processed_count += 1

            del outputs, images, image_embeddings
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"Done. Saved {processed_count} feature files to: {args.dest}")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract CT feature embeddings using a pretrained DenseNet121 3D encoder."
    )

    parser.add_argument(
        "-i",
        "--input",
        "--imgs_path",
        dest="imgs_path",
        type=str,
        default="/workspace/inputs",
        help="Directory containing input CT images in .nii.gz format.",
    )
    parser.add_argument(
        "-o",
        "--output",
        "--dest",
        dest="dest",
        type=str,
        default="/workspace/outputs",
        help="Directory used to save extracted .h5 feature files.",
    )
    parser.add_argument(
        "--masks_path",
        type=str,
        default=None,
        help="Optional directory containing foreground masks with the same filenames as the input images.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the pretrained encoder checkpoint, e.g. best_encoder_only.pth.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for feature extraction.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of dataloader workers.",
    )
    parser.add_argument(
        "--dump_dir",
        type=str,
        default=None,
        help="Optional directory for saving preprocessed images and masks for debugging.",
    )
    parser.add_argument(
        "--window_center",
        type=float,
        default=50.0,
        help="CT window center. Default is 50 for soft-tissue window.",
    )
    parser.add_argument(
        "--window_width",
        type=float,
        default=380.0,
        help="CT window width. Default is 380 for soft-tissue window.",
    )
    parser.add_argument(
        "--target_spacing",
        type=parse_spacing,
        default=(1.0, 1.0, 1.0),
        help="Target spacing in D,H,W order, e.g. 1.0,1.0,1.0.",
    )
    parser.add_argument(
        "--spatial_size",
        type=parse_spatial_size,
        default=(152, 152, 152),
        help="Fixed spatial size in D,H,W order, e.g. 152,152,152.",
    )
    parser.add_argument(
        "--fg_labels",
        type=int,
        nargs="+",
        default=[1],
        help="Foreground labels used for ROI mask center cropping.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout value used to instantiate the DenseNet121 3D encoder.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for feature extraction.",
    )

    return parser.parse_args()


def main() -> None:
    set_multiprocessing_start_method()
    args = get_args()
    extract_features(args)


if __name__ == "__main__":
    main()
