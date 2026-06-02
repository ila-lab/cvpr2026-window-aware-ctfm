"""
Pretrain DenseNet121 3D encoder with window-aware multi-scale global-local SSL.

Example:
    python scripts/pretrain_densenet121.py \
        --data_root /path/to/coreset_data \
        --save_dir ./outputs/pretrain_densenet121
"""

import argparse
import glob
import os
import random
from dataclasses import asdict, dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import rotate
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

#
# Run the script from the repository root:
#   python scripts/pretrain_densenet121.py --data_root ... --save_dir ...
try:
    from models.densenet121_3d_encoder import densenet121_3d_encoder
except ImportError:
    # Fallback for running this file in the same folder as densenet121_3d_encoder.py
    from densenet121_3d_encoder import densenet121_3d_encoder

#Configuration

@dataclass
class TrainConfig:
    """Configuration used for pretraining."""

    data_root: str
    save_dir: str

    seed: int = 42
    target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    global_size: Tuple[int, int, int] = (152, 152, 152)
    local_sizes: Tuple[int, int, int] = (48, 80, 104)
    num_local_per_scale: int = 1

    batch_size: int = 3
    accumulation_steps: int = 5
    epochs: int = 80
    num_workers: int = 4

    lr: float = 3e-4
    weight_decay: float = 1e-4
    temperature: float = 0.2
    dropout: float = 0.2

    proj_dim: int = 128
    proj_hidden_dim: int = 512

    lambda_global: float = 1.0
    lambda_local: float = 1.0
    grad_clip_norm: float = 5.0

    device: str = "cuda"


def parse_args() -> TrainConfig:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Pretrain DenseNet121 3D with window-aware global-local SSL."
    )

    # Required paths
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root directory containing coreset CT volumes in .nii.gz format.")
    parser.add_argument("--save_dir", type=str, required=True,
                        help="Directory for saving checkpoints and training history.")

    # Reproducibility and preprocessing
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target_spacing", type=float, nargs=3, default=(1.0, 1.0, 1.0),
                        help="Target voxel spacing in (D, H, W) order.")
    parser.add_argument("--global_size", type=int, nargs=3, default=(152, 152, 152),
                        help="Final input size in (D, H, W) order.")
    parser.add_argument("--local_sizes", type=int, nargs="+", default=[48, 80, 104],
                        help="Local crop sizes. Default: 48 80 104.")
    parser.add_argument("--num_local_per_scale", type=int, default=1,
                        help="Number of local crops sampled for each local scale.")

    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument("--accumulation_steps", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--proj_dim", type=int, default=128)
    parser.add_argument("--proj_hidden_dim", type=int, default=512)
    parser.add_argument("--lambda_global", type=float, default=1.0)
    parser.add_argument("--lambda_local", type=float, default=1.0)
    parser.add_argument("--grad_clip_norm", type=float, default=5.0)
    parser.add_argument("--device", type=str, default="cuda",
                        help="Training device. Use 'cuda' or 'cpu'.")

    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available. Falling back to CPU.")
        device = "cpu"

    return TrainConfig(
        data_root=args.data_root,
        save_dir=args.save_dir,
        seed=args.seed,
        target_spacing=tuple(args.target_spacing),
        global_size=tuple(args.global_size),
        local_sizes=tuple(args.local_sizes),
        num_local_per_scale=args.num_local_per_scale,
        batch_size=args.batch_size,
        accumulation_steps=args.accumulation_steps,
        epochs=args.epochs,
        num_workers=args.num_workers,
        lr=args.lr,
        weight_decay=args.weight_decay,
        temperature=args.temperature,
        dropout=args.dropout,
        proj_dim=args.proj_dim,
        proj_hidden_dim=args.proj_hidden_dim,
        lambda_global=args.lambda_global,
        lambda_local=args.lambda_local,
        grad_clip_norm=args.grad_clip_norm,
        device=device,
    )


# Reproducibility
def seed_everything(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# IO and preprocessing utilities

def load_sitk_image(path: str) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """
    Load a CT volume using SimpleITK.

    Returns:
        arr: CT volume in (D, H, W) order.
        spacing_dhw: voxel spacing in (D, H, W) order.
    """
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img).astype(np.float32)

    spacing_xyz = img.GetSpacing()  # SimpleITK uses (x, y, z)
    spacing_dhw = (spacing_xyz[2], spacing_xyz[1], spacing_xyz[0])

    return arr, spacing_dhw


def resample_image(
    image: np.ndarray,
    old_spacing: Tuple[float, float, float],
    new_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    """Resample a 3D CT volume to the target spacing."""
    img = sitk.GetImageFromArray(image)
    img.SetSpacing((old_spacing[2], old_spacing[1], old_spacing[0]))

    old_size = np.array(list(img.GetSize()), dtype=np.int32)
    old_spacing_xyz = np.array(list(img.GetSpacing()), dtype=np.float32)
    new_spacing_xyz = np.array(
        [new_spacing[2], new_spacing[1], new_spacing[0]],
        dtype=np.float32,
    )

    new_size = np.round(old_size * old_spacing_xyz / new_spacing_xyz).astype(np.int32)

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(tuple(new_spacing_xyz.tolist()))
    resampler.SetSize([int(x) for x in new_size.tolist()])
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(-1024)  # Air value for CT padding
    resampler.SetInterpolator(sitk.sitkLinear)

    out = resampler.Execute(img)
    return sitk.GetArrayFromImage(out).astype(np.float32)


def center_crop_or_pad(
    image: np.ndarray,
    target_size: Tuple[int, int, int] = (152, 152, 152),
    pad_value: float = -1024,
) -> np.ndarray:
    """Center crop or pad a 3D volume to a fixed size."""
    d, h, w = image.shape
    td, th, tw = target_size

    pad_d = max(td - d, 0)
    pad_h = max(th - h, 0)
    pad_w = max(tw - w, 0)

    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        image = np.pad(
            image,
            (
                (pad_d // 2, pad_d - pad_d // 2),
                (pad_h // 2, pad_h - pad_h // 2),
                (pad_w // 2, pad_w - pad_w // 2),
            ),
            mode="constant",
            constant_values=pad_value,
        )

    d, h, w = image.shape
    sd = max((d - td) // 2, 0)
    sh = max((h - th) // 2, 0)
    sw = max((w - tw) // 2, 0)

    return image[sd:sd + td, sh:sh + th, sw:sw + tw]


# Window-aware augmentation
#### 
def apply_ct_window_hu(x: np.ndarray, center: float, width: float) -> np.ndarray:
    """
    Apply CT window clipping and normalize HU values to [0, 1].
    No z-score normalization is used in this version.
    """
    low = center - width / 2
    high = center + width / 2
    x = np.clip(x, low, high)
    x = (x - low) / max(width, 1e-8)
    return x.astype(np.float32)


def random_window(x: np.ndarray) -> np.ndarray:
    """
    Randomly apply one CT window.

    This augmentation encourages the encoder to learn HU-aware features.
    """
    windows = [
        ("soft_tissue", 50, 380),
        ("lung", -600, 1550),
        ("bone", 450, 1900),
        ("abdomen", 50, 380),
        ("wide", 450, 2100),
    ]

    _, center, width = random.choice(windows)

    # Slightly perturb the selected window for augmentation.
    center = center + random.uniform(-30, 30)
    width = width * random.uniform(0.9, 1.1)

    return apply_ct_window_hu(x, center, width)


def random_intensity_jitter(x: np.ndarray) -> np.ndarray:
    """Apply random intensity scaling and shifting."""
    if random.random() < 0.8:
        x = x * random.uniform(0.9, 1.1) + random.uniform(-0.05, 0.05)
    return x.astype(np.float32)


def random_noise(x: np.ndarray) -> np.ndarray:
    """Add small Gaussian noise."""
    if random.random() < 0.5:
        x = x + np.random.normal(0, 0.01, size=x.shape).astype(np.float32)
    return x.astype(np.float32)


def random_rotate_3d(x: np.ndarray, max_angle: float = 8) -> np.ndarray:
    """Apply small random 3D rotation."""
    if random.random() < 0.5:
        angle = random.uniform(-max_angle, max_angle)
        axes = random.choice([(1, 2), (0, 2), (0, 1)])
        x = rotate(
            x,
            angle=angle,
            axes=axes,
            reshape=False,
            order=1,
            mode="nearest",
        )
    return x.astype(np.float32)


def augment_window_aware(x: np.ndarray) -> np.ndarray:
    """Apply CT window augmentation and light intensity/spatial augmentations."""
    x = random_window(x)
    x = random_intensity_jitter(x)
    x = random_noise(x)
    x = random_rotate_3d(x, max_angle=8)
    return x.astype(np.float32).copy()


# Multi-scale local crop

def random_crop_nonbackground(x: np.ndarray, size: int) -> np.ndarray:
    """
    Randomly crop a local patch while avoiding large background-only regions.

    The standard deviation threshold is used as a simple non-background heuristic.
    """
    D, H, W = x.shape

    if D < size or H < size or W < size:
        x = center_crop_or_pad(x, target_size=(size, size, size), pad_value=-1024)
        D, H, W = x.shape

    last_crop = None

    for _ in range(10):
        d = random.randint(0, D - size)
        h = random.randint(0, H - size)
        w = random.randint(0, W - size)

        crop = x[d:d + size, h:h + size, w:w + size]
        last_crop = crop

        if crop.std() > 30:
            return crop.astype(np.float32)

    return last_crop.astype(np.float32)


# Dataset
class WindowMultiScaleGLDataset(Dataset):
    """
    Dataset for window-aware multi-scale global-local SSL.

    For each CT volume, this dataset returns:
        - two global views
        - multiple local views from different crop sizes
    """

    def __init__(
        self,
        data_root: str,
        target_spacing: Tuple[float, float, float],
        global_size: Tuple[int, int, int],
        local_sizes: Tuple[int, ...],
        num_local_per_scale: int,
    ):
        self.data_root = data_root
        self.target_spacing = target_spacing
        self.global_size = global_size
        self.local_sizes = local_sizes
        self.num_local_per_scale = num_local_per_scale

        self.image_paths = sorted(
            glob.glob(os.path.join(data_root, "**", "*.nii.gz"), recursive=True)
        )

        if len(self.image_paths) == 0:
            raise RuntimeError(f"No .nii.gz files found under: {data_root}")

        print(f"Found {len(self.image_paths)} CT volumes.")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        path = self.image_paths[idx]

        # Load raw HU volume and resample to a common spacing.
        image, spacing = load_sitk_image(path)
        image = resample_image(image, old_spacing=spacing, new_spacing=self.target_spacing)
        image = center_crop_or_pad(image, target_size=self.global_size, pad_value=-1024)

        # Generate two global views.
        global_1 = augment_window_aware(image)
        global_2 = augment_window_aware(image)

        # multi-scale local views.
        local_views = []
        for size in self.local_sizes:
            for _ in range(self.num_local_per_scale):
                crop = random_crop_nonbackground(image, size=size)
                crop = augment_window_aware(crop)

                # Pad local crops to the global input size for the same encoder.
                crop = center_crop_or_pad(crop, target_size=self.global_size, pad_value=0)
                local_views.append(crop)

        global_1 = torch.tensor(global_1[None], dtype=torch.float32)
        global_2 = torch.tensor(global_2[None], dtype=torch.float32)
        local_views = torch.stack(
            [torch.tensor(v[None], dtype=torch.float32) for v in local_views],
            dim=0,
        )

        return {
            "global_1": global_1,
            "global_2": global_2,
            "locals": local_views,
            "path": path,
        }


# SSL model wrapper

class GLSSLModel(nn.Module):
    """Global-local self-supervised learning wrapper."""

    def __init__(self, encoder: nn.Module, projection_dim: int = 128,
                 hidden_dim: int = 512, dropout_prob: float = 0.2):
        super().__init__()
        self.encoder = encoder
        self.projector = nn.Sequential(
            nn.Linear(encoder.out_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, projection_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.encoder.extract_embedding(x)
        z = self.projector(emb)
        return F.normalize(z, dim=1)

    def extract_embedding(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder.extract_embedding(x)


# Loss functions
def info_nce_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    """Global-global InfoNCE loss between two augmented views."""
    B = z1.shape[0]

    z = torch.cat([z1, z2], dim=0)
    sim = torch.matmul(z, z.T) / temperature

    # Remove self-comparisons.
    mask = torch.eye(2 * B, device=z.device).bool()
    sim = sim.masked_fill(mask, -1e9)

    positives = torch.cat([
        torch.arange(B, 2 * B, device=z.device),
        torch.arange(0, B, device=z.device),
    ])

    return F.cross_entropy(sim, positives)


def global_local_loss(global_z: torch.Tensor, local_z: torch.Tensor,
                      temperature: float = 0.2) -> torch.Tensor:
    """Contrastive loss between global features and local patch features."""
    B, K, _ = local_z.shape
    losses = []

    for k in range(K):
        lk = local_z[:, k, :]
        labels = torch.arange(B, device=global_z.device)

        logits_g2l = torch.matmul(global_z, lk.T) / temperature
        loss_g2l = F.cross_entropy(logits_g2l, labels)

        logits_l2g = torch.matmul(lk, global_z.T) / temperature
        loss_l2g = F.cross_entropy(logits_l2g, labels)

        losses.append(0.5 * (loss_g2l + loss_l2g))

    return torch.stack(losses).mean()

# Checkpoint saving

def save_checkpoint(model: GLSSLModel, optimizer: torch.optim.Optimizer,
                    epoch: int, loss: float, cfg: TrainConfig, save_path: str) -> None:
    """Save the full SSL model checkpoint."""
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
            "config": asdict(cfg),
        },
        save_path,
    )

def save_encoder_only(model: GLSSLModel, save_path: str) -> None:
    """Save only the encoder weights for downstream feature extraction."""
    torch.save(model.encoder.state_dict(), save_path)

# Training loop
def train(cfg: TrainConfig) -> None:
    """Run self-supervised pretraining."""
    os.makedirs(cfg.save_dir, exist_ok=True)
    seed_everything(cfg.seed)

    dataset = WindowMultiScaleGLDataset(
        data_root=cfg.data_root,
        target_spacing=cfg.target_spacing,
        global_size=cfg.global_size,
        local_sizes=cfg.local_sizes,
        num_local_per_scale=cfg.num_local_per_scale,
    )

    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.device == "cuda"),
        drop_last=True,
    )

    encoder = densenet121_3d_encoder(
        in_channels=1,
        dropout=cfg.dropout,
    )

    model = GLSSLModel(
        encoder=encoder,
        projection_dim=cfg.proj_dim,
        hidden_dim=cfg.proj_hidden_dim,
        dropout_prob=cfg.dropout,
    ).to(cfg.device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.epochs,
        eta_min=1e-6,
    )

    print("=" * 80)
    print("Training configuration")
    print("=" * 80)
    for k, v in asdict(cfg).items():
        print(f"{k}: {v}")
    print("=" * 80)
    print(model)

    best_loss = float("inf")
    history = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()

        running_loss = 0.0
        running_g_loss = 0.0
        running_l_loss = 0.0
        total_samples = 0

        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(loader, desc=f"Epoch {epoch:03d}/{cfg.epochs:03d}")

        for i, batch in enumerate(pbar):
            global_1 = batch["global_1"].to(cfg.device, non_blocking=True)
            global_2 = batch["global_2"].to(cfg.device, non_blocking=True)
            locals_ = batch["locals"].to(cfg.device, non_blocking=True)

            B, K, C, D, H, W = locals_.shape
            locals_flat = locals_.view(B * K, C, D, H, W)

            # Forward global and local views through the same encoder.
            z_g1 = model(global_1)
            z_g2 = model(global_2)
            z_l_flat = model(locals_flat)
            z_l = z_l_flat.view(B, K, -1)

            # Compute global-global and global-local contrastive losses.
            loss_global = info_nce_loss(z_g1, z_g2, temperature=cfg.temperature)

            loss_local_1 = global_local_loss(z_g1, z_l, temperature=cfg.temperature)
            loss_local_2 = global_local_loss(z_g2, z_l, temperature=cfg.temperature)
            loss_local = 0.5 * (loss_local_1 + loss_local_2)

            loss = cfg.lambda_global * loss_global + cfg.lambda_local * loss_local
            loss = loss / cfg.accumulation_steps
            loss.backward()

            # Gradient accumulation to simulate a larger effective batch size.
            if (i + 1) % cfg.accumulation_steps == 0 or (i + 1) == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            actual_loss = loss.item() * cfg.accumulation_steps
            running_loss += actual_loss * B
            running_g_loss += loss_global.item() * B
            running_l_loss += loss_local.item() * B
            total_samples += B

            pbar.set_postfix({
                "loss": f"{actual_loss:.4f}",
                "g": f"{loss_global.item():.4f}",
                "l": f"{loss_local.item():.4f}",
            })

        scheduler.step()

        epoch_loss = running_loss / total_samples
        epoch_g_loss = running_g_loss / total_samples
        epoch_l_loss = running_l_loss / total_samples
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"[Epoch {epoch:03d}/{cfg.epochs:03d}] "
            f"loss={epoch_loss:.4f}, "
            f"global={epoch_g_loss:.4f}, "
            f"local={epoch_l_loss:.4f}, "
            f"lr={current_lr:.2e}"
        )

        history.append({
            "epoch": epoch,
            "loss": epoch_loss,
            "global_loss": epoch_g_loss,
            "local_loss": epoch_l_loss,
            "lr": current_lr,
        })

        # Save latest checkpoints at every epoch.
        latest_full = os.path.join(cfg.save_dir, "latest_full_model.pth")
        latest_encoder = os.path.join(cfg.save_dir, "latest_encoder_only.pth")
        save_checkpoint(model, optimizer, epoch, epoch_loss, cfg, latest_full)
        save_encoder_only(model, latest_encoder)

        # Save the best checkpoint according to training loss.
        if epoch_loss < best_loss:
            best_loss = epoch_loss

            best_full = os.path.join(cfg.save_dir, "best_full_model.pth")
            best_encoder = os.path.join(cfg.save_dir, "best_encoder_only.pth")
            save_checkpoint(model, optimizer, epoch, epoch_loss, cfg, best_full)
            save_encoder_only(model, best_encoder)

            print(f"  -> saved best full checkpoint: {best_full}")
            print(f"  -> saved best encoder checkpoint: {best_encoder}")

    # Save training history for reproducibility and visualization.
    history_df = pd.DataFrame(history)
    history_csv = os.path.join(cfg.save_dir, "training_history.csv")
    history_df.to_csv(history_csv, index=False)

    print("Training finished.")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Training history saved to: {history_csv}")
    print(f"Outputs saved to: {cfg.save_dir}")


# main
def main() -> None:
    cfg = parse_args()
    train(cfg)


if __name__ == "__main__":
    main()
