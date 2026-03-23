"""
distill.py
----------
Core E2D distillation loop, adapted from Branch_ImageNet_1K/recover/recover.py
for small-scale images (CIFAR-10/100, 32x32).

Key simplifications vs. the original:
  - Single teacher (no ensemble)
  - No BN/Conv statistics collection (no original dataset needed)
  - No distributed training
  - Image size: 32x32 instead of 224x224
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision.transforms import functional as TF
import torchvision.transforms as transforms


# ──────────────────────────────────────────────
#  Pixel clipping (keep values in valid range)
# ──────────────────────────────────────────────
def clip_cifar(image_tensor):
    """Clamp pixel values to the CIFAR normalised range."""
    mean = np.array([0.4914, 0.4822, 0.4465])
    std  = np.array([0.2023, 0.1994, 0.2010])
    for c in range(3):
        image_tensor[:, c] = torch.clamp(
            image_tensor[:, c], -mean[c] / std[c], (1 - mean[c]) / std[c]
        )
    return image_tensor


def denormalize_cifar(image_tensor):
    """Convert normalised floats back to [0,1] pixel values."""
    mean = np.array([0.4914, 0.4822, 0.4465])
    std  = np.array([0.2023, 0.1994, 0.2010])
    out = image_tensor.clone()
    for c in range(3):
        out[:, c] = torch.clamp(out[:, c] * std[c] + mean[c], 0, 1)
    return out


# ──────────────────────────────────────────────
#  Exploration–Exploitation augmentation class
#  (directly adapted from recover.py)
# ──────────────────────────────────────────────
class ExplorationExploitationAug:
    """
    Two-phase random crop augmentation.

    - Exploration  (iteration ≤ K): random crops; high-loss crops are stored
      in a per-image memory buffer.
    - Exploitation (iteration >  K): crops are sampled from the buffer with
      probability proportional to their loss (softmax weighting).
    """

    def __init__(self, batch_size: int, img_size: int = 32):
        # For 32×32 images use a smaller crop scale than the 224×224 original
        scale = (0.5, 1.0)
        self.cropper   = transforms.RandomResizedCrop(img_size, scale=scale)
        self.flipper   = transforms.RandomHorizontalFlip()
        self.last_crops       = [None] * batch_size
        self.selected_indices = [None] * batch_size

    def __call__(self, imgs, iteration, high_loss_crops, high_loss_values, K):
        batch_size   = imgs.shape[0]
        cropped_imgs = []

        for img_idx in range(batch_size):
            if iteration > K and high_loss_crops[img_idx]:
                # ── Exploitation phase ──────────────────────────────────────
                weights = torch.tensor(
                    high_loss_values[img_idx], device=imgs.device
                )
                probs = torch.softmax(weights, dim=0)
                sel   = torch.multinomial(probs, 1).item()
                i, j, h, w = high_loss_crops[img_idx][sel]
                self.selected_indices[img_idx] = sel
            else:
                # ── Exploration phase ───────────────────────────────────────
                i, j, h, w = self.cropper.get_params(
                    imgs[img_idx], self.cropper.scale, self.cropper.ratio
                )
                self.selected_indices[img_idx] = None

            self.last_crops[img_idx] = (i, j, h, w)
            cropped = TF.resized_crop(
                imgs[img_idx], i, j, h, w, self.cropper.size
            )
            cropped = self.flipper(cropped)
            cropped_imgs.append(cropped)

        return torch.stack(cropped_imgs)


# ──────────────────────────────────────────────
#  Main distillation function
# ──────────────────────────────────────────────
def distill_task(
    real_images: torch.Tensor,   # (N, C, H, W)  real images for ONE class
    class_label: int,            # integer class id
    teacher: nn.Module,          # frozen pretrained teacher
    n_synthetic: int,            # how many synthetic images to produce
    device: torch.device,
    iterations:      int   = 500,
    K:               int   = 350,   # exploration / exploitation switch
    loss_threshold:  float = 0.5,   # ε — minimum loss to record a crop
    lr:              float = 0.05,
    img_size:        int   = 32,
) -> torch.Tensor:
    """
    Run the E2D optimisation loop for a single class.

    Returns a detached tensor of shape (n_synthetic, C, H, W) containing
    the distilled synthetic images (still normalised, ready for training).
    """
    teacher = teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    # ── Initialise from random real images (full-image init, Step 1 of E2D) ─
    indices = torch.randint(0, len(real_images), (n_synthetic,))
    inputs  = real_images[indices].clone().to(device).float()
    inputs.requires_grad_(True)

    targets = torch.full((n_synthetic,), class_label,
                         dtype=torch.long, device=device)

    optimizer  = optim.Adam([inputs], lr=lr, betas=(0.5, 0.9))
    criterion  = nn.CrossEntropyLoss(reduction='none')
    aug        = ExplorationExploitationAug(n_synthetic, img_size=img_size)

    # Per-image buffers for the high-loss crop memory
    high_loss_crops  = [[] for _ in range(n_synthetic)]
    high_loss_values = [[] for _ in range(n_synthetic)]

    for iteration in range(iterations):

        # Early stopping: if exploitation phase and buffer is empty → done
        if iteration > K and all(len(c) == 0 for c in high_loss_crops):
            print(f"  [E2D] Early stop at iter {iteration} — buffer empty")
            break

        inputs_aug      = aug(inputs, iteration, high_loss_crops,
                              high_loss_values, K)
        selected_indices = aug.selected_indices

        optimizer.zero_grad()
        outputs  = teacher(inputs_aug)
        loss_all = criterion(outputs, targets)   # (n_synthetic,)
        loss     = loss_all.mean()
        loss.backward()
        optimizer.step()

        # ── Update per-image high-loss crop buffer ─────────────────────────
        loss_vals = loss_all.detach().cpu().numpy()

        for img_idx, (i, j, h, w) in enumerate(aug.last_crops):
            crop = (i, j, h, w)

            # Exploration: record crops whose loss exceeds threshold
            if loss_vals[img_idx] > loss_threshold and iteration <= K:
                if crop in high_loss_crops[img_idx]:
                    idx = high_loss_crops[img_idx].index(crop)
                    high_loss_values[img_idx][idx] = loss_vals[img_idx]
                else:
                    high_loss_crops[img_idx].append(crop)
                    high_loss_values[img_idx].append(loss_vals[img_idx])

            # Exploitation: update or remove selected crop
            sel = selected_indices[img_idx]
            if sel is not None:
                if loss_vals[img_idx] > loss_threshold:
                    high_loss_values[img_idx][sel] = loss_vals[img_idx]
                else:
                    del high_loss_crops[img_idx][sel]
                    del high_loss_values[img_idx][sel]

        # Clip pixel values to valid range
        inputs.data = clip_cifar(inputs.data)

        if iteration % 100 == 0:
            print(f"  [E2D] iter {iteration:4d} | loss {loss.item():.4f}")

    return inputs.detach()
