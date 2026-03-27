"""
distill.py
----------
Fuller E2D distillation loop for continual learning on CIFAR-style images.

This version keeps the original exploration/exploitation crop memory from
recover.py, but also brings back the feature-stat matching term from
recover.py/utils.py by collecting teacher BN/Conv statistics on the real
images of the current class and matching them while optimizing the synthetic
images.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import functional as TF
import torchvision.transforms as transforms

from utils import div_sixteen_mul, get_image_prior_losses, lr_cosine_policy


def _channel_clip_bounds(reference: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Infer per-channel clipping bounds from the real images used to initialize."""
    with torch.no_grad():
        flat = reference.detach().permute(1, 0, 2, 3).reshape(reference.shape[1], -1)
        lower = flat.min(dim=1).values.view(1, -1, 1, 1)
        upper = flat.max(dim=1).values.view(1, -1, 1, 1)
    return lower, upper


def clip_like_reference(image_tensor: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    lower, upper = _channel_clip_bounds(reference)
    lower = lower.to(image_tensor.device, image_tensor.dtype)
    upper = upper.to(image_tensor.device, image_tensor.dtype)
    return torch.max(torch.min(image_tensor, upper), lower)


@dataclass
class E2DStats:
    loss: float
    loss_ce: float
    loss_r_feature: float
    loss_var_l1: float
    loss_var_l2: float
    iterations: int


class ExplorationExploitationAug:
    """Crop memory used by E2D during exploration then exploitation."""

    def __init__(self, batch_size: int, img_size: int = 32, crop_scale: Tuple[float, float] = (0.5, 1.0)):
        self.cropper = transforms.RandomResizedCrop(img_size, scale=crop_scale)
        self.flipper = transforms.RandomHorizontalFlip()
        self.last_crops: List[Optional[Tuple[int, int, int, int]]] = [None] * batch_size
        self.selected_indices: List[Optional[int]] = [None] * batch_size

    def __call__(
        self,
        imgs: torch.Tensor,
        iteration: int,
        high_loss_crops: List[List[Tuple[int, int, int, int]]],
        high_loss_values: List[List[float]],
        K: int,
    ) -> torch.Tensor:
        cropped_imgs: List[torch.Tensor] = []
        for img_idx in range(imgs.shape[0]):
            if iteration > K and high_loss_crops[img_idx]:
                weights = torch.tensor(high_loss_values[img_idx], device=imgs.device, dtype=imgs.dtype)
                probs = torch.softmax(weights, dim=0)
                sel = torch.multinomial(probs, 1).item()
                i, j, h, w = high_loss_crops[img_idx][sel]
                self.selected_indices[img_idx] = sel
            else:
                i, j, h, w = self.cropper.get_params(imgs[img_idx], self.cropper.scale, self.cropper.ratio)
                self.selected_indices[img_idx] = None

            self.last_crops[img_idx] = (i, j, h, w)
            cropped = TF.resized_crop(imgs[img_idx], i, j, h, w, self.cropper.size)
            cropped = self.flipper(cropped)
            cropped_imgs.append(cropped)

        return torch.stack(cropped_imgs, dim=0)


class _BaseFeatureHook:
    def __init__(self, module: nn.Module, momentum: float = 0.4) -> None:
        self.module = module
        self.momentum = momentum
        self.handle = module.register_forward_hook(self._hook_fn)
        self.mode = "collect"
        self.r_feature: Optional[torch.Tensor] = None
        self._target_count = 0

    def _hook_fn(self, module: nn.Module, inputs: Tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        x = inputs[0]
        current = self.compute_current_stats(x)
        if self.mode == "collect":
            self.update_target(current, x.shape[0])
        else:
            self.update_r_feature(current)

    def compute_current_stats(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def update_target(self, current: Dict[str, torch.Tensor], batch_size: int) -> None:
        raise NotImplementedError

    def update_r_feature(self, current: Dict[str, torch.Tensor]) -> None:
        raise NotImplementedError

    def set_collect(self) -> None:
        self.mode = "collect"

    def set_match(self) -> None:
        self.mode = "match"

    def close(self) -> None:
        self.handle.remove()


class BNFeatureMatchingHook(_BaseFeatureHook):
    """Adapted from BNFeatureHook in utils.py for per-class online stats."""

    def __init__(self, module: nn.Module, momentum: float = 0.4) -> None:
        super().__init__(module, momentum)
        self.target_mean: Optional[torch.Tensor] = None
        self.target_var: Optional[torch.Tensor] = None
        self.dd_mean: Optional[torch.Tensor] = None
        self.dd_var: Optional[torch.Tensor] = None

    def compute_current_stats(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        nch = x.shape[1]
        mean = x.mean([0, 2, 3])
        var = x.permute(1, 0, 2, 3).contiguous().reshape(nch, -1).var(1, unbiased=False)
        return {"mean": mean, "var": var}

    def update_target(self, current: Dict[str, torch.Tensor], batch_size: int) -> None:
        mean = current["mean"].detach()
        var = current["var"].detach()
        if self.target_mean is None:
            self.target_mean = mean
            self.target_var = var
            self._target_count = batch_size
        else:
            total = self._target_count + batch_size
            self.target_mean = (self.target_mean * self._target_count + mean * batch_size) / total
            self.target_var = (self.target_var * self._target_count + var * batch_size) / total
            self._target_count = total

    def update_r_feature(self, current: Dict[str, torch.Tensor]) -> None:
        mean = current["mean"]
        var = current["var"]
        with torch.no_grad():
            if self.dd_mean is None:
                self.dd_mean = mean.detach()
                self.dd_var = var.detach()
            else:
                self.dd_mean = self.momentum * self.dd_mean + (1.0 - self.momentum) * mean.detach()
                self.dd_var = self.momentum * self.dd_var + (1.0 - self.momentum) * var.detach()

        assert self.target_mean is not None and self.target_var is not None
        self.r_feature = (
            torch.norm(self.target_var - (self.dd_var + var - var.detach()), 2)
            + torch.norm(self.target_mean - (self.dd_mean + mean - mean.detach()), 2)
        )


class ConvFeatureMatchingHook(_BaseFeatureHook):
    """Adapted from ConvFeatureHook in utils.py for per-class online stats."""

    def __init__(self, module: nn.Module, momentum: float = 0.4) -> None:
        super().__init__(module, momentum)
        self.target_dd_mean: Optional[torch.Tensor] = None
        self.target_dd_var: Optional[torch.Tensor] = None
        self.target_patch_mean: Optional[torch.Tensor] = None
        self.target_patch_var: Optional[torch.Tensor] = None
        self.dd_mean: Optional[torch.Tensor] = None
        self.dd_var: Optional[torch.Tensor] = None
        self.patch_mean: Optional[torch.Tensor] = None
        self.patch_var: Optional[torch.Tensor] = None

    def compute_current_stats(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        nch = x.shape[1]
        dd_mean = x.mean([0, 2, 3])
        dd_var = x.permute(1, 0, 2, 3).contiguous().reshape(nch, -1).var(1, unbiased=False)

        new_h, new_w = div_sixteen_mul(x.shape[2]), div_sixteen_mul(x.shape[3])
        resized = torch.nn.functional.interpolate(x, [new_h, new_w], mode="bilinear", align_corners=False)
        u = max(1, new_h // 16)
        v = max(1, new_w // 16)
        patches = resized.reshape(x.shape[0], x.shape[1], u, 16, v, 16)
        patches = patches.permute(2, 4, 0, 1, 3, 5).contiguous().reshape(u * v, -1)
        patch_mean = patches.mean(dim=1)
        patch_var = patches.var(dim=1, unbiased=False)

        return {
            "dd_mean": dd_mean,
            "dd_var": dd_var,
            "patch_mean": patch_mean,
            "patch_var": patch_var,
        }

    def update_target(self, current: Dict[str, torch.Tensor], batch_size: int) -> None:
        detached = {k: v.detach() for k, v in current.items()}
        if self.target_dd_mean is None:
            self.target_dd_mean = detached["dd_mean"]
            self.target_dd_var = detached["dd_var"]
            self.target_patch_mean = detached["patch_mean"]
            self.target_patch_var = detached["patch_var"]
            self._target_count = batch_size
        else:
            total = self._target_count + batch_size
            self.target_dd_mean = (self.target_dd_mean * self._target_count + detached["dd_mean"] * batch_size) / total
            self.target_dd_var = (self.target_dd_var * self._target_count + detached["dd_var"] * batch_size) / total
            self.target_patch_mean = (self.target_patch_mean * self._target_count + detached["patch_mean"] * batch_size) / total
            self.target_patch_var = (self.target_patch_var * self._target_count + detached["patch_var"] * batch_size) / total
            self._target_count = total

    def update_r_feature(self, current: Dict[str, torch.Tensor]) -> None:
        dd_mean = current["dd_mean"]
        dd_var = current["dd_var"]
        patch_mean = current["patch_mean"]
        patch_var = current["patch_var"]
        with torch.no_grad():
            if self.dd_mean is None:
                self.dd_mean = dd_mean.detach()
                self.dd_var = dd_var.detach()
                self.patch_mean = patch_mean.detach()
                self.patch_var = patch_var.detach()
            else:
                self.dd_mean = self.momentum * self.dd_mean + (1.0 - self.momentum) * dd_mean.detach()
                self.dd_var = self.momentum * self.dd_var + (1.0 - self.momentum) * dd_var.detach()
                self.patch_mean = self.momentum * self.patch_mean + (1.0 - self.momentum) * patch_mean.detach()
                self.patch_var = self.momentum * self.patch_var + (1.0 - self.momentum) * patch_var.detach()

        assert self.target_dd_mean is not None and self.target_dd_var is not None
        assert self.target_patch_mean is not None and self.target_patch_var is not None
        self.r_feature = (
            torch.norm(self.target_dd_var - (self.dd_var + dd_var - dd_var.detach()), 2)
            + torch.norm(self.target_dd_mean - (self.dd_mean + dd_mean - dd_mean.detach()), 2)
            + torch.norm(self.target_patch_mean - (self.patch_mean + patch_mean - patch_mean.detach()), 2)
            + torch.norm(self.target_patch_var - (self.patch_var + patch_var - patch_var.detach()), 2)
        )


def _build_feature_hooks(teacher: nn.Module, momentum: float = 0.4) -> List[_BaseFeatureHook]:
    hooks: List[_BaseFeatureHook] = []
    for module in teacher.modules():
        if isinstance(module, nn.BatchNorm2d):
            hooks.append(BNFeatureMatchingHook(module, momentum=momentum))
        elif isinstance(module, nn.Conv2d):
            hooks.append(ConvFeatureMatchingHook(module, momentum=momentum))
    return hooks


def _collect_class_feature_targets(
    teacher: nn.Module,
    hooks: Sequence[_BaseFeatureHook],
    real_images: torch.Tensor,
    batch_size: int,
) -> None:
    for hook in hooks:
        hook.set_collect()
    with torch.no_grad():
        for start in range(0, len(real_images), batch_size):
            teacher(real_images[start : start + batch_size])


def _feature_regularization(
    hooks: Sequence[_BaseFeatureHook],
    first_multiplier: float,
    device: torch.device,
) -> torch.Tensor:
    if not hooks:
        return torch.zeros((), device=device)
    losses: List[torch.Tensor] = []
    for idx, hook in enumerate(hooks):
        if hook.r_feature is None:
            continue
        multiplier = first_multiplier if idx == 0 else 1.0
        losses.append(hook.r_feature * multiplier)
    if not losses:
        return torch.zeros((), device=device)
    return torch.stack(losses).sum()


def distill_task(
    real_images: torch.Tensor,
    class_label: int,
    teacher: nn.Module,
    n_synthetic: int,
    device: torch.device,
    iterations: int = 500,
    K: int = 350,
    loss_threshold: float = 0.5,
    lr: float = 0.05,
    img_size: int = 32,
    r_loss: float = 0.05,
    first_multiplier: float = 10.0,
    tv_l1_weight: float = 0.0,
    tv_l2_weight: float = 0.0,
    training_momentum: float = 0.4,
    crop_scale: Tuple[float, float] = (0.5, 1.0),
    stats_batch_size: int = 128,
    return_stats: bool = False,
):
    """
    Distill synthetic images for one class using the fuller E2D objective.

    The implementation follows the original recover.py logic more closely than
    the earlier lightweight version:
      - full-image initialization from real images
      - exploration/exploitation crop memory
      - teacher CE objective on augmented views
      - BN/Conv feature-stat regularization adapted from utils.py
      - optional image prior losses
      - cosine LR schedule during image optimization
    """
    teacher = teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    real_images = real_images.to(device=device, dtype=torch.float32)
    if real_images.shape[0] == 0:
        empty = torch.empty((0, 3, img_size, img_size), device=device)
        return (empty, E2DStats(0.0, 0.0, 0.0, 0.0, 0.0, 0)) if return_stats else empty

    hooks = _build_feature_hooks(teacher, momentum=training_momentum)
    try:
        _collect_class_feature_targets(teacher, hooks, real_images, batch_size=stats_batch_size)

        init_indices = torch.randint(0, real_images.shape[0], (n_synthetic,), device=device)
        inputs = real_images[init_indices].clone().detach().requires_grad_(True)
        targets = torch.full((n_synthetic,), class_label, dtype=torch.long, device=device)

        optimizer = optim.Adam([inputs], lr=lr, betas=(0.5, 0.9), eps=1e-8)
        lr_scheduler = lr_cosine_policy(lr, 0, max(1, iterations))
        criterion = nn.CrossEntropyLoss(reduction="none")
        aug = ExplorationExploitationAug(n_synthetic, img_size=img_size, crop_scale=crop_scale)

        high_loss_crops: List[List[Tuple[int, int, int, int]]] = [[] for _ in range(n_synthetic)]
        high_loss_values: List[List[float]] = [[] for _ in range(n_synthetic)]

        best_cost = float("inf")
        best_inputs = inputs.detach().clone()
        final_stats = E2DStats(0.0, 0.0, 0.0, 0.0, 0.0, 0)

        for iteration in range(iterations):
            if iteration > K and all(len(crops) == 0 for crops in high_loss_crops):
                print(f"  [E2D] Early stop at iter {iteration} — crop buffer empty")
                break

            lr_scheduler(optimizer, iteration, iteration)
            optimizer.zero_grad(set_to_none=True)

            inputs_aug = aug(inputs, iteration, high_loss_crops, high_loss_values, K)
            for hook in hooks:
                hook.set_match()

            outputs = teacher(inputs_aug)
            loss_ce_all = criterion(outputs.float(), targets)
            loss_ce = loss_ce_all.mean()
            loss_r_feature = _feature_regularization(hooks, first_multiplier=first_multiplier, device=device)
            loss_var_l1, loss_var_l2 = get_image_prior_losses(inputs_aug)

            loss = loss_ce + r_loss * loss_r_feature + tv_l1_weight * loss_var_l1 + tv_l2_weight * loss_var_l2
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                inputs.copy_(clip_like_reference(inputs, real_images))

            loss_vals = loss_ce_all.detach().cpu().tolist()
            for img_idx, crop in enumerate(aug.last_crops):
                assert crop is not None
                if loss_vals[img_idx] > loss_threshold and iteration <= K:
                    if crop in high_loss_crops[img_idx]:
                        crop_idx = high_loss_crops[img_idx].index(crop)
                        high_loss_values[img_idx][crop_idx] = loss_vals[img_idx]
                    else:
                        high_loss_crops[img_idx].append(crop)
                        high_loss_values[img_idx].append(loss_vals[img_idx])

                selected = aug.selected_indices[img_idx]
                if selected is not None:
                    new_loss = loss_vals[img_idx]
                    if new_loss > loss_threshold:
                        high_loss_values[img_idx][selected] = new_loss
                    else:
                        del high_loss_crops[img_idx][selected]
                        del high_loss_values[img_idx][selected]

            current_loss = float(loss.detach().item())
            if current_loss < best_cost or iteration == 0:
                best_cost = current_loss
                best_inputs = inputs.detach().clone()

            final_stats = E2DStats(
                loss=current_loss,
                loss_ce=float(loss_ce.detach().item()),
                loss_r_feature=float(loss_r_feature.detach().item()),
                loss_var_l1=float(loss_var_l1.detach().item()),
                loss_var_l2=float(loss_var_l2.detach().item()),
                iterations=iteration + 1,
            )

            if iteration % 100 == 0 or iteration == iterations - 1:
                print(
                    "  [E2D] iter {:4d} | total {:.4f} | ce {:.4f} | feat {:.4f} | tv1 {:.4f} | tv2 {:.4f}".format(
                        iteration,
                        final_stats.loss,
                        final_stats.loss_ce,
                        final_stats.loss_r_feature,
                        final_stats.loss_var_l1,
                        final_stats.loss_var_l2,
                    )
                )

        if return_stats:
            return best_inputs.detach(), final_stats
        return best_inputs.detach()
    finally:
        for hook in hooks:
            hook.close()
