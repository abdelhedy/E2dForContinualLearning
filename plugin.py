
"""
plugin.py
---------
Avalanche plugin that integrates E2D dataset distillation into a continual
learning strategy, with:
- plain shuffled replay injection
- synthetic buffer storing teacher logits
- replay KD (DIST / KL / MSE-GT) during student training
"""

from __future__ import annotations

from typing import List, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin

from buffer import E2DBuffer
from distill import distill_task, relabel_synthetic_set


def _to_long_tensor(x) -> torch.Tensor:
    if torch.is_tensor(x):
        return x.long()
    return torch.tensor(x, dtype=torch.long)


def _get_class_tensors(
    dataset,
    class_id: int,
    device: torch.device,
    max_samples: int = 500,
) -> torch.Tensor:
    """Collect up to `max_samples` real images for one class from an Avalanche dataset."""
    imgs: List[torch.Tensor] = []
    total = 0
    loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=0)
    for batch in loader:
        x, y = batch[0], batch[1]
        mask = (y == class_id)
        if mask.any():
            picked = x[mask]
            imgs.append(picked)
            total += picked.shape[0]
        if total >= max_samples:
            break

    if not imgs:
        return torch.empty((0,), device=device)

    return torch.cat(imgs, dim=0)[:max_samples].to(device)


class _MergedDataset(torch.utils.data.Dataset):
    """
    Merges a real Avalanche dataset with a synthetic TensorDataset.

    Returned item format:
        (x, y, t, teacher_logits, is_replay)

    For real samples, `teacher_logits` is a zero vector and `is_replay=False`.
    For synthetic samples, `teacher_logits` comes from the buffer and
    `is_replay=True`.
    """

    def __init__(self, real_dataset, synthetic_dataset: TensorDataset, num_classes: int):
        self.real = real_dataset
        self.synthetic = synthetic_dataset
        self.num_classes = int(num_classes)
        self.real_len = len(real_dataset)
        self.syn_len = len(synthetic_dataset)

    def __len__(self):
        return self.real_len + self.syn_len

    def __getitem__(self, idx):
        if idx < self.real_len:
            sample = self.real[idx]
            if len(sample) >= 3:
                x, y, t = sample[0], sample[1], sample[2]
            elif len(sample) == 2:
                x, y = sample
                t = 0
            else:
                raise ValueError(f"Unexpected real sample length: {len(sample)}")
            y = _to_long_tensor(y)
            t = _to_long_tensor(t)
            logits = torch.zeros(self.num_classes, dtype=torch.float32)
            is_replay = torch.tensor(False, dtype=torch.bool)
            return x, y, t, logits, is_replay

        syn_idx = idx - self.real_len
        sample = self.synthetic[syn_idx]
        if len(sample) != 4:
            raise ValueError(f"Expected synthetic sample of length 4, got {len(sample)}")
        x, y, t, logits = sample
        y = _to_long_tensor(y)
        t = _to_long_tensor(t)
        logits = logits.float()
        is_replay = torch.tensor(True, dtype=torch.bool)
        return x, y, t, logits, is_replay


def cosine_similarity(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + eps)


def pearson_correlation(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    return cosine_similarity(a - a.mean(1, keepdim=True), b - b.mean(1, keepdim=True), eps)


def inter_class_relation(y_s: torch.Tensor, y_t: torch.Tensor) -> torch.Tensor:
    return 1 - pearson_correlation(y_s, y_t).mean()


def intra_class_relation(y_s: torch.Tensor, y_t: torch.Tensor) -> torch.Tensor:
    return inter_class_relation(y_s.transpose(0, 1), y_t.transpose(0, 1))


class DISTLoss(nn.Module):
    def __init__(self, beta: float = 2.0, gamma: float = 2.0, tem: float = 4.0):
        super().__init__()
        self.beta = beta
        self.gamma = gamma
        self.tem = tem

    def forward(self, logits_student: torch.Tensor, logits_teacher: torch.Tensor) -> torch.Tensor:
        y_s = (logits_student / self.tem).softmax(dim=1)
        y_t = (logits_teacher / self.tem).softmax(dim=1)
        inter_loss = (self.tem ** 2) * inter_class_relation(y_s, y_t)
        intra_loss = (self.tem ** 2) * intra_class_relation(y_s, y_t)
        return self.beta * inter_loss + self.gamma * intra_loss


class E2DReplayPlugin(SupervisedPlugin):
    def __init__(
        self,
        teacher: torch.nn.Module,
        num_classes: int = 10,
        buffer_size: int = 200,
        fixed_per_class: bool = True,
        distill_iters: int = 500,
        K: int = 350,
        loss_threshold: float = 0.5,
        lr: float = 0.05,
        img_size: int = 32,
        device: torch.device = torch.device("cpu"),
        r_loss: float = 0.05,
        first_multiplier: float = 10.0,
        tv_l1_weight: float = 0.0,
        tv_l2_weight: float = 0.0,
        training_momentum: float = 0.4,
        crop_scale: Tuple[float, float] = (0.5, 1.0),
        stats_batch_size: int = 128,
        max_real_per_class: int = 500,
        relabel_views: int = 1,
        relabel_temperature: float = 1.0,
        kd_loss: str = "dist",
        kd_weight: float = 0.2,
        kd_temperature: float = 4.0,
        same_crop_across_batch: bool = False,
    ):
        super().__init__()
        self.teacher = teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)

        self.num_classes = int(num_classes)
        self.buffer = E2DBuffer(max_size=buffer_size, fixed_per_class=fixed_per_class)

        self.distill_iters = int(distill_iters)
        self.K = int(K)
        self.loss_threshold = float(loss_threshold)
        self.lr = float(lr)
        self.img_size = int(img_size)
        self.device = device

        self.r_loss = float(r_loss)
        self.first_multiplier = float(first_multiplier)
        self.tv_l1_weight = float(tv_l1_weight)
        self.tv_l2_weight = float(tv_l2_weight)
        self.training_momentum = float(training_momentum)
        self.crop_scale = crop_scale
        self.stats_batch_size = int(stats_batch_size)
        self.max_real_per_class = int(max_real_per_class)

        self.relabel_views = int(relabel_views)
        self.relabel_temperature = float(relabel_temperature)

        self.kd_loss_name = kd_loss
        self.kd_weight = float(kd_weight)
        self.kd_temperature = float(kd_temperature)
        self.dist_loss = DISTLoss(tem=self.kd_temperature)

        self.same_crop_across_batch = bool(same_crop_across_batch)
        self.seen_classes: Set[int] = set()

    def before_training_exp(self, strategy, **kwargs):
        syn_dataset = self.buffer.get_dataset()
        if syn_dataset is None:
            print("[E2DPlugin] First experience — no replay data yet.")
            return

        print(f"[E2DPlugin] Using plain shuffled replay with {len(syn_dataset)} synthetic images.")
        merged = _MergedDataset(strategy.adapted_dataset, syn_dataset, num_classes=self.num_classes)
        strategy.dataloader = DataLoader(
            merged,
            batch_size=strategy.train_mb_size,
            shuffle=True,
            num_workers=0,
            drop_last=False,
        )

    def after_forward(self, strategy, **kwargs):
        if self.kd_weight <= 0:
            return

        mbatch = getattr(strategy, "mbatch", None)
        if mbatch is None or len(mbatch) < 5:
            return

        replay_mask = mbatch[4]
        if not torch.is_tensor(replay_mask):
            replay_mask = torch.as_tensor(replay_mask)
        replay_mask = replay_mask.to(strategy.mb_output.device).bool()

        if replay_mask.numel() == 0 or not replay_mask.any():
            return

        teacher_logits = mbatch[3].to(strategy.mb_output.device).float()
        student_logits = strategy.mb_output[replay_mask]
        teacher_logits = teacher_logits[replay_mask]

        kd = self._compute_kd(student_logits, teacher_logits)
        strategy.loss = strategy.loss + self.kd_weight * kd

    def _compute_kd(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
        if self.kd_loss_name == "dist":
            return self.dist_loss(student_logits, teacher_logits)

        if self.kd_loss_name == "kl":
            t = self.kd_temperature
            log_p_s = F.log_softmax(student_logits / t, dim=1)
            p_t = F.softmax(teacher_logits / t, dim=1)
            return F.kl_div(log_p_s, p_t, reduction="batchmean") * (t ** 2)

        if self.kd_loss_name == "mse_gt":
            p_s = F.softmax(student_logits / self.kd_temperature, dim=1)
            p_t = F.softmax(teacher_logits / self.kd_temperature, dim=1)
            return F.mse_loss(p_s, p_t)

        raise ValueError(f"Unknown kd_loss: {self.kd_loss_name}")

    def after_training_exp(self, strategy, **kwargs):
        exp = strategy.experience
        new_classes = list(exp.classes_in_this_experience)
        dataset = exp.dataset

        self.seen_classes.update(int(c) for c in new_classes)
        total_classes = len(self.seen_classes)

        print(f"\n[E2DPlugin] Distilling classes {new_classes} | total classes seen: {total_classes}")

        budget = self.buffer.budget_per_class(total_classes)
        task_id = int(getattr(exp, "current_experience", 0))

        for class_id in new_classes:
            class_id = int(class_id)
            print(f"  -> class {class_id} | target: {budget} synthetic images")

            real_imgs = _get_class_tensors(
                dataset=dataset,
                class_id=class_id,
                device=self.device,
                max_samples=self.max_real_per_class,
            )
            if real_imgs.numel() == 0:
                print(f"  [E2DPlugin] class {class_id}: no real images found, skipping.")
                continue

            synthetic, stats = distill_task(
                real_images=real_imgs,
                class_label=class_id,
                teacher=self.teacher,
                n_synthetic=budget,
                device=self.device,
                iterations=self.distill_iters,
                K=self.K,
                loss_threshold=self.loss_threshold,
                lr=self.lr,
                img_size=self.img_size,
                r_loss=self.r_loss,
                first_multiplier=self.first_multiplier,
                tv_l1_weight=self.tv_l1_weight,
                tv_l2_weight=self.tv_l2_weight,
                training_momentum=self.training_momentum,
                crop_scale=self.crop_scale,
                stats_batch_size=self.stats_batch_size,
                same_crop_across_batch=self.same_crop_across_batch,
                return_stats=True,
            )

            print(
                f"  [E2DPlugin] class {class_id}: total={stats.loss:.4f} | "
                f"ce={stats.loss_ce:.4f} | feat={stats.loss_r_feature:.4f} | "
                f"iters={stats.iterations}"
            )

            soft_logits = relabel_synthetic_set(
                images=synthetic,
                teacher=self.teacher,
                device=self.device,
                n_views=self.relabel_views,
                temperature=self.relabel_temperature,
                crop_scale=self.crop_scale,
                same_crop_across_batch=self.same_crop_across_batch,
            )

            self.buffer.update(
                class_id=class_id,
                new_images=synthetic,
                new_logits=soft_logits,
                total_classes=total_classes,
                task_id=task_id,
            )

        print(f"[E2DPlugin] Buffer: {self.buffer}\n")
