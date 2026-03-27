"""Avalanche plugin for E2D synthetic replay."""

from __future__ import annotations

from typing import Iterable, Set

import torch
from torch.utils.data import DataLoader, TensorDataset
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin

from buffer import E2DBuffer
from distill import distill_task


def _get_class_tensors(dataset, class_id: int, device: torch.device, max_samples: int = 500) -> torch.Tensor:
    imgs = []
    loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=0)
    total = 0
    for batch in loader:
        x, y = batch[0], batch[1]
        mask = y == class_id
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
    def __init__(self, real_dataset, synthetic_dataset: TensorDataset):
        self.real = real_dataset
        self.synthetic = synthetic_dataset
        self.real_len = len(real_dataset)
        self.syn_len = len(synthetic_dataset)

    def __len__(self):
        return self.real_len + self.syn_len

    def __getitem__(self, idx):
        sample = self.real[idx] if idx < self.real_len else self.synthetic[idx - self.real_len]
        if len(sample) == 3:
            x, y, t = sample
        elif len(sample) == 2:
            x, y = sample
            t = 0
        else:
            raise ValueError(f"Unexpected sample format: len={len(sample)}")
        if not torch.is_tensor(y):
            y = torch.tensor(y, dtype=torch.long)
        else:
            y = y.long()
        if not torch.is_tensor(t):
            t = torch.tensor(t, dtype=torch.long)
        else:
            t = t.long()
        return x, y, t


class E2DReplayPlugin(SupervisedPlugin):
    def __init__(
        self,
        teacher: torch.nn.Module,
        buffer_size: int = 200,
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
        crop_scale=(0.5, 1.0),
        stats_batch_size: int = 128,
        max_real_per_class: int = 500,
    ):
        super().__init__()
        self.teacher = teacher
        self.buffer = E2DBuffer(max_size=buffer_size)
        self.distill_iters = distill_iters
        self.K = K
        self.loss_threshold = loss_threshold
        self.lr = lr
        self.img_size = img_size
        self.device = device
        self.r_loss = r_loss
        self.first_multiplier = first_multiplier
        self.tv_l1_weight = tv_l1_weight
        self.tv_l2_weight = tv_l2_weight
        self.training_momentum = training_momentum
        self.crop_scale = crop_scale
        self.stats_batch_size = stats_batch_size
        self.max_real_per_class = max_real_per_class
        self.seen_classes: Set[int] = set()

    def before_training_exp(self, strategy, **kwargs):
        syn_dataset = self.buffer.get_dataset()
        if syn_dataset is None:
            print("[E2DPlugin] First experience — no replay data yet.")
            return
        print(f"[E2DPlugin] Merging {len(syn_dataset)} synthetic images with real data for replay.")
        merged = _MergedDataset(strategy.adapted_dataset, syn_dataset)
        strategy.dataloader = DataLoader(
            merged,
            batch_size=strategy.train_mb_size,
            shuffle=True,
            num_workers=0,
            drop_last=False,
        )

    def after_training_exp(self, strategy, **kwargs):
        exp = strategy.experience
        new_classes = list(exp.classes_in_this_experience)
        self.seen_classes.update(int(c) for c in new_classes)
        total_classes = len(self.seen_classes)
        budget = self.buffer.budget_per_class(total_classes)

        print(
            f"\n[E2DPlugin] Distilling {len(new_classes)} new class(es): {new_classes} | "
            f"total classes so far: {sorted(self.seen_classes)}"
        )

        for class_id in new_classes:
            print(f"  -> distilling class {class_id} (target: {budget} synthetic images)...")
            real_imgs = _get_class_tensors(exp.dataset, int(class_id), self.device, max_samples=self.max_real_per_class)
            if real_imgs.numel() == 0:
                print(f"  [E2DPlugin] Warning: no images found for class {class_id}, skipping.")
                continue

            synthetic, stats = distill_task(
                real_images=real_imgs,
                class_label=int(class_id),
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
                return_stats=True,
            )
            print(
                f"  [E2DPlugin] class {class_id}: total={stats.loss:.4f}, ce={stats.loss_ce:.4f}, "
                f"feat={stats.loss_r_feature:.4f}, tv1={stats.loss_var_l1:.4f}, tv2={stats.loss_var_l2:.4f}, iters={stats.iterations}"
            )
            self.buffer.update(int(class_id), synthetic, total_classes, task_id=int(exp.current_experience))

        print(f"[E2DPlugin] Buffer after experience: {self.buffer}\n")
