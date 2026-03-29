"""Avalanche plugin for E2D synthetic replay.

Key fix in this version: WeightedRandomSampler
-----------------------------------------------
Root cause of bad results: with 25000 real images and 100 synthetic,
synthetic images were only 0.4% of training data — statistically invisible.

Solution: WeightedRandomSampler gives synthetic images 20x higher sampling
weight so they appear in ~17% of batches regardless of buffer size.
This is standard practice in imbalanced learning and replay methods.
"""

from __future__ import annotations
from typing import Set, List
import torch
from torch.utils.data import (
    DataLoader, TensorDataset, ConcatDataset, WeightedRandomSampler
)
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin

from buffer import E2DBuffer
from distill import distill_task


# ── Collate: handles mixed int/tensor labels from ConcatDataset ───────────────
def _safe_collate(batch: List) -> tuple:
    xs, ys, ts = [], [], []
    for item in batch:
        x, y, t = item[0], item[1], item[2]
        xs.append(x if torch.is_tensor(x) else torch.tensor(x))
        ys.append(torch.tensor(y, dtype=torch.long)
                  if not torch.is_tensor(y) else y.long())
        ts.append(torch.tensor(t, dtype=torch.long)
                  if not torch.is_tensor(t) else t.long())
    return torch.stack(xs), torch.stack(ys), torch.stack(ts)


def _get_class_tensors(
    dataset,
    class_id: int,
    device: torch.device,
    max_samples: int = 500,
) -> torch.Tensor:
    imgs = []
    loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=0)
    total = 0
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


class _SyntheticDataset(torch.utils.data.Dataset):
    """Wraps buffer TensorDataset → returns (x, y_tensor, t_tensor)."""
    def __init__(self, tensor_dataset: TensorDataset):
        self.ds = tensor_dataset

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        x, y, t = self.ds[idx]
        return x, y.long(), t.long()


def _make_weighted_loader(
    real_dataset,
    syn_dataset: _SyntheticDataset,
    batch_size: int,
    synthetic_weight: float = 20.0,
) -> DataLoader:
    """
    Build a DataLoader where synthetic images are oversampled.

    synthetic_weight=20 means each synthetic image is sampled 20x
    more often than each real image per epoch. This ensures synthetic
    images appear in ~17% of every batch regardless of buffer size.

    Formula for expected synthetic fraction in a batch:
        n_syn * w_syn
        ─────────────────────────────────────────────
        n_syn * w_syn + n_real * w_real

    With n_real=25000, n_syn=100, w_real=1, w_syn=20:
        100*20 / (100*20 + 25000*1) = 2000/27000 ≈ 7.4%

    With w_syn=50:
        100*50 / (100*50 + 25000*1) = 5000/30000 ≈ 16.7%
    """
    merged = ConcatDataset([real_dataset, syn_dataset])
    n_real = len(real_dataset)
    n_syn  = len(syn_dataset)

    # Weight: 1.0 for real samples, synthetic_weight for synthetic samples
    weights = torch.cat([
        torch.ones(n_real),
        torch.full((n_syn,), synthetic_weight),
    ])

    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=n_real + n_syn,   # one full "epoch" worth
        replacement=True,
    )

    syn_frac = (n_syn * synthetic_weight) / (
        n_syn * synthetic_weight + n_real * 1.0
    ) * 100

    print(
        f"[E2DPlugin] WeightedSampler: {n_real} real (w=1) + {n_syn} synthetic "
        f"(w={synthetic_weight:.0f}) → synthetic appears in ~{syn_frac:.1f}% of samples"
    )

    return DataLoader(
        merged,
        batch_size=batch_size,
        sampler=sampler,          # sampler handles shuffling — no shuffle=True
        num_workers=0,
        drop_last=False,
        collate_fn=_safe_collate,
    )


class E2DReplayPlugin(SupervisedPlugin):
    def __init__(
        self,
        teacher: torch.nn.Module,
        buffer_size:        int   = 50,
        fixed_per_class:    bool  = True,
        distill_iters:      int   = 500,
        K:                  int   = 350,
        loss_threshold:     float = 0.5,
        lr:                 float = 0.05,
        img_size:           int   = 32,
        device:             torch.device = torch.device("cpu"),
        r_loss:             float = 0.05,
        first_multiplier:   float = 10.0,
        tv_l1_weight:       float = 0.001,
        tv_l2_weight:       float = 0.0001,
        training_momentum:  float = 0.4,
        crop_scale                = (0.5, 1.0),
        stats_batch_size:   int   = 128,
        max_real_per_class: int   = 500,
        synthetic_weight:   float = 50.0,  # oversampling factor for synthetic
    ):
        super().__init__()
        self.teacher           = teacher
        self.buffer            = E2DBuffer(
            max_size=buffer_size,
            fixed_per_class=fixed_per_class,
        )
        self.distill_iters     = distill_iters
        self.K                 = K
        self.loss_threshold    = loss_threshold
        self.lr                = lr
        self.img_size          = img_size
        self.device            = device
        self.r_loss            = r_loss
        self.first_multiplier  = first_multiplier
        self.tv_l1_weight      = tv_l1_weight
        self.tv_l2_weight      = tv_l2_weight
        self.training_momentum = training_momentum
        self.crop_scale        = crop_scale
        self.stats_batch_size  = stats_batch_size
        self.max_real_per_class= max_real_per_class
        self.synthetic_weight  = synthetic_weight
        self.seen_classes: Set[int] = set()

    # ── Inject with weighted sampler before every epoch ───────────────────────
    def before_training_epoch(self, strategy, **kwargs):
        syn_dataset = self.buffer.get_dataset()
        if syn_dataset is None:
            return

        verbose = (strategy.clock.train_exp_epochs == 0)
        syn_wrapped = _SyntheticDataset(syn_dataset)

        strategy.dataloader = _make_weighted_loader(
            real_dataset     = strategy.adapted_dataset,
            syn_dataset      = syn_wrapped,
            batch_size       = strategy.train_mb_size,
            synthetic_weight = self.synthetic_weight,
        ) if verbose else DataLoader(
            ConcatDataset([strategy.adapted_dataset, syn_wrapped]),
            batch_size   = strategy.train_mb_size,
            sampler      = WeightedRandomSampler(
                weights      = torch.cat([
                    torch.ones(len(strategy.adapted_dataset)),
                    torch.full((len(syn_wrapped),), self.synthetic_weight),
                ]),
                num_samples  = len(strategy.adapted_dataset) + len(syn_wrapped),
                replacement  = True,
            ),
            num_workers  = 0,
            drop_last    = False,
            collate_fn   = _safe_collate,
        )

    # ── Distil after each experience ──────────────────────────────────────────
    def after_training_exp(self, strategy, **kwargs):
        exp           = strategy.experience
        new_classes   = list(exp.classes_in_this_experience)
        self.seen_classes.update(int(c) for c in new_classes)
        total_classes = len(self.seen_classes)
        budget        = self.buffer.budget_per_class(total_classes)

        print(
            f"\n[E2DPlugin] Distilling {len(new_classes)} new class(es): "
            f"{new_classes} | total seen: {sorted(self.seen_classes)}"
        )

        for class_id in new_classes:
            print(f"  -> class {class_id} | target: {budget} synthetic images")

            real_imgs = _get_class_tensors(
                exp.dataset, int(class_id),
                self.device, max_samples=self.max_real_per_class,
            )
            if real_imgs.numel() == 0:
                print(f"  [E2DPlugin] Warning: no real images for class {class_id}.")
                continue

            synthetic, stats = distill_task(
                real_images      = real_imgs,
                class_label      = int(class_id),
                teacher          = self.teacher,
                n_synthetic      = budget,
                device           = self.device,
                iterations       = self.distill_iters,
                K                = self.K,
                loss_threshold   = self.loss_threshold,
                lr               = self.lr,
                img_size         = self.img_size,
                r_loss           = self.r_loss,
                first_multiplier = self.first_multiplier,
                tv_l1_weight     = self.tv_l1_weight,
                tv_l2_weight     = self.tv_l2_weight,
                training_momentum= self.training_momentum,
                crop_scale       = self.crop_scale,
                stats_batch_size = self.stats_batch_size,
                return_stats     = True,
            )
            print(
                f"  [E2DPlugin] class {class_id}: "
                f"total={stats.loss:.4f} | ce={stats.loss_ce:.4f} | "
                f"feat={stats.loss_r_feature:.4f} | iters={stats.iterations}"
            )
            self.buffer.update(
                int(class_id), synthetic,
                total_classes,
                task_id=int(exp.current_experience),
            )

        print(f"[E2DPlugin] Buffer: {self.buffer}\n")