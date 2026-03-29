"""Avalanche plugin for E2D synthetic replay."""

from __future__ import annotations
from typing import Set, List
import torch
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin

from buffer import E2DBuffer
from distill import distill_task


# ── Custom collate: handles mixed int/tensor labels from ConcatDataset ────────
def _safe_collate(batch: List) -> tuple:
    """
    Avalanche datasets return labels as plain Python ints.
    Synthetic datasets return labels as tensors.
    This collate handles both so ConcatDataset batches don't crash.
    """
    xs, ys, ts = [], [], []
    for item in batch:
        x, y, t = item[0], item[1], item[2]
        xs.append(x if torch.is_tensor(x) else torch.tensor(x))
        ys.append(torch.tensor(y, dtype=torch.long)
                  if not torch.is_tensor(y)
                  else y.long())
        ts.append(torch.tensor(t, dtype=torch.long)
                  if not torch.is_tensor(t)
                  else t.long())
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


class E2DReplayPlugin(SupervisedPlugin):
    def __init__(
        self,
        teacher: torch.nn.Module,
        buffer_size:       int   = 50,
        fixed_per_class:   bool  = True,
        distill_iters:     int   = 500,
        K:                 int   = 350,
        loss_threshold:    float = 0.5,
        lr:                float = 0.05,
        img_size:          int   = 32,
        device:            torch.device = torch.device("cpu"),
        r_loss:            float = 0.05,
        first_multiplier:  float = 10.0,
        tv_l1_weight:      float = 0.0,
        tv_l2_weight:      float = 0.0,
        training_momentum: float = 0.4,
        crop_scale               = (0.5, 1.0),
        stats_batch_size:  int   = 128,
        max_real_per_class:int   = 500,
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
        self.seen_classes: Set[int] = set()

    # ── Inject synthetic data before every epoch ──────────────────────────────
    def before_training_epoch(self, strategy, **kwargs):
        syn_dataset = self.buffer.get_dataset()
        if syn_dataset is None:
            return

        verbose = (strategy.clock.train_exp_epochs == 0)

        syn_wrapped = _SyntheticDataset(syn_dataset)
        merged      = ConcatDataset([strategy.adapted_dataset, syn_wrapped])

        # Use _safe_collate to handle int vs tensor label mismatch
        strategy.dataloader = DataLoader(
            merged,
            batch_size=strategy.train_mb_size,
            shuffle=True,
            num_workers=0,
            drop_last=False,
            collate_fn=_safe_collate,       # ← THE FIX
        )

        if verbose:
            print(
                f"[E2DPlugin] Injecting {len(syn_wrapped)} synthetic images. "
                f"Total: {len(merged)} "
                f"({len(strategy.adapted_dataset)} real "
                f"+ {len(syn_wrapped)} synthetic)"
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