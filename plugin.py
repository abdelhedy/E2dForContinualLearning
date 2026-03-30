"""Avalanche plugin for E2D replay with teacher-logit KD.

This version removes replay oversampling and instead uses a plain shuffled
merged loader. Replay supervision is strengthened by storing teacher relabeling
logits in the buffer and adding a KD loss on replay samples.
"""

from __future__ import annotations

from typing import List, Sequence, Set, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset, TensorDataset
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin

from buffer import E2DBuffer
from distill import distill_task, relabel_synthetic_set


class DISTLoss(torch.nn.Module):
    def __init__(self, beta: float = 2.0, gamma: float = 2.0, tem: float = 4.0):
        super().__init__()
        self.beta = beta
        self.gamma = gamma
        self.tem = tem

    @staticmethod
    def _cosine_similarity(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + eps)

    def _pearson_correlation(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self._cosine_similarity(a - a.mean(1, keepdim=True), b - b.mean(1, keepdim=True))

    def _inter_class_relation(self, y_s: torch.Tensor, y_t: torch.Tensor) -> torch.Tensor:
        return 1 - self._pearson_correlation(y_s, y_t).mean()

    def _intra_class_relation(self, y_s: torch.Tensor, y_t: torch.Tensor) -> torch.Tensor:
        return self._inter_class_relation(y_s.transpose(0, 1), y_t.transpose(0, 1))

    def forward(self, logits_student: torch.Tensor, logits_teacher: torch.Tensor) -> torch.Tensor:
        y_s = (logits_student / self.tem).softmax(dim=1)
        y_t = (logits_teacher / self.tem).softmax(dim=1)
        inter_loss = (self.tem ** 2) * self._inter_class_relation(y_s, y_t)
        intra_loss = (self.tem ** 2) * self._intra_class_relation(y_s, y_t)
        return self.beta * inter_loss + self.gamma * intra_loss


class _MergedDataset(Dataset):
    def __init__(self, real_dataset, synthetic_dataset: TensorDataset, num_classes: int):
        self.real = real_dataset
        self.synthetic = synthetic_dataset
        self.real_len = len(real_dataset)
        self.syn_len = len(synthetic_dataset)
        self.num_classes = num_classes

    def __len__(self) -> int:
        return self.real_len + self.syn_len

    def __getitem__(self, idx):
        if idx < self.real_len:
            sample = self.real[idx]
            x, y = sample[0], sample[1]
            t = sample[2] if len(sample) > 2 else 0
            soft = torch.zeros(self.num_classes, dtype=torch.float32)
        else:
            x, y, t, soft = self.synthetic[idx - self.real_len]

        if not torch.is_tensor(y):
            y = torch.tensor(y, dtype=torch.long)
        else:
            y = y.long()
        if not torch.is_tensor(t):
            t = torch.tensor(t, dtype=torch.long)
        else:
            t = t.long()
        if not torch.is_tensor(soft):
            soft = torch.tensor(soft, dtype=torch.float32)
        else:
            soft = soft.float()
        return x, y, t, soft


def _safe_collate(batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]):
    xs, ys, ts, softs = zip(*batch)
    xs = [x if torch.is_tensor(x) else torch.tensor(x) for x in xs]
    ys = [y if torch.is_tensor(y) else torch.tensor(y, dtype=torch.long) for y in ys]
    ts = [t if torch.is_tensor(t) else torch.tensor(t, dtype=torch.long) for t in ts]
    softs = [s if torch.is_tensor(s) else torch.tensor(s, dtype=torch.float32) for s in softs]
    return torch.stack(xs), torch.stack(ys).long(), torch.stack(ts).long(), torch.stack(softs).float()


def _get_class_tensors(dataset, class_id: int, device: torch.device, max_samples: int = 500) -> torch.Tensor:
    imgs = []
    total = 0
    loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=0)
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


class E2DReplayPlugin(SupervisedPlugin):
    def __init__(
        self,
        teacher: torch.nn.Module,
        num_classes: int,
        buffer_size: int = 50,
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
        kd_loss: str = "kl",
        kd_weight: float = 1.0,
        kd_temperature: float = 4.0,
        same_crop_across_batch: bool = False,
    ):
        super().__init__()
        self.teacher = teacher
        self.num_classes = num_classes
        self.buffer = E2DBuffer(max_size=buffer_size, fixed_per_class=fixed_per_class)
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
        self.relabel_views = relabel_views
        self.relabel_temperature = relabel_temperature
        self.kd_loss = kd_loss
        self.kd_weight = kd_weight
        self.kd_temperature = kd_temperature
        self.same_crop_across_batch = same_crop_across_batch
        self.seen_classes: Set[int] = set()
        self.dist_loss = DISTLoss(tem=kd_temperature)

    def before_training_exp(self, strategy, **kwargs):
        syn_dataset = self.buffer.get_dataset()
        if syn_dataset is None:
            print("[E2DPlugin] First experience — no replay data yet.")
            return
        merged = _MergedDataset(strategy.adapted_dataset, syn_dataset, num_classes=self.num_classes)
        strategy.dataloader = DataLoader(
            merged,
            batch_size=strategy.train_mb_size,
            shuffle=True,
            num_workers=0,
            drop_last=False,
            collate_fn=_safe_collate,
        )
        print(f"[E2DPlugin] Using plain shuffled replay with {len(syn_dataset)} synthetic images.")

    def before_backward(self, strategy, **kwargs):
        if len(strategy.mbatch) < 4:
            return
        task_ids = strategy.mbatch[2].to(strategy.device)
        soft_targets = strategy.mbatch[3].to(strategy.device)
        replay_mask = task_ids < 0
        if not replay_mask.any():
            return

        student_logits = strategy.mb_output[replay_mask]
        teacher_logits = soft_targets[replay_mask]

        if self.kd_loss == "kl":
            log_p = F.log_softmax(student_logits / self.kd_temperature, dim=1)
            q = F.softmax(teacher_logits / self.kd_temperature, dim=1)
            kd = F.kl_div(log_p, q, reduction="batchmean") * (self.kd_temperature ** 2)
        elif self.kd_loss == "dist":
            kd = self.dist_loss(student_logits, teacher_logits)
        elif self.kd_loss == "mse_gt":
            replay_targets = strategy.mb_y[replay_mask]
            kd = F.mse_loss(student_logits, teacher_logits) + 0.025 * F.cross_entropy(student_logits, replay_targets)
        else:
            raise ValueError(f"Unsupported kd_loss: {self.kd_loss}")

        strategy.loss = strategy.loss + self.kd_weight * kd

    def after_training_exp(self, strategy, **kwargs):
        exp = strategy.experience
        new_classes = list(exp.classes_in_this_experience)
        dataset = exp.dataset
        self.seen_classes.update(new_classes)
        total_classes = len(self.seen_classes)

        print(
            f"\n[E2DPlugin] Distilling classes {new_classes} | total classes seen: {total_classes}"
        )
        budget = self.buffer.budget_per_class(total_classes)

        for class_id in new_classes:
            print(f"  -> class {class_id} | target: {budget} synthetic images")
            real_imgs = _get_class_tensors(dataset, class_id, self.device, max_samples=self.max_real_per_class)
            if real_imgs.shape[0] == 0:
                print(f"  [E2DPlugin] No real images found for class {class_id}; skipping.")
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
                f"  [E2DPlugin] class {class_id}: total={stats.loss:.4f} | ce={stats.loss_ce:.4f} "
                f"| feat={stats.loss_r_feature:.4f} | iters={stats.iterations}"
            )

            soft_logits = relabel_synthetic_set(
                synthetic,
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
                task_id=exp.current_experience,
            )

        print(f"[E2DPlugin] Buffer: {self.buffer}\n")
