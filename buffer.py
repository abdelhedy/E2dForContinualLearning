"""Synthetic replay buffer for E2D continual learning."""

from __future__ import annotations

from typing import Dict, Optional

import torch
from torch.utils.data import TensorDataset


class E2DBuffer:
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.data: Dict[int, torch.Tensor] = {}
        self.task_ids: Dict[int, int] = {}

    @property
    def n_classes(self) -> int:
        return len(self.data)

    @property
    def total_images(self) -> int:
        return sum(t.shape[0] for t in self.data.values())

    def budget_per_class(self, total_classes: int) -> int:
        return max(1, self.max_size // max(1, total_classes))

    def update(self, class_id: int, new_images: torch.Tensor, total_classes: int, task_id: int = 0) -> None:
        budget = self.budget_per_class(total_classes)
        for cid in list(self.data.keys()):
            if self.data[cid].shape[0] > budget:
                self.data[cid] = self.data[cid][:budget].contiguous()
        self.data[class_id] = new_images[:budget].detach().cpu().contiguous()
        self.task_ids[class_id] = int(task_id)
        print(
            f"  [Buffer] class {class_id} -> {self.data[class_id].shape[0]} imgs | "
            f"total {self.total_images}/{self.max_size} | budget {budget}/class across {total_classes} classes"
        )

    def get_dataset(self) -> Optional[TensorDataset]:
        if not self.data:
            return None
        imgs = []
        labels = []
        tasks = []
        for class_id in sorted(self.data.keys()):
            class_imgs = self.data[class_id]
            n = class_imgs.shape[0]
            imgs.append(class_imgs)
            labels.append(torch.full((n,), class_id, dtype=torch.long))
            tasks.append(torch.full((n,), self.task_ids.get(class_id, 0), dtype=torch.long))
        return TensorDataset(torch.cat(imgs, dim=0), torch.cat(labels, dim=0), torch.cat(tasks, dim=0))

    def get_class_images(self, class_id: int) -> Optional[torch.Tensor]:
        return self.data.get(class_id)

    def __repr__(self) -> str:
        counts = {cid: imgs.shape[0] for cid, imgs in self.data.items()}
        return f"E2DBuffer(max_size={self.max_size}, counts={counts})"
