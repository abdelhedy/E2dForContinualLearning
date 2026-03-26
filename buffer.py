"""
buffer.py
---------
Fixed-size synthetic image buffer for continual learning replay.

The buffer stores distilled (synthetic) images for every class seen so far.
When new classes are added, the per-class budget is recalculated so the
total number of stored images never exceeds `max_size`.
"""

from matplotlib.pylab import sample
import torch
from torch.utils.data import TensorDataset
from typing import Dict, List, Optional


class E2DBuffer:
    """
    Manages a fixed-size pool of synthetic images.

    Attributes
    ----------
    max_size : int
        Maximum total number of synthetic images across all classes.
    data : dict[int, torch.Tensor]
        Maps class_id → tensor of shape (n, C, H, W).
    """

    def __init__(self, max_size: int):
        self.max_size: int = max_size
        self.data: Dict[int, torch.Tensor] = {}   # {class_id: images tensor}

    # ── Public API ─────────────────────────────────────────────────────────

    @property
    def n_classes(self) -> int:
        return len(self.data)

    @property
    def total_images(self) -> int:
        return sum(t.shape[0] for t in self.data.values())

    def budget_per_class(self, total_classes: int) -> int:
        """How many images each class gets given the current class count."""
        return max(1, self.max_size // total_classes)

    def update(
        self,
        class_id: int,
        new_images: torch.Tensor,     # (n, C, H, W)
        total_classes: int,           # includes this new class
    ) -> None:
        """
        Add / replace synthetic images for `class_id` and
        trim all classes to the updated per-class budget.
        """
        budget = self.budget_per_class(total_classes)

        # Trim existing classes to new (possibly smaller) budget
        for cid in list(self.data.keys()):
            if self.data[cid].shape[0] > budget:
                self.data[cid] = self.data[cid][:budget]

        # Store new images (CPU to save GPU memory)
        self.data[class_id] = new_images[:budget].cpu()

        print(
            f"  [Buffer] class {class_id} → {self.data[class_id].shape[0]} imgs "
            f"| total {self.total_images}/{self.max_size} "
            f"| budget {budget}/class across {total_classes} classes"
        )
    def get_dataset(self) -> Optional[TensorDataset]:
        if not self.data:
            print("[Buffer] get_dataset called but buffer is empty")
            return None

        all_imgs = []
        all_labels = []
        all_tasks = []

        for class_id, imgs in self.data.items():
            n = imgs.shape[0]
            all_imgs.append(imgs)
            all_labels.append(torch.full((n,), class_id, dtype=torch.long))
            all_tasks.append(torch.zeros(n, dtype=torch.long))  # dummy task ids

        dataset = TensorDataset(
            torch.cat(all_imgs, dim=0),
            torch.cat(all_labels, dim=0),
            torch.cat(all_tasks, dim=0),
        )

        # 🔍 DEBUG LOGGING
        print("\n[DEBUG][Buffer] Dataset created")
        print(f"[DEBUG][Buffer] Total samples: {len(dataset)}")

        sample = dataset[0]
        print(f"[DEBUG][Buffer] Sample type: {type(sample)}")
        print(f"[DEBUG][Buffer] Sample length: {len(sample)}")

        for i, elem in enumerate(sample):
            print(f"[DEBUG][Buffer] element[{i}] → type={type(elem)}, shape={getattr(elem, 'shape', None)}")

        print()

        return dataset

    # def get_dataset(self) -> Optional[TensorDataset]:
    #     """
    #     Return a TensorDataset (images, labels) of all synthetic images,
    #     or None if the buffer is empty.
    #     """
    #     if not self.data:
    #         return None

    #     all_imgs   = []
    #     all_labels = []
    #     for class_id, imgs in self.data.items():
    #         all_imgs.append(imgs)
    #         all_labels.append(
    #             torch.full((imgs.shape[0],), class_id, dtype=torch.long)
    #         )

    #     return TensorDataset(
    #         torch.cat(all_imgs,   dim=0),
    #         torch.cat(all_labels, dim=0),
    #     )
    

    def get_class_images(self, class_id: int) -> Optional[torch.Tensor]:
        """Return the stored images for a single class, or None."""
        return self.data.get(class_id, None)

    def __repr__(self) -> str:
        class_counts = {cid: t.shape[0] for cid, t in self.data.items()}
        return (
            f"E2DBuffer(max_size={self.max_size}, "
            f"classes={list(class_counts.keys())}, "
            f"counts={class_counts})"
        )
