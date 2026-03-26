"""
train.py
--------
Main entry point for E2D-based continual learning on SplitCIFAR10.

Run:
    python train.py
    python train.py --buffer-size 500 --distill-iters 300 --epochs 20

The script trains three strategies and prints a comparison table:
  1. Naive          — fine-tuning, no replay (lower bound)
  2. RandomReplay   — replay with random real images (strong baseline)
  3. E2DReplay      — replay with E2D-distilled synthetic images (ours)
"""

import argparse
import torch
import torch.nn as nn
import torchvision.models as tv_models

from avalanche.benchmarks.classic import SplitCIFAR10
from avalanche.training import Naive
from avalanche.training.plugins import ReplayPlugin
from avalanche.evaluation.metrics import (
    accuracy_metrics, loss_metrics, forgetting_metrics
)
from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin

from plugin import E2DReplayPlugin
from teacher import load_cifar_teacher


# ──────────────────────────────────────────────────────────────────────────────
#  Lightweight ResNet-18 adapted for CIFAR (32×32 input)
# ──────────────────────────────────────────────────────────────────────────────
def build_cifar_resnet18(num_classes: int = 10) -> nn.Module:
    """
    Standard ResNet-18 with the first conv and maxpool replaced for
    32×32 inputs (same modification used in most CIFAR-10 papers).
    """
    model = tv_models.resnet18(weights=None)
    model.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1,
                               padding=1, bias=False)
    model.maxpool  = nn.Identity()
    model.fc       = nn.Linear(512, num_classes)
    return model


# ──────────────────────────────────────────────────────────────────────────────
#  Build a shared evaluation plugin (logs to console)
# ──────────────────────────────────────────────────────────────────────────────
def build_evaluator() -> EvaluationPlugin:
    return EvaluationPlugin(
        accuracy_metrics(experience=True, stream=True),
        loss_metrics(experience=True, stream=True),
        forgetting_metrics(experience=True, stream=True),
        loggers=[InteractiveLogger()],
    )


# ──────────────────────────────────────────────────────────────────────────────
#  Run a strategy through all experiences and return final accuracy
# ──────────────────────────────────────────────────────────────────────────────
def run_strategy(strategy, benchmark, name: str) -> dict:
    print(f"\n{'='*60}")
    print(f"  Running: {name}")
    print(f"{'='*60}")

    results = []
    for experience in benchmark.train_stream:
        print(f"\n--- Training on experience {experience.current_experience} "
              f"(classes {experience.classes_in_this_experience}) ---")
        strategy.train(experience)
        res = strategy.eval(benchmark.test_stream)
        results.append(res)

    # Extract final stream accuracy and average forgetting
    final = results[-1]
    acc_key = [k for k in final if "Top1_Acc_Stream" in k]
    fgt_key = [k for k in final if "StreamForgetting" in k]

    final_acc = final[acc_key[0]] * 100 if acc_key else 0.0
    forgetting = final[fgt_key[0]] * 100 if fgt_key else 0.0
    return {"name": name, "final_acc": final_acc, "forgetting": forgetting}


# ──────────────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="E2D Distillation for Continual Learning — SplitCIFAR10"
    )
    parser.add_argument("--buffer-size",    type=int,   default=200,
                        help="Total synthetic images in E2D buffer")
    parser.add_argument("--distill-iters",  type=int,   default=500,
                        help="E2D optimisation iterations per class")
    parser.add_argument("--K",              type=int,   default=350,
                        help="Exploration→Exploitation switch iteration")
    parser.add_argument("--epochs",         type=int,   default=10,
                        help="Training epochs per experience")
    parser.add_argument("--lr",             type=float, default=0.1,
                        help="Student model learning rate")
    parser.add_argument("--batch-size",     type=int,   default=128)
    parser.add_argument("--seed",           type=int,   default=42)
    parser.add_argument("--n-experiences",  type=int,   default=5,
                        help="Number of experiences to split CIFAR-10 into")
    parser.add_argument("--no-cuda",        action="store_true",
                        help="Disable GPU")
    parser.add_argument("--strategy",       type=str,   default="all",
                        choices=["all", "naive", "random", "e2d"],
                        help="Which strategy to run")
    parser.add_argument("--teacher-ckpt", type=str,
                    default="./cifar_models/resnet50_cifar10_lr01.pth",
                    help="Path to pretrained CIFAR-10 teacher checkpoint")
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )
    print(f"Device: {device}")
    torch.manual_seed(args.seed)

    # ── Benchmark ────────────────────────────────────────────────────────────
    benchmark = SplitCIFAR10(
        n_experiences=args.n_experiences,
        seed=args.seed,
        return_task_id=False,   # class-incremental (harder, more realistic)
        shuffle=True,
    )
    num_classes = 10

    results_table = []

    # ── 1. Naive (no replay) — lower bound ───────────────────────────────────
    if args.strategy in ("all", "naive"):
        model     = build_cifar_resnet18(num_classes).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                    momentum=0.9, weight_decay=1e-4)
        strategy = Naive(
            model=model,
            optimizer=optimizer,
            criterion=nn.CrossEntropyLoss(),
            train_mb_size=args.batch_size,
            train_epochs=args.epochs,
            eval_mb_size=256,
            device=device,
            evaluator=build_evaluator(),
        )
        results_table.append(run_strategy(strategy, benchmark, "Naive"))

    # ── 2. Random replay — strong baseline ───────────────────────────────────
    if args.strategy in ("all", "random"):
        model     = build_cifar_resnet18(num_classes).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                    momentum=0.9, weight_decay=1e-4)
        strategy = Naive(
            model=model,
            optimizer=optimizer,
            criterion=nn.CrossEntropyLoss(),
            train_mb_size=args.batch_size,
            train_epochs=args.epochs,
            eval_mb_size=256,
            device=device,
            evaluator=build_evaluator(),
            plugins=[ReplayPlugin(mem_size=args.buffer_size)],
        )
        results_table.append(
            run_strategy(strategy, benchmark, "RandomReplay")
        )

    # ── 3. E2D Replay — our method ───────────────────────────────────────────
    if args.strategy in ("all", "e2d"):
        # Teacher: pretrained CIFAR-10 ResNet-50 teacher (CIFAR-10 weights, frozen)
        teacher = load_cifar_teacher(device, args.teacher_ckpt)

        e2d_plugin = E2DReplayPlugin(
            teacher        = teacher,
            buffer_size    = args.buffer_size,
            distill_iters  = args.distill_iters,
            K              = args.K,
            loss_threshold = 0.5,
            lr             = 0.05,
            img_size       = 32,
            device         = device,
        )

        model     = build_cifar_resnet18(num_classes).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                    momentum=0.9, weight_decay=1e-4)
        strategy = Naive(
            model=model,
            optimizer=optimizer,
            criterion=nn.CrossEntropyLoss(),
            train_mb_size=args.batch_size,
            train_epochs=args.epochs,
            eval_mb_size=256,
            device=device,
            evaluator=build_evaluator(),
            plugins=[e2d_plugin],
        )
        results_table.append(
            run_strategy(strategy, benchmark, "E2DReplay (ours)")
        )

    # ── Final comparison table ────────────────────────────────────────────────
    print("\n" + "="*55)
    print(f"{'Strategy':<25} {'Final Acc (%)':>14} {'Forgetting (%)':>16}")
    print("="*55)
    for r in results_table:
        print(f"{r['name']:<25} {r['final_acc']:>14.2f} {r['forgetting']:>16.2f}")
    print("="*55)


if __name__ == "__main__":
    main()
