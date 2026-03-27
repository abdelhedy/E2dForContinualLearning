"""Main entry point for E2D-based continual learning on SplitCIFAR10."""

import argparse
import torch
import torch.nn as nn
import torchvision.models as tv_models

from avalanche.benchmarks.classic import SplitCIFAR10
from avalanche.training import Naive
from avalanche.training.plugins import ReplayPlugin
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics, forgetting_metrics
from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin

from plugin import E2DReplayPlugin
from teacher import load_cifar_teacher


def build_cifar_resnet18(num_classes: int = 10) -> nn.Module:
    model = tv_models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(512, num_classes)
    return model


def build_evaluator() -> EvaluationPlugin:
    return EvaluationPlugin(
        accuracy_metrics(experience=True, stream=True),
        loss_metrics(experience=True, stream=True),
        forgetting_metrics(experience=True, stream=True),
        loggers=[InteractiveLogger()],
    )


def run_strategy(strategy, benchmark, name: str) -> dict:
    print(f"\n{'=' * 60}\n  Running: {name}\n{'=' * 60}")
    results = []
    for experience in benchmark.train_stream:
        print(
            f"\n--- Training on experience {experience.current_experience} "
            f"(classes {experience.classes_in_this_experience}) ---"
        )
        strategy.train(experience)
        results.append(strategy.eval(benchmark.test_stream))
    final = results[-1]
    acc_key = [k for k in final if "Top1_Acc_Stream" in k]
    fgt_key = [k for k in final if "StreamForgetting" in k]
    final_acc = final[acc_key[0]] * 100 if acc_key else 0.0
    forgetting = final[fgt_key[0]] * 100 if fgt_key else 0.0
    return {"name": name, "final_acc": final_acc, "forgetting": forgetting}


def main():
    parser = argparse.ArgumentParser(description="Full E2D distillation for continual learning on SplitCIFAR10")
    parser.add_argument("--buffer-size", type=int, default=200)
    parser.add_argument("--distill-iters", type=int, default=500)
    parser.add_argument("--K", type=int, default=350)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-experiences", type=int, default=5)
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--strategy", type=str, default="all", choices=["all", "naive", "random", "e2d"])
    parser.add_argument("--teacher-ckpt", type=str, default="./cifar_models/resnet50_cifar10_lr01.pth")
    parser.add_argument("--e2d-lr", type=float, default=0.05, help="Learning rate for synthetic image optimization")
    parser.add_argument("--e2d-r-loss", type=float, default=0.05, help="Weight of feature-stat regularization")
    parser.add_argument("--e2d-first-multiplier", type=float, default=10.0, help="Extra multiplier for the first feature hook")
    parser.add_argument("--e2d-tv-l1", type=float, default=0.0, help="L1 image prior weight")
    parser.add_argument("--e2d-tv-l2", type=float, default=0.0, help="L2 image prior weight")
    parser.add_argument("--e2d-training-momentum", type=float, default=0.4)
    parser.add_argument("--e2d-loss-threshold", type=float, default=0.5)
    parser.add_argument("--e2d-max-real-per-class", type=int, default=500)
    parser.add_argument("--e2d-stats-batch-size", type=int, default=128)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Device: {device}")
    torch.manual_seed(args.seed)

    benchmark = SplitCIFAR10(
        n_experiences=args.n_experiences,
        seed=args.seed,
        return_task_id=False,
        shuffle=True,
    )
    num_classes = 10
    results_table = []

    if args.strategy in ("all", "naive"):
        model = build_cifar_resnet18(num_classes).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
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

    if args.strategy in ("all", "random"):
        model = build_cifar_resnet18(num_classes).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
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
        results_table.append(run_strategy(strategy, benchmark, "RandomReplay"))

    if args.strategy in ("all", "e2d"):
        teacher = load_cifar_teacher(device, args.teacher_ckpt)
        e2d_plugin = E2DReplayPlugin(
            teacher=teacher,
            buffer_size=args.buffer_size,
            distill_iters=args.distill_iters,
            K=args.K,
            loss_threshold=args.e2d_loss_threshold,
            lr=args.e2d_lr,
            img_size=32,
            device=device,
            r_loss=args.e2d_r_loss,
            first_multiplier=args.e2d_first_multiplier,
            tv_l1_weight=args.e2d_tv_l1,
            tv_l2_weight=args.e2d_tv_l2,
            training_momentum=args.e2d_training_momentum,
            stats_batch_size=args.e2d_stats_batch_size,
            max_real_per_class=args.e2d_max_real_per_class,
        )
        model = build_cifar_resnet18(num_classes).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
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
        results_table.append(run_strategy(strategy, benchmark, "E2DReplay (full)"))

    print("\n" + "=" * 55)
    print(f"{'Strategy':<25} {'Final Acc (%)':>14} {'Forgetting (%)':>16}")
    print("=" * 55)
    for result in results_table:
        print(f"{result['name']:<25} {result['final_acc']:>14.2f} {result['forgetting']:>16.2f}")
    print("=" * 55)


if __name__ == "__main__":
    main()
