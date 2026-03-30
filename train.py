"""Main entry point for E2D-based continual learning on SplitCIFAR10."""

import argparse
import torch
import torch.nn as nn
import torchvision.models as tv_models

from avalanche.benchmarks.classic import SplitCIFAR10
from avalanche.evaluation.metrics import accuracy_metrics, forgetting_metrics, loss_metrics
from avalanche.logging import InteractiveLogger
from avalanche.training import Naive
from avalanche.training.plugins import EvaluationPlugin, ReplayPlugin

from plugin import E2DReplayPlugin
from teacher import load_cifar_teacher


def build_cifar_resnet18(num_classes: int = 10) -> nn.Module:
    model = tv_models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(512, num_classes)
    return model


def build_cifar_resnet50(num_classes: int = 10) -> nn.Module:
    model = tv_models.resnet50(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(2048, num_classes)
    return model


def build_student(arch: str, num_classes: int) -> nn.Module:
    if arch == "resnet18":
        return build_cifar_resnet18(num_classes)
    if arch == "resnet50":
        return build_cifar_resnet50(num_classes)
    raise ValueError(f"Unknown student arch: {arch}")


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
    parser = argparse.ArgumentParser(description="Faithful E2D-style continual learning on SplitCIFAR10")
    parser.add_argument("--buffer-size", type=int, default=50, help="E2D replay images per class when fixed-per-class is on")
    parser.add_argument("--fixed-per-class", action="store_true", default=True)
    parser.add_argument("--distill-iters", type=int, default=500)
    parser.add_argument("--K", type=int, default=350)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-experiences", type=int, default=5)
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--strategy", type=str, default="all", choices=["all", "naive", "random", "e2d"])
    parser.add_argument("--student-arch", type=str, default="resnet50", choices=["resnet18", "resnet50"])
    parser.add_argument("--teacher-ckpt", type=str, default="./cifar_models/resnet50_cifar10_lr01.pth")

    # E2D synthesis
    parser.add_argument("--e2d-lr", type=float, default=0.05)
    parser.add_argument("--e2d-r-loss", type=float, default=0.05)
    parser.add_argument("--e2d-first-multiplier", type=float, default=10.0)
    parser.add_argument("--e2d-tv-l1", type=float, default=0.0)
    parser.add_argument("--e2d-tv-l2", type=float, default=0.0)
    parser.add_argument("--e2d-training-momentum", type=float, default=0.4)
    parser.add_argument("--e2d-loss-threshold", type=float, default=0.5)
    parser.add_argument("--e2d-max-real-per-class", type=int, default=500)
    parser.add_argument("--e2d-stats-batch-size", type=int, default=128)
    parser.add_argument("--e2d-crop-min", type=float, default=0.5)
    parser.add_argument("--e2d-crop-max", type=float, default=1.0)
    parser.add_argument("--e2d-same-crop-across-batch", action="store_true", help="Paper-style option; source code uses per-image crop memory by default")

    # Relabeling / replay KD
    parser.add_argument("--e2d-relabel-views", type=int, default=1)
    parser.add_argument("--e2d-relabel-temperature", type=float, default=1.0)
    parser.add_argument("--e2d-kd-loss", type=str, default="kl", choices=["kl", "dist", "mse_gt"])
    parser.add_argument("--e2d-kd-weight", type=float, default=1.0)
    parser.add_argument("--e2d-kd-temperature", type=float, default=4.0)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Device: {device}")
    print(f"Student architecture: {args.student_arch}")
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
        model = build_student(args.student_arch, num_classes).to(device)
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
        model = build_student(args.student_arch, num_classes).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
        mem_size = args.buffer_size * num_classes if args.fixed_per_class else args.buffer_size
        strategy = Naive(
            model=model,
            optimizer=optimizer,
            criterion=nn.CrossEntropyLoss(),
            train_mb_size=args.batch_size,
            train_epochs=args.epochs,
            eval_mb_size=256,
            device=device,
            evaluator=build_evaluator(),
            plugins=[ReplayPlugin(mem_size=mem_size)],
        )
        results_table.append(run_strategy(strategy, benchmark, "RandomReplay"))

    if args.strategy in ("all", "e2d"):
        teacher = load_cifar_teacher(device, args.teacher_ckpt)
        e2d_plugin = E2DReplayPlugin(
            teacher=teacher,
            num_classes=num_classes,
            buffer_size=args.buffer_size,
            fixed_per_class=args.fixed_per_class,
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
            crop_scale=(args.e2d_crop_min, args.e2d_crop_max),
            stats_batch_size=args.e2d_stats_batch_size,
            max_real_per_class=args.e2d_max_real_per_class,
            relabel_views=args.e2d_relabel_views,
            relabel_temperature=args.e2d_relabel_temperature,
            kd_loss=args.e2d_kd_loss,
            kd_weight=args.e2d_kd_weight,
            kd_temperature=args.e2d_kd_temperature,
            same_crop_across_batch=args.e2d_same_crop_across_batch,
        )
        model = build_student(args.student_arch, num_classes).to(device)
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
        results_table.append(run_strategy(strategy, benchmark, "E2DReplay"))

    if results_table:
        print("\n" + "=" * 60)
        print("Summary")
        print("=" * 60)
        for row in results_table:
            print(
                f"{row['name']:>12} | Final stream acc: {row['final_acc']:.2f}% "
                f"| Forgetting: {row['forgetting']:.2f}%"
            )


if __name__ == "__main__":
    main()
