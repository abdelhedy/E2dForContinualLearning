
import torch
from cifar_models.models.resnet_models import ResNet50


def load_cifar_teacher(device, ckpt_path="./cifar_models/resnet50_cifar10_lr01.pth"):
    model = ResNet50().to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["net"])

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    print(f"Teacher loaded from: {ckpt_path}")
    print(f"Stored checkpoint acc: {ckpt.get('acc', 'N/A')}")
    print(f"Stored epoch: {ckpt.get('epoch', 'N/A')}")

    return model