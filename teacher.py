
import torch
from cifar_models.models.resnet_models import ResNet50


def load_cifar_teacher(device, ckpt_path="./cifar_models/resnet50_cifar10_lr01.pth"):
    model = ResNet50().to(device)

    # PyTorch >= 2.6 changed torch.load default to weights_only=True.
    # This checkpoint stores metadata in a pickled dict, so force full load.
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        # Backward compatibility with older torch versions that don't accept weights_only.
        ckpt = torch.load(ckpt_path, map_location=device)

    state_dict = ckpt["net"] if isinstance(ckpt, dict) and "net" in ckpt else ckpt
    model.load_state_dict(state_dict)

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    print(f"Teacher loaded from: {ckpt_path}")
    if isinstance(ckpt, dict):
        print(f"Stored checkpoint acc: {ckpt.get('acc', 'N/A')}")
        print(f"Stored epoch: {ckpt.get('epoch', 'N/A')}")
    else:
        print("Stored checkpoint acc: N/A")
        print("Stored epoch: N/A")

    return model