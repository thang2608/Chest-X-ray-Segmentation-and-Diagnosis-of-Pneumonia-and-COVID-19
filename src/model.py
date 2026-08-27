# File duy nhất build classifier — Backend (Thắng) và Evaluation (Thịnh)
# đều import build_classifier. KHÔNG đổi tên hàm / thứ tự param positional.
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights

try:
    from .dataset import NUM_CLASSES as _DATASET_NUM_CLASSES
except Exception:  # pragma: no cover
    _DATASET_NUM_CLASSES = 3


# ------------------------------------------------------------------ #
# 1. BUILD CLASSIFIER                                                  #
# ------------------------------------------------------------------ #

def build_classifier(
    num_classes: int = _DATASET_NUM_CLASSES,
    pretrained: bool = True,
    dropout: float = 0.3,
    verbose: bool = True,
) -> nn.Module:
    """EfficientNet-B3 voi head thay bang Linear(num_classes). Chua .to(device)."""
    if num_classes < 2:
        raise ValueError(f"num_classes phai >= 2, nhan duoc {num_classes}.")
    if not (0.0 <= dropout < 1.0):
        raise ValueError(f"dropout phai trong [0, 1), nhan duoc {dropout}.")

    weights = EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
    model = efficientnet_b3(weights=weights)

    # Giu cau truc Sequential (classifier.0=Dropout, classifier.1=Linear)
    # de state_dict keys khong doi khi load weights.
    in_features = model.classifier[1].in_features  # 1536 voi B3
    model.classifier[0] = nn.Dropout(p=dropout, inplace=True)
    model.classifier[1] = nn.Linear(in_features, num_classes)

    if verbose:
        tag = "pretrained=ImageNet" if pretrained else "pretrained=scratch"
        print(f"[model] EfficientNet-B3 | {tag}")
        print(f"        in_features : {in_features}")
        print(f"        num_classes : {num_classes}")
        print(f"        dropout     : {dropout}")
        print(f"        total params: {count_params(model):,}")

    return model


# ------------------------------------------------------------------ #
# 2. FREEZE / UNFREEZE — 3-phase transfer learning                    #
# ------------------------------------------------------------------ #

def freeze_backbone(model: nn.Module, verbose: bool = True) -> None:
    """Freeze toan bo backbone (features), chi train head — Phase 1."""
    for p in model.features.parameters():
        p.requires_grad = False
    if verbose:
        print(f"[model] freeze_backbone   | trainable = {count_trainable_params(model):,}")


def unfreeze_last_blocks(
    model: nn.Module, num_blocks: int = 2, verbose: bool = True
) -> None:
    """Unfreeze N block cuoi cua backbone — Phase 2."""
    total = len(model.features)
    if num_blocks > total:
        raise ValueError(f"num_blocks={num_blocks} > total blocks ({total}).")
    for i, block in enumerate(model.features):
        for p in block.parameters():
            p.requires_grad = i >= total - num_blocks
    if verbose:
        print(
            f"[model] unfreeze_last({num_blocks}) "
            f"blocks [{total - num_blocks}..{total - 1}] "
            f"| trainable = {count_trainable_params(model):,}"
        )


def unfreeze_all(model: nn.Module, verbose: bool = True) -> None:
    """Unfreeze toan bo — Phase 3 (full fine-tune, LR rat thap)."""
    for p in model.parameters():
        p.requires_grad = True
    if verbose:
        print(f"[model] unfreeze_all      | trainable = {count_trainable_params(model):,}")


# ------------------------------------------------------------------ #
# 3. GRAD-CAM TARGET LAYER                                             #
# ------------------------------------------------------------------ #

def get_gradcam_target_layer(model: nn.Module) -> nn.Module:
    """Conv block cuoi truoc GAP — target layer cho gradcam.py."""
    return model.features[-1]


# ------------------------------------------------------------------ #
# 4. TIEN ICH DEM PARAMS                                               #
# ------------------------------------------------------------------ #

def count_trainable_params(model: nn.Module) -> int:
    """So param dang train (requires_grad=True)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_params(model: nn.Module) -> int:
    """Tong so param ke ca frozen."""
    return sum(p.numel() for p in model.parameters())


# ------------------------------------------------------------------ #
# 5. LOAD HELPER — dung khi serve (backend / inference)                #
# ------------------------------------------------------------------ #

def load_classifier(
    weights_path: str,
    num_classes: int = _DATASET_NUM_CLASSES,
    device: str = "cpu",
    verbose: bool = True,
) -> nn.Module:
    """Build model + load state_dict + .eval(). map_location='cpu' truoc roi .to(device)."""
    model = build_classifier(num_classes=num_classes, pretrained=False, verbose=False)
    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state)
    model.to(device).eval()
    if verbose:
        print(f"[model] load_classifier   | path={weights_path!r} device={device} eval=True")
    return model
