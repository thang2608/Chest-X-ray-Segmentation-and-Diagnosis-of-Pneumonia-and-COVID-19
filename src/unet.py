# U-Net segment vung phoi (binary lung mask). shortcut_iou.py va
# train_unet.ipynb import build_unet + loss/metric tu day.
# API contract (KHONG doi sau khi commit):
#   build_unet(in_channels=3, out_channels=1, pretrained=True) -> nn.Module
#   Output: logits shape (N, out_channels, H, W) — CHUA sigmoid (dung BCEWithLogitsLoss).
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


# ------------------------------------------------------------------ #
# 1. BUILD U-NET                                                       #
# ------------------------------------------------------------------ #

def build_unet(
    in_channels: int = 3,
    out_channels: int = 1,
    pretrained: bool = True,
    encoder_name: str = "resnet34",
    verbose: bool = True,
) -> nn.Module:
    """U-Net encoder ResNet-34 (ImageNet). Output logits (N, out_channels, H, W), CHUA sigmoid."""
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights="imagenet" if pretrained else None,
        in_channels=in_channels,
        classes=out_channels,
    )
    if verbose:
        tag = "encoder=ImageNet" if pretrained else "encoder=scratch"
        print(f"[unet] U-Net | {encoder_name} | {tag}")
        print(f"       in_channels : {in_channels}")
        print(f"       out_channels: {out_channels}")
        print(f"       total params: {sum(p.numel() for p in model.parameters()):,}")
    return model


# ------------------------------------------------------------------ #
# 2. LOSS — combo BCE + Dice (robust voi mat can bang foreground)      #
# ------------------------------------------------------------------ #

class BCEDiceLoss(nn.Module):
    """0.5 * BCEWithLogits + 0.5 * (1 - Dice). Input logits (CHUA sigmoid)."""

    def __init__(self, bce_weight: float = 0.5, smooth: float = 1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, target)
        probs = torch.sigmoid(logits)
        p = probs.contiguous().view(probs.size(0), -1)
        t = target.contiguous().view(target.size(0), -1)
        inter = (p * t).sum(1)
        dice = (2 * inter + self.smooth) / (p.sum(1) + t.sum(1) + self.smooth)
        dice_loss = 1 - dice.mean()
        return self.bce_weight * bce_loss + (1 - self.bce_weight) * dice_loss


# ------------------------------------------------------------------ #
# 3. METRIC — Dice & IoU (report ca 2 trong bang metric)              #
# ------------------------------------------------------------------ #

@torch.no_grad()
def dice_score(logits: torch.Tensor, target: torch.Tensor, thresh: float = 0.5) -> float:
    """Dice = 2|A∩B| / (|A|+|B|). smooth=1 tranh chia 0 khi mask rong."""
    pred = (torch.sigmoid(logits) > thresh).float()
    inter = (pred * target).sum()
    return float((2 * inter + 1) / (pred.sum() + target.sum() + 1))


@torch.no_grad()
def iou_score(logits: torch.Tensor, target: torch.Tensor, thresh: float = 0.5) -> float:
    """IoU (Jaccard) = |A∩B| / |A∪B|. smooth=1 tranh chia 0."""
    pred = (torch.sigmoid(logits) > thresh).float()
    inter = (pred * target).sum()
    union = pred.sum() + target.sum() - inter
    return float((inter + 1) / (union + 1))
