import torch
import torch.nn as nn

def dice_loss(pred, target, smooth=1):
    # pred is raw logits [B, 2, H, W], target is [B, H, W] long
    pred = torch.softmax(pred, dim=1)  # convert to probabilities
    pred_fg = pred[:, 1, :, :]  # grab foreground channel probability

    target_float = target.float()

    intersection = (pred_fg * target_float).sum()
    return 1 - (2.0 * intersection + smooth) / (
        pred_fg.sum() + target_float.sum() + smooth
    )


# Combined loss
def combined_loss(pred, target, device, dice_weight=0.5):
    ce = nn.CrossEntropyLoss().to(device)(pred, target)
    dice = dice_loss(pred, target)
    return ce * (1 - dice_weight) + dice * dice_weight