import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalTverskyLoss(nn.Module):
    def __init__(self, n_classes, alpha=0.3, beta=0.7, gamma=1.33, smooth=1e-5, dynamic_beta=True):
        super(FocalTverskyLoss, self).__init__()
        self.n_classes = n_classes
        self.gamma = gamma
        self.smooth = smooth
        self.dynamic_beta = dynamic_beta

        self.target_beta = beta
        self.target_alpha = alpha

        self.start_beta = 0.5
        self.start_alpha = 0.5

        self.rampup_epochs = 50

    def forward(self, probs, targets, current_epoch=None):
        if self.dynamic_beta and current_epoch is not None:
            if current_epoch >= self.rampup_epochs:
                progress = 1.0
            else:
                progress = current_epoch / self.rampup_epochs

            eff_beta = self.start_beta + (self.target_beta - self.start_beta) * progress
            eff_alpha = 1.0 - eff_beta
        else:
            eff_beta = self.target_beta
            eff_alpha = self.target_alpha

        p1 = probs[:, 1, :, :]
        g1 = (targets == 1).float()

        p1 = p1.contiguous().view(-1)
        g1 = g1.contiguous().view(-1)

        tp = (p1 * g1).sum()
        fp = (p1 * (1 - g1)).sum()
        fn = ((1 - p1) * g1).sum()

        tversky_index = (tp + self.smooth) / (tp + eff_alpha * fp + eff_beta * fn + self.smooth)
        focal_tversky_loss = (1 - tversky_index) ** self.gamma

        return focal_tversky_loss


class BoundaryDoULoss(nn.Module):
    def __init__(self, kernel_size=3, smooth=1e-5):
        super(BoundaryDoULoss, self).__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.smooth = smooth

    def forward(self, probs, targets):
        p1 = probs[:, 1, :, :].unsqueeze(1)
        g1 = (targets == 1).float().unsqueeze(1)

        p1_dilate = F.max_pool2d(p1, self.kernel_size, stride=1, padding=self.padding)
        p1_erode = -F.max_pool2d(-p1, self.kernel_size, stride=1, padding=self.padding)
        p1_bound = p1_dilate - p1_erode

        g1_dilate = F.max_pool2d(g1, self.kernel_size, stride=1, padding=self.padding)
        g1_erode = -F.max_pool2d(-g1, self.kernel_size, stride=1, padding=self.padding)
        g1_bound = g1_dilate - g1_erode

        p1_bound = p1_bound.contiguous().view(-1)
        g1_bound = g1_bound.contiguous().view(-1)

        intersection = (p1_bound * g1_bound).sum()
        union = (p1_bound * p1_bound).sum() + (g1_bound * g1_bound).sum()

        boundary_dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - boundary_dice


class NegativePriorPenaltyLoss(nn.Module):
    """
    显式负向先验惩罚:
    在人工确认的负向先验区域内，抑制前景概率。

    公式:
        soft_mask = clamp((neg_prior - tau) / (1 - tau), 0, 1)
        loss = sum(p_fg * soft_mask) / (sum(soft_mask) + eps)

    说明:
    - probs 必须是 softmax 后的概率图，shape = (B, C, H, W)
    - neg_prior shape 支持:
        (B, 1, H, W) 或 (B, H, W)
    - 只使用前景类概率 probs[:, 1]
    """

    def __init__(self, tau=0.20, eps=1e-6, only_when_mask_exists=True):
        super(NegativePriorPenaltyLoss, self).__init__()
        self.tau = tau
        self.eps = eps
        self.only_when_mask_exists = only_when_mask_exists

    def forward(self, probs, neg_prior):
        if probs.ndim != 4:
            raise ValueError(f"probs 应为 4D 张量 (B, C, H, W)，实际得到 {probs.shape}")

        if neg_prior.ndim == 4:
            neg_prior = neg_prior[:, 0, :, :]
        elif neg_prior.ndim != 3:
            raise ValueError(f"neg_prior 应为 3D/4D 张量，实际得到 {neg_prior.shape}")

        p_fg = probs[:, 1, :, :]  # (B, H, W)
        neg_prior = neg_prior.float()

        # 轻微软截断，保留 soft prior 的“软度”
        if self.tau > 0:
            soft_mask = (neg_prior - self.tau) / max(1.0 - self.tau, self.eps)
            soft_mask = torch.clamp(soft_mask, min=0.0, max=1.0)
        else:
            soft_mask = torch.clamp(neg_prior, min=0.0, max=1.0)

        # 每个样本独立算 penalty
        numerator = (p_fg * soft_mask).sum(dim=(1, 2))         # (B,)
        denominator = soft_mask.sum(dim=(1, 2))                # (B,)

        if self.only_when_mask_exists:
            valid = denominator > self.eps
            if valid.any():
                penalty = numerator[valid] / (denominator[valid] + self.eps)
                return penalty.mean()
            else:
                # 没有负向区域时，不产生惩罚
                return probs.new_tensor(0.0)
        else:
            penalty = numerator / (denominator + self.eps)
            return penalty.mean()


def get_neg_lambda(epoch, max_lambda, warmup_epochs):
    """
    线性 warm-up:
    epoch=0 -> 0
    epoch=warmup_epochs 后 -> max_lambda
    """
    if warmup_epochs <= 0:
        return max_lambda
    if epoch >= warmup_epochs:
        return max_lambda
    return max_lambda * (epoch / warmup_epochs)