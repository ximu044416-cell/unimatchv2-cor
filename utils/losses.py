import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalTverskyLoss(nn.Module):
    def __init__(self, n_classes, alpha=0.3, beta=0.7, gamma=1.33, smooth=1e-5, dynamic_beta=True):
        """
        Run 10 专用 Loss: 支持动态 Beta 调整
        :param alpha: 最终目标的 FP 惩罚权重 (默认 0.3)
        :param beta: 最终目标的 FN 惩罚权重 (默认 0.7)
        :param dynamic_beta: 是否开启前50轮的参数爬坡
        """
        super(FocalTverskyLoss, self).__init__()
        self.n_classes = n_classes
        self.gamma = gamma
        self.smooth = smooth
        self.dynamic_beta = dynamic_beta

        # 目标参数 (最终状态: 侧重 Recall)
        self.target_beta = beta
        self.target_alpha = alpha

        # 初始参数 (起始状态: 0.5/0.5 平衡态，防止初期乱画)
        self.start_beta = 0.5
        self.start_alpha = 0.5

        # 爬坡周期 (50个Epoch完成过渡)
        self.rampup_epochs = 50

    def forward(self, inputs, targets, current_epoch=None):
        """
        inputs: (B, C, H, W) Logits
        targets: (B, H, W) Long Tensor (0, 1)
        current_epoch: 当前训练轮数 (用于动态调整 Beta)
        """

        # ================= 1. 动态 Beta 调度逻辑 =================
        if self.dynamic_beta and current_epoch is not None:
            # 计算进度 (0.0 -> 1.0)
            if current_epoch >= self.rampup_epochs:
                progress = 1.0
            else:
                progress = current_epoch / self.rampup_epochs

            # 线性插值: Current = Start + (Target - Start) * Progress
            eff_beta = self.start_beta + (self.target_beta - self.start_beta) * progress
            eff_alpha = 1.0 - eff_beta  # 保持 alpha + beta = 1
        else:
            # 如果不传 epoch 或关闭动态，则直接使用最终设定的目标值
            eff_beta = self.target_beta
            eff_alpha = self.target_alpha
        # =====================================================

        # ================= 2. 准备数据 =================
        inputs = F.softmax(inputs, dim=1)

        # 针对二分类，我们只计算前景类 (Class 1) 的 Loss
        # 这样能避免背景类主导梯度
        p1 = inputs[:, 1, :, :]  # 预测为病灶的概率 (B, H, W)
        g1 = (targets == 1).float()  # 真实的病灶 Mask (B, H, W)

        # 🔥 [优化] 显式展平 (Flatten)，确保维度清晰且兼容性最好
        # 将 (B, H, W) 拉平成一维向量 (B*H*W)
        p1 = p1.contiguous().view(-1)
        g1 = g1.contiguous().view(-1)

        # ================= 3. 计算 Tversky 指标 =================
        tp = (p1 * g1).sum()
        fp = (p1 * (1 - g1)).sum()
        fn = ((1 - p1) * g1).sum()

        # 使用动态调整后的 eff_alpha 和 eff_beta
        tversky_index = (tp + self.smooth) / (tp + eff_alpha * fp + eff_beta * fn + self.smooth)

        # ================= 4. Focal 机制 =================
        focal_tversky_loss = (1 - tversky_index) ** self.gamma

        return focal_tversky_loss