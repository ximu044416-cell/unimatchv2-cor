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

    def forward(self, probs, targets, current_epoch=None):
        """
        🔥 [工程优化] 外部统一传入 Softmax 后的 probs，避免重复计算
        probs: (B, C, H, W) 经过 Softmax 的概率图
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
        # 针对二分类，我们只计算前景类 (Class 1) 的 Loss
        p1 = probs[:, 1, :, :]  # 预测为病灶的概率 (B, H, W)
        g1 = (targets == 1).float()  # 真实的病灶 Mask (B, H, W)

        # 显式展平 (Flatten)，确保维度清晰且兼容性最好
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


class BoundaryDoULoss(nn.Module):
    def __init__(self, kernel_size=3, smooth=1e-5):
        """
        利用形态学池化动态提取预测和标签的软边界，专治“狗牙状”边缘。
        🔥 提示: 如果水肿边界极其模糊，导致 Loss 挂零，可在此处将 kernel_size 改为 5
        :param kernel_size: 池化核大小 (3 代表 1 像素宽，5 代表 2 像素宽)
        :param smooth: 防梯度爆炸的平滑项
        """
        super(BoundaryDoULoss, self).__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.smooth = smooth

    def forward(self, probs, targets):
        """
        🔥 [工程优化] 外部统一传入 Softmax 后的 probs
        probs: (B, C, H, W) 经过 Softmax 的概率图
        targets: (B, H, W) 真实的病灶 Mask
        """
        # 1. 提取病灶类的概率图和标签图，并扩展通道维度以适应池化操作
        p1 = probs[:, 1, :, :].unsqueeze(1)  # (B, 1, H, W)
        g1 = (targets == 1).float().unsqueeze(1)  # (B, 1, H, W)

        # 2. 形态学提取预测的软边界 (膨胀 - 腐蚀)
        p1_dilate = F.max_pool2d(p1, self.kernel_size, stride=1, padding=self.padding)
        p1_erode = -F.max_pool2d(-p1, self.kernel_size, stride=1, padding=self.padding)
        p1_bound = p1_dilate - p1_erode

        # 3. 形态学提取金标准的边界
        g1_dilate = F.max_pool2d(g1, self.kernel_size, stride=1, padding=self.padding)
        g1_erode = -F.max_pool2d(-g1, self.kernel_size, stride=1, padding=self.padding)
        g1_bound = g1_dilate - g1_erode

        # 4. 展平进行 B-DoU 计算
        p1_bound = p1_bound.contiguous().view(-1)
        g1_bound = g1_bound.contiguous().view(-1)

        # 计算边界重合度 (Intersection)
        intersection = (p1_bound * g1_bound).sum()

        # 🔥 [数值安全优化] 采用平方和形式计算 Union，防止 AMP 半精度下溢崩溃
        union = (p1_bound * p1_bound).sum() + (g1_bound * g1_bound).sum()

        # 边界的 Dice 计算公式
        boundary_dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        # Loss 期望边界重叠越高越好，所以用 1 - Dice
        return 1.0 - boundary_dice