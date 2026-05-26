import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class DecoderBlock(nn.Module):
    """标准的 U-Net 解码器模块 (带跳跃连接)"""

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels // 2 + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            # 防御性插值，防止尺寸因为 maxpool 出现奇偶像素偏差
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class ResNet50_UNet(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        # =====================================================================
        # 🔥 核心保留：你的 4->3 先验通道压缩魔法 (Identity Initialization)
        # =====================================================================
        self.prior_conv = nn.Conv2d(4, 3, kernel_size=1)
        # 初始化策略：让前3个通道(RGB)直接穿透，第4个通道(先验)初始权重给极小值让它自己学
        nn.init.dirac_(self.prior_conv.weight[:3, :3])
        nn.init.zeros_(self.prior_conv.weight[:, 3])
        nn.init.zeros_(self.prior_conv.bias)

        # =====================================================================
        # 🧠 替换的心脏：PyTorch 官方预训练 ResNet-50
        # =====================================================================
        resnet = models.resnet50(pretrained=True)  # 吃尽 ImageNet 自然图像红利
        self.encoder = resnet

        # =====================================================================
        # 🧩 组装 U-Net 解码器
        # =====================================================================
        # ResNet50 的层级输出通道数分别是:
        # conv1(64), layer1(256), layer2(512), layer3(1024), layer4(2048)
        self.up1 = DecoderBlock(in_channels=2048, skip_channels=1024, out_channels=512)
        self.up2 = DecoderBlock(in_channels=512, skip_channels=512, out_channels=256)
        self.up3 = DecoderBlock(in_channels=256, skip_channels=256, out_channels=128)
        self.up4 = DecoderBlock(in_channels=128, skip_channels=64, out_channels=64)
        self.up5 = DecoderBlock(in_channels=64, skip_channels=0, out_channels=32)

        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        # 1. 巧妙吃掉 MedSAM-2 的空间先验，压成 3 通道
        if x.shape[1] == 4:
            x = self.prior_conv(x)

        # 2. ResNet 特征提取 (Encoder)
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        skip0 = x  # 提取浅层边缘特征 (64)
        x = self.encoder.maxpool(x)

        skip1 = self.encoder.layer1(x)  # (256)
        skip2 = self.encoder.layer2(skip1)  # (512)
        skip3 = self.encoder.layer3(skip2)  # (1024)
        bottleneck = self.encoder.layer4(skip3)  # 深层语义 (2048)

        # 3. 逐层上采样解码 (Decoder)
        d1 = self.up1(bottleneck, skip3)
        d2 = self.up2(d1, skip2)
        d3 = self.up3(d2, skip1)
        d4 = self.up4(d3, skip0)
        d5 = self.up5(d4)

        # 4. 输出预测 Mask
        out = self.final_conv(d5)
        return out