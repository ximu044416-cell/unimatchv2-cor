import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 🔥 [修改 1] 导入 config，而不是写死 IMG_SIZE
from configs import config

# DINOv2-Small 的特征维度是 384 (Base 是 768, Large 是 1024)
# 如果你确定只用 Small，这里写 384 没问题
EMBED_DIM = 384


class DINOv2Encoder(nn.Module):
    """
    DINOv2 骨干网络 (本地加载 + 严格冻结)
    """

    def __init__(self, local_path):
        super().__init__()

        # 1. 加载定义
        try:
            # 这一步只加载结构，不加载权重
            self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', pretrained=False)
        except Exception as e:
            print(f"❌ 模型定义加载失败: {e}")
            raise e

        # 2. 本地加载权重
        print(f"📥 Loading weights from {local_path}...")
        try:
            state_dict = torch.load(local_path, map_location='cpu')
            self.backbone.load_state_dict(state_dict)
            print("✅ Weights loaded successfully.")
        except FileNotFoundError:
            print(f"❌ 错误: 找不到文件 {local_path}，请检查路径！")
            raise FileNotFoundError

        # 3. 严格冻结
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("❄️ Encoder Frozen (Strict Mode).")

    def forward(self, x):
        # 提取 [2, 5, 8, 11] 层
        return self.backbone.get_intermediate_layers(
            x, n=[2, 5, 8, 11], return_class_token=False
        )

    def train(self, mode=True):
        """强制 Encoder 保持 Eval 模式 (关闭 Dropout/BN 更新)"""
        super().train(mode)
        self.backbone.eval()
        return self


class DINOAdapter(nn.Module):
    """
    适配器: 将柱状特征重塑为金字塔特征
    """

    def __init__(self):
        super().__init__()
        # 1x1 卷积调整通道数
        self.conv1 = nn.Conv2d(EMBED_DIM, 64, 1)
        self.conv2 = nn.Conv2d(EMBED_DIM, 128, 1)
        self.conv3 = nn.Conv2d(EMBED_DIM, 256, 1)
        self.conv4 = nn.Conv2d(EMBED_DIM, 512, 1)

    def reshape_tokens(self, x):
        B, N, C = x.shape
        H = W = int(math.sqrt(N))
        return x.transpose(1, 2).reshape(B, C, H, W)

    def forward(self, features):
        f1 = self.reshape_tokens(features[0])
        f2 = self.reshape_tokens(features[1])
        f3 = self.reshape_tokens(features[2])
        f4 = self.reshape_tokens(features[3])

        # Path 1: 37 -> 148 (Up x4)
        c1 = self.conv1(f1)
        c1 = F.interpolate(c1, scale_factor=4, mode='bilinear', align_corners=False)

        # Path 2: 37 -> 74 (Up x2)
        c2 = self.conv2(f2)
        c2 = F.interpolate(c2, scale_factor=2, mode='bilinear', align_corners=False)

        # Path 3: 37 -> 37 (Identity)
        c3 = self.conv3(f3)

        # Path 4: 37 -> 37 (Identity)
        c4 = self.conv4(f4)

        return [c1, c2, c3, c4]


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            # 🔥 [修改 2] 为了训练稳定性，建议去掉 inplace=True
            nn.ReLU(inplace=False),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False)
        )

    def forward(self, x):
        return self.double_conv(x)


class DINOUNet(nn.Module):
    # 🔥 [修改 3] init 参数使用 config 里的默认值
    def __init__(self, local_path=config.PRETRAINED_PATH, num_classes=config.NUM_CLASSES):
        super().__init__()

        self.encoder = DINOv2Encoder(local_path=local_path)
        self.adapter = DINOAdapter()

        # Decoder Blocks
        self.up1 = nn.Identity()
        self.conv1 = DoubleConv(512 + 256, 256)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv2 = DoubleConv(256 + 128, 128)

        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv3 = DoubleConv(128 + 64, 64)

        # 🔥 [修改 4] 动态读取 Config 中的 IMG_SIZE
        # 这样如果你改了 config.py，这里会自动同步，不会报错
        self.final_up = nn.Upsample(size=(config.IMG_SIZE, config.IMG_SIZE), mode='bilinear', align_corners=False)

        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        features = self.encoder(x)
        c1, c2, c3, c4 = self.adapter(features)

        x = self.up1(c4)
        x = torch.cat([x, c3], dim=1)
        x = self.conv1(x)

        x = self.up2(x)
        x = torch.cat([x, c2], dim=1)
        x = self.conv2(x)

        x = self.up3(x)
        x = torch.cat([x, c1], dim=1)
        x = self.conv3(x)

        x = self.final_up(x)
        logits = self.final_conv(x)

        return logits