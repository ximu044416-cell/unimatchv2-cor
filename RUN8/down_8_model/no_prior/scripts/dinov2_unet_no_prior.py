import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from configs.config_2 import config

EMBED_DIM = config.EMBED_DIM


class DINOv2Encoder(nn.Module):
    """
    DINOv2 encoder for no-prior ablation.
    Input is standard 3-channel MRI mapping only.
    """

    def __init__(self, local_path):
        super().__init__()

        try:
            self.backbone = torch.hub.load(
                "facebookresearch/dinov2",
                "dinov2_vitb14",
                pretrained=False
            )
        except Exception as e:
            print(f"❌ DINOv2 model definition loading failed: {e}")
            raise e

        print(f"📥 Loading DINOv2 weights from {local_path} ...")
        try:
            state_dict = torch.load(local_path, map_location="cpu")
            self.backbone.load_state_dict(state_dict, strict=True)
            print("✅ DINOv2 weights loaded successfully. Strict mode ON.")
        except FileNotFoundError:
            print(f"❌ Weight file not found: {local_path}")
            raise FileNotFoundError

        for param in self.backbone.parameters():
            param.requires_grad = False

        print("❄️ DINOv2 encoder frozen.")

    def forward(self, x):
        return self.backbone.get_intermediate_layers(
            x,
            n=[2, 5, 8, 11],
            return_class_token=False
        )

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self


class DINOAdapter(nn.Module):
    """
    Convert DINOv2 token features into U-Net-like pyramid features.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(EMBED_DIM, 64, 1)
        self.conv2 = nn.Conv2d(EMBED_DIM, 128, 1)
        self.conv3 = nn.Conv2d(EMBED_DIM, 256, 1)
        self.conv4 = nn.Conv2d(EMBED_DIM, 512, 1)

    def reshape_tokens(self, x):
        b, n, c = x.shape
        h = w = int(math.sqrt(n))
        return x.transpose(1, 2).reshape(b, c, h, w)

    def forward(self, features):
        f1 = self.reshape_tokens(features[0])
        f2 = self.reshape_tokens(features[1])
        f3 = self.reshape_tokens(features[2])
        f4 = self.reshape_tokens(features[3])

        c1 = self.conv1(f1)
        c1 = F.interpolate(c1, scale_factor=4, mode="bilinear", align_corners=False)

        c2 = self.conv2(f2)
        c2 = F.interpolate(c2, scale_factor=2, mode="bilinear", align_corners=False)

        c3 = self.conv3(f3)
        c4 = self.conv4(f4)

        return [c1, c2, c3, c4]


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False)
        )

    def forward(self, x):
        return self.double_conv(x)


class DINOUNetNoPrior(nn.Module):
    """
    No-prior DINOv2-UNet ablation model.

    Difference from the final model:
    - Input: only 3 MRI channels.
    - No MedSAM-derived prior channel.
    - No 4-to-3 projection layer.
    """

    def __init__(self, local_path=config.PRETRAINED_PATH, num_classes=config.NUM_CLASSES):
        super().__init__()

        self.encoder = DINOv2Encoder(local_path=local_path)
        self.adapter = DINOAdapter()

        self.up1 = nn.Identity()
        self.conv1 = DoubleConv(512 + 256, 256)

        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv2 = DoubleConv(256 + 128, 128)

        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv3 = DoubleConv(128 + 64, 64)

        self.final_up = nn.Upsample(
            size=(config.IMG_SIZE, config.IMG_SIZE),
            mode="bilinear",
            align_corners=False
        )

        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        if x.shape[1] != 3:
            raise ValueError(
                f"No-prior model expects 3 input channels, but got {x.shape[1]}."
            )

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