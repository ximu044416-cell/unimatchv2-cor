import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class DecoderBlock(nn.Module):
    """U-Net decoder block with optional skip connection."""

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_channels,
            in_channels // 2,
            kernel_size=2,
            stride=2
        )

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels // 2 + skip_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip=None):
        x = self.up(x)

        if skip is not None:
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(
                    x,
                    size=skip.shape[2:],
                    mode="bilinear",
                    align_corners=False
                )
            x = torch.cat([x, skip], dim=1)

        x = self.conv(x)
        return x


class ResNet50_UNet(nn.Module):
    """
    ResNet-50 U-Net baseline.

    Purpose:
    - Used as backbone ablation against DINOv2.
    - Keeps the same 4-channel interface as the final model:
      3 MRI channels + 1 MedSAM-derived prior channel.
    - Uses a 1x1 projection from 4 channels to 3 channels so that
      ImageNet-pretrained ResNet-50 can be used.
    """

    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()

        # 4 -> 3 channel projection.
        # First three channels are initialized as identity-like mapping.
        # Prior channel is initialized as zero contribution.
        self.prior_conv = nn.Conv2d(4, 3, kernel_size=1, bias=True)

        with torch.no_grad():
            self.prior_conv.weight.zero_()
            self.prior_conv.bias.zero_()
            for i in range(3):
                self.prior_conv.weight[i, i, 0, 0] = 1.0
            self.prior_conv.weight[:, 3, 0, 0] = 0.0

        # Torchvision compatibility.
        if pretrained:
            try:
                from torchvision.models import ResNet50_Weights
                resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            except Exception:
                resnet = models.resnet50(pretrained=True)
        else:
            try:
                resnet = models.resnet50(weights=None)
            except Exception:
                resnet = models.resnet50(pretrained=False)

        self.encoder = resnet

        # ResNet50 feature channels:
        # conv1: 64
        # layer1: 256
        # layer2: 512
        # layer3: 1024
        # layer4: 2048
        self.up1 = DecoderBlock(in_channels=2048, skip_channels=1024, out_channels=512)
        self.up2 = DecoderBlock(in_channels=512, skip_channels=512, out_channels=256)
        self.up3 = DecoderBlock(in_channels=256, skip_channels=256, out_channels=128)
        self.up4 = DecoderBlock(in_channels=128, skip_channels=64, out_channels=64)
        self.up5 = DecoderBlock(in_channels=64, skip_channels=0, out_channels=32)

        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        if x.shape[1] == 4:
            x = self.prior_conv(x)
        elif x.shape[1] != 3:
            raise ValueError(f"Expected 3 or 4 input channels, got {x.shape[1]}")

        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)

        skip0 = x
        x = self.encoder.maxpool(x)

        skip1 = self.encoder.layer1(x)
        skip2 = self.encoder.layer2(skip1)
        skip3 = self.encoder.layer3(skip2)
        bottleneck = self.encoder.layer4(skip3)

        d1 = self.up1(bottleneck, skip3)
        d2 = self.up2(d1, skip2)
        d3 = self.up3(d2, skip1)
        d4 = self.up4(d3, skip0)
        d5 = self.up5(d4)

        out = self.final_conv(d5)
        return out