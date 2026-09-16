"""
Compatible 3D ResNet feature encoder for the released Nishizawa et al.
bc_pcr_prediction code.

IMPORTANT
---------
The authors' public repository imports:
    from resnet import resnet18, resnet34, resnet50, resnet101
but does not include resnet.py.

This file restores a standard 3D ResNet feature extractor compatible with
model.py:
    resnet18/resnet34 -> [B, 512]
    resnet50/resnet101 -> [B, 2048]

It is a compatibility reconstruction, NOT the authors' verified original file.
"""

import torch
import torch.nn as nn

__all__ = [
    "ResNet3D",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
]


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        bias=False,
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv1x1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm3d(planes)

        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)

        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNet3D(nn.Module):
    def __init__(self, block, layers, in_channels=1):
        super().__init__()
        self.inplanes = 64

        self.conv1 = nn.Conv3d(
            in_channels,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

        self._initialize_weights()

    def _make_layer(self, block, planes, blocks, stride=1):
        outplanes = planes * block.expansion
        downsample = None

        if stride != 1 or self.inplanes != outplanes:
            downsample = nn.Sequential(
                conv1x1x1(self.inplanes, outplanes, stride),
                nn.BatchNorm3d(outplanes),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = outplanes

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Expected MRI tensor: [B, C, D, H, W].
        # If the released loader supplies [B, D, H, W], add C=1.
        if x.ndim == 4:
            x = x.unsqueeze(1)

        if x.ndim != 5:
            raise ValueError(
                f"3D ResNet expects [B,C,D,H,W] (or [B,D,H,W]); got {tuple(x.shape)}"
            )

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


def resnet18(**kwargs):
    return ResNet3D(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet34(**kwargs):
    return ResNet3D(BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet50(**kwargs):
    return ResNet3D(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet101(**kwargs):
    return ResNet3D(Bottleneck, [3, 4, 23, 3], **kwargs)


if __name__ == "__main__":
    # Small smoke test; intentionally small volume to avoid excessive memory.
    with torch.no_grad():
        x = torch.randn(2, 1, 32, 64, 64)
        for name, factory, expected in [
            ("resnet18", resnet18, 512),
            ("resnet34", resnet34, 512),
            ("resnet50", resnet50, 2048),
            ("resnet101", resnet101, 2048),
        ]:
            model = factory().eval()
            y = model(x)
            print(name, tuple(y.shape))
            assert y.shape == (2, expected)
