import torch.nn as nn
import torch
import torch.nn.functional as F


class UnetConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, is_batchnorm=True):
        super(UnetConv2D, self).__init__()
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
        if is_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
        if is_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))

        self.conv_block = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv_block(x)


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, dilation=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=6, dilation=6),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=12, dilation=12),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=18, dilation=18),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.final_conv = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)

        x5 = self.global_pool(x4)
        x5 = F.interpolate(x5, size=x4.shape[2:], mode='bilinear', align_corners=False)

        x_cat = torch.cat([x1, x2, x3, x4, x5], dim=1)
        out = self.final_conv(x_cat)
        return out


class UnetGatingSignal(nn.Module):
    def __init__(self, in_channels, is_batchnorm=True):
        super(UnetGatingSignal, self).__init__()
        layers = [
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        ]
        if is_batchnorm:
            layers.append(nn.BatchNorm2d(in_channels))
        layers.append(nn.ReLU(inplace=True))
        self.gate = nn.Sequential(*layers)

    def forward(self, x):
        return self.gate(x)


class AttnGatingBlock(nn.Module):
    def __init__(self, x_channels, gating_channels, inter_channels):
        super(AttnGatingBlock, self).__init__()

        self.W_x = nn.Conv2d(x_channels, inter_channels, kernel_size=2, stride=2, padding=0)
        self.W_g = nn.Conv2d(gating_channels, inter_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

        self.psi = nn.Conv2d(inter_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

        self.upsample_mode = 'bilinear'

        self.final_conv = nn.Sequential(
            nn.Conv2d(x_channels, x_channels, kernel_size=1),
            nn.BatchNorm2d(x_channels)
        )

    def forward(self, x, g):
        theta_x = self.W_x(x)  # downsample x
        phi_g = self.W_g(g)

        # upsample g to match theta_x if needed
        if phi_g.shape[2:] != theta_x.shape[2:]:
            phi_g = F.interpolate(phi_g, size=theta_x.shape[2:], mode=self.upsample_mode, align_corners=False)

        concat_xg = self.relu(theta_x + phi_g)
        psi = self.sigmoid(self.psi(concat_xg))

        # upsample attention map
        if psi.shape[2:] != x.shape[2:]:
            psi = F.interpolate(psi, size=x.shape[2:], mode=self.upsample_mode, align_corners=False)

        # expand and apply attention
        y = psi.expand_as(x) * x
        return self.final_conv(y)
