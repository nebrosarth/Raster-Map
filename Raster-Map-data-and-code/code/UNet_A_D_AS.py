import torch.nn as nn


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


import torch
import torch.nn as nn
import torch.nn.functional as F

class T_UNet_Head(nn.Module):
    def __init__(self, in_channels=1):
        super(T_UNet_Head, self).__init__()

        # Downsampling (encoder)
        self.conv1 = UnetConv2D(in_channels, 32, is_batchnorm=True)
        self.drop1 = nn.Dropout2d(0.2)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = UnetConv2D(32, 32, is_batchnorm=True)
        self.drop2 = nn.Dropout2d(0.2)
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = UnetConv2D(32, 64, is_batchnorm=True)
        self.drop3 = nn.Dropout2d(0.2)
        self.pool3 = nn.MaxPool2d(2)

        self.conv4 = UnetConv2D(64, 64, is_batchnorm=True)
        self.drop4 = nn.Dropout2d(0.2)
        self.pool4 = nn.MaxPool2d(2)

        # Center ASPP
        self.center = ASPP(64, 128)

        # Decoder with attention
        self.gate1 = UnetGatingSignal(128, is_batchnorm=True)
        self.attn1 = AttnGatingBlock(64, 128, 128)
        self.up1_trans = nn.ConvTranspose2d(128, 32, kernel_size=3, stride=2, padding=1, output_padding=1)

        self.gate2 = UnetGatingSignal(96, is_batchnorm=True)
        self.attn2 = AttnGatingBlock(64, 96, 64)
        self.up2_trans = nn.ConvTranspose2d(96, 64, kernel_size=3, stride=2, padding=1, output_padding=1)

        self.gate3 = UnetGatingSignal(96, is_batchnorm=True)
        self.attn3 = AttnGatingBlock(32, 96, 32)
        self.up3_trans = nn.ConvTranspose2d(128, 32, kernel_size=3, stride=2, padding=1, output_padding=1)

        self.up4_trans = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)

        # Final output layer
        self.final = nn.Conv2d(64, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        x1 = self.drop1(self.conv1(x))
        x2 = self.drop2(self.conv2(self.pool1(x1)))
        x3 = self.drop3(self.conv3(self.pool2(x2)))
        x4 = self.drop4(self.conv4(self.pool3(x3)))
        x_center = self.center(self.pool4(x4))

        # Decoder
        g1 = self.gate1(x_center)
        attn1 = self.attn1(x4, g1)
        up1 = self.up1_trans(x_center)
        up1 = F.relu(up1)
        up1 = torch.cat([up1, attn1], dim=1)

        g2 = self.gate2(up1)
        attn2 = self.attn2(x3, g2)
        up2 = self.up2_trans(up1)
        up2 = F.relu(up2)
        up2 = torch.cat([up2, attn2], dim=1)

        g3 = self.gate3(up1)
        attn3 = self.attn3(x2, g3)
        up3 = self.up3_trans(up2)
        up3 = F.relu(up3)
        up3 = torch.cat([up3, attn3], dim=1)

        up4 = self.up4_trans(up3)
        up4 = F.relu(up4)
        up4 = torch.cat([up4, x1], dim=1)

        out = self.sigmoid(self.final(up4))
        return out
