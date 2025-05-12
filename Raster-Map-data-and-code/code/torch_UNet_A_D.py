import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_utils import UnetConv2D, UnetGatingSignal, AttnGatingBlock

class UNet_A_D(nn.Module):
    def __init__(self, in_channels=1):
        super(UNet_A_D, self).__init__()

        # Encoder with Dropout
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

        # Center
        self.center = UnetConv2D(64, 128, is_batchnorm=True)

        # Decoder with attention
        self.gate1 = UnetGatingSignal(128, is_batchnorm=True)
        self.attn1 = AttnGatingBlock(64, 128, 64)
        self.up1_trans = nn.ConvTranspose2d(128, 32, kernel_size=3, stride=2, padding=1, output_padding=1)

        self.gate2 = UnetGatingSignal(
            32 + 64, is_batchnorm=True)
        self.attn2 = AttnGatingBlock(64, 32+64, 64)
        self.up2_trans = nn.ConvTranspose2d(32 + 64, 64, kernel_size=3, stride=2, padding=1, output_padding=1)

        self.gate3 = UnetGatingSignal(
            32 + 64, is_batchnorm=True)
        self.attn3 = AttnGatingBlock(32, 32+64, 32)
        self.up3_trans = nn.ConvTranspose2d(64 + 64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)

        self.up4_trans = nn.ConvTranspose2d(32 + 32, 32, kernel_size=3, stride=2, padding=1, output_padding=1)

        # Final output
        self.final = nn.Conv2d(32 + 32, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        x1 = self.drop1(self.conv1(x))            # -> 32
        x2 = self.drop2(self.conv2(self.pool1(x1)))  # -> 32
        x3 = self.drop3(self.conv3(self.pool2(x2)))  # -> 64
        x4 = self.drop4(self.conv4(self.pool3(x3)))  # -> 64

        # Center
        c = self.center(self.pool4(x4))           # -> 128

        # Decoder Stage 1
        g1 = self.gate1(c)
        a1 = self.attn1(x4, g1)                   # -> 64
        u1 = F.relu(self.up1_trans(c))            # -> 32
        u1 = torch.cat([u1, a1], dim=1)           # -> 96

        # Decoder Stage 2
        g2 = self.gate2(u1)
        a2 = self.attn2(x3, g2)                   # -> 64
        u2 = F.relu(self.up2_trans(u1))           # -> 64
        u2 = torch.cat([u2, a2], dim=1)           # -> 128

        # Decoder Stage 3
        g3 = self.gate3(u1)
        a3 = self.attn3(x2, g3)                   # -> 32
        u3 = F.relu(self.up3_trans(u2))           # -> 32
        u3 = torch.cat([u3, a3], dim=1)           # -> 64

        # Decoder Stage 4
        u4 = F.relu(self.up4_trans(u3))           # -> 32
        u4 = torch.cat([u4, x1], dim=1)           # -> 64

        out = self.sigmoid(self.final(u4))
        return out
