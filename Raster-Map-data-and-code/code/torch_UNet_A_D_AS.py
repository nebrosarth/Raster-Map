import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_utils import UnetConv2D, UnetGatingSignal, AttnGatingBlock, ASPP


class UNet_A_D_ASPP(nn.Module):
    def __init__(self, in_channels=1):
        super(UNet_A_D_ASPP, self).__init__()

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
