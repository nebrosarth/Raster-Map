import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_utils import UnetConv2D


class UNet_D(nn.Module):
    def __init__(self, in_channels=1):
        super(UNet_D, self).__init__()

        # Encoder
        self.conv1 = UnetConv2D(in_channels, 32, is_batchnorm=True)
        self.drop1 = nn.Dropout2d(0.2)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = UnetConv2D(32, 64, is_batchnorm=True)
        self.drop2 = nn.Dropout2d(0.2)
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = UnetConv2D(64, 128, is_batchnorm=True)
        self.drop3 = nn.Dropout2d(0.2)
        self.pool3 = nn.MaxPool2d(2)

        self.conv4 = UnetConv2D(128, 256, is_batchnorm=True)
        self.drop4 = nn.Dropout2d(0.2)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck (conv5)
        self.conv5_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        # Decoder
        self.up6_trans = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv6_1 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.conv6_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.up7_trans = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv7_1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.conv7_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.up8_trans = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv8 = nn.Conv2d(128, 64, kernel_size=3, padding=1)

        self.up9_trans = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv9_1 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv9_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)

        # Output layer
        self.final = nn.Conv2d(32, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        x1 = self.drop1(self.conv1(x))  # -> 32
        x2 = self.drop2(self.conv2(self.pool1(x1)))  # -> 64
        x3 = self.drop3(self.conv3(self.pool2(x2)))  # -> 128
        x4 = self.drop4(self.conv4(self.pool3(x3)))  # -> 256
        x5 = self.pool4(x4)  # -> 256

        # Bottleneck
        x5 = F.relu(self.conv5_1(x5))
        x5 = F.relu(self.conv5_2(x5))  # -> 512

        # Decoder
        up6 = self.up6_trans(x5)
        up6 = torch.cat([up6, x4], dim=1)  # -> 512
        x6 = F.relu(self.conv6_1(up6))
        x6 = F.relu(self.conv6_2(x6))  # -> 256

        up7 = self.up7_trans(x6)
        up7 = torch.cat([up7, x3], dim=1)  # -> 256
        x7 = F.relu(self.conv7_1(up7))
        x7 = F.relu(self.conv7_2(x7))  # -> 128

        up8 = self.up8_trans(x7)
        up8 = torch.cat([up8, x2], dim=1)  # -> 128
        x8 = F.relu(self.conv8(up8))  # -> 64

        up9 = self.up9_trans(x8)
        up9 = torch.cat([up9, x1], dim=1)  # -> 64
        x9 = F.relu(self.conv9_1(up9))
        x9 = F.relu(self.conv9_2(x9))  # -> 32

        out = self.sigmoid(self.final(x9))
        return out
