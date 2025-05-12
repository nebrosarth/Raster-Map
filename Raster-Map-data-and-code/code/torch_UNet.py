import torch
import torch.nn as nn
from torch_utils import UnetConv2D


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_filters=32):
        super(UNet, self).__init__()
        # Encoder
        self.conv1 = UnetConv2D(in_channels, base_filters, is_batchnorm=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = UnetConv2D(base_filters, base_filters * 2, is_batchnorm=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = UnetConv2D(base_filters * 2, base_filters * 4, is_batchnorm=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = UnetConv2D(base_filters * 4, base_filters * 8, is_batchnorm=True)
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        # Bottleneck
        self.conv5_1 = nn.Conv2d(base_filters * 8, base_filters * 16, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(base_filters * 16, base_filters * 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        # Decoder
        self.up6 = nn.ConvTranspose2d(base_filters * 16, base_filters * 8, kernel_size=2, stride=2)
        self.conv6_1 = nn.Conv2d(base_filters * 16, base_filters * 8, kernel_size=3, padding=1)
        self.conv6_2 = nn.Conv2d(base_filters * 8, base_filters * 8, kernel_size=3, padding=1)

        self.up7 = nn.ConvTranspose2d(base_filters * 8, base_filters * 4, kernel_size=2, stride=2)
        self.conv7_1 = nn.Conv2d(base_filters * 8, base_filters * 4, kernel_size=3, padding=1)
        self.conv7_2 = nn.Conv2d(base_filters * 4, base_filters * 4, kernel_size=3, padding=1)

        self.up8 = nn.ConvTranspose2d(base_filters * 4, base_filters * 2, kernel_size=2, stride=2)
        self.conv8_1 = nn.Conv2d(base_filters * 4, base_filters * 2, kernel_size=3, padding=1)
        self.conv8_2 = nn.Conv2d(base_filters * 2, base_filters * 2, kernel_size=3, padding=1)

        self.up9 = nn.ConvTranspose2d(base_filters * 2, base_filters, kernel_size=2, stride=2)
        self.conv9_1 = nn.Conv2d(base_filters * 2, base_filters, kernel_size=3, padding=1)
        self.conv9_2 = nn.Conv2d(base_filters, base_filters, kernel_size=3, padding=1)

        # Final conv
        self.final = nn.Conv2d(base_filters, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        c1 = self.conv1(x)
        p1 = self.pool1(c1)

        c2 = self.conv2(p1)
        p2 = self.pool2(c2)

        c3 = self.conv3(p2)
        p3 = self.pool3(c3)

        c4 = self.conv4(p3)
        p4 = self.pool4(c4)

        # Bottleneck
        b = self.relu(self.conv5_1(p4))
        b = self.relu(self.conv5_2(b))

        # Decoder
        u6 = self.up6(b)
        u6 = torch.cat([u6, c4], dim=1)
        c6 = self.relu(self.conv6_1(u6))
        c6 = self.relu(self.conv6_2(c6))

        u7 = self.up7(c6)
        u7 = torch.cat([u7, c3], dim=1)
        c7 = self.relu(self.conv7_1(u7))
        c7 = self.relu(self.conv7_2(c7))

        u8 = self.up8(c7)
        u8 = torch.cat([u8, c2], dim=1)
        c8 = self.relu(self.conv8_1(u8))
        c8 = self.relu(self.conv8_2(c8))

        u9 = self.up9(c8)
        u9 = torch.cat([u9, c1], dim=1)
        c9 = self.relu(self.conv9_1(u9))
        c9 = self.relu(self.conv9_2(c9))

        output = self.sigmoid(self.final(c9))
        return output
