from collections import OrderedDict

import torch
import torch.nn as nn
import numpy as np


class UNet_isolateNode(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(UNet_isolateNode, self).__init__()

        features = init_features
        self.encoder1 = self._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = self._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = self._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = self._block(features * 4, features * 8, name="enc4")

        self.bottleneck = self._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = self._block(features * 16, features * 8, name="upconv4")
        self.decoder4 = self._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = self._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = self._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = self._block(features * 2, features, name="dec1")

        self.weights = nn.Parameter(torch.zeros(200, 200, features, out_channels))
        self.bias = nn.Parameter(torch.zeros(200, 200, out_channels))
        torch.nn.init.xavier_uniform_(self.weights)

        self.norm = nn.InstanceNorm2d(out_channels)

    def forward(self, x_f, x_b):
        x = torch.cat((x_f, x_f), dim=1)
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(enc4)

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        uv_disp = self.decoder1(dec1)
        uv_disp = self.norm(uv_disp)

        uv_disp = uv_disp.permute(0, 2,3,1)
        uv_disp = torch.einsum('bhwc,hwco->bhwo', uv_disp, self.weights)
        uv_disp = uv_disp + self.bias
        uv_disp = uv_disp.permute(0, 3,1,2)

        return uv_disp[:,:3], uv_disp[:,3:]

    def _block(self, in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.InstanceNorm2d(features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.InstanceNorm2d(features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )