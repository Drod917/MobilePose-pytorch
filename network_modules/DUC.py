# -----------------------------------------------------
# Copyright (c) Shanghai Jiao Tong University. All rights reserved.
# Written by Jiefeng Li (jeff.lee.sjtu@gmail.com)
# -----------------------------------------------------

# Modifications made by Daniel Rodriguez
# Senior Design - Spring 2021

import torch.nn as nn 

# The Dense Upsampling Convolution layer rebuilds an image from
# our feature extractor's set of output feature maps.
# It performs classification by returning a heatmap specifying
# which regions are most likely to be which keypoints.
class DUC(nn.Module):

    # Output: (planes // upscale_factor^2) * hw * wd
    def __init__(self, in_channels, out_channels, upscale_factor=2):
        # Initialize the torch nn module
        super(DUC, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pixel_shuffle(x)
        return x