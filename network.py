'''
File: network.py
Project: MobilePose-PyTorch
File Created: Monday, 11th March 2019 12:50:16 am
Author: Yuliang Xiu (yuliangxiu@sjtu.edu.cn)
-----
Last Modified: Monday, 2nd Februrary 2021 1:00:00 pm
Modified By: Daniel Rodriguez (drod11375@knights.ucf.edu>)
-----
Copyright 2018 - 2019 Shanghai Jiao Tong University, Machine Vision and Intelligence Group
'''

from network_modules import DUC, MobileNetV2
import torch.nn as nn 
import dsntnn 

class CoordRegressionNetwork(nn.Module):
    def __init__(self, n_locations, backbone):
        super(CoordRegressionNetwork, self).__init__()

        self.resnet = MobileNetV2.mobilenetv2()
        self.out_channels = 32
        self.hm_conv = nn.Conv2d(
            self.out_channels,
            n_locations,
            kernel_size=1,
            bias=False
        )

    def forward(self, images):
        # Run a frame through our network
        out = self.resnet(images)
        # Use a 1x1 conv to get one unnormalized heatmap 
        # per location
        unnormalized_heatmaps = self.hm_conv(out)
        # Normalize the heatmaps
        heatmaps = dsntnn.flat_softmax(unnormalized_heatmaps)
        # Perform coordinate regression
        coords = dsntnn.dsnt(heatmaps)

        return coords, heatmaps