import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import models
import os
from torch.nn.utils import spectral_norm
import numpy as np

import functools


class ConditionGenerator(nn.Module):
    def __init__(self, clothing_info_channel=4, parsing_dense_channel=16, output_channel=96):
        self.ClothEncoder = nn.Sequential(
            ResBlock(clothing_info_channel, output_channel, norm_layer=output_channel, scale='down'),  # 128
            ResBlock(output_channel, output_channel * 2),  # 64
            ResBlock(output_channel * 2, output_channel * 4),  # 32
            ResBlock(output_channel * 4, output_channel * 4),  # 16
            ResBlock(output_channel * 4, output_channel * 4)  # 8
        )

        self.PoseEncoder = nn.Sequential(
            ResBlock(parsing_dense_channel, output_channel),
            ResBlock(output_channel, output_channel * 2),
            ResBlock(output_channel * 2, output_channel * 4),
            ResBlock(output_channel * 4, output_channel * 4),
            ResBlock(output_channel * 4, output_channel * 4)
        )


class Downsampling(nn.Module):
    def __init__(self, input_channel, output_channel, use_bias=False):
        super(Downsampling, self).__init__()
        self.downsampling = nn.Conv2d(input_channel, output_channel, kernel_size=3, stride=2, padding=1, bias=use_bias)

    def forward(self, x):
        return self.downsampling(x)

class FeatureExtractor(nn.Module):
    def __init__(self, channel):
        super(FeatureExtractor, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(channel,channel,kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channel)
        )

    def forward(self, x):
        return self.feature_extractor(x)
class ResBlock(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(ResBlock, self).__init__()
        self.downsampling_block = Downsampling(input_channel, output_channel)
        self.feature_extractor = FeatureExtractor(output_channel)

    def forward(self, x):
        downsampling = self.downsampling_block(x)
        feature_extractor = self.feature_extractor(downsampling)
        skip_connection = downsampling + feature_extractor
        return nn.ReLU(skip_connection)

