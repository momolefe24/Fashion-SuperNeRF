from config import *

class ConditionGenerator(nn.Module):
    def __init__(self, clothing_info_channel=4, parsing_dense_channel=16, resblock_channel=96, output_channel=13):
        super(ConditionGenerator, self).__init__()
        self.ClothEncoder = nn.Sequential(
            ResBlock(clothing_info_channel, resblock_channel),  # (128, 96, 96)
            ResBlock(resblock_channel, resblock_channel * 2),  # (64, 48, 198)
            ResBlock(resblock_channel * 2, resblock_channel * 4),  # (32, 24, 384)
            ResBlock(resblock_channel * 4, resblock_channel * 4),  # (16, 12, 384)
            ResBlock(resblock_channel * 4, resblock_channel * 4)  # (8, 6, 384)
        )

        self.PoseEncoder = nn.Sequential(
            ResBlock(parsing_dense_channel, resblock_channel),
            ResBlock(resblock_channel, resblock_channel * 2),
            ResBlock(resblock_channel * 2, resblock_channel * 4),
            ResBlock(resblock_channel * 4, resblock_channel * 4),
            ResBlock(resblock_channel * 4, resblock_channel * 4)
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
        return nn.ReLU(inplace=True)(skip_connection)