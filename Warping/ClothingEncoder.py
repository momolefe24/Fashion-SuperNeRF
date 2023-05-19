import torch

from config import *


class ConditionGenerator(nn.Module):
    def __init__(self, clothing_info_channel=4, parsing_dense_channel=16, resblock_channel=96, output_channel=13):
        super(ConditionGenerator, self).__init__()
        self.cloth_flow_list = nn.ModuleList([ResBlock(clothing_info_channel, resblock_channel),
                                              ResBlock(resblock_channel, resblock_channel * 2),
                                              ResBlock(resblock_channel * 2, resblock_channel * 4),
                                              ResBlock(resblock_channel * 4, resblock_channel * 4),
                                              ResBlock(resblock_channel * 4, resblock_channel * 4)])
        self.parse_list = nn.ModuleList([ResBlock(parsing_dense_channel, resblock_channel),
                                         ResBlock(resblock_channel, resblock_channel * 2),
                                         ResBlock(resblock_channel * 2, resblock_channel * 4),
                                         ResBlock(resblock_channel * 4, resblock_channel * 4),
                                         ResBlock(resblock_channel * 4, resblock_channel * 4)])
        self.channel_pooling_list = nn.ModuleList([
            nn.Conv2d(resblock_channel * 4, resblock_channel * 4, kernel_size=1, bias=True),
            nn.Conv2d(resblock_channel * 4, resblock_channel * 4, kernel_size=1, bias=True),
            nn.Conv2d(resblock_channel * 2, resblock_channel * 4, kernel_size=1, bias=True),
            nn.Conv2d(resblock_channel, resblock_channel * 4, kernel_size=1, bias=True),
        ])
        self.bottle_neck_list = nn.ModuleList([
            nn.Sequential(nn.Conv2d(resblock_channel * 4, resblock_channel * 4, kernel_size=3, stride=1, padding=1, bias=True),nn.ReLU()),
            nn.Sequential(nn.Conv2d(resblock_channel * 4, resblock_channel * 4, kernel_size=3, stride=1, padding=1, bias=True),nn.ReLU()),
            nn.Sequential(nn.Conv2d(resblock_channel * 2, resblock_channel * 4, kernel_size=3, stride=1, padding=1, bias=True),nn.ReLU()),
            nn.Sequential(nn.Conv2d(resblock_channel, resblock_channel * 4, kernel_size=3, stride=1, padding=1, bias=True),nn.ReLU()),
        ])
        self.seg_decoder_list = nn.ModuleList([
            ResBlock(resblock_channel * 8, resblock_channel * 4, scale='up'),  # 16
            ResBlock(resblock_channel * 4 * 2 + resblock_channel * 4, resblock_channel * 4, scale='up'),  # 32
            ResBlock(resblock_channel * 4 * 2 + resblock_channel * 4, resblock_channel * 2, scale='up'),  # 64
            ResBlock(resblock_channel * 2 * 2 + resblock_channel * 4, resblock_channel, scale='up'),  # 128
            ResBlock(resblock_channel * 1 * 2 + resblock_channel * 4, resblock_channel, scale='up')  # 256
        ])
        self.out_layer = ResBlock(resblock_channel + clothing_info_channel + parsing_dense_channel, output_channel, scale='same')
        self.channel_pool_same = ResBlock(resblock_channel * 4, resblock_channel * 8, scale='same')

        # Lambda functions
        self.upsample = lambda input: F.interpolate(input, scale_factor=2, mode='bilinear')
        self.torch_to_grid_shape = lambda tensor: tensor.permute(0, 2, 3, 1)
        self.grid_to_torch_shape = lambda tensor: tensor.permute(0, 3, 1, 2)
        self.normalize_flow = lambda flow, iW, iH: torch.cat(
            [flow[:, :, :, 0:1] / ((iW / 2 - 1.0) / 2.0), flow[:, :, :, 1:2] / ((iH / 2 - 1.0) / 2.0)], 3)
        '''
        resblock_channel * 8 = 96 * 8
        Trying to convolve the output of the concatenation between cloth encoder & parse encoder output
        Example; cloth_encoder_output (1, 384, 8,6) & parse_encoder_output (1, 384, 8,6))
        Concatenation: (1, 384 + 384, 8 , 6) 
        '''
        self.flow_conv = nn.Conv2d(resblock_channel * 8, 2, kernel_size=3, stride=1, padding=1, bias=True)
        self.channel_pooling = nn.Sequential(*self.channel_pooling_list)
        self.ClothEncoder = nn.Sequential(*self.cloth_flow_list)
        self.PoseEncoder = nn.Sequential(*self.parse_list)
        self.SegDecoder = nn.Sequential(*self.seg_decoder_list)

    def make_grid(self, N, iH, iW):
        x_linspace = torch.linspace(-1, 1, iH)
        y_linspace = torch.linspace(-1, 1, iW)
        meshx, meshy = torch.meshgrid((x_linspace, y_linspace))
        grid = torch.stack((meshy, meshx), 2)
        grid = grid.expand(N, iH, iW, 2)
        return grid

    def create_stn_grids(self, E1_list):
        grids = []
        for cloth in E1_list[::-1]:
            N, C, iH, iW = cloth.shape
            grid = self.make_grid(N, iH, iW)
            grids.append(grid)
        return grids

    def feature_pyramid(self, input, list):
        E_list = []
        for index, cloth_flow in enumerate(list):
            if index == 0:
                E_list.append(cloth_flow(input))
            else:
                E_list.append(cloth_flow(E_list[index - 1]))
        return E_list

    """
    Recall: The grid sampler shape is (N, h, W, 2)
    """

    def first_flow_list_item(self, cloth_output, parse_output):
        concatenation = torch.cat([cloth_output, parse_output], 1)
        return self.flow_conv(concatenation).permute(0, 2, 3, 1)  # (N, H, W, 2)

    def create_flow_list(self, E1_list, E2_list, grids):
        E1_list_reversed, E2_list_reversed = E1_list[::-1], E2_list[::-1]
        cloth, parse = E1_list_reversed.pop(0), E2_list_reversed.pop(0)
        flow_list = [self.first_flow_list_item(cloth, parse)]
        x = self.channel_pool_same(parse)
        x = self.seg_decoder_list.pop(0)(x)
        for index, (channel_pool, bottle_neck, seg_decoder, flow, dense, grid) in enumerate(
                zip(self.channel_pooling_list,self.bottle_neck_list,self.seg_decoder_list, E1_list_reversed, E2_list_reversed, grids[1:])):
            N, C, iH, iW = flow.shape
            cloth = self.upsample(cloth) + channel_pool(flow)
            parse = self.upsample(parse) + channel_pool(dense) # difference between dense & parse
            flow = self.torch_to_grid_shape(self.upsample(flow_list[index].permute(0, 3, 1, 2)))  # (N, H, W, 2)
            flow_norm = self.normalize_flow(flow, iW, iH)
            warped_cloth = F.grid_sample(cloth, flow_norm + grid, padding_mode='border')
            concatenate_warped_cloth_with_bottle_neck = torch.cat([warped_cloth, bottle_neck(x)],1)
            flow = flow + self.flow_conv(concatenate_warped_cloth_with_bottle_neck).permute(0,2,3,1)
            flow_list.append(flow)
            concatenate_x_with_parse_with_warped_cloth = torch.cat([x, dense, warped_cloth],1)
            x = seg_decoder(concatenate_x_with_parse_with_warped_cloth)
        return flow_list, x
    def forward(self, cloth_input, parsing_dense_input):
        E1_list = self.feature_pyramid(cloth_input, self.cloth_flow_list)
        E2_list = self.feature_pyramid(parsing_dense_input, self.parse_list)
        grids = self.create_stn_grids(E1_list)
        flow_list, x = self.create_flow_list(E1_list, E2_list, grids)
        N, _, iH, iW = cloth_input.shape
        grid = self.make_grid(N, iH, iW)
        flow = self.torch_to_grid_shape(self.upsample(flow_list[-1].permute(0, 3, 1, 2)))
        flow_norm = self.normalize_flow(flow, iW, iH)
        warped_cloth = F.grid_sample(cloth_input, flow_norm + grid, padding_mode='border')
        concatenate_x_with_parse_input_warped_cloth = torch.cat([x, parsing_dense_input, warped_cloth], 1)
        x = self.out_layer(concatenate_x_with_parse_input_warped_cloth)
        warped_cloth = warped_cloth[:,:-1,:,:]
        warped_cloth_mask = warped_cloth[:,-1,:,:]
        return flow_list, x, warped_cloth, warped_cloth_mask


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
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channel)
        )

    def forward(self, x):
        return self.feature_extractor(x)


class ResBlock(nn.Module):
    def __init__(self, input_channel, output_channel, scale='down'):
        super(ResBlock, self).__init__()
        self.feature_extractor = FeatureExtractor(output_channel)
        if scale == 'same':
            self.scale_block = nn.Conv2d(input_channel, output_channel, kernel_size=1,bias=True)
        elif scale == 'up':
            self.scale_block = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'),nn.Conv2d(input_channel, output_channel, kernel_size=1, bias=True))
        elif scale == 'down':
            self.scale_block = nn.Conv2d(input_channel, output_channel, kernel_size=3, stride=2, padding=1, bias=True)


    def forward(self, x):
        scale = self.scale_block(x)
        feature_extractor = self.feature_extractor(scale)
        skip_connection = scale + feature_extractor
        return nn.ReLU(inplace=True)(skip_connection)
