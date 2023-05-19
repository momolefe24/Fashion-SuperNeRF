from Warping.ClothingEncoder import ConditionGenerator, Downsampling, ResBlock, FeatureExtractor
from config import *

# Create model
tocg = ConditionGenerator(clothing_info_channel=cloth_input_channels, parsing_dense_channel=parse_input_channels, resblock_channel=resblock_channel, output_channel=output_channel)


cloth = torch.randn(1, 3, 256, 192) # 8 example images of a cltoh
cloth_masks = torch.randn(1, 1, 256, 192) # 8 example masks of a cltoh
clothing_info = torch.cat([cloth, cloth_masks], 1)
print("---------------------- CLOTHING SHAPE----------------------------")
print(clothing_info.shape)

parse = torch.randn(1, 13, 256, 192) # 8 Examples of human parse segmentation with removed torso
dense = torch.randn(1, 3, 256, 192) # 8 Examples of densepose
parsing_dense = torch.cat([parse, dense], 1)
print("---------------------- Person Agnostic Representatio & Pose SHAPE----------------------------")
print(parsing_dense.shape)

tocg(clothing_info, parsing_dense)