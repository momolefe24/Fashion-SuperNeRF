from Warping.ClothingEncoder import ConditionGenerator, Downsampling, ResBlock, FeatureExtractor
from config import *

# Create model
tocg = ConditionGenerator(clothing_info_channel=cloth_input_channels, parsing_dense_channel=parse_input_channels, resblock_channel=resblock_channel, output_channel=output_channel)


cloth = torch.randn(1, 3, 256, 192) # 8 example images of a cltoh
cloth_masks = torch.randn(1, 1, 256, 192) # 8 example masks of a cltoh
clothing_info = torch.cat([cloth, cloth_masks], 1)
print("---------------------- CLOTHING SHAPE----------------------------")
print(clothing_info.shape)


print("---------------------- TESTING THE DOWNSAMPLING----------------------------")
down = Downsampling(4, 96)(clothing_info)
print(f'Downsampling cloth has shape {down.shape}')

print("---------------------- TESTING THE FEATURE EXTRACTOR----------------------------")
feature_extractor = FeatureExtractor(96)(down)
print(f'Feature extractor cloth has shape {feature_extractor.shape}')

print("---------------------- TESTING THE RESBLOCK----------------------------")
resblock = ResBlock(4, 96)(clothing_info)
print(f'Resblock cloth has shape {resblock.shape}')

print("---------------------- TESTING THE CLOTHING ENCODER----------------------------")
encoding = tocg.ClothEncoder(clothing_info)
print(f'Clothing Encoder cloth has shape {encoding.shape}')
print()
""" Document the output channel of the following"""