from configuration import *

to = lambda inputs,name,i: inputs[name][i].permute(1,2,0).cpu().numpy()
to2 = lambda inputs,name1,name2, i: inputs[name1][name2][i].permute(1,2,0).cpu().numpy()
to3 = lambda inputs,i: inputs[i].permute(1,2,0).detach().cpu().numpy()

cloth = torch.randn(1, 3, 256, 192) # 8 example images of a cltoh
cloth_masks = torch.randn(1, 1, 256, 192) # 8 example masks of a cltoh
clothing_info = torch.cat([cloth, cloth_masks], 1)

parse = torch.randn(1, 13, 256, 192) # 8 Examples of human parse segmentation with removed torso
dense = torch.randn(1, 3, 256, 192) # 8 Examples of densepose
parsing_dense = torch.cat([parse, dense], 1)


# from scratch_model import *
from networks import *
input1_nc = 4  # cloth + cloth-mask
input2_nc = 16  # parse_agnostic + densepose

# tocg = ConditionGenerator(clothing_info_channel=input1_nc, parsing_dense_channel=input2_nc, resblock_channel=96, output_channel=13)
tocg = ConditionGenerator(opt, input1_nc=4, input2_nc=input2_nc, output_nc=opt.output_nc, ngf=96, norm_layer=nn.BatchNorm2d)
flow_list, x, warped_cloth, warped_cloth_mask = tocg(opt, clothing_info, parsing_dense)
print("---------------------- Flow List Shapes----------------------------")
print([flow.shape for flow in flow_list])
print("---------------------- X Shape----------------------------")
print(x.shape)
print("---------------------- Warped Cloth Shape----------------------------")
print(warped_cloth.shape)
print("---------------------- Warped Cloth Mask Shape----------------------------")
print(warped_cloth_mask.shape)
print()
