from configuration import *
from scratch_model import *

to = lambda inputs,name,i: inputs[name][i].permute(1,2,0).cpu().numpy()
to2 = lambda inputs,name1,name2, i: inputs[name1][name2][i].permute(1,2,0).cpu().numpy()
to3 = lambda inputs,i: inputs[i].permute(1,2,0).detach().cpu().numpy()

# Model
input1_nc = 4  # cloth + cloth-mask
input2_nc = 16  # parse_agnostic + densepose
tocg = ConditionGenerator(opt, input1_nc=4, input2_nc=input2_nc, output_nc=opt.output_nc, ngf=96, norm_layer=nn.BatchNorm2d)
load_checkpoint(tocg, opt.tocg_checkpoint, opt)
D = define_D(input_nc=input1_nc + input2_nc + opt.output_nc, Ddownx2 = opt.Ddownx2, Ddropout = opt.Ddropout, n_layers_D=3, spectral = opt.spectral, num_D = opt.num_D)

# Optimizers
optimizer_G = torch.optim.Adam(tocg.parameters(), lr=opt.G_lr, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(D.parameters(), lr=opt.D_lr, betas=(0.5, 0.999))

x = torch.randn(1, 3, 256, 192)
cloth = torch.randn(8, 3, 256, 192) # 8 example images of a cltoh
cloth_masks = torch.randn(8, 1, 256, 192) # 8 example masks of a cltoh
clothing_info = torch.cat([cloth, cloth_masks], 1)
cloth_x = torch.randn(1, 4, 256, 192)

parse = torch.randn(8, 13, 256, 192) # 8 Examples of human parse segmentation with removed torso
dense = torch.randn(8, 3, 256, 192) # 8 Examples of densepose
parsing_dense = torch.cat([parse, dense], 1)
y = torch.randn(1, 16, 256, 192)

'''
Model: Train Condition
Aim: Take the clothing and dowsample it 
Page: 6 of 27
'''
# Testing the downsampling in the cloth encoder
down = Downsampling(4, 96)
out = FeatureExtractor(96, 96)


print("Hello")