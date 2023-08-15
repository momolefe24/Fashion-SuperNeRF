from utils import get_opt, get_transforms_data
import torch
import torch.nn as nn
from dataset import FashionNeRFDatasetTest
import matplotlib.pyplot as plt
import os
""" ================== Data structure imports =================="""
from collections import OrderedDict

""" ================== NeRF imports =================="""
from NeRF.test_helper import render, create_nerf
""" ================== VITON imports =================="""
from VITON.cp_dataset import CPDataset, CPDatasetTest, CPDataLoader
from VITON.networks import ConditionGenerator, load_checkpoint, make_grid
from VITON.network_generator import SPADEGenerator
from VITON.test_generator import test

# ./cihp_pgn.sh data/rail/temp julian gray_long_sleeve julian_gray_long_sleeve_27.jpg
# ./densepose.sh data/rail/temp julian gray_long_sleeve julian_gray_long_sleeve_27.jpg
# ./openpose.sh data/rail/temp julian gray_long_sleeve julian_gray_long_sleeve_27.jpg
# ./parse_agnostic.sh data/rail/temp julian gray_long_sleeve julian_gray_long_sleeve_27.jpg



def load_checkpoint_G(model, checkpoint_path,opt):
    if not os.path.exists(checkpoint_path):
        print("Invalid path!")
        return
    state_dict = torch.load(checkpoint_path)
    new_state_dict = OrderedDict([(k.replace('ace', 'alias').replace('.Spade', ''), v) for (k, v) in state_dict.items()])
    new_state_dict._metadata = OrderedDict([(k.replace('ace', 'alias').replace('.Spade', ''), v) for (k, v) in state_dict._metadata.items()])
    model.load_state_dict(new_state_dict, strict=True)
    if opt.cuda :
        model.cuda()

opt = get_opt()
root_dir = "data/rail/temp"
path = f"{root_dir}/{opt.person}_{opt.clothing}/image"

# input1_nc = 4
# input2_nc = opt.semantic_nc + 3  # parse_agnostic + densepose
# tocg = ConditionGenerator(opt, input1_nc=input1_nc, input2_nc=input2_nc, output_nc=opt.output_nc, ngf=96, norm_layer=nn.BatchNorm2d)
# opt.semantic_nc = 7
# generator = SPADEGenerator(opt, 3 + 3 + 3)
# generator.print_network()
#
# # Load Checkpoint - VITON
# load_checkpoint(tocg, opt.tocg_checkpoint, opt)
# load_checkpoint_G(generator, opt.viton_checkpoint, opt)


# Infer NeRF- create checkpoints, folders for each individual


# Test
near, far = 2., 6.
render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(opt)
bds_dict = {
        'near' : near,
        'far' : far,
    }
render_kwargs_test.update(bds_dict)
save_image = lambda title, torch_img: plt.imsave(f"{title}.png", torch_img.cpu().numpy())
#
def probe(pose, H, W, K, num=0):
	c2w = pose[:3, :4]
	with torch.no_grad():
		rgb, disp, acc, _ = render(H, W, K, chunk=opt.nerf_chunk, c2w=c2w, **render_kwargs_test)
	return rgb
test_dataset = FashionNeRFDatasetTest(opt)
julian = test_dataset.__getitem__(0)
probe(julian['transform_matrix'], int(julian['H']), int(julian['W']), julian['K'])
# test(opt, test_loader, tocg, generator)