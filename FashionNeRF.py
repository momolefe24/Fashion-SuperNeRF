from utils import get_opt, get_transforms_data
import torch
import torch.nn as nn
from dataset import FashionNeRFDatasetTest
import os
""" ================== Data structure imports =================="""
from collections import OrderedDict

""" ================== NeRF imports =================="""

""" ================== VITON imports =================="""
from VITON.cp_dataset import CPDataset, CPDatasetTest, CPDataLoader
from VITON.networks import ConditionGenerator, load_checkpoint, make_grid
from VITON.network_generator import SPADEGenerator
from VITON.test_generator import test



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
input1_nc = 4
input2_nc = opt.semantic_nc + 3  # parse_agnostic + densepose

tocg = ConditionGenerator(opt, input1_nc=input1_nc, input2_nc=input2_nc, output_nc=opt.output_nc, ngf=96, norm_layer=nn.BatchNorm2d)
opt.semantic_nc = 7

# test_dataset = FashionNeRFDatasetTest(opt)
test_dataset = CPDataset(opt)
test_dataset.__getitem__(0)
test_loader = CPDataLoader(opt, test_dataset)

opt.semantic_nc = 7
generator = SPADEGenerator(opt, 3 + 3 + 3)
generator.print_network()

# Load Checkpoint - VITON
load_checkpoint(tocg, opt.tocg_checkpoint, opt)
load_checkpoint_G(generator, opt.viton_checkpoint, opt)

# Train NeRF- create checkpoins, folders for each individual



# Test
test(opt, test_loader, tocg, generator)