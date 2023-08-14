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
test_dataset = FashionNeRFDatasetTest(opt)
# test_dataset.__getitem__(0)

from PIL import Image
import sys
root_dir = "data/rail/temp"
path = f"{root_dir}/{test_dataset.person}_{test_dataset.clothing}/image"
cihp = "./cihp_pgn.sh"
detectron = "./densepose.sh"
openpose = "./openpose.sh"
parse_agnostic = "./parse_agnostic.sh"
# ./cihp_pgn.sh data/rail/temp julian gray_long_sleeve julian_gray_long_sleeve_27.jpg
# ./densepose.sh data/rail/temp julian gray_long_sleeve julian_gray_long_sleeve_27.jpg
# ./openpose.sh data/rail/temp julian gray_long_sleeve julian_gray_long_sleeve_27.jpg
# ./parse_agnostic.sh data/rail/temp julian gray_long_sleeve julian_gray_long_sleeve_27.jpg
for image_name in test_dataset.data_items_:
    if not os.path.exists(path):
        os.makedirs(path)
    image = Image.open(image_name)
    filename = image_name.split("/")[-1]
    save_dir = os.path.join(path, filename)
    if not os.path.isfile(save_dir):
        image.save(save_dir)
    cihp += f" {root_dir} {test_dataset.person} {test_dataset.clothing} {filename}"
    cihp_err = os.system(cihp)

    if cihp_err:
        print("FATAL: CIHP command failed")
        sys.exit(cihp_err)

    detectron += f" {root_dir} {test_dataset.person} {test_dataset.clothing} {filename}"
    densepose_err = os.system(detectron)
    if densepose_err:
        print("FATAL: Densepose command failed")
        sys.exit(densepose_err)

    openpose += f" {root_dir} {test_dataset.person} {test_dataset.clothing} {filename}"
    openpose_err = os.system(openpose)
    if openpose_err:
        print("FATAL: Openpose command failed")
        sys.exit(openpose_err)

    parse_agnostic += f" {root_dir} {test_dataset.person} {test_dataset.clothing} {filename}"
    parse_agnostic_err = os.system(parse_agnostic)
    if parse_agnostic_err:
        print("FATAL: Parse Agnostic command failed")
        sys.exit(parse_agnostic_err)

    #python3 apply_net.py show configs/densepose_rcnn_R_50_FPN_s1x.yaml densepose_rcnn_R_50_FPN_s1x.pkl "$image_file" dp_segm -v --output output/"$1"/"$image_name"


# test_loader = CPDataLoader(opt, test_dataset)
#
# input1_nc = 4
# input2_nc = opt.semantic_nc + 3  # parse_agnostic + densepose
#
# tocg = ConditionGenerator(opt, input1_nc=input1_nc, input2_nc=input2_nc, output_nc=opt.output_nc, ngf=96, norm_layer=nn.BatchNorm2d)
#
# opt.semantic_nc = 7
# generator = SPADEGenerator(opt, 3 + 3 + 3)
# generator.print_network()
#
# # Load Checkpoint - VITON
# load_checkpoint(tocg, opt.tocg_checkpoint, opt)
# load_checkpoint_G(generator, opt.viton_checkpoint, opt)
#
#
# # Train NeRF- create checkpoints, folders for each individual
#
#
# # Test
# test(opt, test_loader, tocg, generator)