import torch
import torch.nn as nn

from torchvision.utils import make_grid
from networks import make_grid as mkgrid

import argparse
import os
import time
from cp_dataset import CPDataset, CPDatasetTest, CPDataLoader
from networks import ConditionGenerator, VGGLoss, GANLoss, load_checkpoint, save_checkpoint, define_D
from tqdm import tqdm
from tensorboardX import SummaryWriter
from utils import *
from torch.utils.data import Subset
import matplotlib.pyplot as plt


def get_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument("--name", default="Molefe")
    parser.add_argument("--gpu_ids", default="")
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('-b', '--batch-size', type=int, default=8)
    parser.add_argument('--fp16', action='store_true', help='use amp')

    # parser.add_argument("--dataroot", default="./data/nerf_people/eric/hr")
    parser.add_argument("--dataroot", default="./data/viton")
    # parser.add_argument("--dataroot", default="./data/molefe")
    parser.add_argument("--datamode", default="train")
    parser.add_argument("--data_list", default="train_pairs.txt")
    parser.add_argument("--fine_width", type=int, default=192)
    parser.add_argument("--fine_height", type=int, default=256)

    parser.add_argument('--tensorboard_dir', type=str, default='tensorboard', help='save tensorboard infos')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='save checkpoint infos')
    # parser.add_argument('--tocg_checkpoint', type=str, default='', help='tocg checkpoint')
    parser.add_argument('--tocg_checkpoint', type=str,
                        default='checkpoints/VITON/Original Virtual Try-On/tocg_step_120000.pth',
                        help='tocg checkpoint')

    parser.add_argument("--tensorboard_count", type=int, default=100)
    parser.add_argument("--display_count", type=int, default=100)
    parser.add_argument("--save_count", type=int, default=10000)
    parser.add_argument("--load_step", type=int, default=0)
    parser.add_argument("--keep_step", type=int, default=300000)
    parser.add_argument("--shuffle", action='store_true', help='shuffle input data')
    parser.add_argument("--semantic_nc", type=int, default=13)
    parser.add_argument("--output_nc", type=int, default=13)

    # network
    parser.add_argument("--warp_feature", choices=['encoder', 'T1'], default="T1")
    parser.add_argument("--out_layer", choices=['relu', 'conv'], default="relu")
    parser.add_argument('--Ddownx2', action='store_true', help="Downsample D's input to increase the receptive field")
    parser.add_argument('--Ddropout', action='store_true', help="Apply dropout to D")
    parser.add_argument('--num_D', type=int, default=2, help='Generator ngf')
    # Cuda availability
    parser.add_argument('--cuda', default=False, help='cuda or cpu')
    # training
    parser.add_argument("--G_D_seperate", action='store_true')
    parser.add_argument("--no_GAN_loss", action='store_true')
    parser.add_argument("--lasttvonly", action='store_true')
    parser.add_argument("--interflowloss", action='store_true', help="Intermediate flow loss")
    parser.add_argument("--clothmask_composition", type=str, choices=['no_composition', 'detach', 'warp_grad'],
                        default='warp_grad')
    parser.add_argument('--edgeawaretv', type=str, choices=['no_edge', 'last_only', 'weighted'], default="no_edge",
                        help="Edge aware TV loss")
    parser.add_argument('--add_lasttv', action='store_true')

    # test visualize
    parser.add_argument("--no_test_visualize", action='store_true')
    parser.add_argument("--num_test_visualize", type=int, default=3)
    parser.add_argument("--test_datasetting", default="unpaired")
    # parser.add_argument("--test_dataroot", default="./data/molefe")
    parser.add_argument("--test_dataroot", default="./data/viton")
    parser.add_argument("--test_data_list", default="test_pairs.txt")

    # Hyper-parameters
    parser.add_argument('--G_lr', type=float, default=0.0002, help='Generator initial learning rate for adam')
    parser.add_argument('--D_lr', type=float, default=0.0002, help='Discriminator initial learning rate for adam')
    parser.add_argument('--CElamda', type=float, default=10, help='initial learning rate for adam')
    parser.add_argument('--GANlambda', type=float, default=1)
    parser.add_argument('--tvlambda', type=float, default=2)
    parser.add_argument('--upsample', type=str, default='bilinear', choices=['nearest', 'bilinear'])
    parser.add_argument('--val_count', type=int, default='1000')
    parser.add_argument('--spectral', action='store_true', help="Apply spectral normalization to D")
    parser.add_argument('--occlusion', action='store_true', help="Occlusion handling")

    opt = parser.parse_args()
    return opt

opt = get_opt()
print(opt)
print("Start to train %s!" % opt.name)
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids
# create train dataset & loader
train_dataset = CPDataset(opt)
train_loader = CPDataLoader(opt, train_dataset)

test_loader = None
if not opt.no_test_visualize:
    train_bsize = opt.batch_size
    opt.batch_size = opt.num_test_visualize
    opt.dataroot = opt.test_dataroot
    opt.datamode = 'test'
    opt.data_list = opt.test_data_list
    test_dataset = CPDatasetTest(opt)
    opt.batch_size = train_bsize
    val_dataset = Subset(test_dataset, np.arange(50))
    test_loader = CPDataLoader(opt, test_dataset)
    val_loader = CPDataLoader(opt, val_dataset)
    # visualization
if not os.path.exists(opt.tensorboard_dir):
    os.makedirs(opt.tensorboard_dir)
board = SummaryWriter(log_dir=os.path.join(opt.tensorboard_dir, opt.name))

# Loss Functions
criterionL1 = nn.L1Loss()
criterionVGG = VGGLoss(opt)
criterionGAN = GANLoss(use_lsgan=True, tensor=torch.Tensor)
