import logging
import sys
import os
import yaml
import json
import imageio
import numpy as np
import torch
from NeRF.load_blender import pose_spherical
import pprint
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

logging.basicConfig(
format="%(asctime)s %(levelname)s %(message)s",
level=logging.DEBUG,
stream=sys.stdout,
)
# --cuda False --name Rail_No_Occlusion -b 4 -j 2 --tocg_checkpoint checkpoints/Rail_RT_No_Occlusion_1/tocg_step_280000.pth
import argparse
def get_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument("--name", default="Inference Pipeline")
    parser.add_argument('--cuda',default=False, help='cuda or cpu')
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('-b', '--batch-size', type=int, default=8)
    parser.add_argument('--person', default="julian")
    parser.add_argument('--clothing', default="gray_long_sleeve")
    # parser.add_argument('--in_shop_clothing', default="molefe_black_shirt.jpg") # soon to be
    parser.add_argument('--in_shop_clothing', default="molefe_black_shirt_53.jpg")
    parser.add_argument('--shuffle', default=False)

    """  ============================================ DATASET ====================== """
    parser.add_argument("--dataroot", default="./data/rail")
    parser.add_argument("--transform_dir", default="transforms")
    # parser.add_argument("--datamode", default="test")
    parser.add_argument("--datamode", default="train")
    # parser.add_argument("--data_list", default="test_pairs.txt")
    parser.add_argument("--data_list", default="train_pairs.txt")
    parser.add_argument("--fine_width", type=int, default=192)
    parser.add_argument("--fine_height", type=int, default=256)

    """  ============================================ CHECKPOINTS ====================== """
    parser.add_argument("--nerf_checkpoint", default="./checkpoints/orginal_nerf/experiment_02_run_03/NeRF.pth")
    parser.add_argument("--viton_checkpoint", default="./checkpoints/Rail_Composition_No_Occlusion/gen_step_050000.pth")
    parser.add_argument("--tocg_checkpoint", default="./checkpoints/Rail_RT_No_Occlusion_1/tocg_step_280000.pth")

    """  ============================================ HYPERPARAMETERS ====================== """
    parser.add_argument("--semantic_nc", type=int, default=13)
    parser.add_argument("--warp_feature", choices=['encoder', 'T1'], default="T1")
    parser.add_argument("--output_nc", type=int, default=13)
    parser.add_argument("--out_layer", choices=['relu', 'conv'], default="relu")

    opt = parser.parse_args()
    return opt

def get_transforms_data(opt):
    transform_string = f"transforms_{opt.person}_{opt.clothing}.json"
    with open(os.path.join(opt.dataroot, opt.transform_dir, transform_string), 'r') as f:
        transform_data = json.load(f)
    return transform_data

def get_transform_matrix(transform_data, image_name):
    transform_matrix = None
    for frame in transform_data['frames']:
        file_string = frame['file_path'].split("/")[-1]
        if image_name == file_string:
            transform_matrix = frame['transform_matrix']
            break
    return transform_matrix

def load_nerf_data(dataset):
    imgs = []
    poses = []
    for data in dataset:
        imgs.append(imageio.imread(data['im_name']))
        poses.append(np.array(data['transform_matrix']))
    imgs = (np.array(imgs) / 255.).astype(np.float32)  # keep all 4 channels (RGBA)
    poses = np.array(poses).astype(np.float32)
    H, W = imgs[0].shape[:2]
    camera_angle_x = float(dataset['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)
    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180, 180, 40 + 1)[:-1]], 0)
    return imgs, poses, render_poses, [H, W, focal], None
