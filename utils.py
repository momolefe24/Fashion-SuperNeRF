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
import cv2
# --cuda False --name Rail_No_Occlusion -b 4 -j 2 --tocg_checkpoint checkpoints/Rail_RT_No_Occlusion_1/tocg_step_280000.pth
import argparse


""" WHEN YOU ARE CURRENTLY TRAINING"""
run_number = 1
experiment_number = 1
experiment_run = f"experiment_{experiment_number}/run_{run_number}"


""" WHEN YOU ARE LOADING CHECKPOINTS"""
run_from_number = 1
experiment_from_number = 1
experiment_from_run = f"experiment_{experiment_from_number}/run_{run_from_number}"
def get_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument("--name", default="Inference Pipeline")
    parser.add_argument('--cuda',default=True, help='cuda or cpu')
    parser.add_argument('-j', '--workers', type=int, default=1)
    parser.add_argument('--viton_workers', type=int, default=1)
    parser.add_argument('--person', default="julian")
    parser.add_argument('--clothing', default="gray_long_sleeve")
    parser.add_argument('--in_shop_clothing', default="molefe_black_shirt_30.jpg")
    parser.add_argument('--shuffle', default=False)
    parser.add_argument('--viton_shuffle', default=True)

    """  ============================================ DATASET ====================== """
    parser.add_argument("--dataroot", default="../data/rail")
    parser.add_argument("--transforms_dir", default="transforms")
    parser.add_argument("--datamode", default="temp")
    parser.add_argument("--output_dir", type=str, default="./Output")
    parser.add_argument("--fine_width", type=int, default=192)
    parser.add_argument("--fine_height", type=int, default=256)
    parser.add_argument('--viton_batch_size', type=int, default=4)
    parser.add_argument('--nerf_batch_size', type=int, default=1)

    """  ============================================ TRAINING IMAGES ====================== """
    parser.add_argument("--nerf_training_images", type=str, default=f"./results/{experiment_run}/NeRF/training")
    parser.add_argument("--nerf_validation_images", type=str, default=f"./results/{experiment_run}/NeRF/validation")
    parser.add_argument("--nerf_testing_images", type=str, default=f"./results/{experiment_run}/NeRF/testing")
    """  ============================================ TENSORBOARD ====================== """
    parser.add_argument("--writer", type=str, default="./tensorboard")
    """  ============================================ CHECKPOINTS ====================== """
    parser.add_argument("--nerf_save_checkpoint", default=f"./checkpoints/{experiment_run}/NeRF.pth")
    parser.add_argument("--nerf_load_checkpoint", default=f"./checkpoints/{experiment_from_run}/NeRF.pth")

    parser.add_argument("--viton_save_step_checkpoint", default=f"./checkpoints/{experiment_run}/gen_step_%06d.pth")
    parser.add_argument("--viton_load_step_checkpoint", default=f"./checkpoints/{experiment_from_run}/gen_step_%06d.pth")

    parser.add_argument("--viton_save_final_checkpoint", default=f"./checkpoints/{experiment_run}/gen_final.pth")
    parser.add_argument("--viton_load_final_checkpoint", default=f"./checkpoints/{experiment_from_run}/gen_final.pth")

    parser.add_argument("--tocg_save_step_checkpoint", default=f"./checkpoints/{experiment_run}/steps/tocg_step_%06d.pth")
    parser.add_argument("--tocg_load_step_checkpoint", default=f"./checkpoints/{experiment_from_run}/steps/tocg_step_%06d.pth")

    parser.add_argument("--tocg_discriminator_save_step_checkpoint", default=f"./checkpoints/{experiment_run}/steps/tocg_step_D%06d.pth")
    parser.add_argument("--tocg_discriminator_load_step_checkpoint", default=f"./checkpoints/{experiment_from_run}/steps/tocg_step_D%06d.pth")

    parser.add_argument("--tocg_save_final_checkpoint", default=f"./checkpoints/{experiment_run}/tocg_final.pth")
    parser.add_argument("--tocg_load_final_checkpoint", default=f"./checkpoints/{experiment_from_run}/tocg_final.pth")
    """  ============================================ HYPERPARAMETERS ====================== """
    parser.add_argument("--semantic_nc", type=int, default=13)
    parser.add_argument("--warp_feature", choices=['encoder', 'T1'], default="T1")
    parser.add_argument("--output_nc", type=int, default=13)
    parser.add_argument("--out_layer", choices=['relu', 'conv'], default="relu")

    """  ============================================ VIRTUAL TRY-ON HYPERPARAMETERS: TRY-ON CONDITION GENERATOR ====================== """
    parser.add_argument('--tocg_Ddownx2', default=True, help="Downsample D's input to increase the receptive field")
    parser.add_argument('--tocg_Ddropout', default=True, help="Apply dropout to D")
    parser.add_argument('--tocg_num_D', type=int, default=2, help='Generator ngf')
    parser.add_argument('--tocg_fp16', default=True, help='use amp')
    """  ============================================ VIRTUAL TRY-ON HYPERPARAMETERS ====================== """
    parser.add_argument("--no_test_visualize", type=bool, default=False)
    parser.add_argument("--num_test_visualize", type=int, default=4)
    parser.add_argument("--test_datasetting", type=str, default="unpaired")
    parser.add_argument("--test_dataroot", type=str, default="../data/rail")
    parser.add_argument("--test_data_list", type=str, default="test_pairs.txt")
    parser.add_argument('--G_lr', type=float, default=0.0002, help='Generator initial learning rate for adam')
    parser.add_argument('--D_lr', type=float, default=0.0002, help='Discriminator initial learning rate for adam')
    parser.add_argument('--CElamda', type=float, default=10, help='initial learning rate for adam')
    parser.add_argument('--GANlambda', type=float, default=1)
    parser.add_argument('--tvlambda', type=float, default=2)
    parser.add_argument('--upsample', type=str, default='bilinear', choices=['nearest', 'bilinear'])
    parser.add_argument('--val_count', type=int, default='1000')
    parser.add_argument('--spectral', default=True, help="Apply spectral normalization to D")
    parser.add_argument('--occlusion', default=True, help="Occlusion handling")
    # training
    parser.add_argument("--G_D_seperate", default=True)
    parser.add_argument("--no_GAN_loss",  default=True,)
    parser.add_argument("--lasttvonly",  default=True)
    parser.add_argument("--interflowloss", default=True, help="Intermediate flow loss")
    parser.add_argument("--clothmask_composition", type=str, choices=['no_composition', 'detach', 'warp_grad'], default='warp_grad')
    parser.add_argument('--edgeawaretv', type=str, choices=['no_edge', 'last_only', 'weighted'], default="no_edge", help="Edge aware TV loss")
    parser.add_argument('--add_lasttv', default=True)
    parser.add_argument("--tocg_basedir", type=str, default=f'./logs/{experiment_run}',
                        help='where to store logs')
    parser.add_argument("--tocg_writer", type=str, default=f'./tensorboard/{experiment_run}/tocg',
                        help='Tensorboard information for try-on condition generator')
    parser.add_argument("--tocg_name", type=str, default='tocg')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='save checkpoint infos')
    # parser.add_argument('--tocg_checkpoint', type=str, default='checkpoints/VITON/Original Virtual Try-On/tocg_step_120000.pth', help='tocg checkpoint')

    parser.add_argument("--tensorboard_count", type=int, default=100)
    parser.add_argument("--display_count", type=int, default=100)
    parser.add_argument("--save_count", type=int, default=10000)
    parser.add_argument("--load_step", type=int, default=0)
    parser.add_argument("--keep_step", type=int, default=300000)
    

    """  ============================================ NEURAL RADIANCE FIELD HYPERPARAMETERS ====================== """
    parser.add_argument('--model', default="NeRF")
    parser.add_argument('--config', default="../data/rail/configs/julian_gray_long_sleeve.txt",
                        help='config file path')
    parser.add_argument("--nerf_basedir", type=str, default=f'./logs/{experiment_run}',
                        help='where to store logs')
    parser.add_argument("--nerf_datadir", type=str, default='./data/rail/temp/',
                        help='input data directory')
    parser.add_argument("--nerf_writer", type=str, default=f'./tensorboard/{experiment_run}/NeRF',
                        help='Tensorboard information')
    parser.add_argument("--nerf_netdepth", type=int, default=8,
                        help='layers in network')
    parser.add_argument("--nerf_netwidth", type=int, default=256,
                        help='channels per layer')
    parser.add_argument("--nerf_netdepth_fine", type=int, default=8,
                        help='layers in fine network')
    parser.add_argument("--nerf_netwidth_fine", type=int, default=256,
                        help='channels per layer in fine network')
    parser.add_argument("--nerf_N_rand", type=int, default=32 * 32 * 4,
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--nerf_lrate", type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument("--nerf_lrate_decay", type=int, default=250,
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--nerf_chunk", type=int, default=1024 * 32,
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--nerf_netchunk", type=int, default=1024 * 64,
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--nerf_no_batching", default=True,
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--nerf_no_reload", default=True,
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--nerf_ft_path", type=str, default=None,
                        help='specific weights npy file to reload for coarse network')

    # rendering options
    parser.add_argument("--nerf_N_samples", type=int, default=64,
                        help='number of coarse samples per ray')
    parser.add_argument("--nerf_N_importance", type=int, default=128,
                        help='number of additional fine samples per ray')
    parser.add_argument("--nerf_perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--nerf_use_viewdirs",default=True,
                        help='use full 5D input instead of 3D')
    parser.add_argument("--nerf_i_embed", type=int, default=0,
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--nerf_multires", type=int, default=10,
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--nerf_multires_views", type=int, default=4,
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--nerf_raw_noise_std", type=float, default=0.,
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    parser.add_argument("--nerf_render_only", default=False,
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--nerf_render_test", default=True,
                        help='render the test set instead of render_poses path')
    parser.add_argument("--nerf_render_factor", type=int, default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # training options
    parser.add_argument("--nerf_precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--nerf_precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops')

    # dataset options
    parser.add_argument("--nerf_dataset_type", type=str, default='blender',
                        help='options: llff / blender / deepvoxels')
    parser.add_argument("--nerf_testskip", type=int, default=8,
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    ## deepvoxels flags
    parser.add_argument("--nerf_shape", type=str, default='greek',
                        help='options : armchair / cube / greek / vase')

    ## blender flags
    parser.add_argument("--nerf_white_bkgd", default=True,
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--nerf_half_res", default=True,
                        help='load blender synthetic data at 400x400 instead of 800x800')

    ## llff flags
    parser.add_argument("--nerf_factor", type=int, default=8,
                        help='downsample factor for LLFF images')
    parser.add_argument("--nerf_no_ndc", default=True,
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--nerf_lindisp", default=True,
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--nerf_spherify", default=True,
                        help='set for spherical 360 scenes')
    parser.add_argument("--nerf_llffhold", type=int, default=8,
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    # logging/saving options
    parser.add_argument("--nerf_i_print", type=int, default=100,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--nerf_i_img", type=int, default=500,
                        help='frequency of tensorboard image logging')
    parser.add_argument("--nerf_i_weights", type=int, default=10000,
                        help='frequency of weight ckpt saving')
    parser.add_argument("--nerf_i_testset", type=int, default=50000,
                        help='frequency of testset saving')
    parser.add_argument("--nerf_i_video", type=int, default=50000,
                        help='frequency of render_poses video saving')
    """  ============================================ SPADE HYPERPARAMETERS ====================== """
    parser.add_argument('--GMM_const', type=float, default=None, help='constraint for GMM module')
    parser.add_argument('--gen_semantic_nc', type=int, default=7, help='# of input label classes without unknown class')
    parser.add_argument('--norm_G', type=str, default='spectralaliasinstance',
                        help='instance normalization or batch normalization')
    parser.add_argument('--norm_D', type=str, default='spectralinstance',
                        help='instance normalization or batch normalization')
    parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
    parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
    parser.add_argument('--num_upsampling_layers', choices=['normal', 'more', 'most'], default='most',
                        help='If \'more\', add upsampling layer between the two middle resnet blocks. '
                             'If \'most\', also add one more (upsampling + resnet) layer at the end of the generator.')
    parser.add_argument('--init_type', type=str, default='xavier',
                        help='network initialization [normal|xavier|kaiming|orthogonal]')
    parser.add_argument('--init_variance', type=float, default=0.02, help='variance of the initialization distribution')

    parser.add_argument('--no_ganFeat_loss', default=True,
                        help='if specified, do *not* use discriminator feature matching loss')
    parser.add_argument('--no_vgg_loss', default=True,
                        help='if specified, do *not* use VGG feature matching loss')
    parser.add_argument('--lambda_l1', type=float, default=1.0, help='weight for feature matching loss')
    parser.add_argument('--lambda_feat', type=float, default=10.0, help='weight for feature matching loss')
    parser.add_argument('--lambda_vgg', type=float, default=10.0, help='weight for vgg loss')
    opt = parser.parse_args()
    return opt

def get_transforms_data(data_path,person_clothing):
    transform_string = f"{data_path}/{person_clothing}.json"
    with open(transform_string, 'r') as f:
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

similarity = lambda n1, n2: 1 - abs(n1 - n2) / (n1 + n2)
labels = {
            0: ['background', [0, 10]],
            1: ['hair', [1, 2]],
            2: ['face', [4, 13]],
            3: ['upper', [5, 6, 7]],
            4: ['bottom', [9, 12]],
            5: ['left_arm', [14]],
            6: ['right_arm', [15]],
            7: ['left_leg', [16]],
            8: ['right_leg', [17]],
            9: ['left_shoe', [18]],
            10: ['right_shoe', [19]],
            11: ['socks', [8]],
            12: ['noise', [3, 11]]
        }

def get_half_res(imgs, focal, H, W):
    H = H//2
    W = W//2
    focal = focal/2.
    depth = imgs[0].shape[-1]
    imgs_half_res = np.zeros((imgs.shape[0], H, W, depth))
    for i, img in enumerate(imgs):
        imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    imgs = imgs_half_res
    return imgs

