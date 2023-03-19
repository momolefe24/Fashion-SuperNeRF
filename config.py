# Python libraries
import os
import yaml
import argparse
import logging
import sys
import imageio
import json
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

# Data-Science Libraries
import numpy as np
from PIL import Image

# Pytorch imports
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision
from torchvision.utils import save_image
from torch.utils.data import Dataset,DataLoader,ConcatDataset
from torch.utils.tensorboard import SummaryWriter

# Albumentations
import albumentations as A
from albumentations.pytorch import ToTensorV2

logging.basicConfig(
format="%(asctime)s %(levelname)s %(message)s",
level=logging.DEBUG,
stream=sys.stdout,
)

def get_parser():
    """Get parser object."""
    parser = ArgumentParser(description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument(
    "-f",
    "--file",
    dest="filename",
    default="experiments/experiment_01_run_01.yaml",
    help="experiment definition file",
    metavar="FILE",
    required=True
    )
    parser.add_argument(
        "-c",
        "--config",
        dest="filename",
        default="Dataset/eric/eric.txt",
        help="Configuration for NeRF",
        required=False
    )
    return parser

def setup_experiment_files(paths):
    for path in paths:
        path_ = os.path.join(experiment_facts['root_path'], path['root_path'] + "/{}".format(experiment_facts['type']))
        path_ += "/{}".format(yaml_filepath.split("/")[1]).replace(".yaml", "")
        paths_.append(path_)
        if not os.path.isdir(path_):
            os.makedirs(path_)
        path_yaml = path_ + "/{}".format("experiment.yaml")
        with open(path_yaml, "w") as out:
            yaml.dump(cfg, out)


def gradient_penalty(critic, real, fake, device):
    BATCH_SIZE, C, H, W = real.shape
    alpha = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * alpha + fake.detach() * (1 - alpha)
    interpolated_images.requires_grad_(True)

    # Calculate critic scores
    mixed_scores = critic(interpolated_images)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty

def save_checkpoint(model, optimizer, checkpoint_file):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, checkpoint_file)

def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=>Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(checkpoint)
    optimizer.load_state_dict(checkpoint)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

paths_ = []
args = get_parser().parse_args()
yaml_filepath = args.filename
with open(yaml_filepath, "r") as stream:
    cfg = yaml.load(stream, Loader=yaml.FullLoader)

"""
Facts
"""
experiment_facts = cfg['experiment_facts']
checkpoint_facts = cfg['checkpoint_facts']
embedding_facts = cfg['embedding_facts']
results_facts = cfg['results_facts']
summary_facts = results_facts['summary_facts']
dataset_facts = cfg['dataset_facts']
hyperparameter_facts = cfg['hyperparameter_facts']
model_facts = cfg['model_facts']

paths = [results_facts, checkpoint_facts, summary_facts]
setup_experiment_files(paths)


"""
CONFIGURATION FILES
"""
nerf_checkpoint = checkpoint_facts['nerf_checkpoints']
nerf_results = results_facts['nerf_results']
nerf_data = dataset_facts['nerf_data']
nerf_embedder = embedding_facts['nerf_embedder']
nerf_summary = summary_facts['nerf_summary']
nerf_hyperparameters = hyperparameter_facts['nerf_hyperparameters']
nerf_model = model_facts['nerf_model']
H, W = eval(nerf_data['image_shape'])

"""
LAMBDA FUNTIONS
"""
random_choice = lambda input: np.random.choice(input)
img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]).to(device))
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)


"""TRANSFORMATIONS"""
transform = A.Compose(
    [
        A.Normalize(mean=list(eval(nerf_data['transforms']['mean'])), std=list(eval(nerf_data['transforms']['std']))),
        ToTensorV2(),
    ]
)


depth_transform = A.Compose(
    [ ToTensorV2()]
)

device = experiment_facts['device']