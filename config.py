import os
import yaml
import argparse
import logging
import sys
import json
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

# Data science libraries
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

to = lambda inputs,name,i: inputs[name][i].permute(1,2,0).cpu().numpy()
to2 = lambda inputs,name1,name2, i: inputs[name1][name2][i].permute(1,2,0).cpu().numpy()
convert_torch_to_numpy = lambda inputs,i: inputs[i].permute(1,2,0).detach().cpu().numpy()

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
    default="experiments/experiment.yaml",
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

paths_ = []
args = get_parser().parse_args()
#yaml_filepath = args.filename
yaml_filepath = "experiments/experiment.yaml"
with open(yaml_filepath, "r") as stream:
    cfg = yaml.load(stream, Loader=yaml.FullLoader)

"""
Facts
"""
checkpoint_facts = cfg['checkpoint_facts']
results_facts = cfg['results_facts']
summary_facts = results_facts['summary_facts']

experiment_facts = cfg['experiment_facts']
dataset_facts = cfg['dataset_facts']
hyperparameter_facts = cfg['hyperparameter_facts']

paths = [results_facts, checkpoint_facts, summary_facts]
setup_experiment_files(paths)

""" DATASET fACTS"""
H, W = eval(dataset_facts['downsample_image_shape']) # (256, 192)


""" HYPERPARAMETER FACTS"""
warping_hyperparameters = hyperparameter_facts['warping_hyperparameters']
cloth_input_channels = warping_hyperparameters['input_channels']['cloth']
parse_input_channels = warping_hyperparameters['input_channels']['person_representation']
resblock_channel = warping_hyperparameters['resblock_channel'] # ngf
output_channel = warping_hyperparameters['output_channel']
