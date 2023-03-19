import logging
import sys
import os
import imageio
import json
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import glob
import cv2
import yaml

# Data-Science Libraries
import numpy as np
from PIL import Image

# Pytorch imports
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torch.optim as optim
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision
from torchvision.utils import save_image
import torchvision.transforms.functional as TF
from torch.utils.tensorboard import SummaryWriter
import torchvision.models as models
from torch import Tensor
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR

# Albumentations
import albumentations as A
from albumentations.pytorch import ToTensorV2


logging.basicConfig(
format="%(asctime)s %(levelname)s %(message)s",
level=logging.DEBUG,
stream=sys.stdout,
)

paths_ = []
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"
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


def load_checkpoint(checkpoint_file, model, optimizer, lr, vgg=False, esrgan=False):
    print("=>Loading checkpoint")

    checkpoint = torch.load(checkpoint_file, map_location=device)
    if vgg:
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    elif esrgan:
        checkpoint = torch.load(checkpoint_file, map_location=device)
        model.load_state_dict(checkpoint)
        optimizer.load_state_dict(checkpoint)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    else:
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def plot_examples(low_res_folder, gen):
    files = os.listdir(low_res_folder)

    gen.eval()
    for file in files:
        image = Image.open("test_images/" + file)
        with torch.no_grad():
            upscaled_img = gen(
                test_transform(image=np.asarray(image))["image"]
                .unsqueeze(0)
                .to(device)
            )
        save_image(upscaled_img, f"saved/{file}")
    gen.train()

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

args = get_parser().parse_args()
yaml_filepath = args.filename
with open(yaml_filepath, "r") as stream:
    cfg = yaml.load(stream, Loader=yaml.FullLoader)


"""  FACTS """
experiment_facts = cfg['experiment_facts']
experiment_type = experiment_facts['type']
checkpoint_facts = cfg['checkpoint_facts']
results_facts = cfg['results_facts']
dataset_facts = cfg['dataset_facts']
model_facts = cfg['model_facts']
training_facts = cfg['training_facts']
summary_facts = cfg['summary_facts']

high_res = (300, 192) # pytorch (height, width)
low_res = (30, 24)
HIGH_RES = 256
# Save Results DIR


# Save experiments
paths = [results_facts, checkpoint_facts, summary_facts]
setup_experiment_files(paths)
# results_dir = results_facts['root_path']/experiment_facts['type']/yaml_filepath.split("/")[1].replace(".yaml","")


""" Get Training Information """
# ESRGAN
esrgan_facts = training_facts["ESRGAN"]


# Transforms
image_transform = A.Compose(
    [A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]), ToTensorV2()]
)


transform_lambda = lambda res: A.Compose(
    [
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
        A.RandomCrop(height=res[0], width=res[1]),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
    ]
)

highres_transform = A.Compose(
    [A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]), ToTensorV2()]
)
lowres_transform = A.Compose(
    [
        A.Resize(width=esrgan_facts['high_res'] // esrgan_facts['upscaling_factor'], height=esrgan_facts['high_res'] // esrgan_facts['upscaling_factor'], interpolation=Image.BICUBIC),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
        ToTensorV2(),
    ]
)

both_transforms_ = A.Compose(
    [
        A.RandomCrop(width=esrgan_facts['high_res'], height=esrgan_facts['high_res']),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
    ]
)


test_transform = A.Compose(
    [
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
        ToTensorV2(),
    ]
)

"""
Summary Creation
"""
writer = SummaryWriter(paths_[-1])

"""
Model Creation
"""