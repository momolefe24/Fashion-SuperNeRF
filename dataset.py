import glob
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from config import *

mean = dataset_facts['transforms']['mean']
std = dataset_facts['transforms']['std']
def denormalize(tensors):
    """ Denormalizes image tensors using mean and std """
    for c in range(3):
        tensors[:, c].mul_(std[c]).add_(mean[c])
    return torch.clamp(tensors, 0, 255)

class ImageDataset(Dataset):
    def __init__(self, root_dir="Dataset/eric", mode="train"):
        self.image_dir = root_dir + "/{}"
        self.lr_dir = self.image_dir.format("lr") + f"/{mode}/*.*"
        self.lr_files = sorted(glob.glob(self.lr_dir))
        self.hr_dir = self.image_dir.format("hr") + f"/{mode}/*.*"
        self.hr_files = sorted(glob.glob(self.hr_dir))
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

    def __len__(self):
        return len(self.hr_files)

    def __getitem__(self, index):
        lr_img = self.transform(Image.open(self.lr_files[index]).convert("RGB"))
        hr_img = self.transform(Image.open(self.hr_files[index]).convert("RGB"))
        return lr_img, hr_img