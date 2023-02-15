import glob
import torchvision
import random
from torch.utils.data import Dataset
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
from config import *
import cv2

mean = dataset_facts['transforms']['mean']
std = dataset_facts['transforms']['std']
to_numpy = lambda x: x.permute(1, 2, 0).numpy()

def get_pairs(lr, hr, highres_shape= (300, 192), divisors = (10, 8)):
    lr = TF.inter
    i, j, h, w = transforms.RandomCrop.get_params(hr, highres_shape)
    hr = TF.crop(hr, i, j, highres_shape[0], highres_shape[1])
    lr = TF.crop(lr, i // divisors[0], j // divisors[1], highres_shape[0] // divisors[0], highres_shape[1] // divisors[1])
    return lr, hr

def closestMultiple(n, x):
    if x > n:
        return x;
    z = (int)(x / 2);
    n = n + z;
    n = n - (n % x);
    return n
def get_pairs_2(lr, hr, highres_shape= (512, 192), divisors = (8, 8)):
    lr = F.interpolate(lr.unsqueeze(dim=0), size=(128, 96)).squeeze(dim=0)
    i, j, h, w = transforms.RandomCrop.get_params(hr, highres_shape)
    i = closestMultiple(i, divisors[0])
    j = closestMultiple(j, divisors[1])
    hr = TF.crop(hr, i, j, highres_shape[0], highres_shape[1])
    lr = TF.crop(lr, i // divisors[0], j // divisors[1], highres_shape[0] // divisors[0], highres_shape[1] // divisors[1])
    save_image(lr, "title.png")
    save_image(hr, "title2.png")
    return lr, hr

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
        self.mode = mode
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        self.read_image = lambda img: cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB)

    def __len__(self):
        return len(self.hr_files)

    # We are using crops to save on memory because a model to consume an entire image is very big
    """
    We are using the ESRGAN Without Gradient Penalty from the cluster
        1) Perform this on crops
        2) Perform this on the whole image
    """
    def __getitem__(self, index):
        if self.mode == "train":
            img = cv2.cvtColor(cv2.imread(self.hr_files[index]), cv2.COLOR_BGR2RGB)
            img = both_transforms_(image=img)['image']
            hr_img = highres_transform(image=img)['image']
            lr_img = lowres_transform(image=img)['image']
        else:
            lr_img = cv2.cvtColor(cv2.imread(self.lr_files[index]), cv2.COLOR_BGR2RGB)
            hr_img = cv2.cvtColor(cv2.imread(self.hr_files[index]), cv2.COLOR_BGR2RGB)
            lr_img = test_transform(image=lr_img)['image']
            hr_img = test_transform(image=hr_img)['image']
        # # lr_img, hr_img = get_pairs(lr_img, hr_img, highres_shape= (256, 256), divisors = (8, 8))
        # lr_img, hr_img = get_pairs_2(lr_img, hr_img)
        return lr_img, hr_img
