import glob
import torchvision
import random
from NeRF_SR.rendering import *
from torch.utils.data import Dataset
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
from config import *
import cv2
import json

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
    def __init__(self, root_dir="Dataset/eric", mode="train", rand=True):
        self.image_dir = root_dir + "/{}"
        self.mode = mode
        self.rand = rand
        def get_images_and_poses(path):
            images = []
            poses = []
            model_dataset = f"{dataset_facts['root_path']}/{dataset_facts['model']}/{path}"
            transforms = json.load(open(f"{model_dataset}/transforms_{mode}.json"))
            for angle in transforms['frames']:
                filename = angle['file_path'].replace(".", f"{model_dataset}") + ".png"
                image = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
                # image = (image / nerf_facts['dynamic_range']).astype(np.float32)
                images.append(image)
                poses.append(angle['transform_matrix'])

            images = np.array(images)
            poses = np.array(poses)
            poses = [pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180, 180, len(poses) + 1)[:-1]]
            return images, poses

        self.lr_images, self.lr_poses = get_images_and_poses("lr")
        self.hr_images, self.hr_poses = get_images_and_poses("hr")
        self.lr_height, self.lr_width = eval(dataset_facts['image']['lr_shape_crop'])[1:]
        self.hr_height, self.hr_width = eval(dataset_facts['image']['hr_shape_crop'])[1:]
        self.lr_focal_length = .5 * self.lr_width / np.tan(.5 * nerf_facts['fc'])
        self.hr_focal_length = .5 * self.hr_width / np.tan(.5 * nerf_facts['fc'])
    def __len__(self):
        return len(self.lr_poses)

    # We are using crops to save on memory because a model to consume an entire image is very big
    """
    We are using the ESRGAN Without Gradient Penalty from the cluster
        1) Perform this on crops
        2) Perform this on the whole image
    """
    def __getitem__(self, index):
        if self.mode == "train":
            # img = cv2.cvtColor(cv2.imread(self.hr_files[index]), cv2.COLOR_BGR2RGB)
            img = self.hr_images[index]
            img = both_transforms_(image=img)['image']
            hr_img = highres_transform(image=img)['image']
            lr_img = lowres_transform(image=img)['image']
        else:
            lr_img = self.lr_images[index]
            hr_img = self.hr_images[index]
            lr_img = test_transform(image=lr_img)['image']
            hr_img = test_transform(image=hr_img)['image']
        # # lr_img, hr_img = get_pairs(lr_img, hr_img, highres_shape= (256, 256), divisors = (8, 8))
        # lr_img, hr_img = get_pairs_2(lr_img, hr_img)
        lr_pose = self.lr_poses[index]
        lr_rays_flat, lr_t_vals = map_fn(lr_pose, self.lr_focal_length, self.lr_width, self.lr_height,
                                         nerf_facts['near'], nerf_facts['far'], nerf_facts['N_samples'], rand=self.rand)
        return hr_img, lr_img, lr_rays_flat, lr_t_vals