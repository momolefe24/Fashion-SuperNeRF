import json
from config import *
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from .rendering import map_fn, pose_spherical
import cv2

class InputPipeline(Dataset):
    # def __init__(self, images, poses, width, height, focal_length, nC=8, near=2.0, far=6.0, rand=True):
    def __init__(self, mode="train", rand=True):

        # dataset_paths  = ["lr","hr"]
        self.rand = rand
        def get_images_and_poses(path):
            images = []
            poses = []
            model_dataset = f"{dataset_facts['root_path']}/{dataset_facts['model']}/{path}"
            transforms = json.load(open(f"{model_dataset}/transforms_{mode}.json"))
            for angle in transforms['frames']:
                filename = angle['file_path'].replace(".", f"{model_dataset}") + ".png"
                image = cv2.imread(filename)[..., ::-1]
                image = (image / nerf_facts['dynamic_range']).astype(np.float32)
                images.append(image)
                poses.append(angle['transform_matrix'])

            images = np.array(images)
            poses = np.array(poses)
            poses = [pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180, 180, len(poses) + 1)[:-1]]
            return images, poses

        self.lr_images, self.lr_poses = get_images_and_poses("lr")
        self.hr_images, self.hr_poses = get_images_and_poses("hr")
        self.lr_height, self.lr_width = eval(dataset_facts['image']['lr_shape'])[1:]
        self.hr_height, self.hr_width = eval(dataset_facts['image']['hr_shape'])[1:]
        self.lr_focal_length = .5 * self.lr_width / np.tan(.5 * nerf_facts['fc'])
        self.hr_focal_length = .5 * self.hr_width / np.tan(.5 * nerf_facts['fc'])
    def __len__(self):
        return len(self.lr_poses)

    def __getitem__(self, index):
        lr_pose, hr_pose = self.lr_poses[index], self.hr_poses[index]
        lr_image, hr_image = self.lr_images[index], self.hr_images[index]
        lr_rays_flat, lr_t_vals = map_fn(lr_pose, self.lr_focal_length, self.lr_width, self.lr_height, nerf_facts['near'], nerf_facts['far'], nerf_facts['N_samples'], rand=self.rand)
        # hr_rays_flat, hr_t_vals = map_fn(hr_pose, self.hr_focal_length, self.hr_width, self.hr_height, nerf_facts['near'], nerf_facts['far'], nerf_facts['N_samples'], rand=self.rand)
        return lr_image, hr_image, lr_rays_flat, lr_t_vals


