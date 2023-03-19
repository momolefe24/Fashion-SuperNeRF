from config import *
from .processing import pose_spherical
from NeRF.rendering import get_rays, render_rays, render, get_image_plane, get_intervals_between_samples, ndc_rays
class NeRF_Dataset(Dataset):
    def __init__(self, quality="lr", mode="train"):
        # Setup directory
        self.data_root = f"Dataset/nerf_{nerf_data['blender_data']}"
        self.model_dir = os.path.join(self.data_root, f"{nerf_data['blender_model']}", quality)
        self.json_file = f"transforms_{mode}.json"
        self.transforms = os.path.join(self.model_dir, self.json_file)

        # Read in transformation
        with open(self.transforms, 'r') as fp:
            json_data = json.load(fp)
        skip = 1

        # Reading frames
        images = []
        poses = []
        for frame in json_data['frames'][::skip]:
            current_frame = os.path.join(self.model_dir, frame['file_path'] + '.png')
            images.append(imageio.imread(current_frame))
            poses.append(np.array(frame['transform_matrix']))
        self.images = (np.array(images) / 255.).astype(np.float32)  # Keep all 4 channels (rgba)
        # self.images = np.array(images)
        self.poses = np.array(poses).astype(np.float32)
        self.H, self.W = images[0].shape[:2]
        self.camera_angle_x = float(json_data['camera_angle_x'])
        self.focal = .5 * W / np.tan(.5 * self.camera_angle_x)
        self.render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180, 180, 40 + 1)[:-1]], 0)
        self.K = np.array([[self.focal, 0, 0.5 * self.W], [0, self.focal, 0.5 * self.H], [0, 0, 1]])
    def __len__(self):
        return len(self.images)

    def __getitem__(self, item): # get_batch_rays
        target = self.images[item] # image
        # rgb = transform(image=target[..., :3])['image']
        rgb = depth_transform(image=target[..., :3])['image']
        if not nerf_data['white_background']:
            target = rgb.permute(1, 2, 0)
        else:
            depth = depth_transform(image=target[..., -1])['image']
            target = torch.cat((rgb, depth), 0).permute(1, 2, 0) # the whole project uses (h,w,3)
        pose = torch.Tensor(self.poses[item, :3, :4])
        return target, pose