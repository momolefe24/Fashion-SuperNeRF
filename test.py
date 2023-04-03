import torch
import torchvision.utils

from config import *
from Dataset.dataset import NeRF_Dataset
from NeRF.model import NeRF, NeRF_Fine

train_dataset = NeRF_Dataset(quality="hr", mode="train")
train_dataloader = DataLoader(dataset=train_dataset, batch_size=hyperparameter_facts['batch_size'], shuffle=True)

valid_dataset = NeRF_Dataset(quality="hr", mode="val")
valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=hyperparameter_facts['batch_size'], shuffle=True)

test_dataloader = NeRF_Dataset(quality="hr", mode="test")
test_dataloader = DataLoader(dataset=test_dataloader, batch_size=hyperparameter_facts['batch_size'], shuffle=False)

K = train_dataset.K
nerf = NeRF(K).to(device)
grad_vars = list(nerf.parameters())

nerf_fine = NeRF_Fine(K).to(device)
grad_vars += list(nerf_fine.parameters())

train_dataset.__getitem__(0)[0].shape

nerf_facts = model_facts['nerf_model']
lrate = nerf_facts['lrate']

nerf_pth = f"Checkpoints/pretrain_nerf/{nerf_checkpoint['checkpoint_nerf']}"
optimizer = torch.optim.Adam(params=grad_vars, lr=lrate, betas=(0.9, 0.999))

nerf_checkpoint = torch.load(nerf_pth)
start = nerf_checkpoint['global_step']
optimizer.load_state_dict(nerf_checkpoint['optimizer_state_dict'])

nerf.load_state_dict(nerf_checkpoint['network_fn_state_dict'])
nerf_fine.load_state_dict(nerf_checkpoint['network_fine_state_dict'])

render_poses = train_dataset.render_poses
H, W, focal = train_dataset.H, train_dataset.W, train_dataset.focal
K = train_dataset.K
hwf = H, W, focal

render_pose = render_poses[0]
c2w = render_pose[:3, :4]
use_viewdirs=False

_, eval_rgb_map, eval_disp_map, eval_acc_map, eval_weights, eval_depth_map, eval_fine_parameters = nerf(c2w)