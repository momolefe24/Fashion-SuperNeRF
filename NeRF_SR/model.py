import torch
import torch.nn as nn
from config import *
from .utils import *
from ESRGAN.model import Generator

class Net(nn.Module):
    '''
    Generates the NeRF neural network
    Args:
        num_layers: The number of MLP layers
        num_pos: The number of dimensions of positional encoding
    '''
    def __init__(self, num_layers, rand=True):
        super().__init__()
        self.input_layer = nn.Sequential(nn.Linear(99, 64), nn.ReLU())
        self.dense_layers = nn.Sequential(nn.Linear(64, 64), nn.ReLU())
        self.regulate_layers = nn.Sequential(nn.Linear(99+64, 64), nn.ReLU())
        self.output_layer = nn.Linear(64, 4)
        self.num_layers = num_layers
        self.layers = [64] * 8
        self.super_resolution = Generator(3, filters=64, num_res_blocks=4)
        # self.depth_super_resolution = Generator(1, filters=64, num_res_blocks=8)
        self.rand = rand

    def forward(self, rays_flat, t_vals):
        out = self.input_layer(rays_flat)
        for i in range(self.num_layers):
            out = self.dense_layers(out)
            if i % 4 == 0 and i > 0:
                out = torch.cat([out, rays_flat], dim=-1)
                out = self.regulate_layers(out)
        out = self.output_layer(out)
        out = out.reshape((training_facts['batch_size'], eval(dataset_facts['image']['lr_shape'])[1], eval(dataset_facts['image']['lr_shape'])[2], nerf_facts['N_samples'], dataset_facts['image']['channels']+1))
        rgb = torch.sigmoid(out[..., :-1])
        sigma_a = nn.ReLU()(out[..., -1])


        delta = t_vals[..., 1:] - t_vals[..., :-1]
        if self.rand:
            delta = torch.cat([delta, torch.broadcast_to(torch.tensor([1e10]), (training_facts['batch_size'], eval(dataset_facts['image']['lr_shape'])[1], eval(dataset_facts['image']['lr_shape'])[2], 1)).to(device)],
                              axis=-1)
            alpha = 1.0 - torch.exp(-sigma_a * delta)
        else:
            delta = torch.cat([delta, torch.broadcast_to(torch.tensor([1e10]), (training_facts['batch_size'], 1)).to(device)], axis=-1)
            alpha = 1.0 - torch.exp(-sigma_a * delta[:, None, None, :])

        exp_term = 1.0 - alpha
        epsilon = 1e-10
        transmittance = torch.cumprod(exp_term + epsilon, -1)
        weights = alpha * transmittance
        rgb = torch.sum(weights[..., None] * rgb, axis=-2)

        if self.rand:
            depth_map = torch.sum(weights * t_vals, axis=-1)
        else:
            depth_map = torch.sum(weights * t_vals[:, None, None], axis=-1)
        rgb = rgb.permute(0, 3, 2, 1)
        rgb = self.super_resolution(rgb)
        # depth_map = self.depth_super_resolution(depth_map)
        return rgb, depth_map

