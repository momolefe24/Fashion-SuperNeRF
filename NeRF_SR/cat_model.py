# One super-resolution network that concats the conv(3, 3) with nerf output and performs further super-resolution

import torch
import torch.nn as nn
from config import *
from .utils import *
import torch.nn.functional as F
import torchvision.models as models
from torch import Tensor
from ESRGAN.model import Generator



class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_act, **kwargs):
        super().__init__()
        self.cnn = nn.Conv2d(
            in_channels,
            out_channels,
            **kwargs,
            bias=True,
        )
        self.act = nn.LeakyReLU(0.2, inplace=True) if use_act else nn.Identity()

    def forward(self, x):
        return self.act(self.cnn(x))


class UpsampleBlock(nn.Module):
    def __init__(self, in_c, scale_factor=2):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode="nearest")
        self.conv = nn.Conv2d(in_c, in_c, 3, 1, 1, bias=True)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.act(self.conv(self.upsample(x)))


class DenseResidualBlock(nn.Module):
    def __init__(self, in_channels, channels=32, residual_beta=0.2):
        super().__init__()
        self.residual_beta = residual_beta
        self.blocks = nn.ModuleList()

        for i in range(5):
            self.blocks.append(
                ConvBlock(
                    in_channels + channels * i,
                    channels if i <= 3 else in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    use_act=True if i <= 3 else False,
                )
            )

    def forward(self, x):
        new_inputs = x
        for block in self.blocks:
            out = block(new_inputs)
            new_inputs = torch.cat([new_inputs, out], dim=1)
        return self.residual_beta * out + x


class RRDB(nn.Module):
    def __init__(self, in_channels, residual_beta=0.2):
        super().__init__()
        self.residual_beta = residual_beta
        self.rrdb = nn.Sequential(*[DenseResidualBlock(in_channels) for _ in range(3)])

    def forward(self, x):
        return self.rrdb(x) * self.residual_beta + x


class SuperNeRF(nn.Module):
    def __init__(self, in_channels=3, num_channels=64, num_blocks=23, num_layers=8):
        super().__init__()
        self.nerf = NerF(num_layers).to(device)
        self.initial = nn.Conv2d(
            in_channels,
            num_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )
        self.residuals = nn.Sequential(*[RRDB(num_channels) for _ in range(num_blocks)])
        self.conv = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1)
        self.upsamples = nn.Sequential(
            *[UpsampleBlock(num_channels) for _ in range(3)]
        )
        self.after_cat = nn.Conv2d(
            68,
            num_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )
        self.final = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_channels, in_channels, 3, 1, 1, bias=True),
        )

    def forward(self, x, rays_flat, t_vals, out_shape=None, mode="train"):
        nerf_output, depth_map = self.nerf(rays_flat, t_vals, mode=mode)
        depth_map = depth_map.unsqueeze(dim=1)
        if out_shape is not None:
            x = F.interpolate(x, size=(out_shape[0]//esrgan_facts['upscaling_factor'], out_shape[1] // esrgan_facts['upscaling_factor']))
            nerf_output = F.interpolate(nerf_output, size=(out_shape[0]//esrgan_facts['upscaling_factor'], out_shape[1] // esrgan_facts['upscaling_factor']))
            depth_map = F.interpolate(depth_map, size=(out_shape[0]//esrgan_facts['upscaling_factor'], out_shape[1] // esrgan_facts['upscaling_factor']))
        initial = self.initial(x)
        x = torch.cat((nerf_output, initial, depth_map), dim=1)
        x = self.after_cat(x)
        x = self.conv(self.residuals(x)) + x
        x = self.upsamples(x)
        if experiment_facts['joint_loss']:
            return self.final(x), nerf_output
        else:
            return self.final(x)


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 64, 64, 128, 128, 128, 256, 256, 512, 512]):
        super().__init__()
        blocks = []
        for idx, feature in enumerate(features):
            blocks.append(
                ConvBlock(
                    in_channels,
                    feature,
                    kernel_size=3,
                    stride=1 + idx % 2,
                    padding=1,
                    use_act=True,
                ),
            )
            in_channels = feature

        self.blocks = nn.Sequential(*blocks)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            nn.Linear(512 * 8 * 8, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
        )

    def forward(self, x):
        x = self.blocks(x)
        return self.classifier(x)

def initialize_weights(model, scale=0.1):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight.data)
            m.weight.data *= scale

        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight.data)
            m.weight.data *= scale

class ContentLoss(nn.Module):
    """Constructs a content loss function based on the VGG19 network.
    Using high-level feature mapping layers from the latter layers will focus more on the texture content of the image.
    Paper reference list:
        -`Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network <https://arxiv.org/pdf/1609.04802.pdf>` paper.
        -`ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks                    <https://arxiv.org/pdf/1809.00219.pdf>` paper.
        -`Perceptual Extreme Super Resolution Network with Receptive Field Block               <https://arxiv.org/pdf/2005.12597.pdf>` paper.
    """

    def __init__(self):
        super(ContentLoss, self).__init__()
        # Load the VGG19 model trained on the ImageNet dataset.
        vgg19 = models.vgg19(pretrained=True, num_classes=1000).eval()
        # Extract the output of the thirty-fifth layer in the VGG19 model as the content loss.
        self.feature_extractor = nn.Sequential(*list(vgg19.features.children())[:35])
        # Freeze model parameters.
        for parameters in self.feature_extractor.parameters():
            parameters.requires_grad = False

        # The preprocessing method of the input data. This is the preprocessing method of the VGG model on the ImageNet dataset.
        self.register_buffer("mean", torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, sr: Tensor, hr: Tensor) -> Tensor:
        # Standardized operations.
        sr = (sr - self.mean) / self.std
        hr = (hr - self.mean) / self.std
        # Find the feature map difference between the two images.
        loss = F.l1_loss(self.feature_extractor(sr), self.feature_extractor(hr))

        return loss

class NerF(nn.Module):
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
        # self.depth_super_resolution = Generator(1, filters=64, num_res_blocks=8)
        self.rand = rand

    def forward(self, rays_flat, t_vals, mode='train'):
        out = self.input_layer(rays_flat)
        for i in range(self.num_layers):
            out = self.dense_layers(out)
            if i % 4 == 0 and i > 0:
                out = torch.cat([out, rays_flat], dim=-1)
                out = self.regulate_layers(out)
        out = self.output_layer(out)
        lr_shape = eval(dataset_facts['image']['lr_shape_crop']) if mode == "train" else eval(dataset_facts['image']['lr_shape'])
        if mode == 'train':
            out = out.reshape((training_facts['batch_size'], lr_shape[1], lr_shape[2], nerf_facts['N_samples'], dataset_facts['image']['channels']+1))
        else:
            out = out.reshape((training_facts['batch_size'], lr_shape[1], lr_shape[2], nerf_facts['N_samples'], dataset_facts['image']['channels']+1))
        rgb = torch.sigmoid(out[..., :-1])
        sigma_a = nn.ReLU()(out[..., -1])


        delta = t_vals[..., 1:] - t_vals[..., :-1]
        if self.rand:
            delta = torch.cat([delta, torch.broadcast_to(torch.tensor([1e10]), (training_facts['batch_size'], lr_shape[1], lr_shape[2], 1)).to(device)],
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
        nerf_output = rgb.permute(0, 3, 2, 1)
        return nerf_output, depth_map

