import torch
from config import *
import numpy as np
from .embedding import get_embeddings

device = 'cuda' if torch.cuda.is_available() else "cpu"


def get_rays(H, W, K, c2w):
    i, j = torch.meshgrid(torch.linspace(0, W - 1, W),
                          torch.linspace(0, H - 1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i - K[0][2]) / K[0][0], -(j - K[1][2]) / K[1][1], -torch.ones_like(i)], -1)
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    # Projection
    o0 = -1. / (W / (2. * focal)) * rays_o[..., 0] / rays_o[..., 2]
    o1 = -1. / (H / (2. * focal)) * rays_o[..., 1] / rays_o[..., 2]
    o2 = 1. + 2. * near / rays_o[..., 2]

    d0 = -1. / (W / (2. * focal)) * (rays_d[..., 0] / rays_d[..., 2] - rays_o[..., 0] / rays_o[..., 2])
    d1 = -1. / (H / (2. * focal)) * (rays_d[..., 1] / rays_d[..., 2] - rays_o[..., 1] / rays_o[..., 2])
    d2 = -2. * near / rays_o[..., 2]

    rays_o = torch.stack([o0, o1, o2], -1)
    rays_d = torch.stack([d0, d1, d2], -1)

    return rays_o, rays_d


def get_image_plane():
    dH = int(H // 2 * nerf_hyperparameters['precrop_factor'])
    dW = int(W // 2 * nerf_hyperparameters['precrop_factor'])
    if nerf_hyperparameters['precrop_iters']:
        coords = torch.stack(torch.meshgrid(torch.linspace(H // 2 - dH, H // 2 + dH - 1, 2 * dH),
                                            torch.linspace(W // 2 - dW, W // 2 + dW - 1, 2 * dW)), -1)
    else:
        coords = torch.stack(torch.meshgrid(torch.linspace(0, H - 1, H), torch.linspace(0, W - 1, W)), -1)  # (H, W, 2)
    return coords


def get_batch_rays(target, pose, K):
    # Shoot rays
    rays_o, rays_d = get_rays(H, W, K, c2w=pose)
    coords = get_image_plane()

    coords = torch.reshape(coords, [-1, 2])  # (H * W, 2)
    select_inds = np.random.choice(coords.shape[0], size=[nerf_hyperparameters['random_samples']], replace=False)
    select_coords = coords[select_inds].long()
    rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]
    rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]
    batch_rays = torch.stack([rays_o, rays_d], 0)
    target_s = target[select_coords[:, 0], select_coords[:, 1]]

    return batch_rays, target_s


def render(H, W, K, rays=None, c2w=None, near=2., far=6., use_viewdirs=None,
           ndc=True):  # 2 - return predictions, z_vals, rays_d
    if c2w is not None:
        rays_o, rays_d = get_rays(H, W, K, c2w)
    else:
        rays_o, rays_d = rays
    if use_viewdirs:
        viewdirs = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1, 3]).float()
    if ndc:
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1, 3]).float()
    rays_d = torch.reshape(rays_d, [-1, 3]).float()

    near, far = near * torch.ones_like(rays_d[..., :1]), far * torch.ones_like(rays_d[..., :1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)
    return rays


def render_rays(batch_rays):  # rays_flat = batch_rays
    num_rays = batch_rays.shape[0]
    rays_o, rays_d = batch_rays[:, 0:3], batch_rays[:, 3:6]
    viewdirs = batch_rays[:, -3:] if batch_rays.shape[-1] > 8 else None
    bounds = torch.reshape(batch_rays[..., 6:8], [-1, 1, 2])
    near, far = bounds[..., 0], bounds[..., 1]  # [-1,1]
    t_vals = torch.linspace(0., 1., steps=nerf_hyperparameters['num_samples'])

    # Z_vals
    z_vals = near * (1. - t_vals) + far * (t_vals)
    z_vals = z_vals.expand([num_rays, nerf_hyperparameters['num_samples']])
    z_vals = get_intervals_between_samples(z_vals) if nerf_hyperparameters['perturb'] == 1.0 else z_vals
    if nerf_hyperparameters['perturb'] > 0:
        # get intervals between samples
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape)
        z_vals = lower + (upper - lower) * t_rand
    sample_points = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
    return sample_points, viewdirs, z_vals, rays_o, rays_d


# Volume Rendering
def volume_rendering(predictions, z_vals, rays_d):
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1. - torch.exp(-act_fn(raw) * dists)
    raw_noise_std = nerf_hyperparameters['raw_noise_std']
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = dists.to(device)
    dists = torch.cat([dists, torch.Tensor([1e10]).to(device).expand(dists[..., :1].to(device).shape)], -1).to(
        device)  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)
    rgb = torch.sigmoid(predictions[..., :3])

    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(predictions[..., 3].shape) * raw_noise_std
        noise = noise.to(device)
    alpha = raw2alpha(predictions[..., 3] + noise, dists).to(device)  # [N_rays, N_samples]
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)).to(device), 1. - alpha + 1e-10], -1),
                                    -1)[:, :-1]
    rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    if nerf_data['white_background']:
        rgb_map = rgb_map + (1. - acc_map[..., None])

    return rgb_map, disp_map, acc_map, weights, depth_map


def hierarchical_sampling(bins, weights, N_samples, det=False):
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    # Invert CDF
    u = u.contiguous().to(device)
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


def resample_points(weights, z_vals, rays_o, rays_d):
    N_importance = nerf_hyperparameters['importance']
    perturb = nerf_hyperparameters['perturb']
    z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
    z_samples = hierarchical_sampling(z_vals_mid, weights[..., 1:-1], N_importance, det=(perturb == 0.))
    z_samples = z_samples.detach()

    z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [N_rays, N_samples + N_importance, 3]
    return pts, z_vals, z_samples

def get_intervals_between_samples(z_vals):
    mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
    upper = torch.cat([mids, z_vals[..., -1:]], -1)
    lower = torch.cat([z_vals[..., :1], mids], -1)

    # stratified samples in those intervals
    t_rand = torch.rand(z_vals.shape)
    z_vals = lower + (upper - lower) * t_rand
    return z_vals
