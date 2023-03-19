from .rendering import *


class NeRF(nn.Module):

    def __init__(self, K):
        super(NeRF, self).__init__()
        self.input_channels = self.get_channels(nerf_embedder['multi_res'])
        self.input_channel_views = self.get_channels(nerf_embedder['multi_res_views'])
        self.output_channel = nerf_model['output_channel']
        self.H, self.W = eval(nerf_data['image_shape'])
        self.K = K
        self.use_viewdirs = nerf_data['use_viewdirs']
        self.skips = [4]
        W = 256
        sample_linear = [nn.Linear(self.input_channels, W)] + [
            nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.input_channels, W) for i in range(8 - 1)]
        self.pts_linears = nn.ModuleList(sample_linear)
        self.views_linears = nn.ModuleList([nn.Linear(self.input_channel_views + W, W // 2)])
        if self.use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W // 2, 3)
        else:
            self.output_linear = nn.Linear(W, self.output_channel)

    def get_channels(self, multi_res=nerf_embedder['multi_res'], num_period_dims=2):
        input_channels = nerf_embedder['input_dim']
        return input_channels * multi_res * num_period_dims + input_channels

    def preprocess(self, pose, target=None, eval=True):
        near = nerf_hyperparameters['near']
        far = nerf_hyperparameters['far']
        if not eval:
            batch_rays, target_s = get_batch_rays(target, pose, self.K)
            rays = render(H, W, self.K, rays=batch_rays, use_viewdirs=True, ndc=False)  # post-processing
        else:
            rays = render(H, W, self.K, c2w=pose, near=near, far=far, use_viewdirs=True, ndc=False)
            target_s = None
        sample_points, viewdirs, z_vals, rays_o, rays_d = render_rays(rays)
        embedded = get_embeddings(sample_points, viewdirs)
        return embedded, sample_points, viewdirs, z_vals, rays_o, rays_d, target_s

    def forward(self, pose, target=None, eval=True):  # input is target & pose
        x, sample_points, viewdirs, z_vals, rays_o, rays_d, target_s = self.preprocess(pose, target,
                                                                                       eval)  # x = embedded
        x, sample_points = x.to(device), sample_points.to(device)
        z_vals, rays_o = z_vals.to(device), rays_o.to(device)
        rays_d = rays_d.to(device)
        input_pts, input_views = torch.split(x, [self.input_channels, self.input_channel_views], dim=-1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)
        outputs = torch.reshape(outputs, list(sample_points.shape[:-1]) + [outputs.shape[-1]]).to(device)
        rgb_map, disp_map, acc_map, weights, depth_map = volume_rendering(outputs, z_vals, rays_d)
        if eval:
            rgb_map = torch.reshape(rgb_map, [100, 100, 3])
            disp_map = torch.reshape(disp_map, [100, 100])
            acc_map = torch.reshape(acc_map, [100, 100])

        fine_parameters = {"viewdirs": viewdirs, "weights": weights, "z_vals": z_vals, "rays_o": rays_o,
                           "rays_d": rays_d}
        return target_s, rgb_map, disp_map, acc_map, weights, depth_map, fine_parameters


class NeRF_Fine(nn.Module):

    def __init__(self, K):
        super(NeRF_Fine, self).__init__()
        self.input_channels = self.get_channels(nerf_embedder['multi_res'])
        self.input_channel_views = self.get_channels(nerf_embedder['multi_res_views'])
        self.output_channel = nerf_model['output_channel']
        self.H, self.W = eval(nerf_data['image_shape'])
        self.K = K
        self.use_viewdirs = nerf_data['use_viewdirs']
        self.skips = [4]
        W = 256
        sample_linear = [nn.Linear(self.input_channels, W)] + [
            nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.input_channels, W) for i in range(8 - 1)]
        self.pts_linears = nn.ModuleList(sample_linear)
        self.views_linears = nn.ModuleList([nn.Linear(self.input_channel_views + W, W // 2)])
        if self.use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W // 2, 3)
        else:
            self.output_linear = nn.Linear(W, self.output_channel)

    def get_channels(self, multi_res=nerf_embedder['multi_res'], num_period_dims=2):
        input_channels = nerf_embedder['input_dim']
        return input_channels * multi_res * num_period_dims + input_channels

    def preprocess(self, viewdirs, weights, z_vals, rays_o, rays_d):
        resampled_points, z_vals, z_samples = resample_points(weights, z_vals, rays_o, rays_d)
        z_std = torch.std(z_samples, dim=-1, unbiased=False)
        resampled_embedded = get_embeddings(resampled_points, viewdirs)
        return resampled_embedded, resampled_points, z_vals, z_std, rays_o, rays_d

    def forward(self, viewdirs, weights, z_vals, rays_o, rays_d, eval=True):  # input is target & pose
        x, resampled_points, z_vals, z_std, rays_o, rays_d = self.preprocess(viewdirs, weights, z_vals, rays_o,
                                                                             rays_d)
        x, resampled_points, z_vals = x.to(device), resampled_points.to(device), z_vals.to(z_vals)
        z_std, rays_o, rays_d = z_std.to(device), rays_o.to(device), rays_d.to(device)
        input_pts, input_views = torch.split(x, [self.input_channels, self.input_channel_views], dim=-1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)
        outputs = torch.reshape(outputs, list(resampled_points.shape[:-1]) + [outputs.shape[-1]]).to(device)
        fine_rgb_map, fine_disp_map, fine_acc_map, fine_weights, fine_depth_map = volume_rendering(outputs, z_vals,
                                                                                                   rays_d)
        if eval:
            fine_rgb_map = torch.reshape(fine_rgb_map, [100, 100, 3])
            fine_disp_map = torch.reshape(fine_disp_map, [100, 100])
            fine_acc_map = torch.reshape(fine_acc_map, [100, 100])
        return fine_rgb_map, fine_disp_map, fine_acc_map, fine_weights, fine_depth_map
