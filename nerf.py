# references:
# https://github.com/yenchenlin/nerf-pytorch
# https://colab.research.google.com/drive/1_51bC5d6m7EFU6U_kkUL2lMYehJqc01R?usp=sharing

import os
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm


class NeRF(nn.Module):
    def __init__(self, D=8, W=256, Lx=10, Ld=4, skip=4, use_pos_encoding=True):
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.Lx = Lx
        self.Ld = Ld
        self.skip = skip
        self.use_pos_encoding = use_pos_encoding

        self.x_dim = self.Lx * 6 if use_pos_encoding else 3
        self.d_dim = self.Ld * 6 if use_pos_encoding else 3

        self.mlp = nn.ModuleList([nn.Linear(self.x_dim, W)])
        for i in range(D - 1):
            if i == skip:
                self.mlp.append(nn.Linear(W + self.x_dim, W))
            else:
                self.mlp.append(nn.Linear(W, W))

        self.sigma_linear = nn.Linear(W, 1)
        self.feature_linear = nn.Linear(W, W)
        self.proj = nn.Linear(W + self.d_dim, W // 2)
        self.rgb_linear = nn.Linear(W // 2, 3)

    def forward(self, input_x, input_d):
        if self.use_pos_encoding:
            x_enc = self.pos_encoding(input_x, L=self.Lx)   # 60d
            d_enc = self.pos_encoding(input_d, L=self.Ld)   # 24d
        else:
            x_enc = input_x.clone()
            d_enc = input_d.clone()
        h = x_enc.clone()
        for i, _ in enumerate(self.mlp):
            h = F.relu(self.mlp[i](h))
            if i == self.skip:
                h = torch.cat([x_enc, h], -1)

        sigma = F.relu(self.sigma_linear(h))
        feature = self.feature_linear(h)

        h = torch.cat([feature, d_enc], -1)

        h = F.relu(self.proj(h))
        rgb = torch.sigmoid(self.rgb_linear(h))

        return rgb, sigma

    @staticmethod
    def pos_encoding(x, L):
        enc = []
        for i in range(L):
            for fn in [torch.sin, torch.cos]:
                enc.append(fn(2. ** i * x))
        return torch.cat(enc, -1)


def generate_rays(H, W, intrinsic, pose):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack(
        [(i - intrinsic[0][2]) / intrinsic[0][0], (j - intrinsic[1][2]) / intrinsic[1][1], np.ones_like(i)], -1)
    rays_d = dirs @ pose[:3, :3].T

    rays_o = np.broadcast_to(pose[:3, -1], np.shape(rays_d))
    return rays_o.astype(np.float32), rays_d.astype(np.float32)


def sample_coarse(t_near, t_far, n_samples, device):
    uniform_ts = torch.linspace(0., 1., n_samples + 1, dtype=torch.float32, device=device)[:-1]
    rand_offsets = torch.rand([n_samples], dtype=torch.float32, device=device) / n_samples
    ts = uniform_ts + rand_offsets
    return t_near + (t_far - t_near) * ts


def sample_fine(bins, weights, n_samples, device):
    pdf = F.normalize(weights, p=1, dim=-1)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)

    u = torch.rand(list(cdf.shape[:-1]) + [n_samples], device=device).contiguous()
    ids = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(ids - 1, device=device), ids - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(ids, device=device), ids)
    ids_g = torch.stack([below, above], -1)     # (batch, n_samples, 2)

    matched_shape = [ids_g.shape[0], ids_g.shape[1], cdf.shape[-1]]
    cdf_val = torch.gather(cdf.unsqueeze(1).expand(matched_shape), -1, ids_g)
    bins_val = torch.gather(bins[None, None, :].expand(matched_shape), -1, ids_g)

    cdf_d = (cdf_val[..., 1] - cdf_val[..., 0])
    cdf_d = torch.where(cdf_d < 1e-5, torch.ones_like(cdf_d, device=device), cdf_d)
    t = (u - cdf_val[..., 0]) / cdf_d
    samples = bins_val[..., 0] + t * (bins_val[..., 1] - bins_val[..., 0])

    return samples


def get_rgb_and_weights(net, pts, rays_d, z_vals, device):
    # pts: (N, N_coarse, 3)
    # rays_d: (N, 3)

    batch_sz, n_samples, _ = pts.shape
    # print(batch_sz, n_samples)

    pts_flat = pts.view(-1, 3)
    dir_flat = F.normalize(torch.reshape(rays_d.unsqueeze(-2).expand_as(pts), [-1, 3]), dim=-1)

    rgb, sigma = net(pts_flat, dir_flat)
    rgb = rgb.view(batch_sz, n_samples, 3)
    sigma = sigma.view(batch_sz, n_samples)

    delta = z_vals[..., 1:] - z_vals[..., :-1]
    delta = torch.cat([delta, torch.ones(delta[..., :1].shape, device=device) * 1e10], -1)
    delta = delta * torch.norm(rays_d, dim=-1, keepdim=True)

    alpha = 1. - torch.exp(-sigma * delta)      # (batch_sz, n_samples)
    weights = alpha * torch.cumprod(
        torch.cat([torch.ones((batch_sz, 1), device=device), 1. - alpha], dim=-1),
        dim=-1
    )[..., :-1]

    return rgb, weights


def render_rays(net_coarse, net_fine, rays, bound, N_samples, device, eval=False):
    rays_o, rays_d = rays       # (N, 3)
    batch_sz = rays_o.shape[0]  # N
    near, far = bound
    N_coarse, N_fine = N_samples
    t_vals_coarse = sample_coarse(near, far, N_coarse, device)  # (N_coarse,)
    pts_coarse = rays_o[..., None, :] + rays_d[..., None, :] * t_vals_coarse[..., None]     # (N, N_coarse, 3)

    if eval:
        # do not need to generate coarse map at test time
        rgb_map_coarse, depth_map_coarse, acc_map_coarse = None, None, None
    else:
        rgb_coarse, weights_coarse = get_rgb_and_weights(net_coarse, pts_coarse, rays_d, t_vals_coarse, device)
        rgb_map_coarse = torch.sum(weights_coarse[..., None] * rgb_coarse, dim=-2)
        depth_map_coarse = torch.sum(weights_coarse * t_vals_coarse, -1)
        acc_map_coarse = torch.sum(weights_coarse, -1)
        rgb_map_coarse = rgb_map_coarse + (1. - acc_map_coarse[..., None])  # white background

    if N_fine > 0:
        with torch.no_grad():
            rgb, weights = get_rgb_and_weights(net_coarse, pts_coarse, rays_d, t_vals_coarse, device)
            t_vals_mid = .5 * (t_vals_coarse[..., 1:] + t_vals_coarse[..., :-1])
            t_vals_fine = sample_fine(t_vals_mid, weights[..., 1:-1], N_fine, device)

        t_vals_coarse = t_vals_coarse.unsqueeze(0).expand([batch_sz, N_coarse])
        t_vals, _ = torch.sort(torch.cat([t_vals_coarse, t_vals_fine], dim=-1), dim=-1)
        pts = rays_o[..., None, :] + rays_d[..., None, :] * t_vals[..., None]
        net = net_fine
    else:
        t_vals = t_vals_coarse
        pts = pts_coarse
        net = net_coarse
    rgb, weights = get_rgb_and_weights(net, pts, rays_d, t_vals, device)

    rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)
    depth_map = torch.sum(weights * t_vals, -1)
    acc_map = torch.sum(weights, -1)
    rgb_map = rgb_map + (1. - acc_map[..., None])   # white background

    return rgb_map, depth_map, rgb_map_coarse, depth_map_coarse
