import os
import torch
import numpy as np
from nerf import NeRF, generate_rays, render_rays
from data import load_poses, load_intrinsic
from utils import set_seed
from PIL import Image
import imageio


def get_pose(theta, phi, r):
    def R_phi(phi):
        return np.array([
            [np.cos(phi), -np.sin(phi), 0., 0.],
            [np.sin(phi), np.cos(phi), 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.],
        ])
    def R_theta(theta):
        return np.array([
            [1., 0., 0., 0.],
            [0., np.cos(theta), -np.sin(theta), 0.],
            [0., np.sin(theta), np.cos(theta), 0.],
            [0., 0., 0., 1.],
        ])
    pose = np.array([
        [1., 0., 0., 0.],
        [0., -1., 0., 0.],
        [0., 0., -1., r],
        [0., 0., 0., 1.]
    ])
    pose[2, -1] = r
    pose = R_theta(theta) @ pose
    pose = R_phi(phi) @ pose
    return pose


if __name__ == "__main__":
    set_seed(7)
    data_dir = "bottles"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device="cpu"
    ckpt = torch.load("exp_out/v17_2nets_2_no_pi_bound=1_5_Nf=128_batchsz=4096chunk_lrdecay0.996_shuffle_white2/checkpoints/ckpt.790000.pth", map_location=device)
    net_coarse = NeRF().to(device)
    net_fine = NeRF().to(device)
    net_coarse.load_state_dict(ckpt['model_state_dict']['net_coarse'])
    net_fine.load_state_dict(ckpt['model_state_dict']['net_fine'])
    del ckpt

    theta = np.pi / 3
    test_poses = [get_pose(theta, phi, 3.) for phi in np.linspace(0., 2 * np.pi, 60 + 1)[:-1]]
    print(len(test_poses))

    # load intrinsic matrix
    # intrinsic = load_intrinsic(os.path.join(data_dir, "intrinsics.txt"))
    intrinsic = np.array([
        [875., 0., 400.],
        [0., 875., 400.],
        [0., 0., 1.]
    ])

    H, W = 800, 800

    s = 2
    H /= s
    W /= s
    intrinsic[:2, :] /= s

    frames = []
    for test_i, test_pose in enumerate(test_poses):
        print("pose", test_i)
        test_rays_o, test_rays_d = generate_rays(H, W, intrinsic, test_pose)
        test_rays_o = torch.tensor(test_rays_o, device=device)
        test_rays_d = torch.tensor(test_rays_d, device=device)

        rgb_pixels, depth_pixels = [], []
        for j in range(test_rays_o.shape[0]):
            # print(j)
            rays_od = (test_rays_o[j], test_rays_d[j])
            rgb_, depth_, _, _ = render_rays(net_coarse, net_fine, rays_od, bound=(1., 5.), N_samples=(64, 128), device=device,
                                       eval=True)
            # print("rgb", rgb.shape)
            # print("depth", depth.shape)
            rgb_pixels.append(rgb_.unsqueeze(0).cpu().detach())
            depth_pixels.append(depth_.unsqueeze(0).cpu().detach())
        rgb = torch.cat(rgb_pixels, dim=0)
        depth = torch.cat(depth_pixels, dim=0)
        # test_rgb_list.append(rgb)

        img = rgb.cpu().detach().numpy()
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)

        frames.append(img)

    imageio.mimwrite("render_360.mp4", frames, fps=30, quality=7)