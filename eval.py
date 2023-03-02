import os
import torch
import numpy as np
from nerf import NeRF, generate_rays, render_rays
from data import load_poses, load_intrinsic
from utils import set_seed
from PIL import Image


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

    test_ids = [0, 16, 55, 93, 160]

    # load test data
    test_pose_files, test_names = [], []
    if test_ids is not None:
        for test_i in test_ids:
            test_name = f"2_test_{test_i:04}"
            test_names.append(test_name)
            test_pose_files.append((os.path.join(data_dir, "pose", f"{test_name}.txt")))
    test_poses = load_poses(test_pose_files)

    # load intrinsic matrix
    intrinsic = load_intrinsic(os.path.join(data_dir, "intrinsics.txt"))

    H, W = 800, 800
    for test_i, test_pose in enumerate(test_poses):
        print(test_i)
        test_rays_o, test_rays_d = generate_rays(H, W, intrinsic, test_pose)
        test_rays_o = torch.tensor(test_rays_o, device=device)
        test_rays_d = torch.tensor(test_rays_d, device=device)

        rgb_pixels, depth_pixels = [], []
        for j in range(test_rays_o.shape[0]):
            print(j)
            rays_od = (test_rays_o[j], test_rays_d[j])
            rgb_, depth_, _, _ = render_rays(net_coarse, net_fine, rays_od, bound=(1., 5.), N_samples=(64, 128), device=device,
                                       eval=True)
            # print("rgb", rgb.shape)
            # print("depth", depth.shape)
            rgb_pixels.append(rgb_.unsqueeze(0))
            depth_pixels.append(depth_.unsqueeze(0))
        rgb = torch.cat(rgb_pixels, dim=0)
        depth = torch.cat(depth_pixels, dim=0)
        # test_rgb_list.append(rgb)

        img = rgb.cpu().detach().numpy()
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        img = Image.fromarray(img)
        img.save(os.path.join("exp_out/eval", f"rgb_{test_names[test_i]}.png"))