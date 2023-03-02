import os
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import argparse
from torch.utils.tensorboard import SummaryWriter

from utils import set_seed
from data import load_data
from nerf import *


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7, help="random seed")
    parser.add_argument("--n_epochs", type=int, default=200, help="total number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4096, help="batch size of rays")
    parser.add_argument("--init_lr", type=float, default=5e-4, help="initial learning rate")
    parser.add_argument("--lr_decay", type=float, default=0.996, help="learning rate exponential decay rate")
    parser.add_argument("--train_ratio", type=float, default=0.95, help="ratio for training data")
    parser.add_argument("--img_size", type=int, default=-1, help="image size for resizing")
    parser.add_argument("--data_dir", type=str, default="./bottles", help="data directory")
    parser.add_argument("--output_dir", type=str, default="./exp_out", help="directory for saving training output")
    parser.add_argument("--exp_id", type=str, default="run", help="identifier for a experiment")
    parser.add_argument("--eval_freq", type=int, default=10000, help="evaluate model every 'eval_freq' iterations")
    parser.add_argument("--save_freq", type=int, default=10000, help="save checkpoints 'save_freq' iterations")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_dir = os.path.join(args.output_dir, args.exp_id)
    exp_ckpt_dir = os.path.join(exp_dir, "checkpoints")
    exp_vis_dir = os.path.join(exp_dir, "visualization")
    exp_log_dir = os.path.join(exp_dir, "log")
    os.makedirs(exp_ckpt_dir)
    os.makedirs(exp_vis_dir)
    os.makedirs(exp_log_dir)
    writer = SummaryWriter(log_dir=exp_log_dir)

    set_seed(args.seed)
    img_size = None
    if args.img_size > 0:
        img_size = (args.img_size, args.img_size)

    test_ids = [0, 16, 55, 93, 160]

    data = load_data(args.data_dir, test_ids=test_ids, img_size=img_size, train_ratio=args.train_ratio)
    poses_train = data["poses_train"]
    poses_val = data["poses_val"]
    rgbs_train = data["rgbs_train"]
    rgbs_val = data["rgbs_val"]
    intrinsic = data["intrinsic"]
    H, W = rgbs_train.shape[1:3]

    # training rays
    rays_o_list, rays_d_list, rays_rgb_list = [], [], []

    print("Generating training rays...")
    for pose, rgb in tqdm(zip(poses_train, rgbs_train), total=poses_train.shape[0]):
        rays_o, rays_d = generate_rays(H, W, intrinsic, pose)

        rays_o_list.append(rays_o.reshape(-1, 3))
        rays_d_list.append(rays_d.reshape(-1, 3))
        rays_rgb_list.append(rgb.reshape(-1, 3))

    rays_o_np = np.concatenate(rays_o_list, axis=0)        # (N, 3)
    rays_d_np = np.concatenate(rays_d_list, axis=0)        # (N, 3)
    rays_rgb_np = np.concatenate(rays_rgb_list, axis=0)    # (N, 3)
    rays = torch.tensor(np.concatenate([rays_o_np, rays_d_np, rays_rgb_np], axis=1),) # device=device)  # (N, 9)

    perm = torch.randperm(rays.shape[0])
    rays = rays[perm, :]

    N = rays.shape[0]
    n_iters_per_epoch = N // args.batch_size

    bound = (1., 5.)
    N_samples = (64, 128)

    net_coarse = NeRF().to(device)
    net_fine = NeRF().to(device)

    #
    # ckpt = torch.load(
    #     "exp_out/v17_2nets_2_no_pi_bound=1_5_Nf=128_batchsz=4096chunk_lrdecay0.996_shuffle_white2/checkpoints/ckpt.790000.pth",
    #     # "exp_out/v14_2nets_2_no_pi_bound=1_5_Nf=128_batchsz=2048_lrdecay0.995_shuffle_white2/checkpoints/ckpt.800000.pth",
    #     map_location=device)
    # net_coarse = NeRF().to(device)
    # net_fine = NeRF().to(device)
    # net_coarse.load_state_dict(ckpt['model_state_dict']['net_coarse'])
    # net_fine.load_state_dict(ckpt['model_state_dict']['net_fine'])
    # del ckpt


    optimizer = torch.optim.Adam(
        params=list(net_coarse.parameters()) + list(net_fine.parameters()),
        lr=args.init_lr
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay)
    mse = torch.nn.MSELoss()

    print("Training starts!")
    iter_cnt = 0
    for ep_i in range(args.n_epochs):
        print(f"[Epoch {ep_i + 1}]")
        rays = rays[torch.randperm(N), :]
        train_iter = iter(torch.split(rays, args.batch_size, dim=0))

        for i in tqdm(range(n_iters_per_epoch)):
            iter_cnt += 1
            train_rays = next(train_iter)
            assert train_rays.shape == (args.batch_size, 9)

            losses = []
            train_rays_chunks = torch.tensor_split(train_rays, 1, dim=0)    # > 1 to split into smaller trunks to fit in CUDA memory
            for train_rays_chunk in train_rays_chunks:
                rays_o, rays_d, target_rgb = torch.chunk(train_rays_chunk.to(device), 3, dim=-1)
                rays_od = (rays_o, rays_d)
                rgb, depth, rgb_coarse, depth_coarse = render_rays(net_coarse, net_fine, rays_od, bound=bound, N_samples=N_samples, device=device)

                loss = mse(rgb, target_rgb) + mse(rgb_coarse, target_rgb)
                losses.append(loss)
            total_loss = torch.mean(torch.stack(losses))
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            writer.add_scalar('train/loss', total_loss.item(), iter_cnt)
            if iter_cnt % 1000 == 0:
                scheduler.step()
                writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], iter_cnt)

            with torch.no_grad():

                if iter_cnt % args.eval_freq == 0:
                    # validation
                    rgb_list, loss_list, psnr_list = [], [], []
                    for val_i, (val_pose, val_rgb) in enumerate(zip(poses_val, rgbs_val)):
                        print(val_i)
                        test_rays_o, test_rays_d = generate_rays(H, W, intrinsic, val_pose)
                        test_rays_o = torch.tensor(test_rays_o, device=device)
                        test_rays_d = torch.tensor(test_rays_d, device=device)

                        pixels = []
                        for j in range(test_rays_o.shape[0]):
                            rays_od = (test_rays_o[j], test_rays_d[j])
                            rgb, _, _, _ = render_rays(net_coarse, net_fine, rays_od, bound=bound, N_samples=N_samples, device=device, eval=True)
                            pixels.append(rgb.unsqueeze(0))
                        rgb = torch.cat(pixels, dim=0)
                        loss = mse(rgb, torch.tensor(val_rgb, device=device)).cpu()
                        psnr = -10. * torch.log(loss).item() / torch.log(torch.tensor([10.]))
                        rgb_list.append(rgb)
                        loss_list.append(loss.item())
                        psnr_list.append(psnr.item())

                    # print(f"[Epoch {ep_i + 1}] loss: {np.mean(loss_list):.5f}, psnr: {np.mean(psnr_list):.5f}")
                    writer.add_scalar('val/loss', np.mean(loss_list), iter_cnt)
                    writer.add_scalar('val/psnr', np.mean(psnr_list), iter_cnt)

                    if iter_cnt % args.save_freq == 0:
                        # save checkpoint
                        torch.save({
                            'epoch': ep_i + 1,
                            'iter': iter_cnt,
                            'model_state_dict': {
                                "net_coarse": net_coarse.state_dict(),
                                "net_fine": net_fine.state_dict()
                            },
                            'optimizer_state_dict': optimizer.state_dict(),
                            'lr_scheduler_state_dict': scheduler.state_dict(),
                            'val_loss': np.mean(loss_list),
                            'val_psnr': np.mean(psnr_list),
                            'args': args
                        }, os.path.join(exp_ckpt_dir, f"ckpt.{iter_cnt}.pth"))

                        # save the first 5 images in the validation set
                        img = rgb_list[0].cpu().detach().numpy()
                        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
                        img = Image.fromarray(img)
                        img.save(os.path.join(exp_vis_dir, f"iter={iter_cnt}_{data['names_val'][0]}_psnr={psnr_list[0]:.5f}.png"))

                        # show validation images in tensorboard
                        writer.add_images('val/render', torch.stack(rgb_list[:5]), iter_cnt, dataformats="NHWC")

                        # generate images for test cases
                        test_rgb_list = []
                        test_depth_list = []
                        for test_i, test_pose in enumerate(data["poses_test"]):
                            test_rays_o, test_rays_d = generate_rays(H, W, intrinsic, test_pose)
                            test_rays_o = torch.tensor(test_rays_o, device=device)
                            test_rays_d = torch.tensor(test_rays_d, device=device)

                            rgb_pixels = []
                            depth_pixels = []
                            for j in range(test_rays_o.shape[0]):
                                rays_od = (test_rays_o[j], test_rays_d[j])
                                rgb, depth, _, _ = render_rays(net_coarse, net_fine, rays_od, bound=bound, N_samples=N_samples, device=device, eval=True)
                                rgb_pixels.append(rgb.unsqueeze(0))
                                depth_pixels.append(depth.unsqueeze(0))
                            rgb = torch.cat(rgb_pixels, dim=0)
                            depth = torch.cat(depth_pixels, dim=0)
                            test_rgb_list.append(rgb)
                            test_depth_list.append(depth)

                            img = rgb.cpu().detach().numpy()
                            img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
                            img = Image.fromarray(img)
                            img.save(os.path.join(exp_vis_dir, f"rgb_iter={iter_cnt}_{data['names_test'][test_i]}.png"))

                            img_depth = depth.cpu().detach().numpy()
                            # print(img_depth.min(), img_depth.max())
                            img_depth = (np.clip(img_depth, 0., 5.) / 5. * 255).astype(np.uint8)
                            img_depth = Image.fromarray(img_depth)
                            img_depth.save(os.path.join(exp_vis_dir, f"depth_iter={iter_cnt}_{data['names_test'][test_i]}.png"))

                        writer.add_images('test/render', torch.stack(test_rgb_list, dim=0), iter_cnt, dataformats="NHWC")


if __name__ == "__main__":
    main()
