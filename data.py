import os
import numpy as np
from PIL import Image


def load_poses(pose_files):
    poses = []
    for pose_file in pose_files:
        with open(pose_file, 'r') as f:
            pose = [[float(x) for x in line.split(' ')] for line in f]
            assert len(pose) == 4 and len(pose[0]) == 4
        poses.append(pose)
    return np.array(poses, dtype=np.float32)


def load_rgbs(rgb_files, img_size=None):
    rgbs = []
    for rgb_file in rgb_files:
        rgba = Image.open(rgb_file)
        if img_size is not None:
            rgba = rgba.resize(img_size)
        rgba = np.array(rgba) / 255.

        rgb = rgba[..., :3] * rgba[..., -1:] + (1. - rgba[..., -1:])    # white background

        rgbs.append(rgb)
    return np.array(rgbs, dtype=np.float32)


def load_intrinsic(intrinsic_path):
    with open(intrinsic_path, 'r') as f:
        intrinsic = np.array([[float(x) for x in line.split(' ')] for line in f])
    return intrinsic


def load_data(data_dir, test_ids=None, img_size=None, train_ratio=0.9):
    # load training/validation data
    pose_files, rgb_files, names = [], [], []
    for prefix in ["0_train", "1_val"]:
        for i in range(100):
            name = f"{prefix}_{i:04}"
            names.append(name)
            pose_files.append(os.path.join(data_dir, "pose", f"{name}.txt"))
            rgb_files.append(os.path.join(data_dir, "rgb", f"{name}.png"))
    poses = load_poses(pose_files)
    rgbs = load_rgbs(rgb_files, img_size=img_size)

    # split data
    n_train = int(train_ratio * poses.shape[0])
    poses_train, poses_val, rgbs_train, rgbs_val, inds_train, inds_val = split_data(poses, rgbs, n_train=n_train)

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
    # scale, TODO
    if img_size is not None:
        s = img_size[0] / 800
        intrinsic[:2, -1] *= s

    data = {
        "poses_train": poses_train,
        "poses_val": poses_val,
        "poses_test": test_poses,
        "rgbs_train": rgbs_train,
        "rgbs_val": rgbs_val,
        "inds_train": inds_train,
        "inds_val": inds_val,
        "names_train": [names[idx] for idx in inds_train],
        "names_val": [names[idx] for idx in inds_val],
        "names_test": test_names,
        "intrinsic": intrinsic,
    }
    print("val names:", data["names_val"])
    return data


def split_data(poses, rgbs, n_train):
    # train/validation split
    assert poses.shape[0] == rgbs.shape[0]
    n_total = poses.shape[0]
    assert n_train < n_total
    shuffled_inds = list(range(n_total))
    np.random.shuffle(shuffled_inds)
    inds_train = shuffled_inds[:n_train]
    inds_val = shuffled_inds[n_train:]
    poses_train = poses[inds_train]
    rgbs_train = rgbs[inds_train]
    poses_val = poses[inds_val]
    rgbs_val = rgbs[inds_val]
    print(f"train size: {n_train}, val size: {n_total - n_train}")
    print("val inds", inds_val)
    return poses_train, poses_val, rgbs_train, rgbs_val, inds_train, inds_val

