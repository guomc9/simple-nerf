import os
import json
import imageio
from torch.utils.data.dataloader import DataLoader
import numpy as np
from tools import get_rays_np

class blenderLoader(DataLoader):
    def __init__(self, meta_path, img_base_dir=None, batch_size=4096, skip=1):
        with open(meta_path, 'r') as fp:
            meta = json.load(fp)
        imgs = []
        poses = []
        if img_base_dir is None:
            img_base_dir = os.getcwd()

        for frame in meta['frames'][::skip]:
            fname = os.path.join(img_base_dir, frame['file_path'])
            imgs.append(imageio.imread(fname))
            poses.append(np.array(frame['transform_matrix']))

        imgs = (np.array(imgs) / 255.).astype(np.float32)[...:3]    # [N_images, H, W, 3], RGBA -> RGB
        poses = np.array(poses).astype(np.float32)                  # [N_images, 4, 4]

        H, W = imgs[0].shape[:2]
        camera_angle_x = float(meta['camera_angle_x'])
        view_z_dir = int(meta['view_z_dir'])
        self.batch_size = batch_size
        self.near = float(meta['near'])
        self.far = float(meta['far'])
        focal = .5 * W / np.tan(.5 * camera_angle_x)
        K = np.array([
                [focal, 0, 0.5*W],
                [0, focal, 0.5*H],
                [0, 0, 1]
            ])
        self.N_images = imgs.shape[0]
        self.N_rays_per_image = H*W
        self.N_rays = self.N_images * self.N_rays_per_image
        self.N_batch = self.N_rays // self.batch_size
        self.rays_o, self.rays_d = get_rays_np(H, W, K, view_z_dir, poses[:,:-1,:]) # [N_images, H*W, 3], [N_images, H*W, 3]
        self.rays_o.reshape(-1, 3)  # [N_images*H*W, 3] = [N_rays, 3]
        self.rays_d.reshape(-1, 3)  # [N_images*H*W, 3] = [N_rays, 3]
        self.view_dirs = self.rays_d / np.linalg.norm(self.rays_d, axis=1, keepdims=True)   # [N_rays, 3]
        self.rays_rgb = imgs.reshape(-1, 3) # [N_images*H*W, 3] = [N_rays, 3]

    def __getitem__(self, index):
        index = index % self.N_batch
        begin = self.batch_size * index
        end = begin + self.batch_size
        end = end if end < self.N_rays else self.N_rays
        rays_o = self.rays_o[begin:end]
        rays_d = self.rays_d[begin:end]
        rays_rgb = self.rays_rgb[begin:end]
        view_dirs = self.view_dirs[begin:end]

        if index == self.N_batch - 1:
            inds = np.random.shuffle(np.arange(self.N_rays))
            self.rays_o = self.rays_o[inds]
            self.rays_d = self.rays_d[inds]
            self.rays_rgb = self.rays_rgb[inds]

        return rays_o, rays_d, view_dirs, rays_rgb     # [batch_size, 3], [batch_size, 3], [batch_size, 3], [batch_size, 3]

    def get_meta(self):
        return {'near':self.near, 'far':self.far}