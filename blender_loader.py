import os
import json
import imageio
from torch.utils.data.dataloader import DataLoader
import numpy as np
from tools import get_rays_np
import logging

class blenderLoader(DataLoader):
    def __init__(self, meta_path, img_base_dir=None, batch_size=4096, skip=1, rand_n=None):
        with open(meta_path, 'r') as fp:
            meta = json.load(fp)
        imgs = []
        poses = []
        if img_base_dir is None:
            img_base_dir = os.getcwd()

        for frame in meta['frames'][::skip]:
            fname = os.path.join(img_base_dir, frame['file_path'])
            if fname[-4:] != '.png':
                fname = fname + '.png'
            imgs.append(imageio.imread(fname))
            poses.append(np.array(frame['transform_matrix']))

        imgs = (np.array(imgs) / 255.).astype(np.float32)[..., :3]      # [N_images, H, W, 3], RGBA -> RGB
        poses = np.array(poses).astype(np.float32)                      # [N_images, 4, 4]

        H, W = imgs[0].shape[:2]
        self.H, self.W = H, W
        camera_angle_x = float(meta['camera_angle_x'])
        view_z_dir = int(meta['view_z_dir'])
        self.near = float(meta['near'])
        self.far = float(meta['far'])
        focal = .5 * W / np.tan(.5 * camera_angle_x)
        K = np.array([
                [focal, 0, 0.5*W],
                [0, focal, 0.5*H],
                [0, 0, 1]
            ])

        self.batch_size = batch_size
        self.N_images = imgs.shape[0]
        self.N_rays_per_image = H*W
        self.N_rays = self.N_images * self.N_rays_per_image
        self.N_batch = self.N_rays // self.batch_size
        self.rays_o, self.rays_d = get_rays_np(H, W, K, view_z_dir, poses[:,:-1,:])     # [N_images, H*W, 3], [N_images, H*W, 3]
        self.rays_o = self.rays_o.reshape(-1, 3)                                        # [N_images*H*W, 3] = [N_rays, 3]
        self.rays_d = self.rays_d.reshape(-1, 3)                                        # [N_images*H*W, 3] = [N_rays, 3]
        self.rays_rgb = imgs.reshape(-1, 3)                                             # [N_images*H*W, 3] = [N_rays, 3]
        self.shuffle_rays()

    def __getitem__(self, index):
        index = index % self.N_batch
        begin = self.batch_size * index
        end = begin + self.batch_size
        end = end if end < self.N_rays else self.N_rays
        rays_o = self.rays_o[begin:end].astype(np.float32)
        rays_d = self.rays_d[begin:end].astype(np.float32)
        rays_rgb = self.rays_rgb[begin:end].astype(np.float32)
        # shuffle rays
        if index == self.N_batch - 1:
            self.shuffle_rays()
        return rays_o, rays_d, rays_rgb         # [batch_size, 3], [batch_size, 3], [batch_size, 3]

    def __len__(self):
        return self.N_batch

    def shuffle_rays(self):
        rays = np.stack((self.rays_o, self.rays_d, self.rays_rgb), axis=1)
        np.random.shuffle(rays)
        rays = np.transpose(rays, (1, 0, 2))
        self.rays_o = rays[0]
        self.rays_d = rays[1]
        self.rays_rgb = rays[2]
        logging.info("Rays shuffled.")

    def get_meta(self):
        return {'H':self.H, 'W':self.W, 'near':self.near, 'far':self.far}