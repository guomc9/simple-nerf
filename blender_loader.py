import os
import json
import imageio
import numpy as np
from tools import get_rays_np
import logging
import cv2

class blenderLoader():
    def __init__(self, meta_path, img_base_dir=None, use_batch=False, N_rand=4096, skip=1, crop_iters=0, crop_frac=0.5, res_half=False):
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

        imgs = (np.array(imgs)[..., :3] / 255.).astype(np.float32)      # [N_images, H, W, 3], RGB
        poses = np.array(poses).astype(np.float32)                      # [N_images, 4, 4]

        self.use_batch = use_batch
        
        H, W = imgs[0].shape[:2]
        camera_angle_x = float(meta['camera_angle_x'])
        focal = .5 * W / np.tan(.5 * camera_angle_x)
        if res_half:
            H, W = H // 2, W // 2
            logging.info(f'Half resolution ({H}, {W}).')
            focal = focal / 2
            imgs_half_res = np.zeros((imgs.shape[0], H, W, 3))
            for i, img in enumerate(imgs):
                imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
            imgs = imgs_half_res
        self.H, self.W = H, W
        view_z_dir = int(meta['view_z_dir'])
        self.near = float(meta['near'])
        self.far = float(meta['far'])
        
        K = np.array([
                [focal, 0, 0.5*W],
                [0, focal, 0.5*H],
                [0, 0, 1]
            ])
        self.N_rays_per_image = H * W
        self.N_images = imgs.shape[0]
        self.N_rand = N_rand
        self.N_rays = self.N_images * self.N_rays_per_image
        self.N_batch = self.N_rays // N_rand
        self.rays_o, self.rays_d = get_rays_np(H, W, K, view_z_dir, poses[:,:-1,:])     # [N_images, H*W, 3], [N_images, H*W, 3]
        self.rays_o = self.rays_o.reshape(-1, 3)                                        # [N_images*H*W, 3] = [N_rays, 3]
        self.rays_d = self.rays_d.reshape(-1, 3)                                        # [N_images*H*W, 3] = [N_rays, 3]
        self.rays_rgb = imgs.reshape(-1, 3)                                             # [N_images*H*W, 3] = [N_rays, 3]
       
        if self.use_batch:
            self.shuffle_rays()
        else:
            self.crop_iters = crop_iters
            self.crop_frac = crop_frac

    def __getitem__(self, index):
        if self.use_batch:
            index = index % self.N_batch
            begin = self.N_rand * index
            end = begin + self.N_rand
            end = end if end < self.N_rays else self.N_rays
            rays_o = self.rays_o[begin:end].astype(np.float32)
            rays_d = self.rays_d[begin:end].astype(np.float32)
            rays_rgb = self.rays_rgb[begin:end].astype(np.float32)
            # shuffle rays
            if index == self.N_batch - 1:
                self.shuffle_rays()
        else:
            img_index = np.random.choice(self.N_images)
            begin = img_index * self.N_rays_per_image
            end = (img_index + 1) * self.N_rays_per_image
            rays_o = self.rays_o[begin:end].astype(np.float32)
            rays_d = self.rays_d[begin:end].astype(np.float32)
            rays_rgb = self.rays_rgb[begin:end].astype(np.float32)
            if index < self.crop_iters:
                dH = int(self.H // 2 * self.crop_frac)
                dW = int(self.W // 2 * self.crop_frac)
                coords = np.stack(
                    np.meshgrid(
                        np.linspace(self.H // 2 - dH, self.H // 2 + dH - 1, 2 * dH),
                        np.linspace(self.W // 2 - dW, self.W // 2 + dW - 1, 2 * dW),
                        indexing='ij'
                    ), -1)
            else:
                coords = np.stack(np.meshgrid(np.linspace(0, self.H-1, self.H), np.linspace(0, self.W-1, self.W), indexing='ij'), -1)  # (H, W, 2)
            
            coords = np.reshape(coords, [-1, 2])                                                # [H * W, 2]
            select_inds = np.random.choice(coords.shape[0], size=[self.N_rand], replace=False)  # [N_rand,]
            select_coords = coords[select_inds].astype(np.int64)                                # [N_rand, 2]
            select_indices_1d = select_coords[:, 0] * self.W + select_coords[:, 1]              # [N_rand,]
            rays_o = rays_o[select_indices_1d]                                                  # [N_rand, 3]
            rays_d = rays_d[select_indices_1d]                                                  # [N_rand, 3]
            rays_rgb = rays_rgb[select_indices_1d]                                              # [N_rand, 3]
        return rays_o, rays_d, rays_rgb                                                         # [N_rand, 3], [N_rand, 3], [N_rand, 3]

    def __len__(self):
        if self.use_batch:
            return self.N_batch
        else:
            return self.N_images

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
    
if __name__ == '__main__':
    loader = blenderLoader(meta_path='./dataset/lego/transforms_train.json', img_base_dir='./dataset/lego', use_batch=False, N_rand=1024, crop_iters=500, crop_frac=0.5, res_half=True)
