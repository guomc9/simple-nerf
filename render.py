from render_loader import renderLoader
from nerf import NeRF
from tools import uniform_sample_rays, integrate, importance_sample_rays
import torch
import numpy as np
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)

if __name__ == '__main__':
    N_iter = 100000
    batch_size = 1024
    Nc_samples = 64
    Nf_samples = 128
    chunk = 16 * 1024
    render_json = './dataset/render.json'
    checkpoint_dir = f'./checkpoints/iter_{N_iter}'
    coarse_nerf_checkpoint = 'coarse_nerf.pt'
    fine_nerf_checkpoint = 'fine_nerf.pt'

    # Load the model parameters from the checkpoint files
    coarse_nerf_params = torch.load(os.path.join(checkpoint_dir, coarse_nerf_checkpoint))
    fine_nerf_params = torch.load(os.path.join(checkpoint_dir, fine_nerf_checkpoint))
    coarse_nerf = NeRF().load_state_dict(coarse_nerf_params).to(device)
    fine_nerf = NeRF().load_state_dict(fine_nerf_params).to(device)

    # Load render tasks
    render_loader = renderLoader(render_json)
    
    while not render_loader.empty():
        rays_rgb = []
        rays_o, rays_d, near, far = render_loader.get_rays()                                            # [N_rays, 3], [N_rays, 3]
        for i in range(0, rays_o.shape[0], batch_size):
            begin, end = i, i + batch_size
            end = end if end < rays_o.shape[0] else rays_o.shape[0]
            batch_rays_o = torch.from_numpy(rays_o[begin:end]).to(device)                               # [batch_size, 3]
            batch_view_dirs = batch_rays_d = torch.from_numpy(rays_d[begin:end]).to(device)             # [batch_size, 3]
            batch_view_dirs = batch_view_dirs / torch.norm(batch_view_dirs, dim=-1, keepdim=True)       # [batch_size, 3]
            
            batch_rays_q, t_vals = uniform_sample_rays(batch_rays_o, batch_rays_d, near, far, Nc_samples)
            
            b = batch_rays_q.shape[0]
            b_s = batch_rays_q.shape[1]
            batch_rays_q_flat = batch_rays_q.reshape(-1, 3)
            n_s = batch_rays_q_flat.shape[0]
            batch_view_dirs_flat = batch_view_dirs[:,None,:].expand(batch_rays_q.shape).reshape(-1, 3)
            all_rgb, all_sigma = [], []
            for j in range(0, n_s, chunk):
                begin = j
                end = j + chunk
                end = end if end < n_s else n_s
                rgb, sigma = coarse_nerf(batch_rays_q_flat[begin:end], batch_view_dirs_flat[begin:end])
                all_rgb.append(rgb)
                all_sigma.append(sigma)
            c_rgb = torch.cat(all_rgb, 0).reshape(b, b_s, 3)                            # [batch_size, N_samples, 3]
            c_sigma = torch.cat(all_sigma, 0).reshape(b, b_s)                           # [batch_size, N_samples]
            c_rgb_map, weights = integrate(c_rgb, c_sigma, rays_d, t_vals)              # [batch_size, 3], [batch_size, N_samples]

            batch_rays_q, imp_t_vals = importance_sample_rays(batch_rays_o, batch_rays_d, t_vals, weights, Nf_samples)  # [batch_size, N_imp_samples+N_samples, 3]
            
            b = batch_rays_q.shape[0]
            b_s = batch_rays_q.shape[1]
            batch_rays_q_flat = batch_rays_q.reshape(-1, 3)
            n_s = batch_rays_q_flat.shape[0]
            batch_view_dirs_flat = batch_view_dirs[:,None,:].expand(batch_rays_q.shape).reshape(-1, 3)
            all_rgb, all_sigma = [], []
            for j in range(0, n_s, chunk):
                begin = j
                end = j+chunk
                end = end if end < n_s else n_s
                rgb, sigma = fine_nerf(batch_rays_q_flat[begin:end], batch_view_dirs_flat[begin:end])
                all_rgb.append(rgb)
                all_sigma.append(sigma)
            f_rgb = torch.cat(all_rgb, 0).reshape(b, b_s, 3)                            # [batch_size, N_samples+N_imp_samples, 3]
            f_sigma = torch.cat(all_sigma, 0).reshape(b, b_s)                           # [batch_size, N_samples+N_imp_samples]
            f_rgb_map, _ = integrate(f_rgb, f_sigma, rays_d, imp_t_vals)                # [batch_size, 3]
            rays_rgb.append(f_rgb_map)
        render_loader.submit(torch.cat(rays_rgb, dim=0))                                # [N_rays, 3]
