from blender_loader import blenderLoader
from nerf import NeRF
from tools import uniform_sample_rays, integrate, importance_sample_rays
import torch
import numpy as np
from tqdm import trange
from loss import get_mse_loss
from metrics import get_psnr
import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)

if __name__ == '__main__':
    N_iter = 100000
    Nc_samples = 64
    Nf_samples = 128
    batch_size = 1 * 1024
    chunk = 16 * 1024
    test_data_loader = blenderLoader(meta_path='./dataset/transforms_test.json', batch_size=batch_size, skip=1)
    checkpoint_dir = f'./checkpoints/iter_{N_iter}'
    coarse_nerf_checkpoint = 'coarse_nerf.pt'
    fine_nerf_checkpoint = 'fine_nerf.pt'

    # Load the model parameters from the checkpoint files
    coarse_nerf_params = torch.load(os.path.join(checkpoint_dir, coarse_nerf_checkpoint))
    fine_nerf_params = torch.load(os.path.join(checkpoint_dir, fine_nerf_checkpoint))
    coarse_nerf = NeRF().load_state_dict(coarse_nerf_params).to(device)
    fine_nerf = NeRF().load_state_dict(fine_nerf_params).to(device)

    # Load meta
    meta = test_data_loader.get_meta()
    near = meta['near']
    far = meta['far']

    loss_history = []
    with trange(0, N_iter) as progress_bar:
        for i in progress_bar:
            # Load rays
            rays_o, rays_d, rays_rgb = test_data_loader[i]
            rays_o = torch.from_numpy(rays_o).to(device)
            view_dirs = rays_d = torch.from_numpy(rays_d).to(device)                # [batch_size, 3]
            view_dirs = view_dirs / torch.norm(view_dirs, dim=-1, keepdim=True)     # [batch_size, 3]
            rays_rgb = torch.from_numpy(rays_rgb).to(device)                        # [batch_size, 3]
            
            # Uniform sampling
            rays_q, t_vals = uniform_sample_rays(rays_o=rays_o, rays_d=rays_d, near=near, far=far, N_samples=Nc_samples)    # [batch_size, N_samples, 3], [batch_size, N_samples]
            b = rays_q.shape[0]
            b_s = rays_q.shape[1]
            rays_q_flat = rays_q.reshape(-1, 3)
            n_s = rays_q_flat.shape[0]
            view_dirs_flat = view_dirs[:,None,:].expand(rays_q.shape).reshape(-1, 3)
            all_rgb, all_sigma = [], []
            for j in range(0, n_s, chunk):
                begin = j
                end = j+chunk
                end = end if end < n_s else n_s
                rgb, sigma = coarse_nerf(rays_q_flat[begin:end], view_dirs_flat[begin:end])
                all_rgb.append(rgb)
                all_sigma.append(sigma)
            c_rgb = torch.cat(all_rgb, 0).reshape(b, b_s, 3)                            # [batch_size, N_samples, 3]
            c_sigma = torch.cat(all_sigma, 0).reshape(b, b_s)                           # [batch_size, N_samples]
            c_rgb_map, weights = integrate(c_rgb, c_sigma, rays_d, t_vals)              # [batch_size, 3], [batch_size, N_samples]
            loss = get_mse_loss(c_rgb_map, rays_rgb)                                    # coarse loss
            
            # Hierarchical volume sampling
            rays_q, imp_t_vals = importance_sample_rays(rays_o=rays_o, rays_d=rays_d, t_vals=t_vals, weights=weights, N_imp_samples=Nf_samples)  # [batch_size, N_imp_samples+N_samples, 3]
            b = rays_q.shape[0]
            b_s = rays_q.shape[1]
            rays_q_flat = rays_q.reshape(-1, 3)
            n_s = rays_q_flat.shape[0]
            view_dirs_flat = view_dirs[:,None,:].expand(rays_q.shape).reshape(-1, 3)
            all_rgb, all_sigma = [], []
            for j in range(0, n_s, chunk):
                begin = j
                end = j+chunk
                end = end if end < n_s else n_s
                rgb, sigma = fine_nerf(rays_q_flat[begin:end], view_dirs_flat[begin:end])
                all_rgb.append(rgb)
                all_sigma.append(sigma)
            f_rgb = torch.cat(all_rgb, 0).reshape(b, b_s, 3)                          # [batch_size, N_samples+N_imp_samples, 3]
            f_sigma = torch.cat(all_sigma, 0).reshape(b, b_s)                         # [batch_size, N_samples+N_imp_samples]
            f_rgb_map, weights = integrate(f_rgb, f_sigma, rays_d, imp_t_vals)        # [batch_size, 3], [batch_size, N_samples]
            loss += get_mse_loss(f_rgb_map, rays_rgb)
            psnr = get_psnr(loss)
            loss_history.append(loss.item())

            progress_bar.set_postfix({"Loss": f"{loss_history[-1]:.4f}", "Avg Loss": f"{sum(loss_history) / len(loss_history):.4f}", "PSNR": f"{psnr.item():.4f}"})

    