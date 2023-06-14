from blender_loader import blenderLoader
from nerf import NeRF
from tools import uniform_sample_rays, integrate, importance_sample_rays
import torch
import numpy as np
from tqdm import trange, tqdm
from loss import get_rmse_loss
import os
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
np.random.seed(0)

def save_model_parameters(coarse_nerf, fine_nerf, iteration):
    save_dir = f'./checkpoints/iter_{iteration}'
    os.makedirs(save_dir, exist_ok=True)
    torch.save(coarse_nerf.state_dict(), os.path.join(save_dir, 'coarse_nerf.pt'))
    torch.save(fine_nerf.state_dict(), os.path.join(save_dir, 'fine_nerf.pt'))
    print(f"Saved model parameters at iteration {iteration}")

if __name__ == '__main__':
    N_iter = 200000
    chunk = 32 * 1024
    Nc_samples = 64
    Nf_samples = 128
    batch_size = 4 * 1024
    chunk = 64 * 1024
    learning_rate = 5e-4
    train_data_loader = blenderLoader(meta_path='./dataset/transforms_train.json', batch_size=batch_size)

    meta = train_data_loader.get_meta()
    near = meta['near']
    far = meta['far']
    coarse_nerf = NeRF().to(device)
    fine_nerf = NeRF().to(device)
    loss_history = []
    save_step = 5000
    optimizer = torch.optim.Adam(params=list(coarse_nerf.parameters()) + list(fine_nerf.parameters()), lr=learning_rate, betas=(0.9, 0.999))
    for i in trange(0, N_iter):
        optimizer.zero_grad()
        rays_o, rays_d, view_dirs, rays_rgb = train_data_loader[i]
        rays_o = torch.tensor(rays_o, dtype=torch.float32).to(device)
        rays_d = torch.tensor(rays_d, dtype=torch.float32).to(device)           # [batch_size, 3]
        view_dirs = torch.tensor(view_dirs, dtype=torch.float32).to(device)     # [batch_size, 3]
        rays_rgb = torch.tensor(rays_rgb, dtype=torch.float32).to(device)       # [batch_size, 3]
        rays_q, t_vals = uniform_sample_rays(rays_o=rays_o, rays_d=rays_d, near=near, far=far, N_samples=Nc_samples)    # [batch_size, N_samples, 3], [batch_size, N_samples]
        
        b = rays_q.shape[0]
        b_s = rays_q.shape[1]
        rays_q_flat = rays_q.reshape(-1, 3).contiguous()
        n_s = rays_q_flat.shape[0]
        view_dirs_flat = view_dirs[:,None,:].expand(rays_q.shape).reshape(-1, 3).contiguous()
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
        loss = get_rmse_loss(c_rgb_map, rays_rgb)                                   # coarse loss
        
        rays_q, imp_t_vals = importance_sample_rays(rays_o=rays_o, rays_d=rays_d, t_vals=t_vals, weights=weights, N_imp_samples=Nf_samples)  # [batch_size, N_imp_samples+N_samples, 3]
        b = rays_q.shape[0]
        b_s = rays_q.shape[1]
        rays_q_flat = rays_q.reshape(-1, 3).contiguous()
        n_s = rays_q_flat.shape[0]
        view_dirs_flat = view_dirs[:,None,:].expand(rays_q.shape).reshape(-1, 3).contiguous()
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
        loss += get_rmse_loss(f_rgb_map, rays_rgb)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
        tqdm.write(f"Iter {i+1}/{N_iter}, Loss: {loss_history[-1]:.4f}, Avg Loss: {sum(loss_history) / len(loss_history):.4f}")

        # Save model parameters
        if (i + 1) % save_step == 0:
            save_model_parameters(coarse_nerf, fine_nerf, i + 1)