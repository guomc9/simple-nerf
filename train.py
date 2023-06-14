from blender_loader import blenderLoader
from nerf import NeRF
from tools import uniform_sample_rays, integrate, importance_sample_rays
import torch
from tqdm import trange, tqdm
from loss import get_rmse_loss
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

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
    optimizer = torch.optim.Adam(params=[coarse_nerf.parameters(), fine_nerf.parameters], lr=learning_rate, betas=(0.9, 0.999))
    for i in trange(0, N_iter):
        optimizer.zero_grad()
        rays_o, rays_d, view_dirs, rays_rgb = train_data_loader[i]
        rays_o = torch.tensor(rays_o, dtype=torch.float32).to(device)
        rays_d = torch.tensor(rays_d, dtype=torch.float32).to(device)           # [batch_size, 3]
        view_dirs = torch.tensor(view_dirs, dtype=torch.float32).to(device)     # [batch_size, 3]
        rays_rgb = torch.tensor(rays_rgb, dtype=torch.float32).to(device)       # [batch_size, 3]
        rays_q, t_vals = uniform_sample_rays(rays_o=rays_o, rays_d=rays_d, near=near, far=far, N_samples=Nc_samples)    # [batch_size, N_samples, 3], [batch_size, N_samples]
        b = rays_q.shape[0]
        n_s = rays_q.shape[1]
        rays_q_flat = rays_q.reshape(-1, 3).contiguous()
        view_dirs_flat = view_dirs[:,None,:].expand(rays_q.shape).reshape(-1, 3).contiguous()

        all_rgb, all_sigma = [], []
        for j in range(0, b, chunk):
            begin = j
            end = j+chunk
            end = end if end < b else b
            rgb, sigma = coarse_nerf(rays_q_flat[begin:end], view_dirs_flat[begin:end])
            all_rgb.append(rgb)
            all_sigma.append(sigma)
        c_rgb = torch.cat(all_rgb, 0).reshape(b, n_s, 3)                            # [batch_size, N_samples, 3]
        c_sigma = torch.cat(all_sigma, 0).reshape(b, n_s)                           # [batch_size, N_samples]
        c_rgb_map, weights = integrate(c_rgb, c_sigma, rays_d, t_vals)              # [batch_size, 3], [batch_size, N_samples]
        loss = get_rmse_loss(c_rgb_map, rays_rgb)                                   # coarse loss
        
        rays_q, _ = importance_sample_rays(rays_o=rays_o, rays_d=rays_d, t_vals=t_vals, weights=weights, N_imp_samples=Nf_samples)  # [batch_size, N_imp_samples+N_samples, 3]
        n_s = rays_q.shape[1]
        rays_q_flat = rays_q.reshape(-1, 3).contiguous()
        all_rgb, all_sigma = [], []
        for j in range(0, b, chunk):
            begin = j
            end = j+chunk
            end = end if end < b else b
            rgb, sigma = fine_nerf(rays_q_flat[begin:end], view_dirs_flat[begin:end])
            all_rgb.append(rgb)
            all_sigma.append(sigma)
        f_rgb = torch.cat(all_rgb, 0).reshape(b, n_s, 3)                          # [batch_size, N_samples, 3]
        f_sigma = torch.cat(all_sigma, 0).reshape(b, n_s)                         # [batch_size, N_samples]
        f_rgb_map, weights = integrate(rgb, sigma, rays_d, t_vals)                # [batch_size, 3], [batch_size, N_samples]
        loss += get_rmse_loss(f_rgb_map, rays_rgb)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
        tqdm.write(f"Iter {i+1}/{N_iter}, Loss: {loss_history[-1]:.4f}, Avg Loss: {sum(loss_history) / len(loss_history):.4f}")
