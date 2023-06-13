from blender_loader import blenderLoader
from nerf import NeRF
from tools import uniform_sample_rays, integrate
import torch
from tqdm import trange
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

    optimizer = torch.optim.Adam(params=[coarse_nerf.parameters(), fine_nerf.parameters], lr=learning_rate, betas=(0.9, 0.999))
    for i in trange(0, N_iter):
        optimizer.zero_grad()
        rays_o, rays_d, view_dirs, rays_rgb = train_data_loader[i]
        rays_o = torch.tensor(rays_o, dtype=torch.float32).to(device)
        rays_d = torch.tensor(rays_d, dtype=torch.float32).to(device)
        view_dirs = torch.tensor(view_dirs, dtype=torch.float32).to(device)
        rays_rgb = torch.tensor(rays_rgb, dtype=torch.float32).to(device)
        rays_q, t_vals = uniform_sample_rays(rays_o=rays_o, rays_d=rays_d, near=near, far=far, N_samples=Nc_samples)
        b = rays_q.shape[0]
        n_s = rays_q.shape[1]
        rays_q_flat = rays_q.reshape(-1, 3)
        view_dirs_flat = view_dirs[:,None,:].expand(rays_q.shape).reshape(-1, 3)
        all_rgb, all_alpha = [], []
        for j in range(0, b, chunk):
            begin = j
            end = j+chunk
            end = end if end < b else b
            rgb, alpha = coarse_nerf(rays_q_flat[begin:end], view_dirs_flat[begin:end])
            all_rgb.append(rgb)
            all_alpha.append(alpha)
        rgb = torch.cat(all_rgb, 0).reshape(b, n_s, 3)  #[batch_size, N_samples, 3]
        alpha = torch.cat(all_alpha, 0).reshape(b, n_s) #[batch_size, N_samples]
        rgb_map, weights = integrate(rgb, alpha, rays_d, t_vals)
        loss_c = get_rmse_loss()
        
            
        