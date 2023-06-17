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

def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()

    # configuration file
    parser.add_argument('--config', is_config_file=True, required=True, 
                        help='config file path')

    # blender data options
    parser.add_argument("--base_dir", type=str, required=True, 
                        help='base directory of blender model')
    parser.add_argument("--meta_file", type=str, required=True, 
                        help='meta description file of blender data')

    # coarse network options
    parser.add_argument("--coarse_net_depth", type=int, default=8, 
                        help='layers in coarse network')
    parser.add_argument("--coarse_net_width", type=int, default=256,
                        help='channels per layer in coarse network')
    parser.add_argument("--coarse_net_skips", nargs='+', type=int, default=[4], 
                        help='layers concat position encoder results in coarse network')
    parser.add_argument("--coarse_net_checkpoint", type=str, required=True, 
                        help='coarse network checkpoint file')

    # fine network options
    parser.add_argument("--fine_net_depth", type=int, default=8, 
                        help='layers in fine network')
    parser.add_argument("--fine_net_width", type=int, default=256, 
                        help='channels per layer in fine network')
    parser.add_argument("--fine_net_skips", nargs='+', type=int, default=[4], 
                        help='layers concat position encoder results in coarse network')
    parser.add_argument("--fine_net_checkpoint", type=str, required=True, 
                        help='fine network checkpoint file')

    # evaluate options
    parser.add_argument("--image_skip", type=int, default=1, 
                        help='skip for image loader')
    parser.add_argument("--batch_size", type=int, default=1*1024, 
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--chunk", type=int, default=1024*64, 
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--Nc_samples", type=int, default=64, 
                        help='number of coarse samples per ray')
    parser.add_argument("--Nf_samples", type=int, default=128,
                        help='number of additional fine samples per ray')
    parser.add_argument("--PE_x", type=int, default=10, 
                        help='number of cos&sin function in position encoder for coordinate')
    parser.add_argument("--PE_d", type=int, default=4, 
                        help='number of cos&sin function in position encoder for directory')
    return parser

if __name__ == '__main__':
    parser = config_parser()
    args = parser.parse_args()
    base_dir = args.base_dir
    meta_file = args.meta_file
    Nc_samples = args.Nc_samples
    Nf_samples = args.Nf_samples
    chunk = args.chunk
    batch_size = args.batch_size
    
    # Load dataset
    eval_data_loader = blenderLoader(os.path.join(base_dir, meta_file), img_base_dir=base_dir, batch_size=batch_size, skip=args.image_skip)
    
    # Load the model parameters from the checkpoint files
    coarse_nerf = NeRF(L_x=args.PE_x, L_d=args.PE_d, L=args.coarse_net_depth, skips=args.coarse_net_skips).to(device)
    coarse_nerf_params = torch.load(args.coarse_net_checkpoint)
    coarse_nerf.load_state_dict(coarse_nerf_params)
    fine_nerf = NeRF(L_x=args.PE_x, L_d=args.PE_d, L=args.fine_net_depth, skips=args.fine_net_skips).to(device)
    fine_nerf_params = torch.load(args.fine_net_checkpoint)
    fine_nerf.load_state_dict(fine_nerf_params)

    # Load meta
    meta = eval_data_loader.get_meta()
    near = meta['near']
    far = meta['far']

    loss_history = []
    with trange(0, len(eval_data_loader)) as progress_bar:
        for i in progress_bar:
            # Load rays
            rays_o, rays_d, rays_rgb = eval_data_loader[i]
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
                end = j + chunk
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

    