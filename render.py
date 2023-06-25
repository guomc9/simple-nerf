from render_loader import renderLoader
from nerf import NeRF
from tools import uniform_sample_rays, integrate, importance_sample_rays
import torch
import numpy as np
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
    parser.add_argument("--script_file", type=str, required=True, 
                        help='script file for rendering')
    parser.add_argument("--batch_size", type=int, default=1*1024, 
                        help='batch size (number of random rays per gradient step)')
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
    parser.add_argument("--res_half", type=bool, default=True, 
                        help='half resolution of images')
    return parser


if __name__ == '__main__':
    parser = config_parser()
    args = parser.parse_args()
    batch_size = args.batch_size
    script = os.path.join(args.base_dir, args.script_file)
    Nc_samples = args.Nc_samples
    Nf_samples = args.Nf_samples
    chunk = args.chunk

    # Load the model parameters from the checkpoint files
    coarse_nerf = NeRF(L_x=args.PE_x, L_d=args.PE_d, L=args.coarse_net_depth, skips=args.coarse_net_skips).to(device)
    coarse_nerf_params = torch.load(args.coarse_net_checkpoint, map_location=device)
    coarse_nerf.load_state_dict(coarse_nerf_params)
    fine_nerf = NeRF(L_x=args.PE_x, L_d=args.PE_d, L=args.fine_net_depth, skips=args.fine_net_skips).to(device)
    fine_nerf_params = torch.load(args.fine_net_checkpoint, map_location=device)
    fine_nerf.load_state_dict(fine_nerf_params)

    # Load render tasks
    render_loader = renderLoader(script)
    
    while not render_loader.empty():
        rays_rgb = []
        rays_o, rays_d, near, far = render_loader.get_rays()                                            # [N_rays, 3], [N_rays, 3]
        for i in range(0, rays_o.shape[0], batch_size):
            begin, end = i, i + batch_size
            end = end if end < rays_o.shape[0] else rays_o.shape[0]
            batch_rays_o = torch.from_numpy(rays_o[begin:end]).to(device)                               # [batch_size, 3]
            batch_view_dirs = batch_rays_d = torch.from_numpy(rays_d[begin:end]).to(device)             # [batch_size, 3]
            batch_view_dirs = batch_view_dirs / torch.norm(batch_view_dirs, dim=-1, keepdim=True)       # [batch_size, 3]
            batch_rays_q, t_vals = uniform_sample_rays(rays_o=batch_rays_o, rays_d=batch_rays_d, near=near, far=far, N_samples=Nc_samples)  # [batch_size, N_samples, 3]
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
            c_rgb = torch.cat(all_rgb, 0).reshape(b, b_s, 3)                                            # [batch_size, N_samples, 3]
            c_sigma = torch.cat(all_sigma, 0).reshape(b, b_s)                                           # [batch_size, N_samples]
            _, weights = integrate(c_rgb, c_sigma, batch_rays_d, t_vals)                        # [batch_size, 3], [batch_size, N_samples]

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
            f_rgb = torch.cat(all_rgb, 0).reshape(b, b_s, 3)                                            # [batch_size, N_samples+N_imp_samples, 3]
            f_sigma = torch.cat(all_sigma, 0).reshape(b, b_s)                                           # [batch_size, N_samples+N_imp_samples]
            f_rgb_map, _ = integrate(f_rgb, f_sigma, batch_rays_d, imp_t_vals)                          # [batch_size, 3]
            f_rgb_map = f_rgb_map.detach().cpu().numpy()
            rays_rgb.append(f_rgb_map)
        render_loader.submit(np.concatenate(rays_rgb, axis=0))                                          # [N_rays, 3]
