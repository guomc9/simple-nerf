from blender_loader import blenderLoader
from nerf import NeRF
from tools import uniform_sample_rays, integrate, importance_sample_rays
import torch
import numpy as np
import random
from tqdm import trange
from loss import get_mse_loss
from metrics import get_psnr
import os
import logging
import ast
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)

def save_model_parameters(save_base_dir, coarse_nerf, fine_nerf, iteration):
    save_dir = os.path.join(save_base_dir, f'iter_{iteration}')
    os.makedirs(save_dir, exist_ok=True)
    torch.save(coarse_nerf.state_dict(), os.path.join(save_dir, 'coarse_nerf.pt'))
    torch.save(fine_nerf.state_dict(), os.path.join(save_dir, 'fine_nerf.pt'))
    logging.info(f"Saved model parameters at iteration {iteration}.")

def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()

    # configuration file
    parser.add_argument('--config', is_config_file=True, required=True, 
                        help='config file path')

    # blender data options
    parser.add_argument("--base_dir", type=str, required=True, 
                        help='base directory of blender model')
    parser.add_argument("--train_meta_file", type=str, required=True, 
                        help='train meta description file of blender data')

    # coarse network options
    parser.add_argument("--coarse_net_depth", type=int, default=8, 
                        help='layers in coarse network')
    parser.add_argument("--coarse_net_width", type=int, default=256,
                        help='channels per layer in coarse network')
    parser.add_argument("--coarse_net_skips", nargs='+', type=int, default=[4], 
                        help='layers concat position encoder results in coarse network')
    parser.add_argument("--coarse_net_use_checkpoint", type=bool, default=False, 
                        help='coarse network checkpoint file')
    parser.add_argument("--coarse_net_checkpoint", type=str, required=False,  
                        help='coarse network checkpoint file')

    # fine network options
    parser.add_argument("--fine_net_depth", type=int, default=8, 
                        help='layers in fine network')
    parser.add_argument("--fine_net_width", type=int, default=256, 
                        help='channels per layer in fine network')
    parser.add_argument("--fine_net_skips", nargs='+', type=int, default=[4], 
                        help='layers concat position encoder results in coarse network')
    parser.add_argument("--fine_net_use_checkpoint", type=bool, default=False, 
                        help='fine network checkpoint file')
    parser.add_argument("--fine_net_checkpoint", type=str, required=False,  
                        help='fine network checkpoint file')

    # train options
    parser.add_argument("--image_skip", type=int, default=1, 
                        help='skip for image loader')
    parser.add_argument("--N_iter", type=int, default=100000, 
                        help='number of train iterations')
    parser.add_argument("--batch_size", type=int, default=1*1024, 
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lr", type=float, default=5e-4, 
                        help='learning rate')
    parser.add_argument("--betas", type=str, default='(0.9, 0.999)', 
                        help='betas for optimizer')
    parser.add_argument("--lr_decay", type=int, default=250, 
                        help='exponential learning rate decay (in 1000 steps)')
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
    
    # saving options
    parser.add_argument("--checkpoints_save_step", type=int, default=1000, 
                        help='checkpoints save step')
    parser.add_argument("--checkpoints_save_dir", type=str, default='./checkpoints', 
                        help='checkpoints save directory')

    # logging options
    parser.add_argument("--log", type=bool, default=True, 
                        help='log or not')
    parser.add_argument("--log_dir", type=str, default='./log', 
                        help='loss and metrics log directory')
    parser.add_argument("--log_step", type=int, default=100, 
                        help='frequency of tensorboard logging')
    
    # test options
    parser.add_argument("--test_meta_file", type=str, default=False, required=True, 
                        help='test meta description file of blender data')
    parser.add_argument("--test_step", type=int, default=500, 
                        help='frequency of test')
    parser.add_argument("--test_rand_n", type=int, default=1, 
                        help='randomly choose n rays for test')
    # parser.add_argument("--i_testset", type=int, default=50000, 
    #                     help='frequency of testset saving')
    # parser.add_argument("--i_video",   type=int, default=50000, 
    #                     help='frequency of render_poses video saving')

    return parser

if __name__ == '__main__':
    parser = config_parser()
    args = parser.parse_args()
    base_dir = args.base_dir
    train_meta_file = args.train_meta_file
    test_meta_file = args.test_meta_file
    lr = args.lr
    Nc_samples = args.Nc_samples
    Nf_samples = args.Nf_samples
    chunk = args.chunk
    lr_decay = args.lr_decay
    batch_size = args.batch_size
    save_dir = args.checkpoints_save_dir
    save_step = args.checkpoints_save_step
    betas = ast.literal_eval(args.betas)
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] - [%(levelname)s] - %(message)s')

    if args.log:
        log_step = args.log_step
        writer = SummaryWriter(log_dir=args.log_dir)
    logging.info('Init train data loader.')
    train_data_loader = blenderLoader(meta_path=os.path.join(base_dir, train_meta_file), img_base_dir=base_dir, batch_size=batch_size, skip=args.image_skip)
    
    logging.info('Init test data loader.')
    test_data_loader = blenderLoader(meta_path=os.path.join(base_dir, test_meta_file), img_base_dir=base_dir, batch_size=batch_size, skip=args.image_skip)
    
    train_meta = train_data_loader.get_meta()
    train_near = train_meta['near']
    train_far = train_meta['far']
    
    test_meta = test_data_loader.get_meta()
    test_near = test_meta['near']
    test_far = test_meta['far']
    
    logging.info('Init coarse net.')
    coarse_nerf = NeRF(L_x=args.PE_x, L_d=args.PE_d, L=args.coarse_net_depth, skips=args.coarse_net_skips).to(device)
    if args.coarse_net_use_checkpoint:
        coarse_nerf_params = torch.load(args.coarse_net_checkpoint)
        coarse_nerf.load_state_dict(coarse_nerf_params)
    
    logging.info('Init fine net.')
    fine_nerf = NeRF(L_x=args.PE_x, L_d=args.PE_d, L=args.fine_net_depth, skips=args.fine_net_skips).to(device)
    if args.fine_net_use_checkpoint:
        fine_nerf_params = torch.load(args.fine_net_checkpoint)
        fine_nerf.load_state_dict(fine_nerf_params)
    
    loss_history = []
    psnr_history = []
    optimizer = torch.optim.Adam(params=list(coarse_nerf.parameters()) + list(fine_nerf.parameters()), lr=lr, betas=betas)
    with trange(0, args.N_iter) as progress_bar:
        for i in progress_bar:
            optimizer.zero_grad()
            # Train
            rays_o, rays_d, rays_rgb = train_data_loader[i]
            rays_o = torch.from_numpy(rays_o).to(device)
            view_dirs = rays_d = torch.from_numpy(rays_d).to(device)                # [batch_size, 3]
            view_dirs = view_dirs / torch.norm(view_dirs, dim=-1, keepdim=True)     # [batch_size, 3]
            rays_rgb = torch.from_numpy(rays_rgb).to(device)                        # [batch_size, 3]
            rays_q, t_vals = uniform_sample_rays(rays_o=rays_o, rays_d=rays_d, near=train_near, far=train_far, N_samples=Nc_samples)    # [batch_size, N_samples, 3], [batch_size, N_samples]
            
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
            f_rgb = torch.cat(all_rgb, 0).reshape(b, b_s, 3)                            # [batch_size, N_samples+N_imp_samples, 3]
            f_sigma = torch.cat(all_sigma, 0).reshape(b, b_s)                           # [batch_size, N_samples+N_imp_samples]
            f_rgb_map, weights = integrate(f_rgb, f_sigma, rays_d, imp_t_vals)          # [batch_size, 3], [batch_size, N_samples]
            loss += get_mse_loss(f_rgb_map, rays_rgb)                                   # fine loss
            psnr = get_psnr(loss)
            loss.backward()
            optimizer.step()
            loss_history.append(loss.item())
            psnr_history.append(psnr.item())
            train_loss = sum(loss_history) / len(loss_history)
            train_psnr = sum(psnr_history) / len(psnr_history)
            progress_bar.set_postfix({"Loss": f"{loss_history[-1]:.4f}", "Avg Loss": f"{train_loss:.4f}", "PSNR": f"{psnr_history[-1]:.4f}", "Avg PSNR": f"{train_psnr:.4f}"})
            
            # Test
            if (i + 1) % args.test_step == 0:
                test_losses = []
                test_psnrs = []
                with trange(0, args.test_rand_n) as test_bar:
                    for j in test_bar:
                        logging.info(f"Test nerf.")
                        k = random.randint(0, len(test_data_loader)-1)
                        rays_o, rays_d, rays_rgb = test_data_loader[k]
                        rays_o = torch.from_numpy(rays_o).to(device)
                        view_dirs = rays_d = torch.from_numpy(rays_d).to(device)                # [batch_size, 3]
                        view_dirs = view_dirs / torch.norm(view_dirs, dim=-1, keepdim=True)     # [batch_size, 3]
                        rays_rgb = torch.from_numpy(rays_rgb).to(device)                        # [batch_size, 3]
                        rays_q, t_vals = uniform_sample_rays(rays_o=rays_o, rays_d=rays_d, near=test_near, far=test_far, N_samples=Nc_samples)    # [batch_size, N_samples, 3], [batch_size, N_samples]
                        
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
                        f_rgb = torch.cat(all_rgb, 0).reshape(b, b_s, 3)                            # [batch_size, N_samples+N_imp_samples, 3]
                        f_sigma = torch.cat(all_sigma, 0).reshape(b, b_s)                           # [batch_size, N_samples+N_imp_samples]
                        f_rgb_map, weights = integrate(f_rgb, f_sigma, rays_d, imp_t_vals)          # [batch_size, 3], [batch_size, N_samples]
                        loss += get_mse_loss(f_rgb_map, rays_rgb)                                   # fine loss
                        psnr = get_psnr(loss)
                        test_loss = loss.item()
                        test_psnr = psnr.item()
                        test_losses.append(test_loss)
                        test_psnrs.append(test_psnr)
                        test_bar.set_postfix({"Loss": f"{test_loss:.4f}", "PSNR": f"{test_psnr:.4f}"})
                writer.add_scalar("Loss/Test", sum(test_losses) / len(test_losses), i + 1)
                writer.add_scalar("PSRN/test", sum(test_psnrs) / len(test_psnrs), i + 1)

            # Log metrics and loss
            if (i + 1) % log_step:
                writer.add_scalar("Loss/train", train_loss, i + 1)
                writer.add_scalar("PSRN/train", train_psnr, i + 1)
                
            # Save model parameters
            if (i + 1) % save_step == 0:
                save_model_parameters(save_base_dir=save_dir, coarse_nerf=coarse_nerf, fine_nerf=fine_nerf, iteration=i+1)

            # Adjust learning rate
            decay_rate = 0.1
            decay_steps = lr_decay * 1000
            new_lrate = lr * (decay_rate ** (i / decay_steps))
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lrate
    
    writer.close()