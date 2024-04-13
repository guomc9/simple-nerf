import numpy as np
import torch
from scipy.spatial.transform import Rotation

def str2bool(x):
    return x.lower() in ('true')

def get_translation_matrix(move):
    transform = np.identity(4, dtype=np.float32)
    transform[0, 3] = move[0]
    transform[1, 3] = move[1]
    transform[2, 3] = move[2]
    return transform

def get_rotation_matrix(axis, angle):
    rotation = Rotation.from_rotvec(axis * np.deg2rad(angle))
    rotation_matrix = rotation.as_matrix()
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = rotation_matrix
    return transform

def get_rays_np(H, W, K, view_dir_z, c2w):
    """get rays(origin, directory) in numpy

    Args:
        H (int): screen height
        W (int): screen width
        K (numpy.ndarray): pixel to camera coordinate system, [3, 3]
            K = np.array([
                [focal, 0, 0.5*W],
                [0, focal, 0.5*H],
                [0, 0, 1]
            ])
        view_dir_z (int): look in postive(1) or negative(-1) Z-axis directory
        c2w (numpy.ndarray): camera to world coordinate system, [N_images, 3, 4]

    Returns:
        numpy.ndarray: rays origin, [N_images, H*W, 3]
        numpy.ndarray: rays directory, [N_images, H*W, 3]
    """
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')                           # i: [H, W], j: [H, W]
    dirs = np.stack([-view_dir_z*(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], view_dir_z*np.ones_like(i)], -1).reshape(-1, 3)     # [H*W, 3]
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.einsum('nij, pj->npi', c2w[..., :3, :3], dirs)                              # [N_images, H*W, 3]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.expand_dims(c2w[...,3], axis=1)                                             # [N_images, 3] -> [N_images, 1, 3]
    rays_o = np.tile(rays_o, (1, H*W, 1))                                                   # [N_images, H*W, 3]
    return rays_o, rays_d

def uniform_sample_rays(rays_o, rays_d, near, far, N_samples):
    """uniform sample rays

    Args:
        rays_o (torch.Tensor): rays origin, [N_rays, 3]
        rays_d (torch.Tensor): rays directory, [N_rays, 3]
        near (float): near plane Z value
        far (float): far plane Z value
        N_samples (int): number of position samples

    Returns:
       torch.Tensor: rays query, [N_rays, N_samples, 3]
       torch.Tensor: t values, [N_rays, N_samples]
    """
    device = rays_o.device
    N_rays = rays_o.shape[0]
    eta = torch.rand(size=[N_rays, N_samples])                                  # [N_rays, N_samples]
    bins = torch.linspace(near, far, steps=N_samples+1)                         # [N_samples+1]
    lower_bins = bins[None,:-1].expand(size=[N_rays, N_samples])                # [N_samples] -> [N_rays, N_samples]
    upper_bins = bins[None,1:].expand(size=[N_rays, N_samples])                 # [N_samples] -> [N_rays, N_samples]
    t_vals = (lower_bins * (1 - eta) + upper_bins * eta).to(device)             # [N_rays, N_samples]
    rays_q = rays_o[:,None,:] + rays_d[:,None,:] * t_vals[...,None]             # [N_rays, N_samples, 3] = [N_rays, 1, 3] +  [N_rays, 1, 3] * [N_rays, N_samples, 1]
    return rays_q, t_vals
    
def integrate(rgb, sigma, rays_d, t_vals):
    """integrate rgb, alpha, rays_d and t_vals to rgb_map

    Args:
        rgb (torch.Tensor): [N_rays, N_samples, 3]
        sigma (torch.Tensor): [N_rays, N_samples]
        rays_d (torch.Tensor): [N_rays, 3]
        t_vals (torch.Tensor): [N_rays, N_samples]

    Returns:
        torch.Tensor: rgb_map, [N_rays, 3]
        torch.Tensor: weights, [N_rays, N_samples]
    """
    device = t_vals.device
    dists = t_vals[...,1:] - t_vals[...,:-1]                                                                            # [N_rays, N_samples-1]
    dists = torch.cat([dists, torch.tensor([1e10], device=device).expand(dists[...,:1].shape)], -1)                     # [N_rays, N_samples]
    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)                                                              # [N_rays, N_samples] = [N_rays, N_samples] * [N_rays, 1]
    alpha = 1 - torch.exp(-sigma * dists)                                                                               # [N_rays, N_samples]
    T = torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)).to(device), 1.-alpha + 1e-10], -1), -1)[:, :-1]        # [N_rays, N_samples]
    weights = alpha * T                                                                                                 # [N_rays, N_samples]
    rgb_map = torch.sum(rgb * weights[...,None], dim=1)                                                                 # [N_rays, N_samples, 3] -> [N_rays, 3]
    return rgb_map, weights

def importance_sample_rays(rays_o, rays_d, t_vals, weights, N_imp_samples):
    """importance sample rays

    Args:
        rays_o (torch.Tensor): rays origin, [N_rays, 3]
        rays_d (torch.Tensor): rays directory, [N_rays, 3]
        t_vals (torch.Tensor): [N_rays, N_samples]
        weights (torch.Tensor): [N_rays, N_samples]
        N_imp_samples (int): number of importance samples

    Returns:
        torch.Tensor: rays query, [N_rays, N_imp_samples+N_samples, 3]
        torch.Tensor: samples, [N_rays, N_imp_samples+N_samples]
    """
    device = rays_o.device
    weights = weights[...,1:-1] + 1e-5                                              # [N_rays, N_samples-2]
    bins = 0.5 * (t_vals[...,:-1] + t_vals[...,1:])                                 # [N_rays, N_samples-1]
    # integrate pdf to cdf
    pdf = weights / torch.sum(weights, dim=-1, keepdim=True)                        # [N_rays, N_samples-2]
    cdf = torch.cumsum(pdf, dim=-1)                                                 # [N_rays, N_samples-2]
    cdf = torch.cat([torch.zeros_like(cdf[...,:1], device=device), cdf], dim=-1)    # [N_rays, N_samples-1]
    # invert transform sampling
    u = torch.rand(list(cdf.shape[:-1]) + [N_imp_samples], device=device)           # uniform sample cdf: [N_rays, N_imp_samples]
    inds = torch.searchsorted(cdf, u, right=True)                                   # bin indices: [N_rays, N_imp_samples]
    below = torch.max(torch.zeros_like(inds-1), inds-1)                             # [N_rays, N_imp_samples]
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)            # [N_rays, N_imp_samples]
    inds_g = torch.stack([below, above], -1)                                        # [N_rays, N_imp_samples, 2]
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]               # matched_shape = [N_rays, N_imp_samples, N_samples-1]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)         # [N_rays, N_imp_samples, 2] <- [N_rays, N_imp_samples, N_samples-1]
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)       # [N_rays, N_imp_samples, 2] <- [N_rays, N_imp_samples, N_samples-1]
    denom = (cdf_g[...,1] - cdf_g[...,0])                                           # [N_rays, N_imp_samples]
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)                # [N_rays, N_imp_samples]
    t = (u - cdf_g[...,0]) / denom                                                  # [N_rays, N_imp_samples]
    imp_samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])                 # [N_rays, N_imp_samples]
    imp_samples.detach()_                                                            # [N_rays, N_imp_samples]
    # hierarchical sampling
    samples, _ = torch.sort(torch.cat([imp_samples, t_vals], dim=-1), dim=-1)       # [N_rays, N_imp_samples+N_samples]
    rays_q = rays_o[:,None,:] + samples[...,None] * rays_d[:,None,:]                # [N_rays, N_imp_samples+N_samples, 3] = [N_rays, 1, 3] + [N_rays, N_imp_samples+N_samples, 1] * [N_rays, 1, 3]
    return rays_q, samples

