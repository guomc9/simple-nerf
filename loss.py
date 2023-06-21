import torch

def get_mse_loss(rgb_map, gt_rgb_map):
    """get mean squared error loss between estimation and ground truth

    Args:
        rgb_map (torch.Tensor): rgb, [batch_size, 3]
        gt_rgb_map (torch.Tensor): ground truth rgb, [batch_size, 3]
    """
    return torch.mean((rgb_map - gt_rgb_map) ** 2)