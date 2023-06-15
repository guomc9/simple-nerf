import torch

def get_psnr(mse):
    """get psnr from rgb mse

    Args:
        mse (torch.Tensor): rgb mean squared error
    Returns:
        torch.Tensor: psnr
    """
    return -10. * torch.log(mse) / torch.log(torch.Tensor([10.]))