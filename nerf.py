import torch

class PositionEncoder():
    def __init__(self, L):
        super().__init__()
        self.L = L

    def __call__(self, input):
        """forward

        Args:
            input (torch.Tensor): [N_rays x N_samples, 3]

        Returns:
            torch.Tensor: [N_rays x N_samples, 3 x 2 x L]
        """
        output = []
        for i in range(0, self.L):
            s = torch.sin((2.**i) * torch.pi * input)
            c = torch.cos((2.**i) * torch.pi * input)
            output.append(s)
            output.append(c)
        return torch.cat(output, dim=1)


class NeRF(torch.nn.Module):
    def __init__(self, L_x=10, L_d=4, L=8, skips=[4], ch=256):
        super().__init__()
        self.pe_x = PositionEncoder(L_x)
        self.pe_d = PositionEncoder(L_d)
        self.skips = skips
        self.L = L
        self.backbone = torch.nn.ModuleList()
        self.backbone.append(torch.nn.Linear(L_x * 6, ch))
        for i in range(1, L-1):
            if i not in skips:
                self.backbone.append(torch.nn.Linear(ch, ch))
            else:
                self.backbone.append(torch.nn.Linear(L_x * 6 + ch, ch))
        self.sigma_linear = torch.nn.Linear(ch, 1)
        self.view_linear = torch.nn.Linear(L_d * 6 + ch, ch // 2)
        self.rgb_linear = torch.nn.Linear(ch // 2, 3)
                

    def forward(self, rays_samples, view_dirs):
        """forward

        Args:
            rays_samples (torch.Tensor): [N_rays x N_samples, 3]
            view_dirs (torch_Tensor): [N_rays x N_samples, 3]

        Returns:
            torch.Tensor: rgb, [N_rays x N_samples, 3]
            torch.Tensor: sigma, [N_rays x N_samples, 1]
        """
        f_x = gamma_x = self.pe_x(rays_samples)   # [N_rays x N_samples, 60]
        f_d = gamma_d = self.pe_d(view_dirs)      # [N_rays x N_samples, 24]
        for i in range(0, self.L):
            if i not in self.skips:
                f_x = self.backbone[i](f_x)     # [N_rays x N_samples, 256/60] -> [N_rays x N_samples, 256]
            else:
                f_x = self.backbone[i](torch.cat([f_x, gamma_x], dim=1))    # [N_rays x N_samples, 256+60] -> [N_rays x N_samples, 256]
            if i < self.L-1:
                f_x = torch.relu(f_x)           # [N_rays x N_samples, 256]

        sigma = self.sigma_linear(f_x)      # [N_rays x N_samples, 1]
        sigma = torch.relu(sigma)           # [N_rays x N_samples, 1]

        f = self.view_linear(torch.cat([f_x, gamma_d], dim=1))
        f = torch.relu(f)
        rgb = torch.rgb_linear(f)   # [N_rays x N_samples, 3]
        rgb = torch.sigmoid(rgb)    # [N_rays x N_samples, 3]
        return rgb, sigma
