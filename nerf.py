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
            s = torch.sin((2.**i) * input)
            c = torch.cos((2.**i) * input)
            output.append(s)
            output.append(c)
        return torch.cat(output, dim=1)


class NeRF(torch.nn.Module):
    def __init__(self, L_x=10, L_d=4, L=8, in_x_ch=3, in_v_ch=3, skips=[4], mlp_ch=256):
        super().__init__()
        self.pe_x = PositionEncoder(L_x)
        self.pe_d = PositionEncoder(L_d)
        self.skips = skips
        self.L = L
        embed_x_ch = in_x_ch * (L_x * 2 + 1)
        embed_v_ch = in_v_ch * (L_d * 2 + 1)
        self.backbone = torch.nn.ModuleList()
        self.backbone.append(torch.nn.Linear(embed_x_ch, mlp_ch))
        for i in range(L-1):
            if i not in skips:
                self.backbone.append(torch.nn.Linear(mlp_ch, mlp_ch))
            else:
                self.backbone.append(torch.nn.Linear(embed_x_ch + mlp_ch, mlp_ch))
        self.feature_linear = torch.nn.Linear(mlp_ch, mlp_ch)
        self.sigma_linear = torch.nn.Linear(mlp_ch, 1)
        self.view_linear = torch.nn.Linear(embed_v_ch + mlp_ch, mlp_ch // 2)
        self.rgb_linear = torch.nn.Linear(mlp_ch // 2, 3)
                

    def forward(self, rays_samples, view_dirs):
        """forward

        Args:
            rays_samples (torch.Tensor): [N_rays x N_samples, 3]
            view_dirs (torch_Tensor): [N_rays x N_samples, 3]

        Returns:
            torch.Tensor: rgb, [N_rays x N_samples, 3]
            torch.Tensor: sigma, [N_rays x N_samples, 1]
        """
        f_x = gamma_x = torch.cat([rays_samples ,self.pe_x(rays_samples)], dim=-1)          # [N_rays x N_samples, 63]
        f_d = gamma_d = torch.cat([view_dirs, self.pe_d(view_dirs)], dim=-1)                # [N_rays x N_samples, 27]
        for i in range(0, self.L):
            f_x = self.backbone[i](f_x)                                 # [N_rays x N_samples, 256/256+63] -> [N_rays x N_samples, 256]
            f_x = torch.relu(f_x) 
            if i in self.skips:
                f_x = torch.cat([f_x, gamma_x], dim=-1)                 # [N_rays x N_samples, 256+63]

        sigma = self.sigma_linear(f_x)                                  # [N_rays x N_samples, 1]
        sigma = torch.relu(sigma)                                       # [N_rays x N_samples, 1]
        f = self.feature_linear(f_x)                                    # [N_rays x N_samples, 256]
        f = self.view_linear(torch.cat([f, gamma_d], dim=-1))           # [N_rays x N_samples, 256+27] -> [N_rays x N_samples, 256]
        f = torch.relu(f)                                               # [N_rays x N_samples, 256]
        rgb = self.rgb_linear(f)                                        # [N_rays x N_samples, 3]
        rgb = torch.sigmoid(rgb)                                        # [N_rays x N_samples, 3]
        return rgb, sigma
