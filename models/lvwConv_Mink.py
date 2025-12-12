import torch
import torch.nn as nn
import MinkowskiEngine as ME
import torch.nn.functional as F

    
class lvwConv3d_learnden_CSatt_2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, den=None,
                 den_init="const", den_value=0.75,
                 stride=1, padding=1, dilation=1, bias=False, dimension=3,
                 enforce_positive=False, alpha_attn="Sigmoid",
                 den_reg_type=None, den_reg_weight=1e-4,
                 att_alpha_range=0.15):
        super().__init__()
        # 基本属性
        self.kernel_size = kernel_size if isinstance(kernel_size, (tuple, list)) else (kernel_size,) * 3
        assert all(k % 2 == 1 for k in self.kernel_size), "kernel_size must be odd"
        self.stride = stride if isinstance(stride, (tuple, list)) else (stride,) * 3
        self.dilation = dilation if isinstance(dilation, (tuple, list)) else (dilation,) * 3
        self.padding = padding if isinstance(padding, (tuple, list)) else (padding,) * 3
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dimension = dimension

        # den / 正则设置
        self.enforce_positive = enforce_positive
        self.den_reg_type = den_reg_type
        self.den_reg_weight = den_reg_weight
        self.alph_attn = alpha_attn
        self.att_alpha_range = float(att_alpha_range)

        # dense core 
        kD, kH, kW = self.kernel_size
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kD, kH, kW))
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')

        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

        if den is None:
            L = (self.kernel_size[0] - 1) // 2
            if den_init == "const":
                den_tensor = torch.full((L,), float(den_value))
            elif den_init == "uniform":
                den_tensor = torch.rand(L) * 0.1 + float(den_value)
            elif den_init == "normal":
                den_tensor = torch.normal(mean=float(den_value), std=0.05, size=(L,))
            else:
                raise ValueError(f"Unsupported den_init: {den_init}")
        else:
            den_tensor = torch.tensor(den, dtype=torch.float32)
        self.den = nn.Parameter(den_tensor)

        self.mink_conv = ME.MinkowskiConvolution(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            dilation=self.dilation,
            bias=bias,
            dimension=dimension
        )

        if hasattr(self.mink_conv, "_parameters") and "kernel" in self.mink_conv._parameters:
            self.mink_conv._parameters.pop("kernel", None)
        object.__setattr__(self.mink_conv, "kernel", torch.empty(0))


        L = (self.kernel_size[0] - 1) // 2
        K = 2 * L + 1
        hidden = max(4, L)
        # alpha attention 
        self.alpha_att_net = nn.Sequential(
            nn.Linear(max(1, L), hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, K),
            nn.Sigmoid()
        )
        self.alpha_raw_att_net = nn.Sequential(
            nn.Linear(max(1, L), hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, K)
        )

        # channel attention
        ch_hidden = max(8, self.in_channels // 4)
        self.channel_att = nn.Sequential(
            nn.Linear(self.in_channels, ch_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(ch_hidden, self.out_channels),
            nn.Sigmoid()
        )

        self.kernel_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.skip_scale = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def forward(self, x: ME.SparseTensor, cross_pos_scale=None, cross_ch_scale=None):
        device = None
        if hasattr(x, 'F') and x.F is not None:
            device = x.F.device
        else:
            device = self.weight.device


        den = F.softplus(self.den) if self.enforce_positive else self.den  # [L]
        center = den.new_tensor([1.0])
        alpha = torch.cat([den, center, torch.flip(den, dims=[0])], dim=0)  # [K]

        
        L = (self.kernel_size[0] - 1) // 2
        if self.alph_attn == "Sigmoid":
            if L > 0:
                att_alpha = self.alpha_att_net(den.view(1, -1).to(device)).squeeze(0)  # [K], in (0,1)
            else:
                att_alpha = alpha.new_ones(alpha.shape).to(device)
            att_alpha = 1.0 + (att_alpha - 0.5) * self.att_alpha_range  # [1-Range/2,1+Range/2]
        else:
            if L > 0:
                raw_alpha = self.alpha_raw_att_net(den.view(1, -1).to(device)).squeeze(0)  # [K]
                att_alpha = F.softmax(raw_alpha, dim=0)
                att_alpha = 1.0 + (att_alpha - (1.0 / att_alpha.numel())) * (self.att_alpha_range * 4/3)
            else:
                att_alpha = alpha.new_ones(alpha.shape).to(device)

        if cross_pos_scale is not None:
            scale = cross_pos_scale.norm(dim=-1, keepdim=True)
            att_alpha = att_alpha * (1.0 + 0.5 * scale.clamp(-0.5, 0.5).to(device))

        
        alpha = (alpha.to(device) * att_alpha).clamp(min=1e-4, max=10.0)

        
        Phi = torch.einsum('i,j,k->ijk', alpha, alpha, alpha)  # [kD,kH,kW]
        Phi = Phi.to(self.weight.device)
        Phi_mean = Phi.mean()
        Phi = Phi / (Phi_mean + 1e-6)

        
        weight_Phi = self.weight * Phi.unsqueeze(0)  # [out, in, kD, kH, kW]
        
        weight_Phi = self.kernel_scale * weight_Phi

        
        if x is not None and getattr(x, "F", None) is not None and x.F.numel() > 0:
            context = x.F.mean(dim=0, keepdim=True).to(self.weight.device)  # [1, in_channels]
            ch_scale = self.channel_att(context).squeeze(0)  # [out_channels]
        else:
            ch_scale = weight_Phi.new_ones(self.out_channels)

        # map to 1 +/- small, then apply cross_ch_scale residual
        ch_scale = 1.0 + (ch_scale - 0.5) * 0.3
        if cross_ch_scale is not None:
            ch_scale = ch_scale * (1.0 + 0.5 * cross_ch_scale.clamp(-0.5, 0.5).to(ch_scale.device))

        ch_scale = ch_scale.clamp(0.7, 1.3).view(self.out_channels, 1, 1, 1, 1)
        weight_Phi = weight_Phi * ch_scale

        
        kD, kH, kW = self.kernel_size
        weight_ME = weight_Phi.permute(2, 3, 4, 1, 0).reshape(kD * kH * kW, self.in_channels, self.out_channels)
        weight_ME = weight_ME.contiguous().to(self.weight.device)

        try:
            if hasattr(self.mink_conv, "_parameters") and "kernel" in self.mink_conv._parameters:
                self.mink_conv._parameters.pop("kernel", None)
        except Exception:
            pass
        object.__setattr__(self.mink_conv, "kernel", weight_ME)
        out = self.mink_conv(x)  

        try:
            if hasattr(out, 'F') and hasattr(x, 'F') and out.F.shape == x.F.shape:
                newF = out.F + x.F * self.skip_scale
                
                if hasattr(out, 'replace_feature'):
                    out = out.replace_feature(newF)
                else:
                    out.F = newF
        except Exception:
            pass

        return out

    def den_regularization_loss(self):
        den = F.softplus(self.den) if self.enforce_positive else self.den
        if self.den_reg_type in [None, "none"]:
            return torch.tensor(0.0, device=den.device)
        elif self.den_reg_type == "l2":
            reg_loss = torch.sum(den ** 2)
        elif self.den_reg_type == "l1":
            reg_loss = torch.sum(torch.abs(den))
        elif self.den_reg_type == "sum1":
            reg_loss = (torch.sum(den) - 1.0) ** 2
        else:
            raise ValueError(f"Unknown den_reg_type: {self.den_reg_type}")
        return self.den_reg_weight * reg_loss
    