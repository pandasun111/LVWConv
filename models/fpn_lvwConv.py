import torch.nn as nn
import torch
import MinkowskiEngine as ME
from MinkowskiEngine import MinkowskiNetwork
from MinkowskiEngine import MinkowskiReLU, MinkowskiInterpolation, MinkowskiELU
import torch.nn.functional as F

from .common import ConvType, NormType, conv, conv_tr, get_norm, sum_pool
from .lvwConv_Mink import lvwConv3d_learnden_CSatt_2

class BasicBlockBase(nn.Module):
    expansion = 1
    NORM_TYPE = NormType.BATCH_NORM

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        dilation=1,
        downsample=None,
        conv_type=ConvType.HYPERCUBE,
        bn_momentum=0.1,
        D=3,
    ):
        super(BasicBlockBase, self).__init__()

        self.conv1 = conv(inplanes, planes, kernel_size=3, stride=stride, dilation=dilation, conv_type=conv_type, D=D)
        self.norm1 = get_norm(self.NORM_TYPE, planes, D, bn_momentum=bn_momentum)
        self.conv2 = conv(
            planes, planes, kernel_size=3, stride=1, dilation=dilation, bias=False, conv_type=conv_type, D=D
        )
        self.norm2 = get_norm(self.NORM_TYPE, planes, D, bn_momentum=bn_momentum)
        self.relu = MinkowskiReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BasicBlock(BasicBlockBase):
    NORM_TYPE = NormType.BATCH_NORM


class BasicBlockIN(BasicBlockBase):
    NORM_TYPE = NormType.INSTANCE_NORM


class BasicBlockINBN(BasicBlockBase):
    NORM_TYPE = NormType.INSTANCE_BATCH_NORM


class ResNetBase(MinkowskiNetwork):
    BLOCK = None
    LAYERS = ()
    INIT_DIM = 64
    PLANES = (64, 128, 256, 512)
    OUT_PIXEL_DIST = 32
    HAS_LAST_BLOCK = False
    CONV_TYPE = ConvType.HYPERCUBE

    def __init__(self, in_channels, out_channels, D, conv1_kernel_size=3, dilations=[1, 1, 1, 1], **kwargs):
        super(ResNetBase, self).__init__(D)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1_kernel_size = conv1_kernel_size
        self.dilations = dilations
        assert self.BLOCK is not None
        assert self.OUT_PIXEL_DIST > 0

        self.network_initialization(in_channels, out_channels, D)
        self.weight_initialization()

    def network_initialization(self, in_channels, out_channels, D):
        def space_n_time_m(n, m):
            return n if D == 3 else [n, n, n, m]

        if D == 4:
            self.OUT_PIXEL_DIST = space_n_time_m(self.OUT_PIXEL_DIST, 1)

        dilations = self.dilations
        bn_momentum = 1
        self.inplanes = self.INIT_DIM
        self.conv1 = conv(
            in_channels, self.inplanes, kernel_size=space_n_time_m(self.conv1_kernel_size, 1), stride=1, D=D
        )

        self.bn1 = get_norm(NormType.BATCH_NORM, self.inplanes, D=self.D, bn_momentum=bn_momentum)
        self.relu = ME.MinkowskiReLU(inplace=True)
        self.pool = sum_pool(kernel_size=space_n_time_m(2, 1), stride=space_n_time_m(2, 1), D=D)

        self.layer1 = self._make_layer(
            self.BLOCK,
            self.PLANES[0],
            self.LAYERS[0],
            stride=space_n_time_m(2, 1),
            dilation=space_n_time_m(dilations[0], 1),
        )
        self.layer2 = self._make_layer(
            self.BLOCK,
            self.PLANES[1],
            self.LAYERS[1],
            stride=space_n_time_m(2, 1),
            dilation=space_n_time_m(dilations[1], 1),
        )
        self.layer3 = self._make_layer(
            self.BLOCK,
            self.PLANES[2],
            self.LAYERS[2],
            stride=space_n_time_m(2, 1),
            dilation=space_n_time_m(dilations[2], 1),
        )
        self.layer4 = self._make_layer(
            self.BLOCK,
            self.PLANES[3],
            self.LAYERS[3],
            stride=space_n_time_m(2, 1),
            dilation=space_n_time_m(dilations[3], 1),
        )

        self.final = conv(self.PLANES[3] * self.BLOCK.expansion, out_channels, kernel_size=1, bias=True, D=D)

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, norm_type=NormType.BATCH_NORM, bn_momentum=0.1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False, D=self.D),
                get_norm(norm_type, planes * block.expansion, D=self.D, bn_momentum=bn_momentum),
            )
        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride=stride,
                dilation=dilation,
                downsample=downsample,
                conv_type=self.CONV_TYPE,
                D=self.D,
            )
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, stride=1, dilation=dilation, conv_type=self.CONV_TYPE, D=self.D))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.final(x)
        return x



class CSAFModule_1(nn.Module):
    """
    轻量级跨尺度注意力生成模块（稳健版）
    返回:
        pos_scale: Tensor (K,)  -> 表示对 alpha 的小幅度“增量”（上游请以 1+pos_scale 融合）
        ch_scale:  Tensor (out_channels,) -> 表示对 channel 的小幅度“增量”
    """
    def __init__(self, in_channels, guide_channels, D=3, kernel_size=3,
                 hidden_dim=None, pos_scale_factor=0.2, ch_scale_factor=0.2, out_channels=None):
        super().__init__()
        self.D = D
        self.in_channels = in_channels
        self.guide_channels = guide_channels
        self.out_channels = out_channels if out_channels is not None else in_channels

        if isinstance(kernel_size, (tuple, list)):
            k0 = kernel_size[0]
        else:
            k0 = kernel_size
        L = (k0 - 1) // 2
        self.K = 2 * L + 1

        # guide_compress
        self.guide_compress = ME.MinkowskiConvolution(guide_channels, in_channels, kernel_size=1, dimension=D)

        if hidden_dim is None:
            hidden_dim = max(16, in_channels // 2)
        self.context_proj = nn.Sequential(
            nn.Linear(in_channels * 2, hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.pos_head = nn.Linear(hidden_dim, self.K)
        self.ch_head = nn.Linear(hidden_dim, self.out_channels)

        self.pos_scale_factor = float(pos_scale_factor)
        self.ch_scale_factor = float(ch_scale_factor)

        nn.init.zeros_(self.pos_head.weight)
        nn.init.zeros_(self.pos_head.bias)
        nn.init.zeros_(self.ch_head.weight)
        nn.init.zeros_(self.ch_head.bias)

    def forward(self, x: ME.SparseTensor, guide_feat):
        device = None
        if hasattr(x, 'F') and x.F is not None:
            device = x.F.device
        elif isinstance(guide_feat, ME.SparseTensor) and guide_feat.F is not None:
            device = guide_feat.F.device
        else:
            device = torch.device('cpu')

        # x 上下文
        if x is None or x.F is None or x.F.numel() == 0:
            x_ctx = torch.zeros(self.in_channels, device=device)
        else:
            x_ctx = x.F.mean(dim=0).to(device)

        # guide context
        g_ctx = torch.zeros(self.in_channels, device=device)
        if guide_feat is not None:
            if isinstance(guide_feat, ME.SparseTensor):
                try:
                    guide_compressed = self.guide_compress(guide_feat)
                    if guide_compressed is not None and guide_compressed.F is not None and guide_compressed.F.numel() > 0:
                        g_ctx = guide_compressed.F.mean(dim=0).to(device)
                except Exception:
                    if hasattr(guide_feat, 'F') and guide_feat.F is not None:
                        t = guide_feat.F.mean(dim=0)
                        n = min(self.in_channels, t.numel())
                        tmp = torch.zeros(self.in_channels, device=device)
                        tmp[:n] = t[:n].to(device)
                        g_ctx = tmp
            elif isinstance(guide_feat, torch.Tensor):
                t = guide_feat.to(device)
                if t.ndim == 1:
                    vec = t
                else:
                    vec = t.mean(dim=0)
                tmp = torch.zeros(self.in_channels, device=device)
                n = min(self.in_channels, vec.numel())
                tmp[:n] = vec[:n].to(device)
                g_ctx = tmp
            else:
                g_ctx = torch.zeros(self.in_channels, device=device)

        ctx = torch.cat([x_ctx, g_ctx], dim=0).unsqueeze(0)
        h = self.context_proj(ctx)

        pos_raw = self.pos_head(h).squeeze(0)
        ch_raw = self.ch_head(h).squeeze(0)

        pos_scale = torch.tanh(pos_raw) * self.pos_scale_factor
        ch_scale = torch.tanh(ch_raw) * self.ch_scale_factor

        # clamp 防止过大
        pos_scale = pos_scale.clamp(-0.5, 0.5)
        ch_scale = ch_scale.clamp(-0.5, 0.5)

        return pos_scale, ch_scale


class Res16FPNBase_lvwConv_4_2_CSattn_2(ResNetBase):
    BLOCK = None
    PLANES = (32, 64, 128, 256)
    LAYERS = (2, 2, 2, 2)
    INIT_DIM = 32
    OUT_PIXEL_DIST = 1
    NORM_TYPE = NormType.BATCH_NORM
    NON_BLOCK_CONV_TYPE = ConvType.SPATIAL_HYPERCUBE
    CONV_TYPE = ConvType.SPATIAL_HYPERCUBE_TEMPORAL_HYPERCROSS

    def __init__(self, in_channels, out_channels, D=3, conv1_kernel_size=5, config=None):
        super().__init__(in_channels, out_channels, D, conv1_kernel_size)
        self.D = D
        self.conv1_kernel_size = conv1_kernel_size
        self.config = config
        self._build_network(in_channels, out_channels)

    def _make_wconv_seq(self, channels, guide_channels=None, do_pool=False):
        conv_layers = [
            lvwConv3d_learnden_CSatt_2(channels, channels, kernel_size=3, den_init="normal",
                                   enforce_positive=True, den_reg_type="l2", den_reg_weight=1e-4, dimension=self.D),
            ME.MinkowskiReLU(inplace=True)
        ]
        if do_pool:
            conv_layers.append(ME.MinkowskiSumPooling(kernel_size=2, stride=2, dimension=self.D))
        conv_layers += [
            lvwConv3d_learnden_CSatt_2(channels, channels, kernel_size=5, den_init="normal",
                                   enforce_positive=True, den_reg_type="l2", den_reg_weight=1e-4, dimension=self.D),
            ME.MinkowskiReLU(inplace=True)
        ]
        conv_seq = nn.Sequential(*conv_layers)
        if guide_channels is not None:
            return nn.ModuleList([
                CSAFModule_1(in_channels=channels, guide_channels=guide_channels, D=self.D,
                           kernel_size=3, pos_scale_factor=0.2, ch_scale_factor=0.2, out_channels=channels),
                conv_seq
            ])
        else:
            return conv_seq

    def _build_network(self, in_channels, out_channels):
        D = self.D
        self.inplanes = self.INIT_DIM
        bn_momentum = 0.02

        def space_n_time_m(n, m):
            return n if D == 3 else [n, n, n, m]

        # conv0p1s1
        self.wconv0 = self._make_wconv_seq(in_channels)  # raw
        self.conv0p1s1 = conv(
            in_channels, self.inplanes,
            kernel_size=space_n_time_m(self.conv1_kernel_size, 1),
            stride=1, dilation=1,
            conv_type=self.NON_BLOCK_CONV_TYPE, D=D
        )
        self.bn0 = get_norm(self.NORM_TYPE, self.inplanes, D, bn_momentum=bn_momentum)

        self.blocks = nn.ModuleList()
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.wconv_layers = nn.ModuleList()

        for idx, (planes, n_layers) in enumerate(zip(self.PLANES, self.LAYERS)):
            guide_channels = self.PLANES[idx-1] if idx > 0 else None
            self.wconv_layers.append(self._make_wconv_seq(self.inplanes, guide_channels=guide_channels))
            conv_layer = conv(
                self.inplanes, self.inplanes,
                kernel_size=space_n_time_m(2, 1),
                stride=space_n_time_m(2, 1),
                dilation=1, conv_type=self.NON_BLOCK_CONV_TYPE, D=D
            )
            self.conv_layers.append(conv_layer)
            self.bn_layers.append(get_norm(self.NORM_TYPE, self.inplanes, D, bn_momentum=bn_momentum))
            
            try:
                block_layer = self._make_layer(self.BLOCK, planes, n_layers,
                                               dilation=1, norm_type=self.NORM_TYPE,
                                               bn_momentum=bn_momentum)
            except Exception:
                # fallback: identity block
                block_layer = nn.Identity()
            self.blocks.append(block_layer)

        self.delayer1 = ME.MinkowskiLinear(256, 128, bias=False)
        self.delayer2 = ME.MinkowskiLinear(128, 128, bias=False)
        self.delayer3 = ME.MinkowskiLinear(64, 128, bias=False)
        self.delayer4 = ME.MinkowskiLinear(32, 128, bias=False)
        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x):
        y = x.sparse()

        if isinstance(self.wconv0, nn.Sequential):
            out = self.wconv0(y)
        else:
            out = self.wconv0(y)

        out = self.conv0p1s1(out)
        out = self.bn0(out)
        out_p1 = self.relu(out)

        out = out_p1
        out_blocks = []
        prev_feat = out  

        for idx, (conv_layer, bn_layer, block_layer, wconv_module) in enumerate(zip(
            self.conv_layers, self.bn_layers, self.blocks, self.wconv_layers
        )):
            pre_down = out

            if idx > 0 and isinstance(wconv_module, nn.ModuleList):
                csaf_module, conv_seq = wconv_module
                cross_pos_scale, cross_ch_scale = csaf_module(pre_down, prev_feat)

                if cross_pos_scale is None:
                    L = (conv_seq[0].kernel_size[0] - 1) // 2 if hasattr(conv_seq[0], 'kernel_size') else (self.conv1_kernel_size - 1) // 2
                    K = 2 * L + 1
                    cross_pos_scale = torch.zeros(K, device=pre_down.F.device)
                if cross_ch_scale is None:
                    cross_ch_scale = torch.zeros(self.PLANES[idx] if idx < len(self.PLANES) else out.F.shape[1], device=pre_down.F.device)

                out_wconv = pre_down
                for m in conv_seq:
                    if isinstance(m, lvwConv3d_learnden_CSatt_2):
                        out_wconv = m(out_wconv, cross_pos_scale, cross_ch_scale)
                    else:
                        out_wconv = m(out_wconv)
                out = out_wconv
            else:
                # 没有 guide 的情况（通常 idx==0）
                out = wconv_module(out)

            # 下采样等
            out = conv_layer(out)
            out = bn_layer(out)
            out = self.relu(out)
            out = block_layer(out)

            out_blocks.append(out)
            prev_feat = pre_down

        # 取四个尺度输出（可能长度小于预期，视网络深度）
        # 若 out_blocks 少于 4，做安全取值
        padded = out_blocks + [out_blocks[-1]] * (4 - len(out_blocks)) if len(out_blocks) > 0 else [out] * 4
        out_b1p2, out_b2p4, out_b3p8, out_b4p16 = padded[:4]

        out = self.delayer1(out_b4p16).interpolate(x)
        dout_b3p8 = self.delayer2(out_b3p8).interpolate(x)
        dout_b2p4 = self.delayer3(out_b2p4).interpolate(x)
        dout_b1p2 = self.delayer4(out_b1p2).interpolate(x)

        # 返回 dense Tensor (N_points, C)
        out = out.F + dout_b3p8.F + dout_b2p4.F + dout_b1p2.F
        return out

    def reg_loss(self):
        total_reg = 0.0
        # wconv0
        try:
            if isinstance(self.wconv0, (list, tuple, nn.ModuleList, nn.Sequential)):
                for m in self.wconv0:
                    if hasattr(m, 'den_regularization_loss'):
                        total_reg += m.den_regularization_loss()
            elif hasattr(self.wconv0, 'den_regularization_loss'):
                total_reg += self.wconv0.den_regularization_loss()
        except Exception:
            pass

        # 每个 stage 的 wconv_layers
        for wconv_layer in self.wconv_layers:
            if isinstance(wconv_layer, nn.ModuleList):
                for m in wconv_layer:
                    if hasattr(m, 'den_regularization_loss'):
                        total_reg += m.den_regularization_loss()
            elif isinstance(wconv_layer, nn.Sequential):
                for m in wconv_layer:
                    if hasattr(m, 'den_regularization_loss'):
                        total_reg += m.den_regularization_loss()
            else:
                if hasattr(wconv_layer, 'den_regularization_loss'):
                    total_reg += wconv_layer.den_regularization_loss()
        return total_reg


class Res16FPN18_lvwConv_4_2_CSattn_2(Res16FPNBase_lvwConv_4_2_CSattn_2):
    BLOCK = BasicBlock
    LAYERS = (2, 2, 2, 2, 2, 2, 2, 2)





def get_block(norm_type, inplanes, planes, stride=1, dilation=1, downsample=None, bn_momentum=0.1, D=3):
    if norm_type == NormType.BATCH_NORM:
        return BasicBlock(
            inplanes=inplanes,
            planes=planes,
            stride=stride,
            dilation=dilation,
            downsample=downsample,
            bn_momentum=bn_momentum,
            D=D,
        )
    elif norm_type == NormType.INSTANCE_NORM:
        return BasicBlockIN(inplanes, planes, stride, dilation, downsample, bn_momentum, D)
    else:
        raise ValueError(f"Type {norm_type}, not defined")
