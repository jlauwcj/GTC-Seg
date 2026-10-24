import torch
import torch.nn as nn
from lib.pointops.functions import pointops
import torch.nn.functional as F
from copy import copy, deepcopy
import numpy as np
from torch.utils.checkpoint import checkpoint as torch_checkpoint
class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0], 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_() 
        return x.div(keep_prob) * random_tensor

@torch.jit.script
def vector_feature_optimized(x: torch.Tensor, knn: torch.Tensor):
    B, N, C = x.shape
    _, _, k = knn.shape
    batch_indices = torch.arange(B, device=x.device).view(B, 1, 1).expand(B, N, k)
    knn_features = x[batch_indices, knn] # (B, N, k, C)
    return knn_features.mean(dim=2)

class VFR(nn.Module):
    """Vector Feature Rotation / Aggregation"""
    def __init__(self, in_dim, out_dim, bn_momentum=0.1, init=0.):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.bn = nn.BatchNorm1d(out_dim, momentum=bn_momentum)
        nn.init.constant_(self.bn.weight, init)

    def forward(self, x, knn):
        B, N, C = x.shape
        x = self.linear(x)
        if self.training and knn is not None:
            x = vector_feature_optimized(x, knn)
        x = self.bn(x.reshape(B*N, -1)).view(B, N, -1)
        return x

class FFN(nn.Module):
    def __init__(self, in_dim, mlp_ratio=4.0, bn_momentum=0.1, act=nn.GELU, init=0.2):
        super().__init__()
        hid_dim = round(in_dim * mlp_ratio)
        self.fc1 = nn.Linear(in_dim, hid_dim)
        self.act = act()
        self.fc2 = nn.Linear(hid_dim, in_dim, bias=False)
        self.bn = nn.BatchNorm1d(in_dim, momentum=bn_momentum)
        nn.init.constant_(self.bn.weight, init)

    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B*N, -1)
        x = self.fc2(self.act(self.fc1(x)))
        x = self.bn(x)
        return x.view(B, N, -1)

class ResLFE_Block(nn.Module):
    def __init__(self, dim, depth=2, drop_path=0.1, mlp_ratio=4.0, use_checkpoint=False):
        super().__init__()
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.VFRs = nn.ModuleList([VFR(dim, dim) for _ in range(depth)])
        self.FFNs = nn.ModuleList([FFN(dim, mlp_ratio) for _ in range(depth)])
        self.drop_paths = nn.ModuleList([
            DropPath(dpr) for dpr in torch.linspace(0., drop_path, depth).tolist()
        ])

    def _forward_block(self, x, pe, knn, i: int):
        x = x + pe
        x = x + self.drop_paths[i](self.VFRs[i](x, knn))
        x = x + self.drop_paths[i](self.FFNs[i](x))
        return x

    def forward(self, x, pe, knn):
        for i in range(self.depth):
            if self.use_checkpoint and self.training:
                x = torch_checkpoint(self._forward_block, x, pe, knn, i, use_reentrant=False)
            else:
                x = self._forward_block(x, pe, knn, i)
        return x
class WeakLayer(nn.Module):
    """LEAP: Label-Efficient Attention for Point clouds"""
    def __init__(self, in_planes, out_planes, share_planes=8, nsample=16):
        super().__init__()
        self.nsample = nsample
        self.linear_q = nn.Linear(in_planes, out_planes)
        self.linear_k = nn.Linear(in_planes, out_planes)
        self.linear_v = nn.Linear(in_planes, out_planes)
        
        self.linear_p = nn.Sequential(
            nn.Linear(3, 3), nn.BatchNorm1d(3), nn.ReLU(inplace=True), nn.Linear(3, out_planes)
        )
        self.linear_uni_spatial = nn.Sequential(
            nn.Linear(in_planes, 1), nn.BatchNorm1d(1), nn.ReLU(inplace=True)
        )
        self.linear_add = nn.Sequential(
            nn.Linear(3, 3), nn.BatchNorm1d(3), nn.ReLU(inplace=True), nn.Linear(3, out_planes)
        )
        self.linear_w = nn.Sequential(
            nn.BatchNorm1d(out_planes), nn.ReLU(inplace=True),
            nn.Linear(out_planes, out_planes),
            nn.BatchNorm1d(out_planes), nn.ReLU(inplace=True),
            nn.Linear(out_planes, out_planes)
        )
        
        self.softmax = nn.Softmax(dim=1)
        self.a = nn.Parameter(torch.zeros(nsample, 1))
        self.b = nn.Parameter(torch.zeros(nsample, out_planes))
        self.d = nn.Parameter(torch.zeros(nsample, out_planes))
        nn.init.xavier_uniform_(self.a.data)
        nn.init.xavier_uniform_(self.b.data)
        nn.init.xavier_uniform_(self.d.data)

    def forward(self, pxo):
        p, x, o = pxo
        x_q, x_k, x_v = self.linear_q(x), self.linear_k(x), self.linear_v(x)
        x_uni_spatial = self.linear_uni_spatial(x)
        x_k = pointops.queryandgroup(self.nsample, p, p, x_k, None, o, o, use_xyz=True)
        x_v_j = pointops.queryandgroup(self.nsample, p, p, x_v, None, o, o, use_xyz=False)
        n, nsample, c = x_v_j.shape      
        e, x_k = x_k[:, :, 0:3], x_k[:, :, 3:]
        e_dis = torch.norm(e, dim=2, keepdim=True).clamp(min=1e-7)
        e_dis_xy = torch.norm(e[:, :, :2], dim=2, keepdim=True).clamp(min=1e-7)
        cos_theta_e = (e_dis_xy / e_dis).clamp(-1, 1)
        cos_theta_a = (e[:, :, 1:2] / e_dis_xy).clamp(-1, 1)
        all_feat = torch.cat([e_dis, cos_theta_a, cos_theta_e], dim=-1)
        
        p_r = e.clone()
        for i, layer in enumerate(self.linear_add):
            if i == 1: # BatchNorm1d 层
                all_feat = layer(all_feat.transpose(1, 2)).transpose(1, 2)
            else:
                all_feat = layer(all_feat)
        
        for i, layer in enumerate(self.linear_p):
            if i == 1:
                p_r = layer(p_r.transpose(1, 2)).transpose(1, 2)
            else:
                p_r = layer(p_r)
        
        p_r = p_r + self.d * all_feat
        w_spatial_d = F.softmax(torch.matmul((x_k - x_q.unsqueeze(1)).transpose(0, 2), x_uni_spatial), dim=1)
        w_spatial_e = F.softmax(torch.matmul(x, w_spatial_d.transpose(0, 1)).transpose(0, 1), dim=1)
        
        w = self.a * w_spatial_d.transpose(0, 2) + self.b * w_spatial_e + p_r
        
        for i, layer in enumerate(self.linear_w):
            if i in [0, 3]: # BatchNorm 层
                w = layer(w.transpose(1, 2)).transpose(1, 2)
            else:
                w = layer(w)
        
        w = self.softmax(w)
        return ((x_v_j + p_r) * w).sum(1).view(n, c)
class Downsampling(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, nsample=16):
        super().__init__()
        self.stride, self.nsample = stride, nsample
        if stride != 1:
            self.linear = nn.Linear(3 + in_planes, out_planes, bias=False)
            self.pool = nn.MaxPool1d(nsample)
        else:
            self.linear = nn.Linear(in_planes, out_planes, bias=False)
        self.bn = nn.BatchNorm1d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, pxo):
        p, x, o = pxo
        if self.stride != 1:
            n_o, count = [o[0].item() // self.stride], o[0].item() // self.stride
            for i in range(1, o.shape[0]):
                count += (o[i].item() - o[i-1].item()) // self.stride
                n_o.append(count)
            n_o = torch.tensor(n_o, dtype=torch.int32, device='cuda')
            idx = pointops.furthestsampling(p, o, n_o)
            n_p = p[idx.long(), :]
            x = pointops.queryandgroup(self.nsample, p, n_p, x, None, o, n_o, use_xyz=True)
            x = self.relu(self.bn(self.linear(x).transpose(1, 2)))
            x = self.pool(x).squeeze(-1)
            p, o = n_p, n_o
        else:
            x = self.relu(self.bn(self.linear(x)))
        return [p, x, o]

class FeaturePropagation(nn.Module):
    def __init__(self, in_planes, out_planes=None):
        super().__init__()
        if out_planes is None:
            self.linear1 = nn.Sequential(nn.Linear(2*in_planes, in_planes), nn.BatchNorm1d(in_planes), nn.ReLU(inplace=True))
            self.linear2 = nn.Sequential(nn.Linear(in_planes, in_planes), nn.ReLU(inplace=True))
        else:
            self.linear1 = nn.Sequential(nn.Linear(out_planes, out_planes), nn.BatchNorm1d(out_planes), nn.ReLU(inplace=True))
            self.linear2 = nn.Sequential(nn.Linear(in_planes, out_planes), nn.BatchNorm1d(out_planes), nn.ReLU(inplace=True))
        
    def forward(self, pxo1, pxo2=None):
        if pxo2 is None:
            _, x, o = pxo1
            x_tmp = []
            for i in range(o.shape[0]):
                s_i, e_i = (0, o[0]) if i == 0 else (o[i-1], o[i])
                cnt = e_i - s_i
                x_b = x[s_i:e_i, :]
                x_b = torch.cat((x_b, self.linear2(x_b.sum(0, True) / cnt).repeat(cnt, 1)), 1)
                x_tmp.append(x_b)
            x = self.linear1(torch.cat(x_tmp, 0))
        else:
            p1, x1, o1 = pxo1; p2, x2, o2 = pxo2
            x = self.linear1(x1) + pointops.interpolation(p2, p1, self.linear2(x2), o2, o1)
        return x

class WeakBlockWithResLFE(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, share_planes=8, nsample=16, 
                 use_reslfe=True, reslfe_depth=2, use_checkpoint=False):
        super().__init__()
        self.use_reslfe = use_reslfe
        self.linear1 = nn.Linear(in_planes, planes, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.weak = WeakLayer(planes, planes, share_planes, nsample)
        self.bn2 = nn.BatchNorm1d(planes)
        
        if use_reslfe:
            self.reslfe = ResLFE_Block(dim=planes, depth=reslfe_depth, use_checkpoint=use_checkpoint)
            self.register_buffer('pe', torch.randn(1, 1, planes) * 0.02)

        self.linear3 = nn.Linear(planes, planes, bias=False)
        self.bn3 = nn.BatchNorm1d(planes)
        self.act = nn.ReLU(inplace=True)
        self.dp = nn.Dropout(0.5)

    def forward(self, pxo):
        p, x, o = pxo
        identity = x
        x = self.act(self.bn1(self.linear1(x)))
        x = self.act(self.bn2(self.weak([p, x, o])))
        
        if self.use_reslfe:
            x_batch = x.unsqueeze(0)
            B, N, C = x_batch.shape
            knn = torch.randint(0, N, (B, N, 16), device=x.device) if N > 12000 else None
            x = self.reslfe(x_batch, self.pe.expand(B, N, -1), knn).squeeze(0)
            
        x = self.bn3(self.linear3(x))
        x = self.dp(x) + identity
        return [p, self.act(x), o]
class WeakSeg(nn.Module):
    def __init__(self, block, blocks, c=6, k=14, unitArch=[32, 64, 128, 256, 512], **kwargs):
        super().__init__()
        self.in_planes = 6 
        
        planes = unitArch
        stride, nsample = [1, 4, 4, 4, 4], [8, 16, 16, 16, 16]
        use_reslfe = kwargs.get('use_reslfe', True)
        reslfe_depth = kwargs.get('reslfe_depth', 2)
        use_checkpoint = kwargs.get('use_checkpoint', False)

        # Encoder Layers
        self.enc1 = self._make_enc(block, planes[0], blocks[0], stride[0], nsample[0], use_reslfe, reslfe_depth, use_checkpoint)
        self.enc2 = self._make_enc(block, planes[1], blocks[1], stride[1], nsample[1], use_reslfe, reslfe_depth, use_checkpoint)
        self.enc3 = self._make_enc(block, planes[2], blocks[2], stride[2], nsample[2], use_reslfe, reslfe_depth, use_checkpoint)
        self.enc4 = self._make_enc(block, planes[3], blocks[3], stride[3], nsample[3], use_reslfe, reslfe_depth, use_checkpoint)
        self.enc5 = self._make_enc(block, planes[4], blocks[4], stride[4], nsample[4], use_reslfe, reslfe_depth, use_checkpoint)
        
        # Decoder Layers
        self.dec5 = self._make_dec(block, planes[4], 1, nsample[4], True, use_reslfe, reslfe_depth, use_checkpoint)
        self.dec4 = self._make_dec(block, planes[3], 1, nsample[3], False, use_reslfe, reslfe_depth, use_checkpoint)
        self.dec3 = self._make_dec(block, planes[2], 1, nsample[2], False, use_reslfe, reslfe_depth, use_checkpoint)
        self.dec2 = self._make_dec(block, planes[1], 1, nsample[1], False, use_reslfe, reslfe_depth, use_checkpoint)
        self.dec1 = self._make_dec(block, planes[0], 1, nsample[0], False, use_reslfe, reslfe_depth, use_checkpoint)
        
        self.cls = nn.Sequential(
            nn.Linear(planes[0], planes[0]), 
            nn.BatchNorm1d(planes[0]), 
            nn.ReLU(inplace=True), 
            nn.Linear(planes[0], k)
        )

    def _make_enc(self, block, planes, blocks, stride, nsample, use_reslfe, depth, cp):
        layers = [Downsampling(self.in_planes, planes, stride, nsample)]
        self.in_planes = planes
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, self.in_planes, 8, nsample, use_reslfe, depth, cp))
        return nn.Sequential(*layers)

    def _make_dec(self, block, planes, blocks, nsample, is_head, use_reslfe, depth, cp):
        layers = [FeaturePropagation(self.in_planes, None if is_head else planes)]
        self.in_planes = planes
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, self.in_planes, 8, nsample, use_reslfe, depth, cp))
        return nn.Sequential(*layers)

    def forward(self, pxo):
        p0, x0, o0 = pxo
        x0 = torch.cat([x0, p0], 1)
        
        p1, x1, o1 = self.enc1([p0, x0, o0]); p2, x2, o2 = self.enc2([p1, x1, o1])
        p3, x3, o3 = self.enc3([p2, x2, o2]); p4, x4, o4 = self.enc4([p3, x3, o3])
        p5, x5, o5 = self.enc5([p4, x4, o4])
        
        x5 = self.dec5[1:]([p5, self.dec5[0]([p5, x5, o5]), o5])[1]
        x4 = self.dec4[1:]([p4, self.dec4[0]([p4, x4, o4], [p5, x5, o5]), o4])[1]
        x3 = self.dec3[1:]([p3, self.dec3[0]([p3, x3, o3], [p4, x4, o4]), o3])[1]
        x2 = self.dec2[1:]([p2, self.dec2[0]([p2, x2, o2], [p3, x3, o3]), o2])[1]
        x1 = self.dec1[1:]([p1, self.dec1[0]([p1, x1, o1], [p2, x2, o2]), o1])[1]
        
        return self.cls(x1)

def weak_seg_repro(custom=None, **kwargs):
    model = WeakSeg(WeakBlockWithResLFE, [1, 3, 3, 6, 3], **kwargs)
    return model
