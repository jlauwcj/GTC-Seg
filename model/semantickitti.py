import torch
import torch.nn as nn
from lib.pointops.functions import pointops
import torch.nn.functional as F
import numpy as np
class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output

class FFN(nn.Module):
    def __init__(self, in_dim, mlp_ratio=4.0):
        super().__init__()
        hid_dim = int(in_dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hid_dim, in_dim, bias=False),
            nn.BatchNorm1d(in_dim)
        )

    def forward(self, x):
        return self.net(x)
class VFR_Layer(nn.Module):
    def __init__(self, in_planes, out_planes, nsample=16):
        super().__init__()
        self.nsample = nsample
        self.linear = nn.Linear(in_planes, out_planes, bias=False)
        self.bn = nn.BatchNorm1d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.pos_mlp = nn.Sequential(
            nn.Linear(3, out_planes // 2),
            nn.ReLU(inplace=True),
            nn.Linear(out_planes // 2, out_planes)
        )

    def forward(self, p, x, o):
        x = self.linear(x) 
        x_k = pointops.queryandgroup(self.nsample, p, p, x, None, o, o, use_xyz=True)
        rel_pos = x_k[:, :, :3] # ΔPij = Pj - Pi
        feat = x_k[:, :, 3:]
        pos_weight = torch.sigmoid(self.pos_mlp(rel_pos))
        feat = (feat * pos_weight).sum(dim=1)
        x = self.relu(self.bn(feat))
        return x

class ResLFE_Block(nn.Module):
    expansion = 1 
    def __init__(self, in_planes, planes, share_planes=8, nsample=16, depth=2, drop_path=0.1):
        super().__init__()
        self.depth = depth
        self.linear_in = nn.Linear(in_planes, planes, bias=False)
        self.bn_in = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        
        self.pe_proj = nn.Sequential(
            nn.Linear(3, planes),
            nn.BatchNorm1d(planes),
            nn.ReLU(inplace=True)
        )

        self.vfrs = nn.ModuleList([VFR_Layer(planes, planes, nsample) for _ in range(depth)])
        self.ffns = nn.ModuleList([FFN(planes) for _ in range(depth)])
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, pxo):
        p, x, o = pxo
        x = self.relu(self.bn_in(self.linear_in(x)))
        pe = self.pe_proj(p)
        x = x + pe
        
        for i in range(self.depth):
            shortcut = x
            x = self.vfrs[i](p, x, o)
            x = shortcut + self.drop_path(x)
            
            shortcut = x
            x = self.ffns[i](x)
            x = shortcut + self.drop_path(x)
            
        return [p, x, o]
class CGCM_Module(nn.Module):
    def __init__(self, feat_dim=32, num_classes=19, target_temp=0.07,
                 base_momentum=0.999, min_momentum=0.90, warmup_steps=2000):
        super().__init__()
        self.num_classes = num_classes
        self.target_temp = target_temp 
        self.base_momentum = base_momentum
        self.min_momentum = min_momentum
        self.warmup_steps = warmup_steps

        self.proj = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim)
        )

        self.register_buffer("prototypes", torch.empty(num_classes, feat_dim))
        nn.init.orthogonal_(self.prototypes)
        self.register_buffer("step", torch.tensor(0, dtype=torch.long))
        self.register_buffer("class_freq", torch.zeros(num_classes))

    @property
    def current_temperature(self):
        progress = min(self.step.item() / 10000.0, 1.0)
        return 0.3 * (1.0 - progress) + self.target_temp * progress

    @property
    def loss_weight(self):
        if self.step.item() <= self.warmup_steps:
            return 0.0
        progress = min((self.step.item() - self.warmup_steps) / 5000.0, 1.0)
        return 0.02 + 0.03 * progress

    def forward(self, x, clip_pseudo_labels=None):
        feat_3d = F.normalize(self.proj(x), p=2, dim=1)

        if self.training and clip_pseudo_labels is not None:
            self.step += 1

            valid_mask = (clip_pseudo_labels >= 0) & (clip_pseudo_labels < self.num_classes)
            noisy_classes = torch.tensor([6,7,8, 9, 10, 11, 14, 16], device=x.device)
            for nc in noisy_classes:
                valid_mask = valid_mask & (clip_pseudo_labels != nc)

            if valid_mask.any():
                filtered_feat = feat_3d[valid_mask]
                filtered_labels = clip_pseudo_labels[valid_mask]
                with torch.no_grad():
                    local_count = torch.zeros(self.num_classes, dtype=torch.float32, device=x.device)
                    local_sum = torch.zeros(self.num_classes, feat_3d.size(-1), dtype=torch.float32, device=x.device)
                    local_count.scatter_add_(
                        0, filtered_labels,
                        torch.ones(filtered_labels.size(0), dtype=torch.float32, device=x.device)
                    )
                    local_sum.scatter_add_(
                        0,
                        filtered_labels.unsqueeze(1).expand_as(filtered_feat),
                        filtered_feat
                    )
                    self.class_freq += local_count

                    if self.step > self.warmup_steps:
                        for c in range(self.num_classes):
                            if local_count[c] > 0:
                                global_mean = F.normalize(
                                    local_sum[c] / local_count[c], p=2, dim=0
                                )
                                m_c = self.min_momentum + \
                                      (self.base_momentum - self.min_momentum) * \
                                      (1.0 - torch.exp(-self.class_freq[c] / 50000.0))
                                updated = m_c * self.prototypes[c] + (1.0 - m_c) * global_mean
                                self.prototypes[c] = F.normalize(updated, p=2, dim=0)

                proto = self.prototypes.detach()
                logits = torch.matmul(filtered_feat, proto.T) / self.current_temperature

                loss_con = F.cross_entropy(logits, filtered_labels, label_smoothing=0.1)
                return loss_con * self.loss_weight
        return feat_3d.sum() * 0.0
class WeakLayer(nn.Module):
    def __init__(self, in_planes, out_planes, share_planes=8, nsample=16):
        super().__init__()
        self.mid_planes = out_planes
        self.out_planes = out_planes
        self.share_planes = share_planes
        self.nsample = nsample
        
        self.linear_q = nn.Linear(in_planes, out_planes)
        self.linear_k = nn.Linear(in_planes, out_planes)
        self.linear_v = nn.Linear(in_planes, out_planes) 
        
        self.pos_enc = nn.Sequential(
            nn.Linear(3, out_planes),
            nn.BatchNorm1d(out_planes),
            nn.ReLU(inplace=True)
        )
        
        self.spatial_weight = nn.Sequential(
            nn.Linear(in_planes, out_planes),
            nn.BatchNorm1d(out_planes),
            nn.ReLU(inplace=True)
        )
        
        self.weight_net = nn.Sequential(
            nn.Linear(out_planes, out_planes),
            nn.BatchNorm1d(out_planes),
            nn.ReLU(inplace=True),
            nn.Linear(out_planes, out_planes)
        )
        
        self.softmax = nn.Softmax(dim=1)

    def forward(self, pxo) -> torch.Tensor:
        p, x, o = pxo 
        
        x_q = self.linear_q(x)
        x_k = self.linear_k(x)
        x_v = self.linear_v(x)
        
        x_k = pointops.queryandgroup(self.nsample, p, p, x_k, None, o, o, use_xyz=True)
        x_v_j = pointops.queryandgroup(self.nsample, p, p, x_v, None, o, o, use_xyz=False)
        
        n, nsample, c = x_v_j.shape
        
        rel_pos = x_k[:, :, :3]
        x_k = x_k[:, :, 3:]
        
        pos_enc = self.pos_enc(rel_pos.view(-1, 3)).view(n, nsample, c)
        feat_diff = x_k - x_q.unsqueeze(1)
        spatial_w = self.spatial_weight(feat_diff.view(-1, c)).view(n, nsample, c)
        
        w = self.weight_net((spatial_w + pos_enc).view(-1, c)).view(n, nsample, c)
        w = self.softmax(w)
        
        x = ((x_v_j + pos_enc) * w).sum(1)
        
        return x

class WeakBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, share_planes=8, nsample=16):
        super(WeakBlock, self).__init__()
        self.linear1 = nn.Linear(in_planes, planes, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.weak = WeakLayer(planes, planes, share_planes, nsample)
        self.bn2 = nn.BatchNorm1d(planes)
        self.linear3 = nn.Linear(planes, planes * self.expansion, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)
        self.act = nn.ReLU(inplace=True)
        self.drop_path = DropPath(0.3)

    def forward(self, pxo):
        p, x, o = pxo
        identity = x
        x = self.act(self.bn1(self.linear1(x)))
        x = self.act(self.bn2(self.weak([p, x, o])))
        x = self.bn3(self.linear3(x))
        x = self.drop_path(x)
        x += identity
        x = self.act(x)
        return [p, x, o]

class Downsampling(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, nsample=16):
        super().__init__()
        self.stride, self.nsample = stride, nsample
        if stride != 1:
            self.linear = nn.Linear(3+in_planes, out_planes, bias=False)
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
            n_o = torch.tensor(n_o, dtype=torch.int32, device=o.device)
            idx = pointops.furthestsampling(p, o, n_o)
            n_p = p[idx.long(), :]
            x = pointops.queryandgroup(self.nsample, p, n_p, x, None, o, n_o, use_xyz=True)
            x = self.relu(self.bn(self.linear(x).transpose(1, 2).contiguous()))
            x = self.pool(x).squeeze(-1)
            p, o = n_p, n_o
        else:
            x = self.relu(self.bn(self.linear(x)))
        return [p, x, o]

class FeaturePropagation(nn.Module):
    def __init__(self, in_planes, out_planes=None):
        super().__init__()
        if out_planes is None:
            self.linear1 = nn.Sequential(
                nn.Linear(2*in_planes, in_planes), 
                nn.BatchNorm1d(in_planes), 
                nn.ReLU(inplace=True)
            )
            self.linear2 = nn.Sequential(
                nn.Linear(in_planes, in_planes), 
                nn.ReLU(inplace=True)
            )
        else:
            self.linear1 = nn.Sequential(
                nn.Linear(out_planes, out_planes), 
                nn.BatchNorm1d(out_planes), 
                nn.ReLU(inplace=True)
            )
            self.linear2 = nn.Sequential(
                nn.Linear(in_planes, out_planes), 
                nn.BatchNorm1d(out_planes), 
                nn.ReLU(inplace=True)
            )
        
    def forward(self, pxo1, pxo2=None):
        if pxo2 is None:
            _, x, o = pxo1
            x_tmp = []
            for i in range(o.shape[0]):
                if i == 0:
                    s_i, e_i, cnt = 0, o[0], o[0]
                else:
                    s_i, e_i, cnt = o[i-1], o[i], o[i] - o[i-1]
                x_b = x[s_i:e_i, :]
                x_b = torch.cat((x_b, self.linear2(x_b.sum(0, True) / cnt).repeat(cnt, 1)), 1)
                x_tmp.append(x_b)
            x = torch.cat(x_tmp, 0)
            x = self.linear1(x)
        else:
            p1, x1, o1 = pxo1
            p2, x2, o2 = pxo2
            x = self.linear1(x1) + pointops.interpolation(p2, p1, self.linear2(x2), o2, o1)
        return x
class HybridWeakSeg(nn.Module):
    def __init__(self, c=4, k=19, unitArch=[32, 64, 128, 256, 512]):
        super().__init__()
        self.c = c
        self.in_planes = c
        planes = unitArch
        share_planes = 8
        stride = [1, 4, 4, 4, 4]
        nsample = [32, 24, 16, 16, 8]
        self.enc1 = self._make_enc(
            ResLFE_Block, planes[0], blocks=1, share_planes=share_planes,
            stride=stride[0], nsample=nsample[0], is_reslfe=True, depth=2
        )
        
        self.enc2 = self._make_enc(
            ResLFE_Block, planes[1], blocks=1, share_planes=share_planes,
            stride=stride[1], nsample=nsample[1], is_reslfe=True, depth=2
        )
        
        self.enc3 = self._make_enc(
            WeakBlock, planes[2], blocks=1, share_planes=share_planes,
            stride=stride[2], nsample=nsample[2]
        )
        
        self.enc4 = self._make_enc(
            WeakBlock, planes[3], blocks=1, share_planes=share_planes,
            stride=stride[3], nsample=nsample[3]
        )
        
        self.enc5 = self._make_enc(
            WeakBlock, planes[4], blocks=1, share_planes=share_planes,
            stride=stride[4], nsample=nsample[4]
        )

        # Decoder -----------------------------------------------------------
        self.dec5 = self._make_dec(WeakBlock, planes[4], 1, share_planes, nsample=nsample[4], is_head=True)
        self.dec4 = self._make_dec(WeakBlock, planes[3], 1, share_planes, nsample=nsample[3])
        self.dec3 = self._make_dec(WeakBlock, planes[2], 1, share_planes, nsample[2])
        self.dec2 = self._make_dec(ResLFE_Block, planes[1], 1, share_planes, nsample=nsample[1], is_reslfe=True, depth=2)
        self.dec1 = self._make_dec(ResLFE_Block, planes[0], 1, share_planes, nsample=nsample[0], is_reslfe=True, depth=2)
        self.cgcm = CGCM_Module(feat_dim=planes[0], num_classes=k)
        self.cls = nn.Sequential(
            nn.Linear(planes[0], planes[0]), 
            nn.BatchNorm1d(planes[0]), 
            nn.ReLU(inplace=True), 
            nn.Linear(planes[0], k)
        )

    def _make_enc(self, block, planes, blocks, share_planes=8, stride=1, nsample=16, is_reslfe=False, depth=2):
        layers = []
        layers.append(Downsampling(self.in_planes, planes, stride, nsample))
        self.in_planes = planes
        for _ in range(blocks):
            if is_reslfe:
                layers.append(block(self.in_planes, self.in_planes, share_planes, nsample, depth=depth))
            else:
                layers.append(block(self.in_planes, self.in_planes, share_planes, nsample))
        return nn.Sequential(*layers)

    def _make_dec(self, block, planes, blocks, share_planes=8, nsample=16, is_head=False, is_reslfe=False, depth=2):
        layers = []
        layers.append(FeaturePropagation(self.in_planes, None if is_head else planes))
        self.in_planes = planes
        for _ in range(blocks):
            if is_reslfe:
                layers.append(block(self.in_planes, self.in_planes, share_planes, nsample, depth=depth))
            else:
                layers.append(block(self.in_planes, self.in_planes, share_planes, nsample))
        return nn.Sequential(*layers)

    def forward(self, pxo, clip_labels=None):
        p0, x0, o0 = pxo
        x0 = torch.cat([x0, p0], 1)
        p1, x1, o1 = self.enc1([p0, x0, o0])
        p2, x2, o2 = self.enc2([p1, x1, o1])
        p3, x3, o3 = self.enc3([p2, x2, o2])
        p4, x4, o4 = self.enc4([p3, x3, o3])
        p5, x5, o5 = self.enc5([p4, x4, o4])
        x5_d = self.dec5[1:]([p5, self.dec5[0]([p5, x5, o5]), o5])[1]
        x4_d = self.dec4[1:]([p4, self.dec4[0]([p4, x4, o4], [p5, x5_d, o5]), o4])[1]
        x3_d = self.dec3[1:]([p3, self.dec3[0]([p3, x3, o3], [p4, x4_d, o4]), o3])[1]
        x2_d = self.dec2[1:]([p2, self.dec2[0]([p2, x2, o2], [p3, x3_d, o3]), o2])[1]
        x1_d = self.dec1[1:]([p1, self.dec1[0]([p1, x1, o1], [p2, x2_d, o2]), o1])[1]
        
        logits = self.cls(x1_d)
        
        if self.training:
            loss_con = self.cgcm(x1_d, clip_labels)
            return logits, loss_con
        
        return logits

def hybrid_weak_seg_repro(custom=None, **kwargs):
    model = HybridWeakSeg(**kwargs)
    return model
