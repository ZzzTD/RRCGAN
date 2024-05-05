from packaging import version
import torch
from torch import nn


class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm + 1e-7)
        return out

class HSL_Loss(nn.Module):
    def __init__(self,):
        super().__init__()
    def forward(self, feat_q, feat_k):

        dim = feat_q.shape[1]
        feat_k = feat_k.detach()
        batch_dim_for_bmm = 1
        feat_k = Normalize()(feat_k)
        feat_q = Normalize()(feat_q)

        feat_q_v = feat_q.view(batch_dim_for_bmm, -1, dim)
        feat_k_v = feat_k.view(batch_dim_for_bmm, -1, dim)

        spatial_q = torch.bmm(feat_q_v, feat_q_v.transpose(2, 1))
        spatial_k = torch.bmm(feat_k_v, feat_k_v.transpose(2, 1))

        spatial_q = nn.Softmax(dim=1)(spatial_q)
        spatial_k = nn.Softmax(dim=1)(spatial_k).detach()
        loss_src = self.get_jsd(spatial_q, spatial_k)
        return loss_src
    def get_jsd(self, p1, p2):
        m = 0.5 * (p1 + p2)
        out = 0.5 * (nn.KLDivLoss(reduction='sum', log_target=True)(torch.log(m), torch.log(p1))+ nn.KLDivLoss(reduction='sum', log_target=True)(torch.log(m), torch.log(p2)))
        return out