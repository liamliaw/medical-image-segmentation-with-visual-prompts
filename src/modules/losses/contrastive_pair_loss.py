import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastivePairLoss(nn.Module):
    def __init__(self, bs, temp=0.5):
        super().__init__()
        self.bs = bs
        self.register_buffer(
            "temp",
            torch.tensor(temp),
        )
        self.register_buffer(
            "neg_mask",
            (~torch.eye(bs * 2, bs * 2, dtype=torch.bool)).float(),
        )

    def forward(self, x_i, x_j):
        z = torch.cat([
            F.normalize(x_i, dim=1),
            F.normalize(x_j, dim=1),
        ])
        sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)
        sim_ij, sim_ji = torch.diag(sim, self.bs), torch.diag(sim, -self.bs)
        # Positive pairs.
        pos = torch.exp(torch.cat([sim_ij, sim_ji], dim=0) / self.temp)
        # Negative pairs.
        neg = self.neg_mask.to(z.device) * torch.exp(sim / self.temp)
        return torch.sum(-torch.log(pos / torch.sum(neg, dim=1))) \
            / (2 * self.bs)