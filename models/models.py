# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# class NetDet(nn.Module):
#     def __init__(self, input_dim=20, hidden=32):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(input_dim, hidden),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(hidden, 1)
#         )
#     def forward(self, x):
#         return self.net(x).squeeze(-1)


class BayesLinear(nn.Module):
    def __init__(self, in_f, out_f, prior_std=1.0):
        super().__init__()
        self.prior_std = prior_std
        self.w_mu  = nn.Parameter(torch.zeros(out_f, in_f))
        self.b_mu  = nn.Parameter(torch.zeros(out_f))
        self.w_rho = nn.Parameter(torch.full((out_f, in_f), -3.0))
        self.b_rho = nn.Parameter(torch.full((out_f,), -3.0))

    def forward(self, x, sample=True):
        if sample:
            w = self.w_mu + F.softplus(self.w_rho) * torch.randn_like(self.w_mu)
            b = self.b_mu + F.softplus(self.b_rho) * torch.randn_like(self.b_mu)
        else:
            w, b = self.w_mu, self.b_mu
        return F.linear(x, w, b)

    def kl(self):
        ws = F.softplus(self.w_rho)
        bs = F.softplus(self.b_rho)
        pv = self.prior_std ** 2
        def _kl(mu, sigma):
            return 0.5 * (sigma**2/pv + mu**2/pv - 1
                          + 2*(np.log(self.prior_std) - torch.log(sigma))).sum()
        return _kl(self.w_mu, ws) + _kl(self.b_mu, bs)


class BNN(nn.Module):
    def __init__(self, input_dim=20, hidden=32, prior_std=1.0):
        super().__init__()
        self.l1 = BayesLinear(input_dim, hidden, prior_std)
        self.l2 = BayesLinear(hidden, 1, prior_std)
        self.log_obs_var = nn.Parameter(torch.tensor(-1.4))

    def forward(self, x, sample=True):
        return self.l2(F.relu(self.l1(x, sample)), sample).squeeze(-1)

    def obs_std(self):
        return torch.exp(0.5 * self.log_obs_var).clamp(min=0.3)

    def kl(self):
        return self.l1.kl() + self.l2.kl()