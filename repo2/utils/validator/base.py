import torch as th
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce


class Validator(nn.Module):
    def __init__(self, n_cls, c_dim, device):
        super().__init__()
        self.n_cls = n_cls
        self.emd = nn.Linear(self.n_cls, 64).to(device)

        self.fc = nn.Sequential(
            nn.Linear(64+c_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        ).to(device)

    def forward(self, x: th.Tensor, c: th.Tensor): 
        assert x.ndim in [2, 3]
        assert c.ndim == 2

        if x.dtype in [th.int32, th.int64]:
            assert x.ndim == 2, "x must be 2D if it is an index"
            x = F.one_hot(x, num_classes=self.n_cls).float()
        
        x = reduce(self.emd(x), 'b l e -> b e', 'sum')
        x_c = th.cat([x, c], dim=-1)

        return self.fc(x_c)

    def logits(self, x: th.Tensor, c: th.Tensor):
        return self.forward(x, c)

    def probs(self, x: th.Tensor, c: th.Tensor):
        return th.sigmoid(self.forward(x, c))
