import torch as th
from einops import repeat
from einops.layers.torch import Rearrange
from survae.distributions import ConditionalDistribution
from survae.flows.cond_flow import ConditionalFlow
from survae.transforms import ConditionalTransform
from survae.utils import context_size
from torch import nn

from ..transforms.cond_surjection import ConditionalSurjection


class ConditionalFlow_v2(ConditionalFlow):
    """
    Base class for ConditionalFlow.
        Flows use the forward transforms to transform data to noise.
        The inverse transforms can subsequently be used for sampling.
        These are typically useful as generative models of data.

    This class add a function `sample_softmax` to sample from the flow using the softmax trick.
    """
    def __init__(self, base_dist, transforms, context_init=None):
        super().__init__(base_dist, transforms, context_init)
    
    @th.no_grad()
    def sample(self, context):
        return super().sample(context)

    def sample_enable_grad(self, context):
        return super().sample(context)

    def sample_softmax(self, context):
        pass_thru_sur = False

        if self.context_init: context = self.context_init(context)
        if isinstance(self.base_dist, ConditionalDistribution):
            z = self.base_dist.sample(context)
        else:
            z = self.base_dist.sample(context_size(context))
        for transform in reversed(self.transforms):
            if pass_thru_sur: 
                raise RuntimeError("Has passed through a surjection which should be the last layer.")
            if isinstance(transform, ConditionalSurjection):
                z_ori = transform.inverse(z, context)
                z_soft = transform.inverse_soft(z, context)
                pass_thru_sur = True
            elif isinstance(transform, ConditionalTransform):
                z = transform.inverse(z, context)
            else:
                z = transform.inverse(z)
        return z_ori, z_soft

class ContextNet(nn.Module):
    """Context initialization network for the observation conditional 
    posterior Q(.|a,s). This network takes as input the observation and
    discrete action. It outputs the embedding of the two inputs.

    This network is used for coupling flows.
    """
    def __init__(self, act_dim, obs_dim, n_d_act=False, hidden_dim=128, 
                 output_dim=64,num_layers=1, dropout=0.0): 
        assert num_layers >= 1, "num_layers must be at least 1"
        super().__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Dropout(p=dropout))
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.GELU())

        self.act_embedding = nn.Sequential(
            nn.Embedding(act_dim, hidden_dim//2),
            Rearrange("b (l h) -> b l h", l=1) if not n_d_act else nn.Identity(), # (B,L,H/2)
        )
        self.obs_embedding = nn.Linear(obs_dim, hidden_dim//2)
        output_layer = Rearrange("b l p -> b p l") if n_d_act else nn.Identity() # n_d_act: (B,P,L), otherwise: (B,L,P)
        self.context_net = nn.Sequential(
            *layers,
            nn.Linear(hidden_dim, output_dim),
            output_layer
        )
    
    def forward(self, x):
        act, obs = x
        L = act.size(-1)
        act_embedding = self.act_embedding(act.squeeze(1)) # (B,L) -> (B,L,H/2)
        obs_embedding = repeat(self.obs_embedding(obs), "b h -> b l h", l=L) # (B,L,H/2)
        context = th.cat([act_embedding, obs_embedding], dim=-1) # (B,L,H)
        context = self.context_net(context) # (B,L,H) -> (B,L,P) or (B,P,L)
        return context

class IdxContextNet(nn.Module):
    """Context initialization network for the observation conditional 
    posterior Q(.|a,s). This network takes as input the observation and
    discrete action. It outputs the embedding of the two inputs.

    This network is used for AR flows.
    """
    def __init__(self, act_dim, obs_dim, hidden_dim=128,
                 output_dim=64, num_layers=1, dropout=0.0) -> None:
        super().__init__()
        self.act_embedding = nn.Embedding(act_dim, hidden_dim//2) # (B,L,H/2)
        self.obs_embedding = nn.Linear(obs_dim, hidden_dim//2)
        self.context_net = nn.Sequential(
            Rearrange("b l h -> l b h"),
            LayerLSTM(hidden_dim, hidden_dim, num_layers=num_layers, dropout=dropout, bidirectional=True), # (L,B,H) -> (L,B,2*H)
            nn.Linear(2*hidden_dim, output_dim), # (L,B,2*H) -> (L,B,P)
            Rearrange("l b p -> b p l") # (L,B,P) -> (B,P,L)
        )

    def forward(self, x):
        act, obs = x
        L = act.size(-1)
        act_embedding = self.act_embedding(act.squeeze(1)) # (B,L) -> (B,L,H/2)
        obs_embedding = repeat(self.obs_embedding(obs), "b h -> b l h", l=L) # (B,L,H/2)
        context = th.cat([act_embedding, obs_embedding], dim=-1) # (B,L,H)
        context = self.context_net(context) # (B,L,H) -> (B,L,P) or (B,P,L)
        return context

class LayerLSTM(nn.LSTM):
    def forward(self, x):
        output, _ = super(LayerLSTM, self).forward(x) # output, (c_n, h_n)
        return output
