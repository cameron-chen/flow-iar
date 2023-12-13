import torch as th
from survae.transforms.bijections import Bijection
from survae.utils import sum_except_batch
from torch.nn import functional as F


class Threshold(Bijection):
    """Thresholding bijection described in [1].
    
    [1] Hoogeboom, E. 2021. Argmax flows and multinomial diffusion: 
        Learning categorical distributions.
    """
    def __init__(self, eps=1e-7):
        super(Threshold, self).__init__()
        self.eps = eps

    def forward(self, x, one_hot: th.Tensor):
        '''
        z = T - softplus(T-x) = T - log(1+exp(T-x))
        ldj = log(d(T-softplus(T-x))/dx) = log(sigmoid(T-x))

        Args:
            x: tensor to be processed
            one_hot: one-hot encoding of a discrete variable, on which dimension
                the operation is not performed

        Note:
            The thresholding operation is not performed on the one_hot-th dimension.
        '''
        one_hot_not = 1 - one_hot

        z_no_op = x * one_hot
        T = z_no_op.sum(dim=-1, keepdim=True)
        z_op = (T - F.softplus(T - x)) * one_hot_not
        ldj = sum_except_batch(F.logsigmoid(T - x)*one_hot_not)
        z = z_op + z_no_op
        return z, ldj

    def inverse(self, z, one_hot: th.Tensor):
        '''
        softplus_inv(t) = log(exp(t)-1) = t + log(1-exp(-t))
        x = T-softplus_inv(T-z) = z-log(1-exp(-T+z))
        '''
        one_hot_not = 1-one_hot
        
        z_no_op = z * one_hot
        T = z_no_op.sum(dim=-1, keepdim=True)
        T_z = (T - z).clamp(self.eps)
        z_op = (z - th.log1p(-th.exp(-T_z))) * one_hot_not
        return z_no_op + z_op


if __name__ == '__main__':
    import torch as th

    threshold = Threshold()
    instances = {
        0: th.tensor([[0.1,0.2,0.3,0.4],[0.1,0.2,0.3,0.4]]),
        1: th.tensor([[0.1,0.2,0.3,0.4],[0.1,0.2,0.3,0.4]]),
        2: th.tensor([[0.1,0.2,0.3,0.4],[0.1,0.2,0.3,0.4]]),
        3: th.tensor([[0.1,0.2,0.3,0.4],[0.1,0.2,0.3,0.4]]),
    }

    sol = {
        0: [th.tensor([[0.1, -0.5444, -0.4981, -0.4544],[0.1, -0.5444, -0.4981, -0.4544]]),
            th.tensor([-2.3969,-2.3969])],
        1: [th.tensor([[-0.5444, 0.2, -0.4444, -0.3981],[-0.5444, 0.2, -0.4444, -0.3981]]),
            th.tensor([-2.1868999999999996,-2.1868999999999996])],
        2: [th.tensor([[-0.4981, -0.4444, 0.3, -0.3444],[-0.4981, -0.4444, 0.3, -0.3444]]),
            th.tensor([-1.9868999999999999,-1.9868999999999999])],
        3: [th.tensor([[-0.4544, -0.3981, -0.3444, 0.4],[-0.4544, -0.3981, -0.3444, 0.4]]),
            th.tensor([-1.7969,-1.7969])],
    }

    print('##############\n',
        '# Testing start\n')
    for k,v in instances.items():
        one_hot = F.one_hot(th.tensor([k,k]).view(2), 4)
        z, ldj = threshold(v, one_hot)
        v_p = threshold.inverse(z, one_hot)
        try:
            assert z[0].sum().round(decimals=2) == sol[k][0][0].sum().round(decimals=2) # function value
            assert ldj.sum().round(decimals=2) == sol[k][1].sum().round(decimals=2) # ldj
            assert th.isclose(v, v_p).all() # inverse
            print('Test case {}: OK\n'.format(k))
        except:
            print('Test case {}: FAIL'.format(k))
            print('z: {}\nldj: {}\n'.format(z, ldj))
            print('z_sol: {}\nldj_sol: {}\n'.format(sol[k][0], sol[k][1]))
            print('v: {}\nv_p: {}\n'.format(v, v_p))

    print('##############\n',
        '# Testing end\n')