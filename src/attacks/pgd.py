from attacks.attack import Attack
from attacks.fga import FastGradientAttack

import torch.nn.functional as F

class PGD(Attack):

    def __init__(self, **params):
        self.fga = FastGradientAttack(**params)
    
    def attackSample(self, x, y, epsilon=0, num_iter=1, norm='inf', loss_fn=F.nll_loss)
        raise Exception("Not vectorized - TODO")
        for i in range(num_iter):
            x = self.fga.attackSample(x, y, epsilon, norm, loss_fn)
        return x 
