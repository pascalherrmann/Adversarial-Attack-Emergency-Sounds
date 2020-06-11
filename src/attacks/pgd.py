from attacks.attack import Attack
from attacks.fga import FastGradientAttack

import torch.nn.functional as F

class PGD(Attack):

    def __init__(self, model, data_loader,
                    attack_parameters, early_stopping=-1,
                    device='cuda', save_samples=True):

        self.fga=FastGradientAttack(model, data_loader,
                                    attack_parameters, early_stopping=early_stopping,
                                    device=device, save_samples=save_samples)
    
    def attackSample(self, x, y, epsilon=0, num_iter=1, norm='inf', loss_fn=F.nll_loss):
        raise Exception("Not vectorized - TODO")
        for i in range(num_iter):
            x = self.fga.attackSample(x, y, epsilon, norm, loss_fn)
        return x 
