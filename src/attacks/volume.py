from attacks.attack import Attack

import torch.nn.functional as F
import torch

class VolumeAttack(Attack):
    
    """
        lower: lower bound for volume increase 
        upper: upper bound for volume increase 
    """
    def attackSample(self, x, y, lower=0.2, upper=2, epsilon=0, num_iter=1):
        a = torch.tensor(1.0).cuda()

        for i in range(num_iter):
            a.requires_grad_()

            loss = F.nll_loss(self.model(a * x), y)
            self.model.zero_grad()
            loss.backward()

            a = a + epsilon * a.grad.data
            a = clamp(lower, upper).detach()

        return (a * x).clamp(-1, 1)
