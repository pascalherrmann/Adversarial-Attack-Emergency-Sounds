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

            x_pert = {'audio': a * x['audio'], 'sample_rate': x['sample_rate']}
            loss = F.nll_loss(self.model(x_pert), y)
            self.model.zero_grad()
            loss.backward()

            # some models cannot backpropagate/are robust by default
            assert not torch.isnan(a.grad.data)

            a = a + epsilon * a.grad.data
            a = a.clamp(lower, upper).detach()

        return {'audio': (a * x['audio']).clamp(-1, 1), 'sample_rate': x['sample_rate']}