from attacks.attack import Attack

import torch.nn.functional as F
import torch

class VolumeAttack(Attack):
    
    """
        lower: lower bound for volume increase 
        upper: upper bound for volume increase 
    """
    def attackSample(self, x, y, lower=0.2, upper=2, epsilon=0, num_iter=1):

        a = torch.ones(x['audio'].size(0)).unsqueeze(1).cuda()

        for _ in range(num_iter):
            a.requires_grad_()

            tmp_audio = x['audio']
            x['audio'] = (a * x['audio']).clamp(-1, 1)
            loss = F.nll_loss(self.model(x), y)
            self.model.zero_grad()
            loss.backward()

            # we change a absolutely, 
            # otherwise we would get powers of a (a^num_iter)
            x['audio'] = tmp_audio 

            a = a + epsilon * a.grad.sign()
            a = a.clamp(lower, upper).detach()

        x['audio'] = (a * x['audio']).clamp(-1, 1)
        return x 