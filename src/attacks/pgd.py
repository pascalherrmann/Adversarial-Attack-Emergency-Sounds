from attacks.attack import Attack

import torch.nn.functional as F

class PGD(Attack):
    
    def attackSample(self, x, y, epsilon=0, num_iter=1):
        raise Exception("Not vectorized - TODO")
        for i in range(num_iter):
            x['audio'].requires_grad_()
            loss = F.nll_loss(self.model(x), y)
            self.model.zero_grad()
            loss.backward()

            x['audio'] = x['audio'] + epsilon * x['audio'].grad.data

            # projection in case epsilon is too large
            x['audio'] = x['audio'].clamp(-1, 1).detach() 

        return x 
