from attacks.attack import Attack

import torch.nn.functional as F

class PGD(Attack):
    
    def attackSample(self, x, y, epsilon=0, num_iter=1):
        for i in range(num_iter):
            x[0].requires_grad_()
            loss = F.nll_loss(self.model(x), y)
            self.model.zero_grad()
            loss.backward()

            x[0] = x[0] + epsilon * x[0].grad.data

            # projection in case epsilon is too large
            x[0] = x[0].clamp(-1, 1).detach() 

        return x 
