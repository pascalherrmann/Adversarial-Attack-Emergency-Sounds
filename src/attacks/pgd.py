from attacks.attack import Attack

import torch.nn.functional as F

class ProjectedGradientDescent(Attack):
    
    '''
        - Epsilon step in gradient direction (step size defined by norm)
        - Fully vectorized 
        - For num_iter=1 and norm='inf': Degrades to Fast Gradient Sign Method (FGSM)
    '''
    def attackSample(self, x, y, epsilon=0, num_iter=1, norm='inf', loss_fn=F.nll_loss):
        for _ in range(num_iter):
            x['audio'].requires_grad_()

            loss = loss_fn(self.model(x), y)
            self.model.zero_grad()
            loss.backward()

            if norm == "inf":
            x['audio'] = x['audio'] + epsilon * x['audio'].grad.sign()
            else:
            normed_grad = x.grad.norm(p=float(norm), dim=[2]).unsqueeze(2)
            x['audio'] = x['audio'] + epsilon * x.grad/normed_grad

            # projection in case epsilon is too large
            x['audio'] = x['audio'].clamp(-1, 1).detach()
        return x
