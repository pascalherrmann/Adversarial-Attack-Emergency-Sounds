from attacks.attack import Attack

import torch.nn.functional as F

class FastGradientAttack(Attack):
    
    '''
        - Single epsilon step in gradient direction (step size defined by norm)
        - Fully vectorized 
        - For norm='inf': Degrades to Fast Gradient Sign Method (FGSM)
    '''
    def attackSample(self, x, y, epsilon=0, norm='inf', loss_fn=F.nll_loss):
        x['audio'].requires_grad_()
        
        print(x['audio'].shape)
        print(y.shape)
        loss = loss_fn(self.model(x), y)
        self.model.zero_grad()
        loss.backward()

        if norm == "inf":
            x['audio'] = x['audio'] + epsilon * x['audio'].grad.sign()
        else:
            normed_grad = x.grad.norm(p=float(norm), dim=[2,3]).unsqueeze(2).unsqueeze(3)
            x['audio'] = x['audio'] + epsilon * x.grad/normed_grad

        # projection in case epsilon is too large
        x['audio'] = x['audio'].clamp(-1, 1).detach()  
        return x