from attacks.attack import Attack

import torch.nn.functional as F

class FGSM(Attack):
    
    def attackSample(self, x, y, norm='inf', epsilon=0):
        x['audio'].requires_grad_()
        
        loss = F.nll_loss(self.model(x), y)
        self.model.zero_grad()
        loss.backward()

        if norm == "inf":
            x['audio'] = x['audio'] + epsilon * x['audio'].grad.data.sign()
        else:
            normed_grad = x.grad.data.norm(p=float(norm), dim=[2,3]).unsqueeze(2).unsqueeze(3)
            x['audio'] = x['audio'] + epsilon * x.grad.data/normed_grad

        # projection in case epsilon is too large
        x['audio'] = x['audio'].clamp(-1, 1).detach()  
        return x