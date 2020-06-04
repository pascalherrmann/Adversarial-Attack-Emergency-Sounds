from attacks.attack import Attack
import torch.nn.functional as F

class FGSM(Attack):
    
    def attackSample(self, x, y, epsilon=0):
        x.requires_grad_()
        
        loss = F.nll_loss(self.model(x), y)
        self.model.zero_grad()
        loss.backward()

        x = x + epsilon * x.grad.data.sign()
        x = x.clamp(-1, 1).detach()  # projection
        return x