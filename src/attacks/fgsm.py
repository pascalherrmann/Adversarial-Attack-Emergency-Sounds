from attacks.attack import Attack
import torch.nn.functional as F

class FGSM(Attack):
    
    def attackSample(self, x, y, epsilon=0):
        x.requires_grad = True
        
        loss = F.nll_loss(self.model(x), y)
        self.model.zero_grad()
        loss.backward()
        
        perturbed_sample = x + epsilon * x.grad.data.sign()
        return perturbed_sample

if __name__ == '__main__':
    pass