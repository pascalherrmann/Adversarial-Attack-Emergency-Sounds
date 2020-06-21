from attacks.attack import Attack

import torch.nn.functional as F

class FunctionalVolumeAttack(Attack):

    '''
        - Epsilon step in gradient direction (step size defined by norm)
        - Fully vectorized 
        - For num_iter=1 and norm='inf': Degrades to Fast Gradient Sign Method (FGSM)
    '''
    def attackSample(self, x, y, epsilonVolume=0, epsilonPGD=0, num_iter=1,
                        norm='inf', loss_fn=F.nll_loss, lower=0.2, upper=2):
        
        batch_size = x['audio'].size(0)
        clip_length = x['audio'].size(1)
        
        a = torch.ones(batch_size).unsqueeze(1).to(self.device)
        delta = torch.zeros(batch_size, clip_length).to(self.device)
        
        for _ in range(num_iter):
           
            a.requires_grad_()
            delta.requires_grad_()
            
            tmp_audio = x['audio']
            x['audio'] = (a * x['audio'] + delta).clamp(-1, 1)
            
            loss = loss_fn(self.model(x), y)
            self.model.zero_grad()
            loss.backward()
            
            x['audio'] = tmp_audio

            if norm == "inf":
                a = a + epsilonVolume * a.grad.sign()
                a = a.clamp(lower, upper).detach()
                
                delta = delta + epsilonPGD * delta.grad.sign()
                delta = delta.detach()
            else:
                a = a + epsilonVolume * a.grad/a.grad.norm(p=foat(norm), dim=[1]).unsqueeze(1)
                a = a.clamp(lower, upper).detach()
                delta = delta + epsilonPGD * delta.grad/delta.grad.norm(p=foat(norm), dim=[1]).unsqueeze(1)
                delta = delta.detach()
            
        x['audio'] = (a * x['audio'] + delta).clamp(-1, 1)
        return x