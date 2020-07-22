from attacks.attack import Attack

import torch.nn.functional as F
import torch 

'''
    - In the Wiki, we call this Overlay attacks
    - Note the term interpolation is actually more accurate, since:
        - For alpha=1, beta=0: we get only the original sample
        - For alpha=0, beta=1: we get only the overlay
'''
class InterpolationAttack(Attack):
    '''
    Interpolation attack: 
        - take gradient wrt. the interpolation parameters a,b 
        
    Parameters:
        - overlay_sound: sound to be inserted
        - epsilon: gradient step size
        - num_iterations: PGD iterations
        - clamping parameters lowerX, upperX (4 in total):
             define the max/min of interpolation volume
    '''
    def attackSample(self, x, y, overlay_sound, epsilon=0, num_iter=1,
                               lower1=0.8, upper1=1, lower2=0.05, upper2=0.1):
        batch_size = x['audio'].size(0)
        a = torch.ones(batch_size).unsqueeze(1).to(self.device) # original sound volume (alpha)
        b = torch.ones(batch_size).unsqueeze(1).to(self.device) # inserted sound volume (beta)
        overlay_sound = overlay_sound.repeat(batch_size,1).to(self.device)

        for i in range(num_iter):
            a.requires_grad_()
            b.requires_grad_()

            tmp_audio = x['audio']
            
            x['audio'] = (a * x['audio'] + b * overlay_sound).clamp(-1,1)
            loss = F.nll_loss(self.model(x), y)
            self.model.zero_grad()
            loss.backward()

            # otherwise we would get powers of a (a^num_iter)
            x['audio'] = tmp_audio 

            a = (a + epsilon * a.grad.sign()).clamp(lower1, upper1).detach()
            b = (b + epsilon * b.grad.sign()).clamp(lower2, upper2).detach()

        x['audio'] = (a * x['audio'] + b * overlay_sound).clamp(-1, 1)
        return x 