from attacks.attack import Attack

import torch.nn.functional as F
import torch 

class InterpolationAttack(Attack):
    '''
    Interpolation attack: 
        - take gradient wrt. the interpolation parameters a,b 
        
    Parameters:
        - tum_sound: sound to be inserted
        - epsilon: gradient step size
        - num_iterations: PGD iterations
        - clamping parameters lowerX, upperX (4 in total):
             define the max/min of interpolation volume
    '''
    def attackSample(self, x, y, overlay_sound, epsilon=0, num_iter=1,
                               lower1=0.8, upper1=1, lower2=0.05, upper2=0.1):
        raise Exception("Not vectorized - TODO")
        a = torch.tensor(1.0).cuda()  # original sound volume (alpha)
        b = torch.tensor(1.0).cuda()  # inserted sound volume (beta)
        overlay_sound = overlay_sound.cuda()

        for i in range(num_iter):
            a.requires_grad_()
            b.requires_grad_()

            x_pert = {'audio': a * x['audio'] + b * overlay_sound, 'sample_rate': x[1]}
            loss = F.nll_loss(self.model(x_pert), y)
            self.model.zero_grad()
            loss.backward()

            assert not torch.isnan(a.grad.data)
            assert not torch.isnan(b.grad.data)

            a = (a + epsilon * a.grad.data).clamp(lower1, upper1).detach()
            b = (b + epsilon * b.grad.data).clamp(lower2, upper2).detach()

            x_pert = {'audio': (a * x['audio'] + b * overlay_sound).clamp(-1, 1)}
            x_pert['sample_rate']: x['sample_rate']
        return x_pert