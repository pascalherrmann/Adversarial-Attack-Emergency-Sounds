from attacks.attack import Attack

import torch.nn.functional as F
import torch 

class FunctionalInterpolationAttack(Attack):
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
    def attackSample(self, x, y, overlay_sound, epsilonInterpolation=0, epsilonNoise,
                        num_iter=1, lower1=0.8, upper1=1, lower2=0.05, upper2=0.1):
        batch_size = x['audio'].size(0)
        overlay_sound = overlay_sound.repeat(batch_size,1).cuda()

        a = torch.ones(batch_size).unsqueeze(1).cuda() # original sound volume (alpha)
        b = torch.ones(batch_size).unsqueeze(1).cuda() # inserted sound volume (beta)
        delta = torch.zeros(batch_size, clip_length).to(self.device)

        for i in range(num_iter):
            a.requires_grad_()
            b.requires_grad_()
            delta.requires_grad_()

            tmp_audio = x['audio']
            
            x['audio'] = (a * x['audio'] + b * overlay_sound + delta).clamp(-1,1)
            loss = F.nll_loss(self.model(x), y)
            self.model.zero_grad()
            loss.backward()

            # otherwise we would get powers of a (a^num_iter)
            x['audio'] = tmp_audio 

            if norm == "inf":
                a = (a + epsilonInterpolation * a.grad.sign()).clamp(lower1, upper1).detach()
                b = (b + epsilonInterpolation * b.grad.sign()).clamp(lower2, upper2).detach()
                
                delta = delta + epsilonNoise * delta.grad.sign()
                delta = delta.detach()
            else:
                a = a + epsilonInterpolation * a.grad/a.grad.norm(p=foat(norm), dim=[1]).unsqueeze(1)
                a = a.clamp(lower1, upper1).detach()

                b = b + epsilonInterpolation * b.grad/b.grad.norm(p=foat(norm), dim=[1]).unsqueeze(1)
                b = b.clamp(lower2, upper2).detach()

                delta = delta + epsilonNoise * delta.grad/delta.grad.norm(p=foat(norm), dim=[1]).unsqueeze(1)
                delta = delta.detach()


        x['audio'] = (a * x['audio'] + b * overlay_sound + delta).clamp(-1, 1)
        return x 