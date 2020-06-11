import torch
import math
import random

#
# Attack on whole batch
#
    
# sampling epsilon for adversarial training
def sample_epsilon(lo=1e-4, hi=1):
    log_lo = math.log10(lo)
    log_hi = math.log10(hi)
    log_sample = random.uniform(log_lo, log_hi)
    return 10**log_sample

''' use fga.py instead '''
@DeprecationWarning
def fast_gradient_attack(model, x, y, epsilon, norm,
                         loss_fn=torch.nn.functional.cross_entropy):
    
    norm = str(norm)
    assert norm in ["1", "2", "inf"]

    def project(l, vals):
        flattened = vals.view(vals.shape[0], -1)
        n = torch.norm(flattened, p=l, dim=1)
        normalized = flattened / n.unsqueeze(1) 
        return normalized.view(vals.shape)

    model.zero_grad() 
    x.requires_grad = True
    logits = model(x)  
    loss = loss_fn(logits, y)
    loss.backward()
    x.requires_grad = False
    model.zero_grad() 

    if norm == "inf":
        x_pert = x + x.grad.data.sign() * epsilon
    else:
        normalized = project(int(norm), x.grad.data)
        x_pert = x + normalized * epsilon

    x_pert = torch.clamp(x_pert, -1, 1) 

    return x_pert.detach()