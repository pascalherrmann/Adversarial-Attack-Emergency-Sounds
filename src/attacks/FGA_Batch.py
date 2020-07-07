import torch
import math
import numpy as np
import random

#
# Attack on whole batch
#
    
def scale_with_random_epsilon(vals, low=0.001, high=0.5):
    log_low = np.log10(low)
    log_high = np.log10(high)
    log_samples = np.random.uniform(log_low, log_high, [vals.shape[0],1])
    sampled = 10**log_samples
    
    result = torch.tensor(sampled).float().cuda() * vals
    return result

def project(l, vals):
    flattened = vals.view(vals.shape[0], -1)
    n = torch.norm(flattened, p=l, dim=1)
    normalized = flattened / n.unsqueeze(1) 
    return normalized.view(vals.shape)

''' use fga.py instead '''
def fast_gradient_attack(model, x, y, norm, epsilon=0.1, 
                         loss_fn=torch.nn.functional.cross_entropy):
    
    norm = str(norm)
    assert norm in ["1", "2", "inf"]

    # compute gradient w.r.t. x
    model.zero_grad() 
    x.requires_grad = True
    logits = model(x)  
    loss = loss_fn(logits, y)
    loss.backward()
    x.requires_grad = False
    model.zero_grad() 
    
    # project to unit ball according to specified norm
    if norm == "inf":
        normalized = x.grad.data.sign()
    else:
        normalized = project(int(norm), x.grad.data)
        
    # scale normalized according to specified epsilon
    if type(epsilon) == list:
        scaled = scale_with_random_epsilon(normalized, low=epsilon[0], high=epsilon[-1])
    else:
        scaled = normalized * epsilon
        
    # perturb
    x_pert = x + scaled
    
    # clip
    x_pert = torch.clamp(x_pert, -1, 1) 

    return x_pert.detach()