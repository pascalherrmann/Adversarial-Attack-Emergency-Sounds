import random
import math

import numpy as np
import torch

def get_n_random_epsilons(N, low=0.001, high=0.5, mode = "log"):
    log_low = np.log10(low)
    log_high = np.log10(high)
    log_samples = np.random.uniform(log_low, log_high, [int(N),1])
    sampled = 10**log_samples
    
    result = torch.tensor(sampled).float()
    return result.cuda()


def sample(sampling_object, N=None):
    
    mode = sampling_object["SAMPLING_MODE"]
    assert mode in ["log", "log_batch", "u_batch", "choice"]
    
    if mode == "choice":
        return random.choice(sampling_object["options"])
    
    low, high = sampling_object["l"], sampling_object["u"]
    
    if mode == "log":
        low_log, up_log = math.log10(low), math.log10(high)
        sample_log = random.uniform(low_log, up_log)
        return 10 ** sample_log
    
    if mode == "log_batch":
        return get_n_random_epsilons(N, low, high)
    
    if mode == "u_batch":
        return torch.tensor(np.random.uniform(low, high, [int(N),1])).float().cuda()
    

def sample_dict_values(sample_dict, N_batch):
    new_dict = {}
    for key in sample_dict.keys():
        if type(sample_dict[key]) == dict and "SAMPLING_MODE" in sample_dict[key]:
            new_dict[key] = sample(sample_dict[key], N=N_batch)
        else: 
            new_dict[key] = sample_dict[key]
    return new_dict