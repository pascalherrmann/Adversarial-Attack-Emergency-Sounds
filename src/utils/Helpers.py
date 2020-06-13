import random
import math

def sample_dict_values(sample_dict):
    new_dict = {}
    for key in sample_dict.keys():
        if type(sample_dict[key]) == dict and "SAMPLING_MODE" in sample_dict[key]:
            low_log, up_log = math.log10(sample_dict[key]["l"]), math.log10(sample_dict[key]["u"])
            sample_log = random.uniform(low_log, up_log)
            new_dict[key] = 10 ** sample_log
        else: 
            new_dict[key] = sample_dict[key]
    return new_dict