from abc import ABC, abstractmethod
import random

import IPython.display as ipd
from tqdm import tqdm
from copy import deepcopy
import torch

class Attack(ABC):
    
    def __init__(self, model, data_loader, attack_parameters, early_stopping=-1, device='cuda'):
        self.model = model
        self.data_loader = data_loader
        self.attack_parameters = attack_parameters
        self.early_stopping = early_stopping # -1=disabled 
        self.device = device

        self.success = 0
        self.failed = 0
        self.totalAttacked = 0
        self.totalProcessed = 0
        self.adversarial_examples = []
        
    def attack(self):
        assert self.totalProcessed == 0 # only attack once

        for i, batch in tqdm(list(enumerate(self.data_loader,0)), position=0):
            x = {k: batch[k].to(self.device) for k in batch}
            y_true = batch['label'].to(self.device) 

            y_initial = self.predictClass(x)

            self.totalProcessed += 1

            # we only attack correctly classified samples (TPs and TNs)  
            samples_to_attack = y_initial != y_true:

            x_to_perturb = {k: x[k][samples_to_attack] for k in x}
            x_to_perturb['audio'] = x['audio'].clone() # preserve original sample
            x_perturbed = self.attackSample(x_to_perturb, y_true, **self.attack_parameters)
            y_perturbed = self.predictClass(x_perturbed)

            self.evaluateAttack(i, x, x_perturbed, y_perturbed, y_initial)

            if self.early_stopping <= self.success and self.early_stopping > 0:
                print("Early stopping")
                return

    def evaluateAttack(self, i, x, x_perturbed, y_perturbed, y_initial):
        self.totalAttacked += 1
        
        if y_perturbed == y_initial:
            self.failed += 1
        else:
            self.success += 1
            adversarial_example = (i, y_initial.cpu(), y_perturbed.cpu(),
                                    {k: x[k].cpu() for k in x},
                                    {k: x[k].cpu() for k in x_perturbed})
            self.adversarial_examples.append(adversarial_example)

    def showAdversarialExample(self, target_class=0):
        allOfOneClass = [s for s in self.adversarial_examples if s[1]==target_class]
        if len(allOfOneClass) == 0:
            print("not enough adversarial samples for this class")
            return 
        random_sample = random.sample(allOfOneClass,1)[0]
        original = random_sample[-2]
        adversarial = random_sample[-1]
        ipd.display(ipd.Audio(original[0].cpu(),    rate=original[1].item(), normalize=True))
        ipd.display(ipd.Audio(adversarial[0].cpu(), rate=original[1].item(), normalize=False))
    
    def predictClass(self, x):
        self.model.eval().to(self.device)
        return torch.max(self.model(x).data, 1)[1]

    def report(self):
        print(f"Attack-Parameters:\t{self.attack_parameters}")
        print(f"Early stopping: \t{self.early_stopping > 0} ({self.early_stopping})\n")
        print(f"Successfully attacked:\t{self.success}")
        print(f"Total attacked: \t{self.totalAttacked}")
        print(f"Total processed:\t{self.totalProcessed}\n")
        print(f"Success-Rate: \t\t{round(self.getSuccessRate(), 2)}")
        print(f"Perturbed Accurracy: \t{round(self.getAccuracy(), 2)}")
    
    def getSuccessRate(self):
        assert self.totalAttacked > 0
        return self.success/float(self.totalAttacked)

    def getAccuracy(self):
        assert self.totalProcessed > 0
        # attack_failed = model still correct
        return self.failed/float(self.totalProcessed)
    
    def to(device='cuda'):
        self.device = device
        return self

    '''
        - expected to be fully vectorized
    '''
    @abstractmethod
    def attackSample(self, x, target, **attack_parameters):
        pass # Implement attack in subclass