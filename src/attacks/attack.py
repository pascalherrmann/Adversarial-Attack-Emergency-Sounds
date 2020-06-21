from abc import ABC, abstractmethod
import random

import IPython.display as ipd
from tqdm import tqdm
from copy import deepcopy
import torch

class Attack(ABC):
    
    # TODO: refactor camelCase: attackSample -> attack_sample
    def __init__(self, model, data_loader,
                    attack_parameters, early_stopping=-1,
                    device='cuda', save_samples=True):
        self.model = model
        self.data_loader = data_loader
        self.attack_parameters = attack_parameters
        self.early_stopping = early_stopping # -1=disabled 
        self.device = device
        self.save_samples = save_samples

        self.success = 0
        self.failed = 0
        self.totalAttacked = 0
        self.totalProcessed = 0
        self.adversarial_examples = []
        
    def attack(self):
        assert self.totalProcessed == 0 # only attack once

        for i, batch in tqdm(list(enumerate(self.data_loader,0)), position=0):
            x = {k: batch[k].to(self.device) for k in batch if k != 'label'}
            y_true = batch['label'].to(self.device)

            y_initial = self.predictClass(x)
            self.totalProcessed += y_initial.size(0)

            # we only attack correctly classified samples (TPs and TNs)  
            samples_to_attack = (y_initial == y_true)
            if samples_to_attack.sum() == 0:
                continue # no correct classified sample in this batch
            assert (y_true[samples_to_attack] != y_initial[samples_to_attack]).sum() == 0
            x = {k: x[k][samples_to_attack] for k in x}
            y_initial = y_initial[samples_to_attack]
            y_true = y_true[samples_to_attack].to(self.device)

            # preserve original sample
            x_to_perturb = {k: x[k].clone().to(self.device) for k in x}

            # perform actual attack
            x_perturbed = self.attackSample(x_to_perturb, y_true, **self.attack_parameters)

            # evaluate attack
            y_perturbed = self.predictClass(x_perturbed)
            self.evaluateAttack(i, x, x_perturbed, y_perturbed, y_initial)

            if self.early_stopping <= self.success and self.early_stopping > 0:
                print("Early stopping")
                return

    def evaluateAttack(self, i, x, x_perturbed, y_perturbed, y_initial):
        self.totalAttacked += y_perturbed.size(0)
        
        self.failed += (y_perturbed == y_initial).sum().item()
        self.success += (y_perturbed != y_initial).sum().item()

        if not self.save_samples:
            return # don't save s

        batch_adversarial_examples = [ \
            ( # shift examples to cpu (otherwise dumps GPU)
                y_initial[i].cpu(),
                y_perturbed[i].cpu(),
                {k: x[k][i].cpu() for k in x},
                {k: x_perturbed[k][i].cpu() for k in x_perturbed}
            ) for i in range(y_perturbed.size(0)) \
              if (y_perturbed != y_initial)[i] \
            ]

        self.adversarial_examples.extend(batch_adversarial_examples)

    def showAdversarialExample(self, target_class=0):
        if not self.save_samples:
            raise Exception("you don't save adversarial examples currently")

        allOfOneClass = [s for s in self.adversarial_examples if s[0]==target_class]
        if len(allOfOneClass) == 0:
            print("not enough adversarial samples for this class")
            return 
        random_sample = random.sample(allOfOneClass,1)[0]
        original = random_sample[-2]
        adversarial = random_sample[-1]
        ipd.display(ipd.Audio(original['audio'],
                                rate=original['sample_rate'].item(),
                                normalize=False))
        ipd.display(ipd.Audio(adversarial['audio'],
                                rate=original['sample_rate'].item(),
                                normalize=False))
    
    def predictClass(self, x):
        self.model.eval().to(self.device)
        return torch.max(self.model(x).data, 1)[1]

    def report(self, log=True):
        if log:
            print(f"Attack-Parameters:\t{self.attack_parameters}")
            print(f"Early stopping: \t{self.early_stopping > 0} ({self.early_stopping})\n")
            print(f"Successfully attacked:\t{self.success}")
            print(f"Total attacked: \t{self.totalAttacked}")
            print(f"Total processed:\t{self.totalProcessed}\n")
            print(f"Success-Rate: \t\t{round(self.getSuccessRate(), 2)}")
            print(f"Perturbed Accurracy: \t{round(self.getAccuracy(), 2)}")
        return {"success_rate": self.getSuccessRate(), "acc": self.getAccuracy()}
    
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