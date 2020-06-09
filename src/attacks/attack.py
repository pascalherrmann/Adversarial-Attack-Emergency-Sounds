from abc import ABC, abstractmethod
import random

import IPython.display as ipd
from tqdm import tqdm
import torch

class Attack(ABC):
    
    def __init__(self, model, data_loader, attack_parameters, early_stopping=-1):
        self.model = model
        self.data_loader = data_loader
        self.attack_parameters = attack_parameters
        self.early_stopping = early_stopping # -1=disabled 

        assert self.data_loader.batch_size == 1

        self.success = 0
        self.failed = 0
        self.totalAttacked = 0
        self.totalProcessed = 0
        self.adversarial_examples = []
        
    def attack(self):
        assert self.totalProcessed == 0 # only attack once

        for i, data in tqdm(list(enumerate(self.data_loader,0)), position=0):
            data = [item.cuda() for item in data] # move to gpu
            x, y_true = data[:2], data[-1]
            y_initial = self.predictClass(x)

            self.totalProcessed += 1
            if y_initial != y_true:
                continue # we only attack correctly classified samples (TPs and TNs)  

            x_perturbed = self.attackSample(x, y_true, **self.attack_parameters)
            y_perturbed = self.predictClass(x_perturbed)

            self.evaluateAttack(i, x, x_perturbed, y_perturbed, y_initial)

            if self.early_stopping >= self.success && self.early_stopping > 0:
                print("Early stopping")
                return

    def evaluateAttack(self, i, x, x_perturbed, y_perturbed, y_initial):
        self.totalAttacked += 1
        
        if y_perturbed == y_initial:
            self.failed += 1
        else:
            self.success += 1
            adversarial_example = (i, y_initial, y_perturbed, x, x_perturbed)
            self.adversarial_examples.append(adversarial_example)

    def showAdversarialExample(self, sr, target_class=0):
        allOfOneClass = [s for s in self.adversarial_examples if s[1]==target_class]
        if len(allOfOneClass) == 0:
            print("not enough adversarial samples for this class")
            return 
        random_sample = random.sample(allOfOneClass,1)[0]
        original = random_sample[-2].cpu()
        adversarial = random_sample[-1].cpu()
        ipd.display(ipd.Audio(original,    rate=sr, normalize=False))
        ipd.display(ipd.Audio(adversarial, rate=sr, normalize=False))
    
    def predictClass(self, x):
        self.model.eval().cuda()
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

    @abstractmethod
    def attackSample(self, x, target, **attack_parameters):
        pass # Implement attack in subclass

if __name__ == '__main__':
    pass