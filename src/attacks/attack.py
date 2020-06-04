from abc import ABC, abstractmethod

import torch
from tqdm import tqdm

class Attack(ABC):
    
    def __init__(self, model, data_loader, early_stopping=-1):
        self.model = model
        self.data_loader = data_loader
        self.early_stopping = early_stopping # -1=disabled 

        self.success = 0
        self.failed = 0
        self.total = 0
        self.adversarial_examples = []
        
    def attack(self, attack_parameters):
        assert data_loader.batch_size == 1

        for i, data in tqdm(list(enumerate(self.data_loader,0)), position=0):
            x, y_true = [x.cuda() for x in data]
            y_initial = self.predictClass(model, x)

            if y_initial != y_true:
                continue # we only attack correctly classified samples (TPs and TNs)  

            x_perturbed = self.attackSample(model, x, y_true, **attack_parameters)
            y_perturbed = self.predictClass(model, x_perturbed)

            self.evaluateAttack(i, y_perturbed, y_initial)
            
            if self.early_stopping <= self.success:
                print("Early stopping")
                return

    def evaluateAttack(self, i, y_perturbed, y_initial):
        self.total += 1
        
        if y_perturbed == y_initial:
            self.failed += 1
        else:
            self.success += 1
            self.adversarial_examples.append(i)
    
    def predictClass(self, x):
        self.model.eval().cuda()
        return self.torch.max(self.model(x).data, 1)[1]
    
    @abstractmethod
    def attackSample(self, x, target, **attack_parameters):
        pass # Implement attack in subclass

if __name__ == '__main__':
    pass