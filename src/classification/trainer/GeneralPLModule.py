import pytorch_lightning as pl
import random
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from utils.Helpers import sample_dict_values

class GeneralPLModule(pl.LightningModule):        
    #
    # these functions need to be overwritten:
    #
    
    # set your model
    def __init__(self, hparams):
        super().__init__()
        # set hyperparams
        self.hparams = hparams
        self.attack = None
        self.model = None
        self.dataset = {}

    #
    # Optional
    #
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), 
                                     lr = self.hparams["learning_rate"], weight_decay = self.hparams["weight_decay"])
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=self.hparams["lr_decay"])
        
        return [optimizer], [scheduler]
    
    #
    # These functions do not need to be modified
    #
    def validation_end(self, outputs):
        avg_loss, acc = self.general_end(outputs, "validation")
        print("Val-Acc={}".format(acc))
        tensorboard_logs = {'val_loss': avg_loss, 'val_acc': acc}
        return {'val_loss': avg_loss, 'val_acc': acc, 'log': tensorboard_logs} 
    
    def trainindg_end(self, outputs):
        avg_loss, acc = self.general_end(outputs, "training")
        print("Train-Acc={}".format(acc))
        tensorboard_logs = {'train_loss': avg_loss, 'train_acc': acc}
        return {'train_loss': avg_loss, 'train_acc': acc, 'log': tensorboard_logs} 
    
    def general_step(self, batch, batch_idx, mode):
        x, y = batch, batch["label"]
            
        if mode == "training" and self.attack:
            x = self.attack.attackSample(x, y, **sample_dict_values(self.attack.attack_parameters))

        # forward pass
        scores = self.model.forward(x) # should be of shape [batch_size, 2]

        # compute loss
        loss = F.cross_entropy(scores, y)

        # predictions
        preds = scores.argmax(axis=1) #[0] is max, [1] is arg max
        
        n_correct = preds.eq(y).sum()

        return loss, n_correct
    
    def training_step(self, batch, batch_idx):
        loss, n_correct = self.general_step(batch, batch_idx, "training")
        tensorboard_logs = {'loss': loss}
        return {'loss': loss, 'train_n_correct': n_correct, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        loss, n_correct = self.general_step(batch, batch_idx, "validation")
        return {'val_loss': loss, 'val_n_correct': n_correct}
    

    
    def forward(self, x):
        x = self.model(x)
        return x#F.log_softmax(x, dim = 2)

    def general_end(self, outputs, mode):
        # average over all batches aggregated during one epoch
        avg_loss = torch.stack([x[mode + '_loss'] for x in outputs]).mean()
        total_correct = torch.stack([x[mode + '_n_correct'] for x in outputs]).sum().cpu().numpy()
        acc = total_correct / len(self.dataset[mode])
        return avg_loss, acc
    
    def save(self, path):
        torch.save( {"state_dict": self.model.state_dict(), "hparams": self.hparams, "attack_args": None if not self.attack else self.attack.attack_parameters}, path)
        print("Saved model to \"{}\"".format(path))
    
    #
    # convenience
    #
    
    def setAttack(self, attack_class, attack_args):
        self.attack = attack_class(self.model, self.val_dataloader(), attack_args, early_stopping=-1, device='cuda', save_samples=False)
    
    def report(self, loader=None, attack=None, attack_args=None, log=True):
        self.model.to(self.device)

        tp, fp, tn, fn, correct = 0, 0, 0, 0, 0
        if not loader: loader = self.val_dataloader()

        self.model.eval()

        for batch in loader:
            data, targets = batch, batch["label"]
            data = data.to(self.device)
            
            if attack:
                data = self.attack.attackSample(x = data, y = targets, **self.attack.attack_parameters)

            scores = self.model(data)

            preds = scores.argmax(axis=1).cpu()

            correct += preds.eq(targets).sum()

            with torch.no_grad():
                tp += torch.sum(preds & targets)
                tn += torch.sum((preds == 0) & (targets == 0))
                fp += torch.sum(((preds == 1) & (targets == 0)))
                fn += torch.sum((targets == 1) & (preds == 0))

        tp, fp, tn, fn, correct = tp.numpy(), fp.numpy(), tn.numpy(), fn.numpy(), correct.numpy()
        acc = (tp + tn) / (fp + fn + tp + tn)
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        f1 = 2*(prec*rec)/(prec+rec)
        p_rate = (tp+fp)/(tp+fp+tn+fn)

        if log:
            print("Accuracy: \t{:.2f}".format(acc))
            print('Precision: \t{:.2f}'.format(prec))
            print('Recall: \t{:.2f}'.format(rec))
            print('F1-Score: \t{:.2f}'.format(f1))
            print('\nVAL-ACC: {}/{} ({}%)\n'.format(correct, len(loader.dataset),
                100. * correct / len(loader.dataset)))
            print("P-Rate: \t{:.2f}".format(p_rate))
        
        return {"tp":tp, "fp":fp, "tn":tn, "fn":fn, "correct":correct, "n":len(loader.dataset), "acc":acc, "prec":prec, "rec":rec, "f1":f1, "attack_args":attack_args, "p_rate":p_rate}
        

    '''
     for dataset/data loading
    '''
    def set_dataset(self, split_mode, dataset):
        self.dataset[split_mode] = dataset
        
    # more general method for dataloader
    def get_dataloader(self, split_mode, **params):
        return DataLoader(self.datasets[split_mode], **params)

    @pl.data_loader
    def train_dataloader(self):
        return DataLoader(self.dataset["training"], shuffle=True, batch_size=self.hparams["batch_size"])

    @pl.data_loader
    def val_dataloader(self):
        return DataLoader(self.dataset["validation"], batch_size=self.hparams["batch_size"])