import pytorch_lightning as pl
import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

class GeneralPLModule(pl.LightningModule):        
    #
    # these functions need to be overwritten:
    #
    
    # set your model
    def __init__(self, hparams):
        super().__init__()
        # set hyperparams
        self.hparams = hparams
        self.model = None
        self.attack_fn = False
        self.attack_args = {}

    # set self.dataset["train"] , self.dataset["val"] 
    '''
    def prepare_data(self):
        X_train, y_train, paths_train, X_val, y_val, paths_val = PrepareData.get_preprocessed_data()
        kwargs = {'num_workers': 1, 'pin_memory': True} if self.device == 'cuda' else {} #needed for using datasets on gpu
        self.dataset = {}
        self.dataset["train"] = EmergencyDataset(X_train, y_train, paths_train, **kwargs)
        self.dataset["val"] = EmergencyDataset(X_val, y_val, paths_val, **kwargs)
    '''
    
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
        avg_loss, acc = self.general_end(outputs, "val")
        print("Val-Acc={}".format(acc))
        tensorboard_logs = {'val_loss': avg_loss, 'val_acc': acc}
        return {'val_loss': avg_loss, 'val_acc': acc, 'log': tensorboard_logs} 
    
    def trainindg_end(self, outputs):
        avg_loss, acc = self.general_end(outputs, "train")
        print("Train-Acc={}".format(acc))
        tensorboard_logs = {'train_loss': avg_loss, 'train_acc': acc}
        return {'train_loss': avg_loss, 'train_acc': acc, 'log': tensorboard_logs} 
    
    def general_step(self, batch, batch_idx, mode):
        x, y = batch

        # load X, y to device!
        x, y = x.to(self.device), y.to(self.device)
        
        if mode == "train" and self.attack_fn: # create adversarial sample.
            x = self.attack_fn(self.model, x, y, **self.attack_args)

        # forward pass
        scores = self.model.forward(x) # should be of shape [batch_size, 2]

        # compute loss
        loss = F.cross_entropy(scores, y)

        # predictions
        preds = scores.argmax(axis=1) #[0] is max, [1] is arg max
        
        n_correct = preds.eq(y).sum()

        return loss, n_correct
    
    def training_step(self, batch, batch_idx):
        loss, n_correct = self.general_step(batch, batch_idx, "train")
        tensorboard_logs = {'loss': loss}
        return {'loss': loss, 'train_n_correct': n_correct, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        loss, n_correct = self.general_step(batch, batch_idx, "val")
        return {'val_loss': loss, 'val_n_correct': n_correct}
    
    @pl.data_loader
    def train_dataloader(self):
        return DataLoader(self.dataset["train"], shuffle=True, batch_size=self.hparams["batch_size"])

    @pl.data_loader
    def val_dataloader(self):
        return DataLoader(self.dataset["val"], batch_size=self.hparams["batch_size"])
    
    def forward(self, x):
        x = self.model(x)
        return F.log_softmax(x, dim = 2)

    def general_end(self, outputs, mode):
        # average over all batches aggregated during one epoch
        avg_loss = torch.stack([x[mode + '_loss'] for x in outputs]).mean()
        total_correct = torch.stack([x[mode + '_n_correct'] for x in outputs]).sum().cpu().numpy()
        acc = total_correct / len(self.dataset[mode])
        return avg_loss, acc