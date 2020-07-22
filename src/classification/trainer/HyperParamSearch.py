from pytorch_lightning import Callback
import os
import pickle

class MetricsCallback(Callback):
    """PyTorch Lightning metric callback."""
    def __init__(self):
        super().__init__()
        self.metrics = []

    def on_validation_end(self, trainer, pl_module):
        self.metrics.append(trainer.callback_metrics)
        
def save_model(model, file_name, directory = "models"):
    model_dict = {"state_dict":model.state_dict(), "hparams": model.hparams}
    if not os.path.exists(directory):
        os.makedirs(directory)
    pickle.dump(model_dict, open(os.path.join(directory, file_name), 'wb', 4))
    

class SaveCallback(Callback): 
    def __init__(self, model_name = "newest_model", directory = "models"):
        super().__init__()
        self.model_name = model_name
        self.best_val_acc = None
        self.directory = directory
        
        if not os.path.exists(directory):
            print("created models directory!")
            os.makedirs(directory)

    def on_epoch_end(self, trainer, pl_module):
        if not self.best_val_acc or pl_module.val_results_history[-1]["val_acc"] > self.best_val_acc:
            print("new best val acc", pl_module.val_results_history[-1]["val_acc"])
            self.best_val_acc = pl_module.val_results_history[-1]["val_acc"]
            file_name = self.model_name + "_{:.4f}.p".format(self.best_val_acc)
            save_path = os.path.join(self.directory, file_name)
            pl_module.save(save_path, overwrite_if_exists=True)
            print("Saved checkpoint at epoch {} at \"{}\"".format((trainer.current_epoch + 1), save_path))