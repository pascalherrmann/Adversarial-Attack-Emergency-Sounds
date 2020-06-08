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