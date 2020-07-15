from .dataset import Dataset
from .EmergencyDataset import EmergencyDataset
import config

'''
    Model-specific dataset
        - This dataset class ensures that each (different) dataset is loaded at most once
        - Each model keeps reference to its specific dataset
        
    Model is required to implement two methods:
    
    def getDatasetInfo(self, split_mode)
        - returns information about dataset to be loaded
        
    def setDataset(self, dataset)
        - to assign the model a reference to its specific dataset

    Further, each model should define this method:

    def getDataLoader(self, split_mode, **params)
        - returns DataLoader 
'''
class DatasetHandler():
    
    def __init__(self):
        self.datasets = {}
    
    '''
        Load all datasets (train, valid, test) needed for a model.
        (Loads Emegency Dataset by default)
    '''
    def load_datasets(self, model, dataset_id=config.DATASET_EMERGENCY, old_data=False):
        self.load(model, split_mode='training', dataset_id=dataset_id, old_data=old_data)
        self.load(model, split_mode='validation', dataset_id=dataset_id, old_data=old_data)
        self.load(model, split_mode='testing', dataset_id=dataset_id, old_data=old_data)

    '''
        Load specific dataset.
    '''
    def load(self, model, split_mode='training', dataset_id=config.DATASET_EMERGENCY, old_data=False):
        dataset_type, dataset_params = model.dataset_info()
        dataset_params = {"split_mode": split_mode, **dataset_params}
        dataset_key = str((dataset_id,{ **dataset_type, **dataset_params}))#, dataset_id})
        
        if dataset_key in self.datasets:
            model.set_dataset(split_mode, self.datasets[dataset_key])
            return
        
        if old_data:
            self.datasets[dataset_key] = EmergencyDataset(split_mode=split_mode)
        else:
            self.datasets[dataset_key] = Dataset(dataset_id, **dataset_type, **dataset_params)
        model.set_dataset(split_mode, self.datasets[dataset_key])