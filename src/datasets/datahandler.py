from .audioset import Audioset
from .EmergencyDataset import EmergencyDataset

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
    
    def load(self, model, split_mode = 'training'):
        dataset_type, dataset_params = model.getDatasetInfo()
        dataset_params = {"split_mode": split_mode, **dataset_params}
        dataset_id = str({**dataset_type, **dataset_params})
        
        if dataset_id in self.datasets:
            model.setDataset(split_mode, self.datasets[dataset_id])
            return
            
        if dataset_type['sample_rate'] == 48000:
            dataset = Audioset(**dataset_params)
        elif dataset_type['sample_rate'] == 8000:
            dataset = EmergencyDataset(**dataset_params)

        self.datasets[dataset_id] = dataset
        model.setDataset(split_mode, dataset)