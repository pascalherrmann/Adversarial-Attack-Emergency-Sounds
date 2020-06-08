import os
import pickle
import torch
import torch.utils.data as data
from torchvision import transforms
from .custom_transforms import ToTensorAudio
import numpy as np


class Audioset(data.Dataset):
    """
    Input: 
        pickle_folder_path: Folder which has training.p and validation.p
        split_mode: "Training" or "Validation"
        transform: PyTorch transformations
    
    Output:
        audio: Raw audio signal on time domain
        sr: Sample rate
        label: 1 if labeled as siren, else 0

    Further explanation about how dataset is obtained
    https://wiki.tum.de/pages/viewpage.action?spaceKey=mllab&title=4%3A+Datasets
    """
    def __init__(self, 
                pickle_folder_path="/nfs/students/summer-term-2020/project-4/data/dataset1/dataset_resampled",
                split_mode='training',
                fixed_padding = True,
                transforms=transforms.Compose([ToTensorAudio()])):
        """
        Input:
            pickle_folder_path: str, folder which has `training.p` and `validation.p`
            split_mode='training': 'training' or 'validation'
            fixed_padding = Zero padding to self.max_length_sample which is the longest seq in training, validation dataset.

                            Pytorch RNN's use torch.rnn.pad_sequence and does padding based on max len of batch.
                            !! If you want to use RNN, set `fixed_padding=False` and set argument of dataloader as
                            `dataloader(collate_fn=dataset.pad_seq)`

            transforms: see custom_transforms.py to add new one
        """
        
        self.max_length_sample = 481489 # Will be used for padding
        self.pickle_path = pickle_folder_path
        if split_mode == 'training':
            self.pickle_path = os.path.join(pickle_folder_path, "training.p")
        elif split_mode == 'validation':
            self.pickle_path = os.path.join(pickle_folder_path, "validation.p")      
        self.fixed_padding = fixed_padding
        self.transforms = transforms
        
        self.whole_data = pickle.load(open(self.pickle_path, 'rb'))
        

    def __len__(self):
        return len(self.whole_data)

    def __getitem__(self, index):
        instance = self.whole_data[index]
        seqs, sample_rate = instance['data']
        label = instance['binary_class']
        label = 1 if label == 'positive' else 0
        data = [seqs, sample_rate]
        
        # TODO: Should it be after transforms?
        if self.fixed_padding:
            data[0] = np.pad(data[0], [0, self.max_length_sample - len(data[0])])
        
        
        if self.transforms is not None:
            # !! If sample rate is changed during transforms, make sure you return it
            audio, sr = self.transforms(data)
            return audio, sr, label

        return data[0], data[1], label # raw audio signal, sample rate, label

        

    def pad_seq(self, batch):
        """
        Needed for RNN like networks
        Get the batch and sort across samples according to length
        """
        # sort_ind should point to length
        sort_ind = 0
        sorted_batch = sorted(batch, key=lambda x: x[0].size(sort_ind), reverse=True)
        seqs, srs, labels = zip(*sorted_batch)
        
        lengths, srs, labels = map(torch.LongTensor, [[x.size(sort_ind) for x in seqs], srs, labels])

        # seqs_pad -> (batch, time, channel) 
        seqs_pad = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=0)
        #seqs_pad = seqs_pad_t.transpose(0,1)
        return seqs_pad, lengths, srs, labels
    
    

if __name__ == '__main__':
    pass