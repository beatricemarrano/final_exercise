import torch
#import wget
from torch.utils.data import Dataset
import numpy as np
import os

class CorruptMnist(Dataset):
    def __init__(self, train):
        path= '/Users/mac/Documents/GitHub/final_exercise/data/processed'
        if train:
            data = torch.load(os.path.join(path, "data_train.pkl"))
            targets = torch.load(os.path.join(path, "targets_train.pkl"))
        else:
            data = torch.load(os.path.join(path, "data_test.pkl"))
            targets = torch.load(os.path.join(path, "targets_test.pkl"))
            
        self.data = data
        self.targets = targets
    
    
    def __len__(self):
        return self.targets.numel()
    
    def __getitem__(self, idx):
        return self.data[idx].float(), self.targets[idx]


if __name__ == "__main__":
    dataset_train = CorruptMnist(train=True)
    dataset_test = CorruptMnist(train=False)
    print(dataset_train.data.shape)
    print(dataset_train.targets.shape)
    print(dataset_test.data.shape)
    print(dataset_test.targets.shape)